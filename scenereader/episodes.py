

"""episodes.py - L1 places, L2 anonymous linking, L3 episodes, L4 narrative.

Turns the per-camera captures in `events` into the required story:

    "Person A entered Field 1 (cam04) at 09:02, stayed 12 min; then went to
     Store (cam07) at 09:20 with Person B; left at 09:35."

Layering (plans/event-scene-reader.md §4) - each layer is a pure function of the
one below, so any layer can be disabled without breaking the others:

  L1 `Places`            camera + Frigate zone -> a human place name, plus the
                         camera adjacency used by L2. Frigate owns the camera and
                         zone identifiers; this is only the translation.
  L2 `build_episodes`    links per-camera person tracks into one anonymous
                         person chain using an ended->started gap and adjacency.
  L3 ...                 derives the ordered `visits` (place, enter, leave,
                         duration) and the co-presence ("with Person B").
  L4 `compose_narrative` writes the deterministic sentence. No model, no
                         hallucination; an optional text LLM may rephrase later.

Everything is deterministic and rebuildable: `build_episodes` re-derives all
episodes from `events`, so changing places.conf or a gap costs no re-capture and
no model run.
"""
import datetime
import json
import time

# default labels that form a person episode
DEFAULT_LABELS = ("person",)

# Frigate label -> Arabic noun, for the Arabic narrative. A label with no entry
# keeps its English name, so the table can grow without breaking anything.
_LABEL_AR = {
    "person": "شخص", "people": "أشخاص", "dog": "كلب", "cat": "قط", "cow": "بقرة",
    "sheep": "خروف", "horse": "حصان", "bird": "طائر", "car": "سيارة",
    "truck": "شاحنة", "bus": "حافلة", "motorcycle": "دراجة نارية",
    "bicycle": "دراجة", "train": "قطار", "boat": "قارب", "fire": "حريق",
    "smoke": "دخان", "false_positive": "إنذار كاذب",
}


# ---------------------------------------------------------------------------
# L1 - places
# ---------------------------------------------------------------------------
def _kv(path):
    """Tiny KEY=VALUE reader (same shape as the other service .conf files)."""
    cfg = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                cfg[key.strip()] = value.strip()
    except OSError:
        return {}
    return cfg


def _pairs(value):
    """`a=1,b=2` -> {'a': '1', 'b': '2'} (keys/values lower-cased)."""
    out = {}
    for item in (value or "").split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, _, val = item.partition("=")
        if key.strip():
            out[key.strip().lower()] = val.strip()
    return out


def _route_pairs(value):
    """`a>b,c>d` -> {('a','b'), ('c','d')} (undirected: both orders stored)."""
    edges = set()
    for item in (value or "").split(","):
        item = item.strip()
        if ">" not in item:
            continue
        left, _, right = item.partition(">")
        left, right = left.strip().lower(), right.strip().lower()
        if left and right and left != right:
            edges.add((left, right))
            edges.add((right, left))
    return edges


class Places:
    """The human layer over Frigate's identifiers (config/places.conf)."""

    def __init__(self, camera_places=None, zone_places=None, adjacency=None,
                 coords=None, place_ar=None):
        # camera (lower) -> place
        self.camera_places = {k.lower(): v for k, v in (camera_places or {}).items()}
        # "camera.zone" (lower) -> place
        self.zone_places = {k.lower(): v for k, v in (zone_places or {}).items()}
        self.adjacency = set(adjacency or ())          # {("cam04","cam07"), ...}
        self.coords = coords or {}                     # place(lower) -> (x, y)
        # place (lower) -> Arabic place name, for the Arabic narrative
        self.place_ar = {k.lower(): v for k, v in (place_ar or {}).items()}

    @classmethod
    def load(cls, path):
        """Build from config/places.conf. A missing file = identity mapping."""
        cfg = _kv(path) if path else {}
        coords = {}
        for item in cfg.get("PLACE_COORDS", "").split(";"):
            item = item.strip()
            if ":" not in item:
                continue
            name, _, xy = item.partition(":")
            try:
                x, y = (float(p) for p in xy.split(",")[:2])
            except (TypeError, ValueError):
                continue
            coords[name.strip().lower()] = (x, y)
        return cls(_pairs(cfg.get("CAMERA_PLACES")),
                   _pairs(cfg.get("ZONE_PLACES")),
                   _route_pairs(cfg.get("ADJACENCY")),
                   coords,
                   _pairs(cfg.get("PLACE_AR")))

    # -- L1 lookups ---------------------------------------------------------
    def resolve(self, camera, zones=()):
        """Human place for a camera+zone hit. Zone beats camera; camera name last."""
        cam = (camera or "").strip().lower()
        if isinstance(zones, str):
            try:
                zones = json.loads(zones) if zones else []
            except (TypeError, ValueError):
                zones = []
        for zone in (zones or []):
            hit = self.zone_places.get(cam + "." + str(zone).strip().lower())
            if hit:
                return hit
        return self.camera_places.get(cam) or (camera or "").strip() or cam

    def adjacent(self, cam_a, cam_b, max_dist=0.0):
        """Are two cameras a plausible walking hop apart?

        Explicit ADJACENCY wins. If it is empty and PLACE_COORDS is supplied,
        fall back to a straight-line distance between the two resolved places.
        A camera is always adjacent to itself (same-place continuation).
        """
        a = (cam_a or "").strip().lower()
        b = (cam_b or "").strip().lower()
        if a == b:
            return True
        if (a, b) in self.adjacency:
            return True
        if not self.adjacency and self.coords and max_dist > 0:
            pa = self.coords.get(self.resolve(a).lower())
            pb = self.coords.get(self.resolve(b).lower())
            if pa and pb:
                return ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5 <= max_dist
        return False

    def unmapped(self, cameras):
        """Cameras with no CAMERA_PLACES entry (reported in reader_status.json)."""
        return [c for c in cameras if (c or "").strip().lower() not in self.camera_places]

    def place_name_ar(self, place):
        """Arabic name of a place, falling back to the English name verbatim.

        A missing translation is never an error - the Arabic sentence simply
        carries the English place name so the narrative stays readable.
        """
        text = str(place or "").strip()
        return self.place_ar.get(text.lower(), text)


# ---------------------------------------------------------------------------
# L2/L3 - episodes
# ---------------------------------------------------------------------------
def _anon_name(index):
    """0 -> 'Person A' ... 25 -> 'Person Z', 26 -> 'Person AA' (spreadsheet style)."""
    letters = ""
    n = int(index)
    while True:
        letters = chr(ord("A") + (n % 26)) + letters
        n = n // 26 - 1
        if n < 0:
            break
    return "Person " + letters


def _utc_day(epoch):
    return datetime.datetime.fromtimestamp(float(epoch), datetime.timezone.utc).strftime("%Y-%m-%d")


# Arabic alphabet order, used for anonymous person labels ("الشخص ا").
_AR_LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"


def _anon_name_ar(index):
    """0 -> 'الشخص ا' ... 27 -> 'الشخص ي', then 'الشخص ا2' (mirrors the Latin scheme)."""
    n = int(index)
    letter = _AR_LETTERS[n % len(_AR_LETTERS)]
    suffix = "" if n < len(_AR_LETTERS) else str(n // len(_AR_LETTERS) + 1)
    return "الشخص " + letter + suffix


def _anon_ar_from_en(anon_name):
    """'Person B' -> 'الشخص ب'; a real name (e.g. 'Ali' or 'علي') passes through."""
    text = str(anon_name or "").strip()
    if not text.lower().startswith("person "):
        return text
    index = 0
    for ch in text[7:].strip().upper():
        if "A" <= ch <= "Z":
            index = index * 26 + (ord(ch) - ord("A") + 1)
    return _anon_name_ar(max(0, index - 1))


def person_events(conn, labels=DEFAULT_LABELS, after=None, before=None):
    """Capture events that can form a person episode, oldest first."""
    labels = [str(x).strip().lower() for x in (labels or DEFAULT_LABELS) if str(x).strip()]
    if not labels:
        return []
    marks = ",".join("?" * len(labels))
    sql = ("SELECT id, camera, label, zones, start_time, end_time, description_attr,"
           " motion_disp FROM events WHERE label IN (" + marks + ")")
    params: list = list(labels)
    if after is not None:
        sql += " AND start_time >= ?"
        params.append(float(after))
    if before is not None:
        sql += " AND start_time <= ?"
        params.append(float(before))
    sql += " ORDER BY start_time ASC, id ASC"
    return conn.execute(sql, params).fetchall()


def _same_place(a, b):
    """Do two visits describe the same human place? (camera is the fallback.)

    Compared case-insensitively on the RESOLVED place, so a zone hit and a camera
    hit that name the same place still count as one place.
    """
    pa = str(a.get("place") or a.get("camera") or "").strip().lower()
    pb = str(b.get("place") or b.get("camera") or "").strip().lower()
    return bool(pa) and pa == pb


def _disp(row):
    """The event's OWN trajectory displacement (0.0 when unknown)."""
    try:
        return float(row["motion_disp"] or 0.0)
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0


def _mover_index(foreign_movers, min_disp):
    """camera (lower) -> sorted [(start, end, label)] of FOREIGN movers.

    `foreign_movers` are events whose own trajectory moved and whose label is not
    one the episode builder chains (e.g. dog / cow / truck while chaining
    `person`). They are the movement-ownership index: a stationary person visit
    overlapping one of these did not travel - something else moved.
    """
    index = {}
    threshold = float(min_disp or 0.0)
    for row in foreign_movers or ():
        try:
            disp = float(row["motion_disp"] or 0.0)
            camera = str(row["camera"] or "").strip().lower()
            label = str(row["label"] or "").strip().lower()
            start = float(row["start_time"] or 0)
            end = float(row["end_time"] or 0) or start
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if not camera or disp < threshold:
            continue
        index.setdefault(camera, []).append((start, end, label))
    for items in index.values():
        items.sort()
    return index


def _overlapping_movers(index, camera, start, end):
    """Labels of foreign movers whose window overlaps [start, end) in this camera."""
    labels = set()
    for m_start, m_end, label in index.get(str(camera or "").strip().lower(), ()):
        if m_start < end and start < m_end:
            labels.add(label)
    return sorted(labels)


def _mark_incidental(visit, index, min_disp):
    """Flag a NON-moving visit that overlaps a foreign mover as incidental.

    It stays a real presence - we never drop a person - but the person did not
    travel, so the composer says "was present at" instead of "entered/went to" and
    names the object that actually moved. This is what stops a moving dog/cow/
    truck from being narrated as a person's journey
    (plans/adaptive-scene-narrative.md section 4).
    """
    visit["moved"] = float(visit.get("disp") or 0.0) >= float(min_disp or 0.0)
    visit["foreign_movers"] = []
    visit["incidental"] = False
    if visit["moved"]:
        return
    movers = _overlapping_movers(index, visit.get("camera"),
                                 float(visit["enter_time"]),
                                 float(visit["leave_time"]))
    if movers:
        visit["foreign_movers"] = movers
        visit["incidental"] = True


def merge_visits(visits, gap_s=120.0):
    """Collapse CONSECUTIVE captures of one person at ONE place into one stay.

    WHY this exists: Frigate emits a NEW capture event every time detection
    re-triggers, so a person who stands in the gateway for three minutes arrives as
    five or six events. Narrating one clause per event produced:

        "Person FO entered Estraha (cam01) at 22:34, stayed 1 min; then returned to
         Estraha (cam01) at 22:35, stayed 1 min; then returned to Estraha ..."

    A stay is therefore ONE visit: `enter_time` = the first capture, `leave_time` =
    the last one, and `duration_s` = the **span** - never the sum, which would count
    a single presence several times. All captures are kept as `event_ids` (so the
    store still links each of them to the episode) and the LONGEST single capture
    becomes the representative `event_id`, i.e. the thumbnail for that stay.

    `gap_s` bounds the merge: a longer gap means the person really left and came
    back, which is its own visit and reads as "returned to ...".
    """
    out = []
    for visit in visits or []:
        enter = float(visit["enter_time"])
        leave = float(visit["leave_time"])
        current = out[-1] if out else None
        if current is not None and _same_place(current, visit):
            gap = enter - float(current["leave_time"])
            if 0.0 <= gap <= float(gap_s or 0.0):
                current["enter_time"] = min(float(current["enter_time"]), enter)
                current["leave_time"] = max(float(current["leave_time"]), leave)
                current["duration_s"] = max(
                    0.0, current["leave_time"] - current["enter_time"])
                current["event_ids"].append(visit["event_id"])
                current["n_events"] += 1
                current["disp"] = max(float(current.get("disp") or 0.0),
                                      float(visit.get("disp") or 0.0))
                current["_best"].append(
                    (max(0.0, leave - enter), visit["event_id"], visit["camera"]))
                _best_s, best_id, best_cam = max(current["_best"])
                current["event_id"] = best_id
                current["camera"] = best_cam
                continue
        out.append({
            "event_id": visit["event_id"],
            "camera": visit["camera"],
            "place": visit["place"],
            "enter_time": enter,
            "leave_time": leave,
            "duration_s": max(0.0, float(visit.get("duration_s") or (leave - enter))),
            # the strongest OWN movement seen in any capture of this stay
            "disp": float(visit.get("disp") or 0.0),
            "event_ids": [visit["event_id"]],
            "n_events": 1,
            "_best": [(max(0.0, leave - enter), visit["event_id"], visit["camera"])],
        })
    for visit in out:
        visit.pop("_best", None)
    return out


def build_episodes(rows, places, reid_max_gap_s=90.0, episode_gap_s=600.0,
                   visit_min_s=0.0, visit_merge_gap_s=120.0, move_min_disp=0.05,
                   foreign_movers=None):
    """Link person captures into anonymous cross-camera episodes.

    Deterministic and explainable: a capture joins an OPEN episode only when that
    episode's previous visit has already ENDED, the gap to it is within
    `reid_max_gap_s`, and the two cameras are adjacent. A still-open previous visit
    is never merged (two people in two places at once must not become one chain).

    Returns a list of dicts ordered by start time:
        {day, anon_name, label, start_time, end_time, link_confidence,
         visits: [{event_id, event_ids, n_events, camera, place, enter_time,
                   leave_time, duration_s, partners, disp, moved, incidental,
                   foreign_movers}],
         partners: [anon_name, ...]}

    `visit_merge_gap_s` is handed to `merge_visits`, which turns repeated captures
    of the same place into ONE stay (see its docstring) - without it the narrative
    repeats "returned to X" once per capture.

    MOVEMENT OWNERSHIP: `foreign_movers` (see `_mover_index`) marks a stay
    incidental when the person did not move but another object did - the case a
    moving dog in a crowded frame used to turn into a person's journey.
    """
    movers_index = _mover_index(foreign_movers, move_min_disp)
    open_eps = []   # list of dicts, each with 'visits'
    done = []
    for row in rows:
        camera = row["camera"]
        start = float(row["start_time"] or 0)
        end = float(row["end_time"]) if row["end_time"] else start
        place = places.resolve(camera, row["zones"])
        if end < start:
            end = start

        # close episodes that have gone quiet for longer than episode_gap_s
        still_open = []
        for ep in open_eps:
            if start - ep["last_end"] > float(episode_gap_s):
                done.append(ep)
            else:
                still_open.append(ep)
        open_eps = still_open

        best, best_gap = None, None
        for ep in open_eps:
            if ep["last_end"] > start:
                continue                       # previous visit still running -> not a link
            gap = start - ep["last_end"]
            if gap > float(reid_max_gap_s):
                continue
            same_cam = ep["last_camera"] == camera
            if not same_cam and not places.adjacent(ep["last_camera"], camera):
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = ep, gap

        if best is None:
            ep = {"label": row["label"], "start": start, "last_end": end,
                  "last_camera": camera, "confidence": None,
                  "visits": [{"event_id": row["id"], "camera": camera,
                              "place": place, "enter_time": start,
                              "leave_time": end,
                              "duration_s": max(0.0, end - start),
                              "disp": _disp(row)}]}
            open_eps.append(ep)
            continue

        best["visits"].append({"event_id": row["id"], "camera": camera,
                               "place": place, "enter_time": start,
                               "leave_time": end,
                               "duration_s": max(0.0, end - start),
                               "disp": _disp(row)})
        best["last_end"] = max(best["last_end"], end)
        best["last_camera"] = camera
        if best_gap and float(reid_max_gap_s) > 0:
            confidence = max(0.0, 1.0 - best_gap / float(reid_max_gap_s))
            best["confidence"] = confidence if best["confidence"] is None else min(
                best["confidence"], confidence)

    done.extend(open_eps)
    done.sort(key=lambda e: e["start"])

    # anonymous names, assigned by first sighting within each day
    counters = {}
    for ep in done:
        ep["day"] = _utc_day(ep["start"])
        idx = counters.get(ep["day"], 0)
        counters[ep["day"]] = idx + 1
        ep["anon_name"] = _anon_name(idx)
        # Collapse repeated captures of the same place BEFORE anything reads the
        # visit list: `end_time`, the confidence fallback below, the co-presence
        # overlap, the stored visits and the narrative all use the merged stays.
        ep["visits"] = merge_visits(ep["visits"], visit_merge_gap_s)
        ep["end_time"] = max(v["leave_time"] for v in ep["visits"])
        if ep["confidence"] is None:
            ep["confidence"] = 1.0 if len(ep["visits"]) == 1 else 0.5
        # Movement ownership, applied AFTER merging so it sees the whole stay
        # window (a stay is only incidental if NO capture in it showed own motion).
        for visit in ep["visits"]:
            _mark_incidental(visit, movers_index, move_min_disp)

    # drop episodes that never really appeared (configurable floor)
    if visit_min_s and float(visit_min_s) > 0:
        done = [e for e in done
                if (e["end_time"] - e["start"]) >= float(visit_min_s)
                or len(e["visits"]) > 1]

    _add_partners(done)
    return done


def _add_partners(episodes):
    """Co-presence: two episodes sharing a place with an OVERLAPPING window.

    Attached to the VISITS it covers, not only to the episode, so the narrative
    says "with Person B" on the clause where they actually shared the place instead
    of tacking every partner onto the final clause. The episode-level list is kept
    as well (it is the one the portal and older data carry).
    """
    for ep in episodes:
        ep["partners"] = []
        for visit in ep["visits"]:
            visit["partners"] = []
    for i, a in enumerate(episodes):
        for b in episodes[i + 1:]:
            # episodes are time-sorted: once b starts after a ends, no later one overlaps
            if b["start"] >= a["end_time"]:
                break
            for visit_a in a["visits"]:
                for visit_b in b["visits"]:
                    if not _same_place(visit_a, visit_b):
                        continue
                    overlap = min(float(visit_a["leave_time"]),
                                  float(visit_b["leave_time"])) - max(
                        float(visit_a["enter_time"]), float(visit_b["enter_time"]))
                    if overlap < 0:
                        continue          # touching windows do not count as "with"
                    if b["anon_name"] not in visit_a["partners"]:
                        visit_a["partners"].append(b["anon_name"])
                    if a["anon_name"] not in visit_b["partners"]:
                        visit_b["partners"].append(a["anon_name"])
            if any(v["partners"] for v in a["visits"]):
                a["partners"] = sorted({p for v in a["visits"] for p in v["partners"]})
                b["partners"] = sorted({p for v in b["visits"] for p in v["partners"]})


def write_episodes(conn, episodes, store):
    """Persist a build result: episodes + visits (events.episode_id) in one go.

    A merged visit writes ONE row (the representative capture, whose thumbnail
    stands for the whole stay) but links EVERY capture of that stay to the episode,
    so `events.episode_id` stays complete without inflating the visit list the
    portal renders.
    """
    store.clear_episodes(conn)
    for ep in episodes:
        eid = store.insert_episode(conn, ep["day"], ep["anon_name"], ep["label"],
                                   ep["start"], ep["end_time"],
                                   link_confidence=ep.get("confidence"))
        for seq, visit in enumerate(ep["visits"]):
            store.add_visit(conn, eid, visit["event_id"], seq, visit["place"],
                            visit["enter_time"], visit["leave_time"],
                            visit["duration_s"],
                            event_ids=visit.get("event_ids"))
    return len(episodes)


# ---------------------------------------------------------------------------
# L4 - narrative
# ---------------------------------------------------------------------------
def _clock(epoch, tz_offset_h=0.0):
    dt = datetime.datetime.fromtimestamp(float(epoch), datetime.timezone.utc)
    if tz_offset_h:
        dt = dt + datetime.timedelta(hours=float(tz_offset_h))
    return dt.strftime("%H:%M")


# A stay shorter than this is not described in minutes: "stayed 1 min" for a 20 s
# capture was one of the things that made the narrative read as nonsense.
SHORT_STAY_S = 45.0


def _minutes(seconds):
    return max(1, int(round(float(seconds) / 60.0)))


def _duration_en(seconds):
    """'under a minute' below SHORT_STAY_S, else rounded minutes."""
    secs = max(0.0, float(seconds))
    if secs < SHORT_STAY_S:
        return "under a minute"
    return "{} min".format(_minutes(secs))


def _minutes_ar(seconds):
    """Arabic duration phrase, honouring its plural rules.

    3-10 take the plural (دقائق) while 11+ take the singular (دقيقة); 1 and 2 have
    their own forms. Getting this wrong reads as broken Arabic to a native
    speaker, so it is done explicitly rather than with a naive format().
    """
    if max(0.0, float(seconds)) < SHORT_STAY_S:
        return "أقل من دقيقة"
    n = _minutes(seconds)
    if n == 1:
        return "دقيقة واحدة"
    if n == 2:
        return "دقيقتان"
    if 3 <= n <= 10:
        return "{} دقائق".format(n)
    return "{} دقيقة".format(n)


def compose_narrative(episode, aliases=None, tz_offset_h=0.0):
    """One deterministic sentence over the ordered visits. No model involved.

    Example: "Person A entered Field 1 (cam04) at 09:02, stayed 12 min; then went
    to Store (cam07) at 09:20 with Person B; left at 09:35."
    """
    aliases = aliases or {}
    name = aliases.get(episode["anon_name"]) or episode["anon_name"]
    visits = episode.get("visits") or []
    if not visits:
        return ""
    parts = []
    seen_places = set()
    for idx, visit in enumerate(visits):
        place = visit["place"] or visit["camera"]
        clock = _clock(visit["enter_time"], tz_offset_h)
        # The subject is named once; later visits read as "then went to ...",
        # matching how the required sentence is spoken ("... then go to the store").
        # "returned to" is kept for a place met EARLIER: after `merge_visits` a
        # continuous stay is a single visit, so a repeat here means the person
        # really left that place and came back.
        #
        # MOVEMENT OWNERSHIP: an INCIDENTAL visit is a non-moving presence whose
        # window overlaps a foreign mover (a dog/cow/truck), so it must NOT read as
        # a journey - "was present at" states the fact and names the mover
        # (plans/adaptive-scene-narrative.md section 4).
        if idx == 0:
            if visit.get("incidental"):
                segment = "{} was present at {} ({}) at {}".format(
                    name, place, visit["camera"], clock)
            else:
                segment = "{} entered {} ({}) at {}".format(
                    name, place, visit["camera"], clock)
        elif visit.get("incidental"):
            segment = "then was present at {} ({}) at {}".format(
                place, visit["camera"], clock)
        elif str(place).strip().lower() in seen_places:
            segment = "then returned to {} ({}) at {}".format(
                place, visit["camera"], clock)
        else:
            segment = "then went to {} ({}) at {}".format(
                place, visit["camera"], clock)
        if visit.get("duration_s") is not None:
            segment += ", stayed {}".format(_duration_en(visit["duration_s"]))
        partners = [aliases.get(p) or p for p in (visit.get("partners") or [])]
        if partners:
            segment += " with " + ", ".join(partners)
        if visit.get("foreign_movers"):
            segment += " (while {} moved)".format(
                ", ".join(str(m) for m in visit["foreign_movers"]))
        parts.append(segment)
        seen_places.add(str(place).strip().lower())
    # A partner known only at EPISODE level (data built before per-visit partners,
    # or a co-presence that merged away) still has to be surfaced: the last clause
    # is where it reads best.
    if not any(v.get("partners") for v in visits):
        partners = [aliases.get(p) or p for p in (episode.get("partners") or [])]
        if partners:
            parts[-1] += " with " + ", ".join(partners)
    # a trailing "left at HH:MM" only reads well for a single-visit episode
    if len(visits) == 1:
        text = parts[0] + ", left at {}.".format(
            _clock(visits[-1]["leave_time"], tz_offset_h))
    else:
        text = "; ".join(parts) + "; left at {}.".format(
            _clock(visits[-1]["leave_time"], tz_offset_h))
    # capitalise the leading name the way the sentence reads naturally
    return (text[0].upper() + text[1:]) if text else text


def compose_narrative_ar(episode, aliases=None, tz_offset_h=0.0, places=None):
    """The Arabic narrative, from the SAME structured facts as the English one.

    Deliberately built with verbal nouns ("وصول ... ثم الانتقال ...") rather than
    gendered verbs, so it reads correctly for a person of any gender and needs no
    model. Example:

        "وصول الشخص ا إلى الحقل 1 (cam04) في 22:36، مدة البقاء 12 دقيقة؛
         ثم الانتقال إلى المخزن (cam07) في 22:49 برفقة الشخص ب؛ المغادرة في 22:51."
    """
    aliases = aliases or {}
    visits = episode.get("visits") or []
    if not visits:
        return ""
    name = aliases.get(episode["anon_name"]) or _anon_ar_from_en(episode["anon_name"])

    def place_text(visit):
        raw = visit["place"] or visit["camera"]
        return places.place_name_ar(raw) if places is not None else raw

    parts = []
    seen_places = set()
    for idx, visit in enumerate(visits):
        clock = _clock(visit["enter_time"], tz_offset_h)
        raw_place = str(visit["place"] or visit["camera"]).strip().lower()
        # An INCIDENTAL (non-moving) visit uses the verbal noun "حضور" instead of
        # "وصول/الانتقال", so it never reads as travel (see the English composer).
        if idx == 0:
            if visit.get("incidental"):
                segment = "حضور {} في {} ({}) في {}".format(
                    name, place_text(visit), visit["camera"], clock)
            else:
                segment = "وصول {} إلى {} ({}) في {}".format(
                    name, place_text(visit), visit["camera"], clock)
        elif visit.get("incidental"):
            segment = "ثم الحضور في {} ({}) في {}".format(
                place_text(visit), visit["camera"], clock)
        elif raw_place in seen_places:
            # a place met EARLIER = a real return (العودة), not "moving on"
            segment = "ثم العودة إلى {} ({}) في {}".format(
                place_text(visit), visit["camera"], clock)
        else:
            segment = "ثم الانتقال إلى {} ({}) في {}".format(
                place_text(visit), visit["camera"], clock)
        if visit.get("duration_s") is not None:
            segment += "، مدة البقاء {}".format(_minutes_ar(visit["duration_s"]))
        partners = [_anon_ar_from_en(aliases.get(p) or p)
                    for p in (visit.get("partners") or [])]
        if partners:
            segment += " برفقة " + "، ".join(partners)
        if visit.get("foreign_movers"):
            segment += " بينما كانت حركة {}".format(
                "، ".join(_LABEL_AR.get(str(m).lower(), str(m))
                          for m in visit["foreign_movers"]))
        parts.append(segment)
        seen_places.add(raw_place)
    # episode-level partners (older data, or a co-presence merged away) last
    if not any(v.get("partners") for v in visits):
        partners = [aliases.get(p) or _anon_ar_from_en(p)
                    for p in (episode.get("partners") or [])]
        if partners:
            parts[-1] += " برفقة " + "، ".join(partners)
    return "؛ ".join(parts) + "؛ المغادرة في {}.".format(
        _clock(visits[-1]["leave_time"], tz_offset_h))


def rebuild(conn, places, store, labels=DEFAULT_LABELS, reid_max_gap_s=90.0,
            episode_gap_s=600.0, visit_min_s=0.0, tz_offset_h=0.0,
            visit_merge_gap_s=120.0, move_min_disp=0.05, foreign_movers=None):
    """Re-derive every episode from `events`, write it, and return the count.

    Each episode gets BOTH narratives (English + Arabic) from the same facts, so
    the portal can show either without a second pass or any model.
    """
    rows = person_events(conn, labels=labels)
    episodes = build_episodes(rows, places, reid_max_gap_s=reid_max_gap_s,
                              episode_gap_s=episode_gap_s, visit_min_s=visit_min_s,
                              visit_merge_gap_s=visit_merge_gap_s,
                              move_min_disp=move_min_disp,
                              foreign_movers=foreign_movers)
    write_episodes(conn, episodes, store)
    for ep in episodes:
        aliases = store.get_aliases(conn, ep["day"])
        store.set_narratives(conn, ep["day"], ep["anon_name"],
                             compose_narrative(ep, aliases, tz_offset_h),
                             compose_narrative_ar(ep, aliases, tz_offset_h, places))
    conn.commit()
    return len(episodes)
