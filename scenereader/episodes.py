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
                 coords=None):
        # camera (lower) -> place
        self.camera_places = {k.lower(): v for k, v in (camera_places or {}).items()}
        # "camera.zone" (lower) -> place
        self.zone_places = {k.lower(): v for k, v in (zone_places or {}).items()}
        self.adjacency = set(adjacency or ())          # {("cam04","cam07"), ...}
        self.coords = coords or {}                     # place(lower) -> (x, y)

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
                   coords)

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


def person_events(conn, labels=DEFAULT_LABELS, after=None, before=None):
    """Capture events that can form a person episode, oldest first."""
    labels = [str(x).strip().lower() for x in (labels or DEFAULT_LABELS) if str(x).strip()]
    if not labels:
        return []
    marks = ",".join("?" * len(labels))
    sql = ("SELECT id, camera, label, zones, start_time, end_time, description_attr"
           " FROM events WHERE label IN (" + marks + ")")
    params: list = list(labels)
    if after is not None:
        sql += " AND start_time >= ?"
        params.append(float(after))
    if before is not None:
        sql += " AND start_time <= ?"
        params.append(float(before))
    sql += " ORDER BY start_time ASC, id ASC"
    return conn.execute(sql, params).fetchall()


def build_episodes(rows, places, reid_max_gap_s=90.0, episode_gap_s=600.0,
                   visit_min_s=0.0):
    """Link person captures into anonymous cross-camera episodes.

    Deterministic and explainable: a capture joins an OPEN episode only when that
    episode's previous visit has already ENDED, the gap to it is within
    `reid_max_gap_s`, and the two cameras are adjacent. A still-open previous visit
    is never merged (two people in two places at once must not become one chain).

    Returns a list of dicts ordered by start time:
        {day, anon_name, label, start_time, end_time, link_confidence,
         visits: [{event_id, camera, place, enter_time, leave_time, duration_s}],
         partners: [anon_name, ...]}
    """
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
                              "leave_time": end, "duration_s": max(0.0, end - start)}]}
            open_eps.append(ep)
            continue

        best["visits"].append({"event_id": row["id"], "camera": camera,
                               "place": place, "enter_time": start,
                               "leave_time": end, "duration_s": max(0.0, end - start)})
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
        ep["end_time"] = max(v["leave_time"] for v in ep["visits"])
        if ep["confidence"] is None:
            ep["confidence"] = 1.0 if len(ep["visits"]) == 1 else 0.5

    # drop episodes that never really appeared (configurable floor)
    if visit_min_s and float(visit_min_s) > 0:
        done = [e for e in done
                if (e["end_time"] - e["start"]) >= float(visit_min_s)
                or len(e["visits"]) > 1]

    _add_partners(done)
    return done


def _add_partners(episodes):
    """Co-presence: two episodes overlapping in time at a shared place."""
    for ep in episodes:
        ep["partners"] = []
    for i, a in enumerate(episodes):
        for b in episodes[i + 1:]:
            # episodes are time-sorted: once b starts after a ends, no later one overlaps
            if b["start"] >= a["end_time"]:
                break
            places_a = {v["place"] for v in a["visits"]}
            if any(v["place"] in places_a for v in b["visits"]):
                a["partners"].append(b["anon_name"])
                b["partners"].append(a["anon_name"])


def write_episodes(conn, episodes, store):
    """Persist a build result: episodes + visits (events.episode_id) in one go."""
    store.clear_episodes(conn)
    for ep in episodes:
        eid = store.insert_episode(conn, ep["day"], ep["anon_name"], ep["label"],
                                   ep["start"], ep["end_time"],
                                   link_confidence=ep.get("confidence"))
        for seq, visit in enumerate(ep["visits"]):
            store.add_visit(conn, eid, visit["event_id"], seq, visit["place"],
                            visit["enter_time"], visit["leave_time"],
                            visit["duration_s"])
    return len(episodes)


# ---------------------------------------------------------------------------
# L4 - narrative
# ---------------------------------------------------------------------------
def _clock(epoch, tz_offset_h=0.0):
    dt = datetime.datetime.fromtimestamp(float(epoch), datetime.timezone.utc)
    if tz_offset_h:
        dt = dt + datetime.timedelta(hours=float(tz_offset_h))
    return dt.strftime("%H:%M")


def _minutes(seconds):
    return max(1, int(round(float(seconds) / 60.0)))


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
    prev_place = None
    for idx, visit in enumerate(visits):
        place = visit["place"] or visit["camera"]
        clock = _clock(visit["enter_time"], tz_offset_h)
        # The subject is named once; later visits read as "then went to ...",
        # matching how the required sentence is spoken ("... then go to the store").
        if idx == 0:
            segment = "{} entered {} ({}) at {}".format(
                name, place, visit["camera"], clock)
        else:
            verb = "returned to" if place == prev_place else "went to"
            segment = "then {} {} ({}) at {}".format(verb, place, visit["camera"], clock)
        if visit.get("duration_s") is not None:
            segment += ", stayed {} min".format(_minutes(visit["duration_s"]))
        parts.append(segment)
        prev_place = place
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


def rebuild(conn, places, store, labels=DEFAULT_LABELS, reid_max_gap_s=90.0,
            episode_gap_s=600.0, visit_min_s=0.0, tz_offset_h=0.0):
    """Re-derive every episode from `events`, write it, and return the count."""
    rows = person_events(conn, labels=labels)
    episodes = build_episodes(rows, places, reid_max_gap_s=reid_max_gap_s,
                              episode_gap_s=episode_gap_s, visit_min_s=visit_min_s)
    write_episodes(conn, episodes, store)
    # narrate, honouring the day-scoped aliases
    for ep in episodes:
        aliases = store.get_aliases(conn, ep["day"])
        conn.execute("UPDATE episodes SET narrative = ?, updated_at = ?"
                     " WHERE day = ? AND anon_name = ?",
                     (compose_narrative(ep, aliases, tz_offset_h), time.time(),
                      ep["day"], ep["anon_name"]))
    conn.commit()
    return len(episodes)
