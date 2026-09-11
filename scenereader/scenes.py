"""scenes.py - L1 adaptive scenes + L2 camera story (deterministic, no model).

The unit the narrator was missing. Frigate gives per-OBJECT captures, so today a
scene holding a person, a dog, a cow and a truck arrives as four unrelated event
streams and the episode builder reads only the `person` one. When ANY of them
moves, detection re-fires and the co-present person gets fresh captures, which
the narrative then turns into a confident journey. That is an error AMPLIFIED by
composition (plans/adaptive-scene-narrative.md section 1).

This module adds the level between a capture and a camera story:

  L1 SCENE    the maximal run of events in ONE camera whose consecutive
              inactivity gap is <= `gap_s`, capped at `max_s`. A scene knows the
              objects that COEXISTED and - from each object's OWN trajectory
              (Frigate `data.path_data`, see `frigate.path_motion`) - which of
              them actually MOVED.
  L2 CAMERA STORY  the ordered scenes of one camera, each with place, window and
              the EN/AR sentence.

MOVEMENT OWNERSHIP is the point. An object is a mover only if its own
displacement >= `move_min_disp`; every other object in the scene is `present`.
A moving dog therefore produces "dog moved; person, cow present" instead of a
person journey. All of it is a pure function of `events`, so it is rebuildable
with no re-capture and no model run.
"""
import datetime
import json

import episodes

# The Arabic label table lives in episodes.py (`_LABEL_AR`) and is shared here, so
# the scene sentence and the episode sentence can never drift apart.


def _g(row, key, default=None):
    """Read a field from a mapping OR a sqlite3.Row (Row has no .get())."""
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _utc_day(epoch):
    return datetime.datetime.fromtimestamp(
        float(epoch), datetime.timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# L1 - adaptive grouping
# ---------------------------------------------------------------------------
def group_scenes(rows, gap_s=120.0, max_s=600.0):
    """Split `rows` (ordered by camera, start_time) into scenes. Pure.

    A scene grows while each new event starts within `gap_s` of the previous
    event's end; it closes on a longer inactivity gap, when the camera changes, or
    when it would exceed `max_s` (a hard cap so a permanently-busy pasture does
    not become one all-day scene). Input order is trusted - `store.all_events`
    supplies it.
    """
    gap_s = max(0.0, float(gap_s))
    max_s = max(0.0, float(max_s))
    scenes = []
    current = None
    for row in rows:
        camera = str(_g(row, "camera", "") or "")
        start = float(_g(row, "start_time", 0) or 0)
        end = float(_g(row, "end_time", 0) or 0) or start
        if end < start:
            end = start
        if current is not None:
            inactive = start - current["last_end"]
            span = max(end, current["end_time"]) - current["start_time"]
            if (current["camera"] != camera or inactive > gap_s
                    or (max_s > 0 and span > max_s)):
                scenes.append(current)
                current = None
        if current is None:
            current = {"camera": camera, "start_time": start, "end_time": end,
                       "last_end": end, "rows": [row]}
        else:
            current["end_time"] = max(current["end_time"], end)
            current["last_end"] = max(current["last_end"], end)
            current["rows"].append(row)
    if current is not None:
        scenes.append(current)
    return scenes


def _scene_place(places, camera, rows):
    """The scene's place: the most common resolved place across its captures."""
    counts = {}
    for row in rows:
        try:
            place = places.resolve(camera, _g(row, "zones"))
        except Exception:  # noqa: BLE001 - a missing/odd places map must not fail
            place = camera
        counts[place] = counts.get(place, 0) + 1
    if not counts:
        return camera
    return max(counts, key=lambda name: counts[name])


def _pick_keyframes(rows, k):
    """Up to `k` event ids spread evenly across the scene (for a later montage)."""
    if k is None or int(k) <= 0 or not rows:
        return []
    ordered = sorted(rows, key=lambda r: float(_g(r, "start_time", 0) or 0))
    k = int(k)
    if len(ordered) <= k:
        return [int(_g(r, "id", 0)) for r in ordered]
    if k == 1:
        picks = [0]
    else:
        step = (len(ordered) - 1) / float(k - 1)
        picks = sorted({int(round(i * step)) for i in range(k)})
    return [int(_g(ordered[i], "id", 0)) for i in picks]


def build_scenes(rows, places, gap_s=120.0, max_s=600.0, move_min_disp=0.05,
                 keyframes=4):
    """Group `rows` into scenes and resolve MOVEMENT OWNERSHIP per scene.

    Returns dicts ordered by (camera, start):
        {camera, place, day, start_time, end_time, duration_s, n_events,
         objects: [{label, n, disp, best_score, moved}],  # movers first
         movers: [label], stationary: [label], event_ids: [...],
         rep_event_id, key_event_ids}
    """
    out = []
    for sc in group_scenes(rows, gap_s, max_s):
        camera = sc["camera"]
        objects = {}
        event_ids = []
        for row in sc["rows"]:
            label = (str(_g(row, "label", "") or "").strip().lower()) or "object"
            disp = float(_g(row, "motion_disp", 0) or 0)
            score = _g(row, "score")
            if score is None:
                score = _g(row, "top_score")
            slot = objects.setdefault(
                label, {"label": label, "n": 0, "disp": 0.0, "best_score": None})
            slot["n"] += 1
            slot["disp"] = max(slot["disp"], disp)
            if isinstance(score, (int, float)):
                best = float(score)
                slot["best_score"] = best if slot["best_score"] is None \
                    else max(slot["best_score"], best)
            event_ids.append(int(_g(row, "id", 0)))

        # Movement ownership: only an object whose OWN trajectory moved is a mover.
        for slot in objects.values():
            slot["moved"] = slot["disp"] >= float(move_min_disp)
        objects_list = sorted(objects.values(),
                              key=lambda o: (not o["moved"],
                                             -(o["best_score"] or 0.0), o["label"]))
        movers = [o["label"] for o in objects_list if o["moved"]]
        stationary = [o["label"] for o in objects_list if not o["moved"]]

        rep = max(sc["rows"], key=lambda r: (
            (float(_g(r, "end_time", 0) or 0) or float(_g(r, "start_time", 0) or 0))
            - float(_g(r, "start_time", 0) or 0), int(_g(r, "id", 0))))

        out.append({
            "camera": camera,
            "place": _scene_place(places, camera, sc["rows"]),
            "day": _utc_day(sc["start_time"]),
            "start_time": sc["start_time"],
            "end_time": sc["end_time"],
            "duration_s": max(0.0, sc["end_time"] - sc["start_time"]),
            "n_events": len(sc["rows"]),
            "objects": objects_list,
            "movers": movers,
            "stationary": stationary,
            "event_ids": event_ids,
            "rep_event_id": int(_g(rep, "id", 0)),
            "key_event_ids": _pick_keyframes(sc["rows"], keyframes),
        })
    return out


# ---------------------------------------------------------------------------
# L2 - the camera story
# ---------------------------------------------------------------------------
def _join_en(labels):
    """'a' / 'a and b' / 'a, b and c'."""
    items = [str(x) for x in labels if str(x).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _join_ar(labels):
    """'أ' / 'أ وب' / 'أ، ب وج'."""
    items = [str(x) for x in labels if str(x).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return "، ".join(items[:-1]) + " و" + items[-1]


def _label_ar(label):
    return episodes._LABEL_AR.get(str(label).strip().lower(), str(label))


def _head(camera, place, clock, duration):
    """'cam04 (Field 1) 09:02, 12 min' - the place is omitted when unmapped."""
    head = str(camera)
    if place and str(place).strip().lower() != str(camera).strip().lower():
        head += " (" + str(place) + ")"
    return "{} {}, {}".format(head, clock, duration)


def compose_scene_narrative(scene, tz_offset_h=0.0):
    """One deterministic English sentence for a scene. No model, no invention.

    Examples:
        cam08 (Barn) 09:02, 2 min: dog moved; person, cow present.
        cam04 (Field 1) 09:02, 12 min: person and dog moved; cow present.
        cam04 (Field 1) 09:02, 1 min: person present.
    """
    clock = episodes._clock(scene["start_time"], tz_offset_h)
    duration = episodes._duration_en(scene["duration_s"])
    clauses = []
    if scene.get("movers"):
        clauses.append("{} moved".format(_join_en(scene["movers"])))
    if scene.get("stationary"):
        clauses.append("{} present".format(_join_en(scene["stationary"])))
    if not clauses:
        clauses.append("nothing tracked")
    return "{}: {}.".format(
        _head(scene["camera"], scene.get("place"), clock, duration),
        "; ".join(clauses))


def compose_scene_narrative_ar(scene, tz_offset_h=0.0, places=None):
    """The Arabic scene sentence, from the SAME facts (verbal nouns, no model)."""
    clock = episodes._clock(scene["start_time"], tz_offset_h)
    duration = episodes._minutes_ar(scene["duration_s"])
    place = scene.get("place") or scene.get("camera")
    place_txt = places.place_name_ar(place) if places is not None else place
    clauses = []
    if scene.get("movers"):
        clauses.append("حركة " + _join_ar([_label_ar(x) for x in scene["movers"]]))
    if scene.get("stationary"):
        clauses.append("حضور " + _join_ar([_label_ar(x) for x in scene["stationary"]]))
    if not clauses:
        clauses.append("لا كائنات مرصودة")
    return "{}: {}.".format(
        _head(scene["camera"], place_txt, clock, duration), "؛ ".join(clauses))


def write_scenes(conn, scene_list, store, tz_offset_h=0.0, places=None):
    """Persist a build result: scenes + their member captures, in one go.

    Rebuild-safe: `store.clear_scenes` drops the derived rows first, and every
    insert is keyed/idempotent, so re-running after a `SCENE_GAP_S` change costs
    nothing but CPU.
    """
    store.clear_scenes(conn)
    for sc in scene_list:
        scene_id = store.insert_scene(
            conn, sc["camera"], sc["day"], sc["place"], sc["start_time"],
            sc["end_time"], sc["duration_s"], sc["n_events"],
            json.dumps(sc["objects"], ensure_ascii=False), ",".join(sc["movers"]),
            sc["rep_event_id"])
        for event_id in sc["event_ids"]:
            store.link_scene_event(conn, scene_id, event_id)
        store.set_scene_narratives(
            conn, scene_id,
            compose_scene_narrative(sc, tz_offset_h),
            compose_scene_narrative_ar(sc, tz_offset_h, places))
    return len(scene_list)


def rebuild(conn, places, store, gap_s=120.0, max_s=600.0, move_min_disp=0.05,
            keyframes=4, tz_offset_h=0.0):
    """Re-derive every scene from `events`, write it, and return the count.

    Each scene gets BOTH narratives from the same facts, exactly like an episode,
    so the portal shows either without a second pass or a model run.
    """
    rows = store.all_events(conn)
    scene_list = build_scenes(rows, places, gap_s=gap_s, max_s=max_s,
                              move_min_disp=move_min_disp, keyframes=keyframes)
    write_scenes(conn, scene_list, store, tz_offset_h=tz_offset_h, places=places)
    conn.commit()
    return len(scene_list)
