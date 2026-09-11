#!/usr/bin/env python3
"""test_scene_narrative.py - verify L1 scenes + movement ownership locally.

No model, no network, no host: a synthetic store exercising the cases from
plans/adaptive-scene-narrative.md, including the HOST recon samples (a stationary
person capture overlapping a moving motorcycle / a moving dog in a multi-object
scene). Run:

    python3 dev_scripts/test_scene_narrative.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "scenereader"))

import episodes      # noqa: E402
import scenes        # noqa: E402
import store         # noqa: E402

BASE = 1_700_000_000.0
PLACES = episodes.Places(camera_places={"cam01": "Gate", "cam04": "Field 1",
                                        "cam08": "Barn"})
FAILS = []


def check(label, condition, detail=""):
    print("{} {}".format("PASS" if condition else "FAIL", label))
    if not condition:
        FAILS.append(label)
        if detail:
            print("     -> {}".format(detail))


def add(conn, event_id, camera, label, start, end, disp, score=0.8):
    store.upsert_event(conn, {
        "frigate_event_id": event_id, "camera": camera, "label": label,
        "start_time": start, "end_time": end, "motion_disp": disp,
        "zones": "[]", "score": score,
    })


def main():
    tmp = tempfile.mkdtemp(prefix="scenetest-")
    conn = store.open_writer(os.path.join(tmp, "events.db"))

    # 1) the HOST recon sample: a stationary person whose captures overlap a
    #    MOVING motorcycle in the same camera (35 such cases exist on the host).
    add(conn, "cam01-a", "cam01", "person", BASE + 0, BASE + 4, 0.0)
    add(conn, "cam01-m", "cam01", "motorcycle", BASE + 1, BASE + 8, 0.30)
    add(conn, "cam01-b", "cam01", "person", BASE + 10, BASE + 14, 0.0)
    # a genuinely moving person elsewhere (their own trajectory moved)
    add(conn, "cam04-a", "cam04", "person", BASE + 200, BASE + 212, 0.40)
    # 2) a multi-object scene: the DOG moves, the person only stands there
    add(conn, "cam08-p", "cam08", "person", BASE + 100, BASE + 104, 0.0)
    add(conn, "cam08-d", "cam08", "dog", BASE + 100, BASE + 106, 0.40)
    conn.commit()

    # ---- L1 scenes + movement ownership ------------------------------------
    count = scenes.rebuild(conn, PLACES, store, gap_s=120, max_s=600,
                           move_min_disp=0.05, keyframes=4)
    check("three scenes built (cam01, cam04, cam08)", count == 3, "got {}".format(count))

    rows = {r["camera"]: r for r in conn.execute("SELECT * FROM scenes")}
    cam08 = dict(rows["cam08"])
    check("cam08 scene: dog is the mover",
          cam08["movers"] == "dog", "movers={!r}".format(cam08["movers"]))
    check("cam08 scene: person is PRESENT, not a mover (EN sentence)",
          "dog moved" in cam08["narrative"] and "person present" in cam08["narrative"],
          cam08["narrative"])
    check("cam08 scene: Arabic sentence names the dog's movement",
          "حركة" in cam08["narrative_ar"] and "كلب" in cam08["narrative_ar"],
          cam08["narrative_ar"])
    check("cam08 place resolves to Barn", cam08["place"] == "Barn", cam08["place"])

    cam04 = dict(rows["cam04"])
    check("cam04 scene: the moving person IS the mover",
          cam04["movers"] == "person" and "person moved" in cam04["narrative"],
          cam04["narrative"])

    cam01 = dict(rows["cam01"])
    check("cam01 scene: motorcycle is the only mover",
          cam01["movers"] == "motorcycle", "movers={!r}".format(cam01["movers"]))

    # ---- the EPISODE narrative must not invent a person journey ------------
    foreign = store.moving_events(conn, exclude_labels=["person"], min_disp=0.05)
    episodes.rebuild(conn, PLACES, store, labels=["person"], reid_max_gap_s=90,
                     episode_gap_s=600, move_min_disp=0.05, foreign_movers=foreign)
    eps = {r["anon_name"]: dict(r) for r in conn.execute("SELECT * FROM episodes")}
    gate = next(e for e in eps.values() if "Gate" in (e.get("narrative") or ""))
    check("stationary person @ Gate reads 'was present at', not 'entered'",
          "was present at Gate" in gate["narrative"]
          and "entered" not in gate["narrative"], gate["narrative"])
    check("the narrative NAMES the mover (motorcycle)",
          "while motorcycle moved" in gate["narrative"], gate["narrative"])
    check("Arabic episode uses the presence verbal noun (حضور)",
          "حضور" in (gate.get("narrative_ar") or ""), gate.get("narrative_ar"))
    # the truly-moving person keeps the normal journey wording
    journey = [e for e in eps.values() if "Field 1" in (e.get("narrative") or "")]
    check("a real mover keeps 'entered' wording",
          bool(journey) and "entered Field 1" in journey[0]["narrative"],
          journey[0]["narrative"] if journey else "(none)")

    # ---- grouping rules ----------------------------------------------------
    from scenes import group_scenes
    seq = [{"id": 1, "camera": "c", "start_time": 0, "end_time": 10},
           {"id": 2, "camera": "c", "start_time": 250, "end_time": 260}]
    check("gap 200s > 120s -> 2 scenes", len(group_scenes(seq, 120, 600)) == 2)
    check("gap 200s <= 300s -> 1 scene", len(group_scenes(seq, 300, 1000)) == 1)
    check("max_s cap splits one long burst",
          len(group_scenes([{"id": 1, "camera": "c", "start_time": 0, "end_time": 5},
                            {"id": 2, "camera": "c", "start_time": 6, "end_time": 40}],
                           120, 30)) == 2)
    check("a camera change always splits",
          len(group_scenes([{"id": 1, "camera": "a", "start_time": 0, "end_time": 5},
                            {"id": 2, "camera": "b", "start_time": 6, "end_time": 9}],
                           120, 600)) == 2)

    conn.close()
    print("\n{}".format("ALL PASSED" if not FAILS else "FAILURES: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
