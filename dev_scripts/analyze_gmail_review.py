#!/usr/bin/env python3
"""analyze_gmail_review.py - scenario analysis of the fire-alert screenshot review.

Reads the user-verdict review table (media/Gmail/fire_alerts_scores.csv) produced by
dev_scripts/score_gmail_screenshots.py and analyses it under TWO deployment scenarios,
because the checkpoint that raised an alert is NOT always the active v4:

  * Scenario A ("v1-only", repo-consistent): the ACTIVE model was v1 for the whole
    2026-09-06/07 alert window (v1 go-live 09-05; v4 promoted 09-07 20:12; v2 never
    promoted in git/models/fire/VERSIONS.md). All 28 "printed" scores are v1's.
  * Scenario B ("v1->v2->v4", user recollection): v2 ran live for part of the window.
    v2 has no promotion commit, so the exact boundary is unknown; this script marks
    the temporally-plausible cut (v2 takes over after the 06/09 daytime FP burst) as
    a HYPOTHESIS to be confirmed against host firewatch logs. V4's own re-score on the
    CCTV-frame crops is the "would the active model fire on this frame today" column.

Per alert it also derives:
  cluster      - observable time/content grouping, no model assumption
  v4_fire_gate - would the deployed v4 fire-only gate (>=0.50) alert today?
  v4_any_gate  - would v4 fire-OR-smoke gate (>=0.50) alert today?
  scn_model    - running checkpoint attributed under each scenario

Writes media/Gmail/fire_alerts_analysis.csv (one row per alert) and prints a
summary. Reproducible sibling of score_gmail_screenshots.py.
"""
import csv
import os
from collections import Counter

SRC_CSV = os.path.join("media", "Gmail", "fire_alerts_scores.csv")
OUT_CSV = os.path.join("media", "Gmail", "fire_alerts_analysis.csv")

GATE = 0.50  # firewatch SCORE_THRESHOLD / alert gate (config/firewatch.conf)


def parse_printed(raw):
    """'fire 0.94' -> float; unknown -> None."""
    if not raw:
        return None
    s = raw.lower().replace("fire", "").replace("smoke", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def classify_cluster(ts, note, verdict):
    """Observable time/content grouping (no model assumption). Rows whose burned-in
    clock could not be OCR'd are inferred from verdict + note so they join the
    obvious event bursts."""
    def tod_mins(t):
        try:
            hh, mm = t.split(":")[:2]
            return int(hh) * 60 + int(mm)
        except Exception:
            return None

    d = None
    mins = None
    if ts:
        try:
            date_part, _, tod = ts.replace("/", "-").partition(" ")
            d = date_part
            mins = tod_mins(tod)
        except Exception:
            d = None
    note = (note or "").lower()
    if d is None:
        # burned-in clock not readable - infer from the verdict/note
        if verdict == "TRUE":
            return "07/09 evening TRUE-fire burst (ts unreadable)"
        return "06/09 FP burst (ts unreadable)"
    if d.startswith("06-09"):
        if mins is not None and 16 * 60 <= mins <= 18 * 60:
            return "06/09 afternoon FP burst (sunset/sand)"
        if mins is not None and mins >= 19 * 60:
            return "06/09 night palm-leaf FP burst"
        return "06/09 FP burst (other)"
    if d.startswith("07-09"):
        if mins is not None and mins <= 12 * 60:
            return "07/09 early-morning FP burst"
        if mins is not None and 18 * 60 + 30 <= mins <= 21 * 60 + 30:
            return "07/09 evening TRUE-fire burst"
        return "07/09 other"
    return "unknown"


# Scenario B hypothesis: which checkpoint fired each cluster. "*" = unconfirmed.
SCN_B = {
    "06/09 afternoon FP burst (sunset/sand)": "v1 (documented daytime sun/warm-object FPs)",
    "06/09 FP burst (ts unreadable)": "v1*",
    "06/09 night palm-leaf FP burst": "v2* (night palm-leaf FPs)",
    "06/09 FP burst (other)": "v1/v2*",
    "07/09 early-morning FP burst": "v2*",
    "07/09 evening TRUE-fire burst": "v2* (fires v2 caught)",
    "07/09 evening TRUE-fire burst (ts unreadable)": "v2* (fires v2 caught)",
    "07/09 other": "v2/v4*",
    "unknown": "?",
}


def scn_b_model(cluster):
    return SCN_B.get(cluster, "?")


def main():
    rows = []
    with open(SRC_CSV, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd)
        for r in rd:
            if not r or not r[0]:
                continue
            rows.append(r)

    out = []
    for r in rows:
        file, ts, raw_print, v4f, v4s, verdict = r[0], r[1], r[2], r[3], r[4], r[5]
        note = r[6] if len(r) > 6 else ""
        printed = parse_printed(raw_print)
        v4f = float(v4f or 0)
        v4s = float(v4s or 0)
        cluster = classify_cluster(ts, note, verdict)
        out.append({
            "file": file, "cctv_ts": ts,
            "printed": "" if printed is None else f"{printed:.2f}",
            "v4_fire": f"{v4f:.3f}", "v4_smoke": f"{v4s:.3f}",
            "verdict": verdict, "note": note, "cluster": cluster,
            "v4_fire_gate": "ALERT" if v4f >= GATE else "no",
            "v4_any_gate": "ALERT" if (v4f >= GATE or v4s >= GATE) else "no",
            "scenarioA_model": "v1 (v1 ACTIVE 09-05..09-07 20:12 - all alerts)",
            "scenarioB_model": scn_b_model(cluster),
        })

    # ---- summary stats -------------------------------------------------
    def g(verdict):
        return [o for o in out if o["verdict"] == verdict]

    tru, fal = g("TRUE"), g("FALSE")

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"loaded {len(out)} alerts from {SRC_CSV}\n")
    print("== head-to-head: running model (printed) vs ACTIVE v4 re-score ==")
    print(f"  running model fired {len(out)} alerts: {len(tru)} TRUE / {len(fal)} FALSE "
          f"(alert precision {len(tru)}/{len(out)} = {100*len(tru)/len(out):.0f}%)")
    print(f"    printed conf:   TRUE avg {avg([float(x['printed']) for x in tru if x['printed']]):.3f}"
          f"  FALSE avg {avg([float(x['printed']) for x in fal if x['printed']]):.3f}")
    print(f"    v4 fire re-score: TRUE avg {avg([float(x['v4_fire']) for x in tru]):.3f}"
          f" (max {max(float(x['v4_fire']) for x in tru):.3f})  "
          f"FALSE avg {avg([float(x['v4_fire']) for x in fal]):.3f}")
    print(f"    v4 smoke re-score: TRUE avg {avg([float(x['v4_smoke']) for x in tru]):.3f}  "
          f"FALSE avg {avg([float(x['v4_smoke']) for x in fal]):.3f}")
    n_fg = sum(1 for o in out if o["v4_fire_gate"] == "ALERT")
    n_ag = sum(1 for o in out if o["v4_any_gate"] == "ALERT")
    print(f"  v4 fire-only gate (as deployed): would alert {n_fg}/{len(out)} "
          f"(TRUE {sum(1 for o in tru if o['v4_fire_gate']=='ALERT')})")
    print(f"  v4 fire|smoke gate (if TRACK_SMOKE=true): would alert {n_ag}/{len(out)} "
          f"(TRUE {sum(1 for o in tru if o['v4_any_gate']=='ALERT')})")

    print("\n== cluster breakdown (observable, model-agnostic) ==")
    for cl, n in Counter(o["cluster"] for o in out).most_common():
        t = sum(1 for o in out if o["cluster"] == cl and o["verdict"] == "TRUE")
        print(f"  {cl:<45} n={n:>2}  TRUE={t} FALSE={n-t}")

    print("\n== scenario A (all alerts = v1): precision "
          f"{len(tru)}/{len(out)} = {100*len(tru)/len(out):.0f}% ==")
    print("== scenario B (v1->v2->v4 HYPOTHESIS; boundary after the 06/09 "
          "afternoon burst; * = unconfirmed) ==")
    for model in ("v1", "v2", "v1/v2", "v2/v4", "?"):
        grp = [o for o in out if o["scenarioB_model"].split(" (")[0].rstrip("*") == model]
        if grp:
            t = sum(1 for o in grp if o["verdict"] == "TRUE")
            print(f"    {model:<6} alerts={len(grp):>2}  TRUE={t} FALSE={len(grp)-t}"
                  + ("   <-- v1 the whole window" if model == "v1" else ""))

    print("\n== v4 false positives that would PERSIST if smoke tracking were enabled ==")
    for o in out:
        if o["v4_fire_gate"] == "no" and o["v4_any_gate"] == "ALERT":
            print(f"    {o['file'][-12:-4]} {o['verdict']:<5} v4_smoke={o['v4_smoke']} "
                  f"| {o['note']}")

    keys = ["file", "cctv_ts", "printed", "v4_fire", "v4_smoke", "verdict",
            "note", "cluster", "v4_fire_gate", "v4_any_gate",
            "scenarioA_model", "scenarioB_model"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for o in out:
            wr.writerow({k: o[k] for k in keys})
    print(f"\nwrote {len(out)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
