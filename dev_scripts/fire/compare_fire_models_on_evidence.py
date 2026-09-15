#!/usr/bin/env python3
"""compare_fire_models_on_evidence.py - score ANY fire models on the labelled evidence.

WHAT IT COMPARES
----------------
The photo collection downloaded by dev_scripts/deploy/pull_firewatch_evidence.sh
(./firewatch-evidence/) with the user's manual labels:

  * POSITIVES = every JPEG in <evidence>/true-positives/   (a real fire is visible)
  * NEGATIVES = every OTHER evidence JPEG in <evidence>/<cam>/ - i.e. the frames
    firewatch stored that are NOT a fire (warm-object / IR / reflection misses).

The positives were COPIED out of the store, so <cam>/ still contains them: the
negatives are "manifest raw images minus the positive basenames" (an exact,
byte-identical split - see the copy check in the plan doc).

Every `*_annotated.jpg` (firewatch's box-overlay twin) is EXCLUDED - both as an
input and as an output - per the user's instruction: the models must be judged on
the ORIGINAL frames only, never on an image that already has boxes drawn on it.

METRIC SEMANTICS (production, not mAP)
--------------------------------------
firewatch tracks FIRE only (`TRACK_SMOKE=false`) and stores/alerts on the fire
class, so the decision under test is image-level:

    "does the model put at least one FIRE box above the threshold in this frame?"

Thresholds come from config/firewatch.conf:
  * SCORE_THRESHOLD = 0.50  (the alert + evidence bar)
  * SCORE_HIGH      = 0.85  (the bar for a fire with NO nearby motion)

Reported per model: positive image-recall and negative image-FP-rate at both
bars, the max-fire-confidence distributions, precision/recall treating the
49 positives vs 151 negatives as a binary classification, a per-camera FP
breakdown, and (V4 -> V6) a transition matrix of the individual flips:
fixed / new-FP / lost-recall, with the exact filenames in differences.csv.

The filename itself encodes the confidence the ORIGINAL model reported when it
stored the frame (`..._conf0.58.jpg`), so the re-measured V4 score is also
checked against that stored value as a sanity check on this harness.

Usage:
  .venv/bin/python dev_scripts/fire/compare_fire_models_on_evidence.py
  .venv/bin/python dev_scripts/fire/compare_fire_models_on_evidence.py \
      --models v4=models/fire/best.pt v6=models/fire/versions/v6-.../model.pt

Options:
  --evidence DIR      pulled collection            [./firewatch-evidence]
  --pos-subdir NAME   positives subdir            [true-positives]
  --models N=P ...    models to compare, in order [v4=models/fire/best.pt,
                                                   newest versions/v6-*/model.pt]
                      Any checkpoint whose classes include one named "fire" works
                      (its index is treated as fire; non-3-class layouts are noted).
  --conf-floor F      predict floor (box recall)  [0.02 - low on purpose, so the
                      threshold sweep can see sub-threshold boxes]
  --fire-thresh F     alert/evidence bar          [0.50]
  --high-thresh F     no-motion bar               [0.85]
  --imgsz N           inference size              [640]
  --device D          cpu|0                       [cpu]
  --out DIR           output dir                  [model-training/firewatch_eval/run-<utc>]
  --annotate          ALSO write box overlays (never reads *_annotated.jpg)
"""
import argparse
import csv
import datetime
import json
import os
import re
import statistics
import sys
from typing import Any, cast

CONF_RE = re.compile(r"_conf([0-9]+(?:\.[0-9]+)?)\.jpe?g$", re.I)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def is_annotated(name):
    """True for firewatch's box-overlay twin - never an input, never an output."""
    return name.lower().endswith("_annotated.jpg") or name.lower().endswith("_annotated.jpeg")


def utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# collection discovery
# ---------------------------------------------------------------------------
def collect(evidence, pos_subdir):
    """Return (positives, negatives, info) as lists of dicts.

    negatives come from manifest.csv (the container-written index) so the split
    is exact: manifest RAW images minus the positive basenames.  Falls back to a
    directory walk when manifest.csv is absent.
    """
    pos_dir = os.path.join(evidence, pos_subdir)
    if not os.path.isdir(pos_dir):
        sys.exit("positives dir not found: %s" % pos_dir)

    pos = []
    for name in sorted(os.listdir(pos_dir)):
        p = os.path.join(pos_dir, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in IMG_EXT \
                and not is_annotated(name):
            pos.append({"path": p, "name": name, "camera": camera_of(name),
                        "store_conf": store_conf(name), "label": 1})
    pos_names = {r["name"] for r in pos}

    neg = []
    manifest = os.path.join(evidence, "manifest.csv")
    if os.path.isfile(manifest):
        with open(manifest, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("kind") != "raw":
                    continue
                name = os.path.basename(row["file"])
                if name in pos_names or is_annotated(name):
                    continue
                path = os.path.join(evidence, row["file"])
                if not os.path.isfile(path):
                    continue
                neg.append({"path": path, "name": name,
                            "camera": row.get("camera") or camera_of(name),
                            "store_conf": store_conf(name), "label": 0,
                            "in_db": row.get("in_db")})
    else:
        for cam in sorted(os.listdir(evidence)):
            d = os.path.join(evidence, cam)
            if not os.path.isdir(d) or cam == pos_subdir:
                continue
            for name in sorted(os.listdir(d)):
                if name in pos_names or is_annotated(name):
                    continue
                if os.path.splitext(name)[1].lower() not in IMG_EXT:
                    continue
                neg.append({"path": os.path.join(d, name), "name": name,
                            "camera": cam, "store_conf": store_conf(name), "label": 0})

    info = {"positives": len(pos), "negatives": len(neg), "pos_dir": pos_dir}
    if not pos:
        sys.exit("no positive images found in %s" % pos_dir)
    return pos, neg, info


def camera_of(name):
    m = re.search(r"_(cam[0-9]+)_", name)
    return m.group(1) if m else "?"


def store_conf(name):
    m = CONF_RE.search(name)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def discover_models(specs, root):
    """Return [(tag, path)] - from --models or the v4/v6 defaults."""
    if specs:
        out = []
        for s in specs:
            if "=" not in s:
                sys.exit("--models wants NAME=PATH, got %r" % s)
            tag, path = s.split("=", 1)
            if not os.path.isfile(path):
                sys.exit("model not found: %s" % path)
            out.append((tag, path))
        return out

    out = []
    v4 = os.path.join(root, "models", "fire", "best.pt")
    if os.path.isfile(v4):
        out.append(("v4", v4))
    vers = os.path.join(root, "models", "fire", "versions")
    if os.path.isdir(vers):
        cands = []
        for d in sorted(os.listdir(vers)):
            if not d.startswith("v6"):
                continue
            p = os.path.join(vers, d, "model.pt")
            if os.path.isfile(p):
                cands.append((d, p))
        if cands:
            out.append(("v6", cands[-1][1]))
    if len(out) < 2:
        sys.exit("could not resolve 2 models; pass --models NAME=PATH NAME2=PATH2")
    return out


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
def run_model(tag, path, pos, neg, args, out_dir):
    from ultralytics import YOLO
    model = YOLO(path)
    names = model.names
    fire_idx = next((i for i, n in names.items() if str(n).strip().lower() == "fire"), None)
    smoke_idx = next((i for i, n in names.items() if str(n).strip().lower() == "smoke"), None)
    if fire_idx is None:
        sys.exit("model %s has NO 'fire' class: %s" % (tag, names))
    print("  %-6s %s" % (tag, path))
    if {names.get(0), names.get(1), names.get(2)} != {"fire", "other", "smoke"}:
        print("         NOTE: non-production layout %s -> class %d is treated as fire"
              % (names, fire_idx))
    else:
        print("         classes 0=%s 1=%s 2=%s" % (names[0], names[1], names[2]))

    ann_dir = None
    if args.annotate:
        ann_dir = os.path.join(out_dir, "annotated", tag)
        os.makedirs(ann_dir, exist_ok=True)

    rows = []
    todo = pos + neg
    for k, item in enumerate(todo, 1):
        # ultralytics types predict() as Iterator[Results | Tensor]; cast away that
        # union (we always pass a single image path, so result 0 is a Results).
        results = cast(Any, model.predict(item["path"], conf=args.conf_floor,
                                          imgsz=args.imgsz, device=args.device,
                                          verbose=False))
        r = results[0]
        n_fire = n_smoke = n_other = 0
        mx_fire = mx_smoke = 0.0
        boxes = []
        if len(r.boxes):
            for cls, cf, xy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                   r.boxes.xyxy.tolist()):
                ci = int(cls)
                boxes.append((ci, float(cf), [float(v) for v in xy]))
                if ci == fire_idx:
                    n_fire += 1
                    mx_fire = max(mx_fire, float(cf))
                elif smoke_idx is not None and ci == smoke_idx:
                    n_smoke += 1
                    mx_smoke = max(mx_smoke, float(cf))
                else:
                    n_other += 1
        rows.append({
            "image": item["name"], "label": item["label"], "camera": item["camera"],
            "store_conf": item["store_conf"], "max_fire": round(mx_fire, 4),
            "max_smoke": round(mx_smoke, 4), "n_fire": n_fire, "n_smoke": n_smoke,
            "n_other": n_other,
        })
        if ann_dir is not None:
            _annotate(item["path"], boxes, model.names,
                      os.path.join(ann_dir, item["name"]))
        if k % 25 == 0:
            print("    %d/%d" % (k, len(todo)))

    with open(os.path.join(out_dir, "per_image_%s.csv" % tag), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return {r["image"]: r for r in rows}


def _annotate(img_path, boxes, names, out_path):
    """Optional visual: draw the model's boxes (input is never an *_annotated.jpg)."""
    from PIL import Image, ImageDraw
    im = Image.open(img_path).convert("RGB")
    dr = ImageDraw.Draw(im)
    for ci, cf, xy in boxes:
        color = (255, 0, 0) if ci == 0 else (255, 165, 0) if ci == 2 else (0, 128, 255)
        dr.rectangle(xy, outline=color, width=3)
        dr.text((xy[0], max(0, xy[1] - 12)), "%s %.2f" % (names.get(ci, ci), cf), fill=color)
    im.save(out_path, quality=85)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def dist(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": round(statistics.fmean(vals), 4),
            "median": round(statistics.median(vals), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def metrics_for(rows, thr, high):
    """rows: {image: row} - image-level fire-only metrics at the two bars."""
    pos = [r for r in rows.values() if r["label"] == 1]
    neg = [r for r in rows.values() if r["label"] == 0]
    hit = lambda r, t: r["max_fire"] >= t          # noqa: E731
    tp = sum(1 for r in pos if hit(r, thr))
    fp = sum(1 for r in neg if hit(r, thr))
    fn = len(pos) - tp
    tn = len(neg) - fp
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
    per_cam = {}
    for r in neg:
        st = per_cam.setdefault(r["camera"], {"images": 0, "fp": 0})
        st["images"] += 1
        st["fp"] += 1 if hit(r, thr) else 0
    return {
        "conf_bars": {"alert_scoredthreshold": thr, "no_motion_scorehigh": high},
        "pos_images": len(pos), "neg_images": len(neg),
        "pos_recall_at_thr": (tp, len(pos), tp / len(pos) if pos else None),
        "pos_recall_at_high": (sum(1 for r in pos if hit(r, high)), len(pos),
                               sum(1 for r in pos if hit(r, high)) / len(pos) if pos else None),
        "neg_fp_at_thr": (fp, len(neg), fp / len(neg) if neg else None),
        "neg_fp_at_high": (sum(1 for r in neg if hit(r, high)), len(neg),
                           sum(1 for r in neg if hit(r, high)) / len(neg) if neg else None),
        "precision": prec, "recall": rec, "f1": f1,
        "confusion_at_thr": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "max_fire_positive": dist([r["max_fire"] for r in pos]),
        "max_fire_negative": dist([r["max_fire"] for r in neg]),
        "neg_fp_per_camera": per_cam,
    }


def v4_sanity(rows):
    """max_fire re-measured by V4 vs the confidence the store recorded in the filename."""
    pairs = [(r["store_conf"], r["max_fire"]) for r in rows.values()
             if r["store_conf"] is not None and r["max_fire"] is not None]
    if not pairs:
        return None
    diffs = [abs(a - b) for a, b in pairs]
    return {"n": len(pairs), "mean_abs_diff": round(statistics.fmean(diffs), 4),
            "max_abs_diff": round(max(diffs), 4),
            "within_0.05": sum(1 for d in diffs if d <= 0.05)}


def transitions(a_rows, b_rows, thr):
    """V4(a) -> V6(b) flips, split by label."""
    out = {"pos_fixed": [], "pos_lost": [], "pos_ok_both": [], "pos_fail_both": [],
           "neg_fixed": [], "neg_newfp": [], "neg_clean_both": [], "neg_fp_both": []}
    for img, ra in sorted(a_rows.items()):
        rb = b_rows.get(img)
        if rb is None:
            continue
        a_hit = ra["max_fire"] >= thr
        b_hit = rb["max_fire"] >= thr
        lab = ra["label"]
        if lab == 1:
            key = ("pos_ok_both" if a_hit and b_hit else
                   "pos_lost" if a_hit and not b_hit else
                   "pos_fixed" if b_hit and not a_hit else "pos_fail_both")
        else:
            key = ("neg_newfp" if b_hit and not a_hit else
                   "neg_fixed" if a_hit and not b_hit else
                   "neg_fp_both" if a_hit and b_hit else "neg_clean_both")
        out[key].append({"image": img, "camera": ra["camera"],
                         "v4": ra["max_fire"], "v6": rb["max_fire"]})
    return out


def pct(x):
    return "n/a" if x is None else "%.1f%%" % (100.0 * x)


SWEEP = [round(0.05 * i, 2) for i in range(1, 18)]        # 0.05 .. 0.85


def sweep_at(rows, bar):
    """Image-level fire stats at one bar (the bar is a free choice, unlike production)."""
    pos = [r for r in rows.values() if r["label"] == 1]
    neg = [r for r in rows.values() if r["label"] == 0]
    tp = sum(1 for r in pos if r["max_fire"] >= bar)
    fp = sum(1 for r in neg if r["max_fire"] >= bar)
    recall = tp / len(pos) if pos else None
    fpr = fp / len(neg) if neg else None
    prec = tp / (tp + fp) if tp + fp else None
    f1 = (2 * prec * recall / (prec + recall)) if prec and recall else None
    return {"bar": bar, "tp": tp, "fp": fp, "recall": recall, "fp_rate": fpr,
            "precision": prec, "f1": f1}


def sweep_table(rows):
    return {bar: sweep_at(rows, bar) for bar in SWEEP}


def best_threshold(table):
    """Highest bar with the best F1 (ties -> the tighter bar)."""
    best = None
    for bar in sorted(table):
        s = table[bar]
        if s["f1"] is None:
            continue
        if best is None or (s["f1"], bar) > (best["f1"], best["bar"]):
            best = s
    return best or {"bar": 0.0, "f1": None, "recall": None, "fp_rate": None}


def at_fp_budget(table, budget):
    """Best recall reachable while keeping the negative-FP rate at/below `budget`."""
    best = None
    for bar in sorted(table):
        s = table[bar]
        if s["fp_rate"] is None or s["recall"] is None:
            continue
        if s["fp_rate"] <= budget + 1e-9 and (best is None or s["recall"] > best["recall"]):
            best = s
    return best


def build_report(models, metrics, trans, sanity, sweep, info, args, paths):
    L = []
    L.append("# %s on the firewatch evidence collection"
             % " vs ".join(t.upper() for t, _ in models))
    L.append("")
    L.append("**Generated:** %s UTC · **inference**: imgsz=%d device=%s conf-floor=%.2f"
             % (utc_stamp(), args.imgsz, args.device, args.conf_floor))
    L.append("")
    L.append("**Collection:** %d positives (`%s/`) vs %d negatives "
             "(every other evidence frame; `*_annotated.jpg` excluded)"
             % (info["positives"], args.pos_subdir, info["negatives"]))
    L.append("")
    L.append("**Decision under test** (production semantics: firewatch tracks FIRE only,"
             " `TRACK_SMOKE=false`): *does the model put a FIRE box over the bar in this"
             " frame?* Bars from `config/firewatch.conf`: `SCORE_THRESHOLD=%.2f`"
             " (alert+evidence), `SCORE_HIGH=%.2f` (fire with no nearby motion)."
             % (args.fire_thresh, args.high_thresh))
    L.append("")
    L.append("## 1. Headline")
    L.append("")
    L.append("| model | positives detected @%.2f | positives detected @%.2f | negatives flagged @%.2f"
             " | negatives flagged @%.2f | precision | recall | F1 |"
             % (args.fire_thresh, args.high_thresh, args.fire_thresh, args.high_thresh))
    L.append("|---|---|---|---|---|---|---|---|")
    for tag, _ in models:
        m = metrics[tag]
        L.append("| **%s** | %d/%d (%s) | %d/%d (%s) | **%d/%d (%s)** | %d/%d (%s) | %s | %s | %s |"
                 % (tag, m["pos_recall_at_thr"][0], m["pos_recall_at_thr"][1],
                    pct(m["pos_recall_at_thr"][2]),
                    m["pos_recall_at_high"][0], m["pos_recall_at_high"][1],
                    pct(m["pos_recall_at_high"][2]),
                    m["neg_fp_at_thr"][0], m["neg_fp_at_thr"][1], pct(m["neg_fp_at_thr"][2]),
                    m["neg_fp_at_high"][0], m["neg_fp_at_high"][1], pct(m["neg_fp_at_high"][2]),
                    pct(m["precision"]), pct(m["recall"]), pct(m["f1"])))
    L.append("")
    L.append("## 2. Max fire confidence (what the model reports on each set)")
    L.append("")
    L.append("| model | positives mean/median/min | negatives mean/median/max |")
    L.append("|---|---|---|")
    for tag, _ in models:
        m = metrics[tag]
        p = m["max_fire_positive"]
        n = m["max_fire_negative"]
        L.append("| %s | %.3f / %.3f / %.3f | %.3f / %.3f / %.3f |"
                 % (tag, p.get("mean", 0), p.get("median", 0), p.get("min", 0),
                    n.get("mean", 0), n.get("median", 0), n.get("max", 0)))
    L.append("")
    L.append("## 3. Per-image flips at the %.2f bar (%s -> %s)"
             % (args.fire_thresh, models[0][0].upper(), models[1][0].upper()))
    L.append("")
    L.append("| group | count |")
    L.append("|---|---|")
    for key, label in (("pos_ok_both", "POSITIVES still detected by both (kept)"),
                       ("pos_lost", "POSITIVES lost (V4 hit, V6 missed)  <-- recall regression"),
                       ("pos_fixed", "POSITIVES gained (V4 missed, V6 hit)"),
                       ("pos_fail_both", "POSITIVES missed by both  <-- standing recall gap"),
                       ("neg_fixed", "NEGATIVES fixed (V4 flagged, V6 clean)  <-- FP removed"),
                       ("neg_newfp", "NEGATIVES newly flagged by V6  <-- new FP"),
                       ("neg_clean_both", "NEGATIVES clean in both"),
                       ("neg_fp_both", "NEGATIVES flagged by both  <-- standing FP")):
        L.append("| %s | %d |" % (label, len(trans[key])))
    L.append("")
    L.append("## 4. Negative-set false positives per camera (at %.2f)" % args.fire_thresh)
    L.append("")
    header = "| camera | images | " + " | ".join("%s FP" % t for t, _ in models) + " |"
    L.append(header)
    L.append("|---" * (2 + len(models)) + "|")
    cams = sorted({c for tag, _ in models for c in metrics[tag]["neg_fp_per_camera"]})
    for cam in cams:
        n = metrics[models[0][0]]["neg_fp_per_camera"].get(cam, {}).get("images", 0)
        cells = []
        for tag, _ in models:
            st = metrics[tag]["neg_fp_per_camera"].get(cam, {"images": 0, "fp": 0})
            cells.append("%d/%d (%s)" % (st["fp"], st["images"],
                                         pct(st["fp"] / st["images"] if st["images"] else None)))
        L.append("| %s | %d | %s |" % (cam, n, " | ".join(cells)))
    L.append("")
    L.append("## 5. Threshold sweep - is there ANY bar where the model separates the sets?")
    L.append("")
    L.append("`fire` boxes only (recall = positives detected, FP = negatives flagged)."
             " A model that cannot separate the two sets has no useful operating point.")
    L.append("")
    L.append("| bar | " + " | ".join("%s rec / FP / F1" % t for t, _ in models) + " |")
    L.append("|---" * (1 + len(models)) + "|")
    for bar in SWEEP:
        cells = []
        for tag, _ in models:
            s = sweep[tag][bar]
            cells.append("%s / %s / %s" % (pct(s["recall"]), pct(s["fp_rate"]), pct(s["f1"])))
        L.append("| %.2f | %s |" % (bar, " | ".join(cells)))
    L.append("")
    for tag, _ in models:
        b = best_threshold(sweep[tag])
        L.append("- **%s** best F1 = %s at bar %.2f (recall %s, FP %s)"
                 % (tag, pct(b["f1"]), b["bar"], pct(b["recall"]), pct(b["fp_rate"])))
    L.append("")
    L.append("## 6. Best recall at a FIXED false-positive budget")
    L.append("")
    L.append("The decision-relevant view when FPs are what bother the operator: at each FP"
             " budget, the recall achievable by moving the bar (`n/a` = the model can never"
             " get under that budget).")
    L.append("")
    L.append("| FP budget | " + " | ".join("%s bar -> recall" % t for t, _ in models) + " |")
    L.append("|---" * (1 + len(models)) + "|")
    for budget in (0.0, 0.05, 0.10, 0.20, 0.35):
        cells = []
        for tag, _ in models:
            b = at_fp_budget(sweep[tag], budget)
            cells.append("n/a" if b is None
                         else "%.2f -> %s (FP %s)" % (b["bar"], pct(b["recall"]),
                                                      pct(b["fp_rate"])))
        L.append("| <= %s | %s |" % (pct(budget), " | ".join(cells)))
    L.append("")
    L.append("## 7. Harness sanity - re-measured V4 vs the confidence stored in the filename")
    L.append("")
    for tag, _ in models:
        s = sanity.get(tag)
        if s:
            L.append("- **%s**: n=%d, mean |diff| = %.4f, max |diff| = %.4f,"
                     " within 0.05 on %d/%d frames"
                     % (tag, s["n"], s["mean_abs_diff"], s["max_abs_diff"],
                        s["within_0.05"], s["n"]))
    L.append("")
    L.append("## 8. Files")
    L.append("")
    for p in paths:
        L.append("- `%s`" % p)
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence", default="./firewatch-evidence")
    ap.add_argument("--pos-subdir", default="true-positives")
    ap.add_argument("--models", nargs="*", default=None, metavar="NAME=PATH")
    ap.add_argument("--conf-floor", type=float, default=0.02,
                    help="predict() floor - kept LOW so sub-threshold boxes are visible "
                         "to the sweep (production floors at 0.35)")
    ap.add_argument("--fire-thresh", type=float, default=0.50)
    ap.add_argument("--high-thresh", type=float, default=0.85)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--annotate", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    evidence = os.path.abspath(args.evidence)
    if not os.path.isdir(evidence):
        sys.exit("evidence dir not found: %s" % evidence)
    out_dir = args.out or os.path.join(root, "model-training", "eval", "firewatch_eval",
                                       "run-%s" % utc_stamp())
    os.makedirs(out_dir, exist_ok=True)

    models = discover_models(args.models, root)
    pos, neg, info = collect(evidence, args.pos_subdir)
    print("collection: %d positives, %d negatives  (%s)"
          % (info["positives"], info["negatives"], evidence))
    print("models:")
    all_rows = {}
    for tag, path in models:
        all_rows[tag] = run_model(tag, path, pos, neg, args, out_dir)

    metrics = {tag: metrics_for(all_rows[tag], args.fire_thresh, args.high_thresh)
               for tag, _ in models}
    sanity = {tag: v4_sanity(all_rows[tag]) for tag, _ in models}
    sweep = {tag: sweep_table(all_rows[tag]) for tag, _ in models}

    trans = {}
    if len(models) >= 2:
        trans = transitions(all_rows[models[0][0]], all_rows[models[1][0]], args.fire_thresh)

    flips = {}
    for key, items in trans.items():
        for it in items:
            flips[it["image"]] = key
    diffs_path = os.path.join(out_dir, "differences.csv")
    with open(diffs_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "label", "camera", "store_conf"]
                   + ["%s_max_fire" % t for t, _ in models]
                   + ["%s_hit@%.2f" % (t, args.fire_thresh) for t, _ in models]
                   + ["transition"])
        for img in sorted(all_rows[models[0][0]]):
            ra = all_rows[models[0][0]][img]
            w.writerow([img, ra["label"], ra["camera"], ra["store_conf"]]
                       + [all_rows[t][img]["max_fire"] for t, _ in models]
                       + [int(all_rows[t][img]["max_fire"] >= args.fire_thresh)
                          for t, _ in models]
                       + [flips.get(img, "")])

    paths = [os.path.join(out_dir, "per_image_%s.csv" % t) for t, _ in models]
    paths += [diffs_path, os.path.join(out_dir, "summary.md"),
              os.path.join(out_dir, "summary.json")]
    report = build_report(models, metrics, trans, sanity, sweep, info, args, paths)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"generated": utc_stamp(), "args": vars(args), "collection": info,
                   "models": [{"tag": t, "path": p} for t, p in models],
                   "metrics": metrics, "sanity": sanity, "sweep": sweep,
                   "transitions": {k: len(v) for k, v in trans.items()}}, fh,
                  indent=2, default=str)
    print()
    print(report)
    print("\nwrote:", out_dir)


if __name__ == "__main__":
    main()
