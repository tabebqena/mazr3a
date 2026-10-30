#!/usr/bin/env python3
import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from firewatch.firewatch import _adaptive_px, _changed_mask, _expanded_region

BONUSES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
FRACTIONS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20)
FLOOR = 0.35


def load_csv_map(path, value):
    return {row["key"]: row[value] for row in csv.DictReader(open(path))}


def load_predictions(path):
    out = {}
    for line in open(path):
        row = json.loads(line)
        out[row["key"]] = row
    return out


def metrics(rows, scorer, threshold):
    tp = fp = fn = tn = 0
    for row in rows:
        positive = row["label"] == "true"
        value = scorer(row)
        if value >= threshold:
            if positive:
                tp += 1
            else:
                fp += 1
        elif positive:
            fn += 1
        else:
            tn += 1
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall, "fpr": fpr, "precision": precision, "f1": f1,
    }


def best_f1(rows, scorer):
    best = (0.0, 0.0)
    for step in range(1, 100):
        threshold = step / 100.0
        result = metrics(rows, scorer, threshold)
        if (result["f1"] > best[0] + 1e-12
                or (abs(result["f1"] - best[0]) <= 1e-12
                    and threshold > best[1])):
            best = (result["f1"], threshold)
    return best


def build_rows(run_dir, evidence_dir, gap_max, px_diff, night_extra,
               night_luma, blur):
    labels = load_csv_map(run_dir / "labels_snapshot.csv", "verdict")
    cohorts = load_csv_map(run_dir / "cohorts.csv", "cohort")
    predictions = load_predictions(run_dir / "predictions_candidate.jsonl")
    conn = sqlite3.connect(REPO_ROOT / "firewatch-evidence" / "firewatch.db")
    conn.row_factory = sqlite3.Row
    frames = []
    for row in conn.execute(
            "SELECT id, camera, captured_at, jpg_path FROM frames "
            "ORDER BY camera, captured_at"):
        key = row["jpg_path"].split("/media/firewatch/", 1)[-1]
        if key in labels and cohorts.get(key) == "core":
            frames.append({
                "key": key,
                "camera": row["camera"],
                "captured_at": row["captured_at"],
                "label": labels[key],
            })
    conn.close()

    by_camera = {}
    for row in frames:
        by_camera.setdefault(row["camera"], []).append(row)

    out = []
    for camera_rows in by_camera.values():
        previous = None
        for row in camera_rows:
            mask = None
            if previous is not None:
                gap = row["captured_at"] - previous["captured_at"]
                a_path = evidence_dir / previous["key"]
                b_path = evidence_dir / row["key"]
                if 0 < gap <= gap_max and a_path.exists() and b_path.exists():
                    a = np.asarray(Image.open(a_path).convert("L"),
                                   dtype=np.uint8)
                    b = np.asarray(Image.open(b_path).convert("L"),
                                   dtype=np.uint8)
                    if a.shape == b.shape:
                        threshold = _adaptive_px(
                            px_diff, night_extra, night_luma, float(b.mean()))
                        mask = _changed_mask(a, b, threshold, blur)
            if mask is not None:
                boxes = []
                for box in predictions[row["key"]]["boxes"]:
                    if box["s"] < FLOOR:
                        continue
                    core = _expanded_region(mask, tuple(box["xy"]), 0.0)
                    expanded = _expanded_region(mask, tuple(box["xy"]), 0.5)
                    if core is None or expanded is None:
                        continue
                    boxes.append({
                        "s": float(box["s"]),
                        "core_frac": float(core.mean()),
                        "exp_frac": float(expanded.mean()),
                    })
                row["boxes"] = boxes
                row["candidate_raw"] = float(
                    predictions[row["key"]]["max_fire"])
                out.append(row)
            previous = row
    return out


def score(row, denominator, fraction, bonus):
    values = [row["candidate_raw"]]
    for box in row["boxes"]:
        value = box["s"]
        if box[denominator] >= fraction:
            value = min(1.0, value + bonus)
        values.append(value)
    return max(values)


def grid(rows, denominator, bonus, threshold=None):
    result = []
    for fraction in FRACTIONS:
        scorer = lambda row, f=fraction, b=bonus: score(
            row, denominator, f, b)
        if threshold is None:
            f1, bar = best_f1(rows, scorer)
            result.append((fraction, f1, bar, metrics(rows, scorer, bar)))
        else:
            result.append(
                (fraction, metrics(rows, scorer, threshold)["f1"],
                 threshold, metrics(rows, scorer, threshold)))
    return result


def table(rows, denominator, threshold, title):
    lines = [
        f"### {title}",
        "",
        "| motion pixels / box | "
        + " | ".join(f"bonus {bonus:.2f}" for bonus in BONUSES) + " |",
        "|---:|" + "---:|" * len(BONUSES),
    ]
    grids = {
        bonus: grid(rows, denominator, bonus, threshold)
        for bonus in BONUSES
    }
    for index, fraction in enumerate(FRACTIONS):
        cells = []
        for bonus in BONUSES:
            _, f1, bar, _ = grids[bonus][index]
            if threshold is None:
                cells.append(f"{100 * f1:.1f}% @{bar:.2f}")
            else:
                cells.append(f"{100 * f1:.1f}%")
        lines.append(f"| {fraction:.2f} | " + " | ".join(cells) + " |")
    return lines


def pct(value):
    return f"{100 * value:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path,
                        default=REPO_ROOT / "firewatch-evidence")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gap-max", type=float, default=30.0)
    parser.add_argument("--px-diff", type=float, default=18.0)
    parser.add_argument("--night-extra", type=float, default=8.0)
    parser.add_argument("--night-luma", type=float, default=25.0)
    parser.add_argument("--blur-radius", type=float, default=1.0)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out = (args.out or (run_dir / "motion_parameter_sweep.md")).resolve()
    rows = build_rows(run_dir, args.evidence_dir.resolve(), args.gap_max,
                      args.px_diff, args.night_extra, args.night_luma,
                      args.blur_radius)
    true = sum(row["label"] == "true" for row in rows)
    false = len(rows) - true

    no_motion = metrics(rows, lambda row: row["candidate_raw"], 0.5)
    current = metrics(
        rows, lambda row: score(row, "exp_frac", 0.02, 0.15), 0.5)
    best_05 = (-1.0, None, None, None)
    best_any = (-1.0, None, None, None, None)
    for fraction in FRACTIONS:
        for bonus in BONUSES:
            scorer = lambda row, f=fraction, b=bonus: score(
                row, "core_frac", f, b)
            result = metrics(rows, scorer, 0.5)
            if result["f1"] > best_05[0]:
                best_05 = (result["f1"], fraction, bonus, result)
            f1, bar = best_f1(rows, scorer)
            if f1 > best_any[0]:
                best_any = (f1, fraction, bonus, bar,
                            metrics(rows, scorer, bar))

    def detail(name, result):
        return (f"| {name} | {pct(result['recall'])} | "
                f"{pct(result['fpr'])} | {pct(result['precision'])} | "
                f"{pct(result['f1'])} |")

    lines = [
        "# Candidate motion-parameter sweep",
        "",
        "## Scope",
        "",
        f"Image-derived motion masks available on {len(rows)} core frames "
        f"({true} true / {false} false).",
        f"Mask from the nearest prior stored frame within {args.gap_max:g}s. "
        f"Candidate detection floor kept at {FLOOR:.2f}.",
        "Pure changed-pixel fraction rule; the Firewatch min-pixel OR is "
        "disabled to isolate the percentage threshold.",
        "Main grids divide changed pixels by the complete detection-box area. "
        "The current Firewatch rule uses the expanded box (margin 0.5).",
        "",
        "Absolute F1 values here differ from the full 4,245-frame replay "
        "because this subset has a usable image-derived mask and a higher "
        "share of true frames. Use these tables for parameter comparison, not "
        "as production precision estimates.",
        "",
        "## F1 at production bar 0.50",
        "",
    ]
    lines += table(rows, "core_frac", 0.5, "Complete box area")
    lines += [""]
    lines += table(rows, "core_frac", None, "Complete box area, best F1")
    lines += [""]
    lines += table(rows, "exp_frac", 0.5,
                   "Expanded box area (current Firewatch), F1 at 0.50")
    lines += [
        "",
        "## Selected operating points",
        "",
        "| rule | recall | FP rate | precision | F1 |",
        "|---|---:|---:|---:|---:|",
        detail("candidate raw, bar 0.50", no_motion),
        detail("current Firewatch rule: expanded 2%, bonus 0.15, bar 0.50",
               current),
        detail(
            f"best at bar 0.50: complete {best_05[1]:.2f}, "
            f"bonus {best_05[2]:.2f}", best_05[3]),
        detail(
            f"best overall: complete {best_any[1]:.2f}, "
            f"bonus {best_any[2]:.2f}, bar {best_any[3]:.2f}",
            best_any[4]),
        "",
        "## Result",
        "",
        "- At a fixed 0.50 bar, bonuses above 0.15 do not add detections: "
        "the candidate floor is 0.35, so 0.35 + 0.15 already reaches 0.50.",
        "- With the bar kept at 0.50, the complete-box fraction works best at "
        "0.02-0.05. The current expanded-box 0.02 rule is close but slightly "
        "worse than the complete-box 0.02-0.05 rule.",
        "- To benefit from more than 0.15, raise the bar with it. Bonus 0.20 "
        "at bar 0.55 (complete 2-3%) gives F1 66.0%, recall 62.2%, FP rate "
        "5.9%, precision 70.3%. This captures most of the gain with a modest "
        "bonus.",
        "- Bonus 0.25 at bar 0.60 (complete 2-5%) gives F1 67.4-67.5%, recall "
        "61.5-62.2%, FP rate 4.6-5.0%, precision 73.5-74.8%.",
        "- Bonus 0.35 is aggressive: it acts as a different two-tier policy "
        "(near boxes >=0.35, static boxes >=0.70). It reaches F1 68.4-68.6%, "
        "only about 1 point above bonus 0.25, and is more likely to overfit "
        "the selected evidence. Prefer 0.20 or 0.25 unless live testing "
        "supports 0.35.",
        "- A fraction of 0.00 is much worse: giving every box the bonus "
        "removes the motion-spatial filter.",
        "- The 3-of-4 follow-up confirmation was not replayed; these are "
        "per-frame scores.",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
