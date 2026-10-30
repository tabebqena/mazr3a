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

MARGINS = (0.0, 0.5, 1.0, 1.5, 2.0)
SEARCH_MARGINS = (0.5, 1.0, 1.5, 2.0)
BONUSES = (0.10, 0.15, 0.20, 0.25)
SCALES = (0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20)
POWERS = (0.5, 1.0, 2.0)
FLOOR = 0.35


def load_map(path, field):
    return {row["key"]: row[field] for row in csv.DictReader(open(path))}


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
    labels = load_map(run_dir / "labels_snapshot.csv", "verdict")
    cohorts = load_map(run_dir / "cohorts.csv", "cohort")
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
                    fractions = {}
                    for margin in MARGINS:
                        region = _expanded_region(
                            mask, tuple(box["xy"]), margin)
                        if region is None:
                            fractions = {}
                            break
                        fractions[margin] = float(region.mean())
                    if fractions:
                        boxes.append({"s": float(box["s"]),
                                      "fractions": fractions})
                row["boxes"] = boxes
                row["candidate_raw"] = float(
                    predictions[row["key"]]["max_fire"])
                out.append(row)
            previous = row
    return out


def fixed_score(row, margin, threshold, bonus):
    values = [row["candidate_raw"]]
    for box in row["boxes"]:
        value = box["s"]
        if box["fractions"][margin] >= threshold:
            value = min(1.0, value + bonus)
        values.append(value)
    return max(values)


def scaled_score(row, margin, scale, bonus, power):
    values = [row["candidate_raw"]]
    for box in row["boxes"]:
        fraction = box["fractions"][margin]
        value = min(1.0, fraction / scale) ** power
        values.append(min(1.0, box["s"] + bonus * value))
    return max(values)


def best_config(rows, make_scorer, bonuses, scales, power=None):
    best_05 = (-1.0, None, None, None)
    best_any = (-1.0, None, None, None, None)
    for bonus in bonuses:
        for scale in scales:
            if power is None:
                scorer = lambda row, b=bonus, s=scale: make_scorer(
                    row, s, b)
            else:
                scorer = lambda row, b=bonus, s=scale, p=power: make_scorer(
                    row, s, b, p)
            result = metrics(rows, scorer, 0.5)
            if result["f1"] > best_05[0]:
                best_05 = (result["f1"], bonus, scale, result)
            f1, bar = best_f1(rows, scorer)
            if f1 > best_any[0]:
                best_any = (f1, bonus, scale, bar,
                            metrics(rows, scorer, bar))
    return best_05, best_any


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
    out = (args.out or (run_dir / "motion_scaled_bonus_sweep.md")).resolve()
    rows = build_rows(run_dir, args.evidence_dir.resolve(), args.gap_max,
                      args.px_diff, args.night_extra, args.night_luma,
                      args.blur_radius)
    true = sum(row["label"] == "true" for row in rows)
    false = len(rows) - true

    raw = metrics(rows, lambda row: row["candidate_raw"], 0.5)
    current = metrics(
        rows, lambda row: fixed_score(row, 0.5, 0.02, 0.15), 0.5)
    complete = metrics(
        rows, lambda row: fixed_score(row, 0.0, 0.03, 0.15), 0.5)

    scaled_summary = {}
    fixed_summary = {}
    for margin in SEARCH_MARGINS:
        fixed_05, fixed_any = best_config(
            rows,
            lambda row, s, b, m=margin: fixed_score(row, m, s, b),
            BONUSES, SCALES)
        fixed_summary[margin] = (fixed_05, fixed_any)
        for power in POWERS:
            scaled_05, scaled_any = best_config(
                rows,
                lambda row, s, b, p=power, m=margin: scaled_score(
                    row, m, s, b, p),
                BONUSES, SCALES, power)
            scaled_summary[(margin, power)] = (scaled_05, scaled_any)

    def cell05(value):
        return (f"{pct(value[3]['f1'])} "
                f"({value[1]:.2f}/{value[2]:.2f})")

    def cellany(value):
        return (f"{pct(value[4]['f1'])} @{value[3]:.2f} "
                f"({value[1]:.2f}/{value[2]:.2f})")

    lines = [
        "# Scaled motion bonus in a larger box",
        "",
        "## Scope",
        "",
        f"{len(rows)} core frames with an image-derived motion mask "
        f"({true} true / {false} false).",
        f"Mask from nearest prior stored frame within {args.gap_max:g}s. "
        f"Candidate detection floor {FLOOR:.2f}.",
        "Search box = detection box expanded by the margin per side. "
        "Scaled bonus = max_bonus * min(1, motion_fraction / scale)^power.",
        "Fixed bonus = max_bonus when motion_fraction >= scale.",
        "F1@0.50 is the production bar; best F1 uses its own best bar.",
        "",
        "## Fixed baselines",
        "",
        "| rule | recall | FP rate | precision | F1 |",
        "|---|---:|---:|---:|---:|",
        f"| candidate raw | {pct(raw['recall'])} | {pct(raw['fpr'])} | "
        f"{pct(raw['precision'])} | {pct(raw['f1'])} |",
        f"| current fixed: expanded 2%, bonus 0.15 | "
        f"{pct(current['recall'])} | {pct(current['fpr'])} | "
        f"{pct(current['precision'])} | {pct(current['f1'])} |",
        f"| fixed complete box 3%, bonus 0.15 | "
        f"{pct(complete['recall'])} | {pct(complete['fpr'])} | "
        f"{pct(complete['precision'])} | {pct(complete['f1'])} |",
        "",
        "## Larger search boxes: fixed vs scaled",
        "",
        "Cells show F1 (max_bonus / fraction-scale).",
        "",
        "| search margin | fixed F1@0.50 | scaled p=0.5 | scaled p=1 | "
        "scaled p=2 | fixed best F1 | scaled best p=1 | scaled best p=2 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for margin in SEARCH_MARGINS:
        cells = []
        fixed_05, fixed_any = fixed_summary[margin]
        cells.append(cell05(fixed_05))
        for power in POWERS:
            cells.append(cell05(scaled_summary[(margin, power)][0]))
        cells.append(cellany(fixed_any))
        cells.append(cellany(scaled_summary[(margin, 1.0)][1]))
        cells.append(cellany(scaled_summary[(margin, 2.0)][1]))
        lines.append(f"| {margin:.1f} | " + " | ".join(cells) + " |")

    best_scaled_05 = max(
        (value[0] for value in scaled_summary.values()), key=lambda x: x[0])
    best_scaled_any = max(
        (value[1] for key, value in scaled_summary.items() if key[1] == 2.0),
        key=lambda x: x[0])
    lines += [
        "",
        "## Result",
        "",
        f"- Best scaled result at bar 0.50: F1 {pct(best_scaled_05[0])} "
        f"(bonus {best_scaled_05[1]:.2f}, scale {best_scaled_05[2]:.2f}).",
        f"- Best scaled result overall: F1 {pct(best_scaled_any[0])} at bar "
        f"{best_scaled_any[3]:.2f} (bonus {best_scaled_any[1]:.2f}, scale "
        f"{best_scaled_any[2]:.2f}, power 2).",
        "- At the current 0.5 margin, scaling is slightly worse than the "
        "fixed bonus. At larger margins scaling can beat the fixed rule at "
        "that same margin, but the larger box itself loses more than scaling "
        "recovers; no scaled/larger-box result beats the current 0.5 rule.",
        "- Larger boxes dilute the motion fraction: margins 1.0-2.0 perform "
        "worse than 0.5 because neighboring movement enters the search box.",
        "- Linear scaling under-rewards boxes whose raw score needs the full "
        "0.15 to reach the 0.50 bar. Power 2 is the least-bad curve, but still "
        "below fixed.",
        "- Recommendation: keep the fixed near-threshold bonus. Searching in "
        "a larger box and scaling the bonus is not supported by this evidence.",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
