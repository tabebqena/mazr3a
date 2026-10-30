#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_motion_parameters import best_f1, build_rows, metrics

BONUSES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
FRACTIONS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25,
             0.30, 0.40, 0.50)


def score(row, fraction, bonus):
    values = [row["candidate_raw"]]
    for box in row["boxes"]:
        value = box["s"]
        if box["core_frac"] >= fraction:
            value = min(1.0, value + bonus)
        values.append(value)
    return max(values)


def pct(value):
    return f"{100 * value:.1f}%"


def table(rows, threshold):
    lines = [
        "| motion pixels / box | "
        + " | ".join(f"bonus {bonus:.2f}" for bonus in BONUSES) + " |",
        "|---:|" + "---:|" * len(BONUSES),
    ]
    for fraction in FRACTIONS:
        cells = []
        for bonus in BONUSES:
            scorer = lambda row, f=fraction, b=bonus: score(row, f, b)
            if threshold is None:
                f1, bar = best_f1(rows, scorer)
                cells.append(f"{pct(f1)} @{bar:.2f}")
            else:
                cells.append(pct(metrics(rows, scorer, threshold)["f1"]))
        lines.append(f"| {fraction:.2f} | " + " | ".join(cells) + " |")
    return lines


def detail(name, result):
    return (f"| {name} | {pct(result['recall'])} | {pct(result['fpr'])} | "
            f"{pct(result['precision'])} | {pct(result['f1'])} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path,
                        default=Path(__file__).resolve().parents[2]
                        / "firewatch-evidence")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gap-max", type=float, default=30.0)
    parser.add_argument("--px-diff", type=float, default=18.0)
    parser.add_argument("--night-extra", type=float, default=8.0)
    parser.add_argument("--night-luma", type=float, default=25.0)
    parser.add_argument("--blur-radius", type=float, default=1.0)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out = (args.out or (run_dir / "motion_inside_box_fixed_bonus.md")).resolve()
    rows = build_rows(run_dir, args.evidence_dir.resolve(), args.gap_max,
                      args.px_diff, args.night_extra, args.night_luma,
                      args.blur_radius)
    true = sum(row["label"] == "true" for row in rows)
    false = len(rows) - true

    raw = metrics(rows, lambda row: row["candidate_raw"], 0.5)
    expanded = metrics(
        rows, lambda row: min(
            1.0, row["candidate_raw"]
            + (0.15 if any(box["exp_frac"] >= 0.02 for box in row["boxes"])
               else 0.0)), 0.5)
    inside_2 = metrics(rows, lambda row: score(row, 0.02, 0.15), 0.5)
    inside_5 = metrics(rows, lambda row: score(row, 0.05, 0.15), 0.5)
    moderate = metrics(rows, lambda row: score(row, 0.02, 0.20), 0.55)
    higher = metrics(rows, lambda row: score(row, 0.05, 0.25), 0.60)

    best_05 = (-1.0, None, None, None)
    best_any = (-1.0, None, None, None, None)
    for fraction in FRACTIONS:
        for bonus in BONUSES:
            scorer = lambda row, f=fraction, b=bonus: score(row, f, b)
            result = metrics(rows, scorer, 0.5)
            if result["f1"] > best_05[0]:
                best_05 = (result["f1"], fraction, bonus, result)
            f1, bar = best_f1(rows, scorer)
            if f1 > best_any[0]:
                best_any = (f1, fraction, bonus, bar,
                            metrics(rows, scorer, bar))

    lines = [
        "# Inside-box fixed motion bonus",
        "",
        "## Scope",
        "",
        f"{len(rows)} core frames with an image-derived motion mask "
        f"({true} true / {false} false).",
        f"Mask from nearest prior stored frame within {args.gap_max:g}s. "
        "Candidate detection floor 0.35.",
        "Motion is measured only inside the complete detection box. "
        "No expanded search box and no min-pixel OR.",
        "A fixed bonus is added when motion pixels / box area >= threshold.",
        "",
        "## F1 at production bar 0.50",
        "",
    ]
    lines += table(rows, 0.5)
    lines += ["", "## Best F1 and its best bar", ""]
    lines += table(rows, None)
    lines += [
        "",
        "## Selected operating points",
        "",
        "| rule | recall | FP rate | precision | F1 |",
        "|---|---:|---:|---:|---:|",
        detail("candidate raw, bar 0.50", raw),
        detail("current Firewatch: expanded 2%, bonus 0.15, bar 0.50",
               expanded),
        detail("inside box 2%, bonus 0.15, bar 0.50", inside_2),
        detail("inside box 5%, bonus 0.15, bar 0.50", inside_5),
        detail("inside box 2%, bonus 0.20, bar 0.55", moderate),
        detail("inside box 5%, bonus 0.25, bar 0.60", higher),
        detail(
            f"best at bar 0.50: {best_05[1]:.2f} threshold, "
            f"bonus {best_05[2]:.2f}", best_05[3]),
        detail(
            f"best overall: {best_any[1]:.2f} threshold, "
            f"bonus {best_any[2]:.2f}, bar {best_any[3]:.2f}",
            best_any[4]),
        "",
        "## Result",
        "",
        "- Inside the complete box, the best threshold at bar 0.50 is "
        "0.02-0.05 with bonus 0.15. Best F1 is 63.6-63.7%, slightly above "
        "the current expanded-box rule at 63.1%.",
        "- Higher motion thresholds lose recall. F1 at bar 0.50 drops from "
        "63.7% at 5% motion to 62.7% at 10%, 59.3% at 20%, 52.5% at 30%, "
        "46.9% at 40%, and 44.1% at 50%.",
        "- At bar 0.50, fixed bonuses above 0.15 do not change detections: "
        "the candidate floor 0.35 + 0.15 already reaches 0.50.",
        "- Higher bonuses only matter if the bar is raised. Bonus 0.20 at "
        "bar 0.55 gives F1 65.8-66.0%; bonus 0.25 at bar 0.60 gives "
        "67.4-67.5%. Bonus 0.35 at bar 0.70 reaches 68.4-68.6%, but is "
        "aggressive.",
        "- Recommendation: use inside-box motion at 3-5%, fixed bonus 0.15 "
        "if the bar stays 0.50. If retuning, bonus 0.20/bar 0.55 is the "
        "moderate option, or bonus 0.25/bar 0.60 for more precision.",
        "- These are per-frame scores; the 3-of-4 follow-up confirmation was "
        "not replayed.",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
