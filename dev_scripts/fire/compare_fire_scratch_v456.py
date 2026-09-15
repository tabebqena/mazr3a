#!/usr/bin/env python3
"""compare_fire_scratch_v456.py - score scratch vs v4 vs v5 vs v6 on the 2000-image
augmented benchmark built by build_fire_aug_benchmark2000.py, and write a matrix report.

Image-level, model-agnostic comparison (max class confidence per image, no NMS-threshold
tuning) - the same semantics as compare_pretrained_fire_models.py, but pre-configured for
the scratch campaign and capable of a fire-only checkpoint (`nc=1`, smoke columns -> N/A).

Default model set:
    scratch = model-training/runs/scratch-v1-dfire/scratch-v1-dfire.pt   (fire-only, nc=1)
    v4      = models/fire/best.pt                                        (fire/other/smoke)
    v5      = models/fire/versions/v5/best_v5_finetuned_not_verified.pt  (fire/other/smoke)
    v6      = models/fire/versions/v6-2026-09-13-fireviewer-hfcorpus-ft15ep/model.pt

Usage:
    .venv/bin/python dev_scripts/fire/compare_fire_scratch_v456.py \
        [--manifest model-training/eval/fire-benchmark-aug2000/manifest.csv] \
        [--out model-training/eval/fire-benchmark-aug2000/compare] \
        [--models scratch=PATH v4=PATH ...] [--conf 0.5] \
        [--thresholds 0.25,0.35,0.5,0.7] [--imgsz 0] [--batch 16] \
        [--conf-floor 0.001] [--force]

Outputs (git-ignored): <out>/report.md, <out>/summary.csv, <out>/<tag>/per_image.csv.
"""
import argparse
import collections
import csv
import datetime
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_pretrained_fire_models import (  # noqa: E402
    category_metrics, fmt_pct, load_probe, md5_file, md_table, score_model,
)

DEFAULT_MANIFEST = os.path.join(ROOT, "model-training", "eval/fire-benchmark-aug2000",
                                "manifest.csv")
DEFAULT_OUT = os.path.join(ROOT, "model-training", "eval/fire-benchmark-aug2000", "compare")

DEFAULT_MODELS = [
    ("scratch", "model-training/runs/scratch-v1-dfire/scratch-v1-dfire.pt"),
    ("v4", "models/fire/best.pt"),
    ("v5", "models/fire/versions/v5/best_v5_finetuned_not_verified.pt"),
    ("v6", "models/fire/versions/v6-2026-09-13-fireviewer-hfcorpus-ft15ep/model.pt"),
]

CATEGORIES = ["fire_smoke", "fire_only", "smoke_only", "other"]

_T0 = time.time()


def step(msg):
    print("[step %6.1fs] %s" % (time.time() - _T0, msg), flush=True)


def abs_repo(rel):
    return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)


def resolve_models(tokens):
    """Return [(tag, path)] - from --models NAME=PATH or the built-in defaults."""
    if tokens:
        out = []
        for s in tokens:
            if "=" not in s:
                sys.exit("--models wants NAME=PATH, got %r" % s)
            tag, path = s.split("=", 1)
            p = abs_repo(path)
            if not os.path.isfile(p):
                sys.exit("model not found: %s" % path)
            out.append((tag, p))
        return out
    out = []
    for tag, path in DEFAULT_MODELS:
        p = abs_repo(path)
        if os.path.isfile(p):
            out.append((tag, p))
        else:
            step("  ! skipping missing default %s (%s)" % (tag, path))
    return out


def mask_metrics(mt, has_fire, has_smoke):
    """Return a copy of category_metrics() with missing-class columns set to None."""
    mt = dict(mt)
    smoke_keys = ["smoke_recall_smoke_only", "smoke_recall_fire_smoke",
                  "both_recall_fire_smoke", "smoke_recall_all_smoke_images",
                  "fp_smoke_other"]
    fire_keys = ["fire_recall_fire_only", "fire_recall_fire_smoke",
                 "fire_recall_all_fire_images", "fp_fire_other"]
    if not has_smoke:
        for k in smoke_keys:
            mt[k] = None
        # a fire-only model never emits smoke, so fp_any == fp_fire (keep the numeric value)
    if not has_fire:
        for k in fire_keys:
            mt[k] = None
        mt["fp_any_other"] = None
    return mt


def metric_order():
    return ["n_fire_smoke", "n_fire_only", "n_smoke_only", "n_other",
            "fire_recall_fire_only", "fire_recall_fire_smoke",
            "smoke_recall_smoke_only", "smoke_recall_fire_smoke",
            "both_recall_fire_smoke", "fire_recall_all_fire_images",
            "smoke_recall_all_smoke_images", "fp_any_other", "fp_fire_other",
            "fp_smoke_other"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="*", default=None,
                    help="NAME=PATH pairs (default: scratch/v4/v5/v6 built-ins)")
    ap.add_argument("--conf", type=float, default=0.5, help="headline threshold [0.5]")
    ap.add_argument("--thresholds", default="0.25,0.35,0.5,0.7")
    ap.add_argument("--imgsz", type=int, default=0,
                    help="force one inference size for all models; 0 = each ckpt's own [0]")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf-floor", type=float, default=0.001,
                    help="ultralytics conf floor used to keep raw per-class maxima [0.001]")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    models = resolve_models(args.models)
    if len(models) < 2:
        sys.exit("need at least 2 models to compare")

    man_path = abs_repo(args.manifest)
    if not os.path.isfile(man_path):
        sys.exit("manifest not found: %s\nRun build_fire_aug_benchmark2000.py first."
                 % args.manifest)
    with open(man_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    step("manifest: %d images (%s)" % (
        len(rows), ", ".join("%s=%d" % (c, sum(1 for r in rows if r["category"] == c))
                             for c in CATEGORIES)))

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    if args.conf not in thresholds:
        thresholds = sorted(set(thresholds + [args.conf]))

    out_dir = abs_repo(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # ---- load + probe ----
    us = {}
    na = []
    step("loading + probing %d checkpoint(s) ..." % len(models))
    for tag, path in models:
        info = load_probe(path)
        info["pt"] = path
        info["tag"] = tag
        info["md5"] = md5_file(path)
        if info["fire_i"] is None and info["smoke_i"] is None:
            na.append(info)
            step("  %-10s NO fire/smoke class (nc=%s) -> N/A" % (tag, info["nc"]))
        else:
            us[tag] = info
            step("  %-10s nc=%s fire_i=%s smoke_i=%s imgsz=%s"
                 % (tag, info["nc"], info["fire_i"], info["smoke_i"],
                    args.imgsz or info["imgsz"]))

    # ---- score (cached per model) ----
    scored_all = {}
    for tag, m in us.items():
        cell_dir = os.path.join(out_dir, tag)
        os.makedirs(cell_dir, exist_ok=True)
        cache = os.path.join(cell_dir, "per_image.csv")
        if os.path.isfile(cache) and not args.force:
            step("[cached] %s <- %s" % (tag, cache))
            with open(cache, newline="", encoding="utf-8") as fh:
                scored = {r["bench_path"]: (float(r["fire"]), float(r["smoke"]), int(r["nbox"]))
                          for r in csv.DictReader(fh)}
            scored_all[tag] = scored
            continue
        isz = args.imgsz or m["imgsz"]
        step("[run   ] %s  (imgsz=%d, %s)" % (tag, isz, os.path.relpath(m["pt"], ROOT)))
        scored = score_model(m["model"], rows, m["fire_i"], m["smoke_i"], isz,
                             args.batch, args.conf_floor)
        scored_all[tag] = scored
        with open(cache, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["category", "source", "bench_path", "gt_fire", "gt_smoke",
                        "fire", "smoke", "nbox", "rot_deg", "bright_factor"])
            for r in rows:
                f, s, nb = scored[r["bench_path"]]
                w.writerow([r["category"], r["source"], r["bench_path"], r["gt_fire"],
                            r["gt_smoke"], "%.5f" % f, "%.5f" % s, nb,
                            r.get("rot_deg", ""), r.get("bright_factor", "")])

    # ---- summary.csv ----
    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "threshold"] + metric_order())
        for tag, m in us.items():
            for thr in thresholds:
                mt = category_metrics(scored_all[tag], rows, thr)
                mt = mask_metrics(mt, m["fire_i"] is not None, m["smoke_i"] is not None)
                w.writerow([tag, thr] + ["" if mt[k] is None else "%.4f" % mt[k]
                                         for k in metric_order()])
    step("wrote %s" % summary_path)

    # ---- report ----
    L = []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L.append("# Fire model — scratch vs v4/v5/v6 (2000-image augmented benchmark)\n")
    L.append("**Generated:** %s · **manifest:** `%s` (%d imgs) · **headline conf:** %.2f · "
             "**harness:** `dev_scripts/fire/compare_fire_scratch_v456.py`\n"
             % (now, os.path.relpath(man_path, ROOT), len(rows), args.conf))
    L.append("> Image-level max class confidence (no NMS-threshold tuning) across the 4 "
             "benchmark categories. Each model runs at its own training imgsz (unless "
             "`--imgsz`). `scratch` is **fire-only** (`nc=1`), so its smoke columns are N/A. "
             "See [`plans/fire-model-scratch-v456-aug-benchmark.md`]"
             "(../plans/fire-model-scratch-v456-aug-benchmark.md).\n")

    # models table
    L.append("\n## Models\n")
    mh = ["Model", "Checkpoint", "md5", "nc", "classes", "imgsz"]
    mrows = []
    for tag, m in us.items():
        cls = ", ".join("%d:%s" % (i, n) for i, n in sorted(m["names"].items()))
        mrows.append([tag, os.path.relpath(m["pt"], ROOT), m["md5"][:12], m["nc"], cls,
                      args.imgsz or m["imgsz"]])
    for m in na:
        mrows.append([m["tag"], os.path.relpath(m["pt"], ROOT), m["md5"][:12], m["nc"],
                      "no fire/smoke class", "—"])
    L.append(md_table(mh, mrows) + "\n")

    # composition
    L.append("\n## Benchmark composition\n")
    ch = ["Category", "N"] + sorted({r["source"] for r in rows})
    crows = []
    for c in CATEGORIES:
        sub = [r for r in rows if r["category"] == c]
        counts = collections.Counter(r["source"] for r in sub)
        crows.append([c, len(sub)] + [counts.get(s, 0)
                                      for s in sorted({r["source"] for r in rows})])
    L.append(md_table(ch, crows) + "\n")

    # headline
    L.append("\n## Headline (conf ≥ %.2f)\n" % args.conf)
    hh = ["Model", "fire recall\nfire-only", "fire recall\nfire+smoke",
          "smoke recall\nsmoke-only", "smoke recall\nfire+smoke",
          "both recall\nfire+smoke", "FP any\n(other)", "FP fire\n(other)",
          "FP smoke\n(other)"]
    hrows = []
    for tag, m in us.items():
        mt = category_metrics(scored_all[tag], rows, args.conf)
        mt = mask_metrics(mt, m["fire_i"] is not None, m["smoke_i"] is not None)
        hrows.append([tag,
                      fmt_pct(mt["fire_recall_fire_only"]),
                      fmt_pct(mt["fire_recall_fire_smoke"]),
                      fmt_pct(mt["smoke_recall_smoke_only"]),
                      fmt_pct(mt["smoke_recall_fire_smoke"]),
                      fmt_pct(mt["both_recall_fire_smoke"]),
                      fmt_pct(mt["fp_any_other"]),
                      fmt_pct(mt["fp_fire_other"]),
                      fmt_pct(mt["fp_smoke_other"])])
    L.append(md_table(hh, hrows) + "\n")
    L.append("Higher is better for every column **except** `FP …` (lower is better).\n")

    # threshold sweep
    L.append("\n## Threshold sweep (recall over all fire / all smoke images, and FP on other)\n")
    for thr in thresholds:
        L.append("### conf ≥ %.2f\n" % thr)
        th = ["Model", "fire recall\n(all fire imgs)", "smoke recall\n(all smoke imgs)",
              "FP any\n(other)"]
        trows = []
        for tag, m in us.items():
            mt = category_metrics(scored_all[tag], rows, thr)
            mt = mask_metrics(mt, m["fire_i"] is not None, m["smoke_i"] is not None)
            trows.append([tag, fmt_pct(mt["fire_recall_all_fire_images"]),
                          fmt_pct(mt["smoke_recall_all_smoke_images"]),
                          fmt_pct(mt["fp_any_other"])])
        L.append(md_table(th, trows) + "\n")

    # scratch fire-only note
    L.append("\n## Caveats\n")
    L.append("- `scratch` is **fire-only** (`nc=1`): it has no smoke class, so all smoke "
             "columns are N/A by construction, not 0 %.\n")
    L.append("- Positives come only from held-out **test** splits (D-Fire test, FireViewer "
             "test, CCTV Emergency), so they are unseen by every model scored here; negatives "
             "are re-rendered through the seeded rotation/brightness transform into unseen "
             "pixels (see the builder).\n")
    L.append("- `fire+smoke` recall needs BOTH classes ≥ conf on the same image (strict); the "
             "per-class columns are the fairer read.\n")
    L.append("- Image-level max confidence (no NMS-threshold tuning) — a domain probe, not a box "
             "mAP benchmark. For box mAP use `compare_fire_models.py`.\n")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    step("wrote report: %s" % report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
