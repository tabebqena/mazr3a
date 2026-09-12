#!/usr/bin/env python3
"""score_fire_images.py - run the PRODUCTION fire model over one or more image
trees (or a sample) and review the fire confidence.

Bulk sibling of [`analyze_fire_alert_frames.py`](analyze_fire_alert_frames.py):
that one prints raw+bonus boxes (+ a COCO cross-check) for a handful of frames;
THIS one scores whole trees and summarises the distribution - the tool for
"does the model fire on this negative set?" (e.g. the Stanford Dogs download, or
a mixed folder of sunsets / beaches / animals).

It reuses the production decoder (`FireModel` from `firewatch/firewatch.py`,
`models/fire` OpenVINO IR) and reads the model's raw class scores directly, so
the reported `fire` value is the true max fire confidence (no threshold / NMS
bias). Writes a per-image CSV + a summary, optionally copying the flagged images
for visual review.

Execution is logged STEP BY STEP with timings and progress/ETA.

HIERARCHIES
-----------
Any nesting is supported. `--per-dir N` samples N images from EACH directory that
directly holds images (at any depth) - so a tree like
``places/beach``, ``sunset/nighttime``, ``camels-horse/train/horse`` is covered
group-by-group. `--sample N` instead draws N at random across the whole tree.
Multiple roots may be given at once.

SKIPPING
--------
`test` / `tests` / `testing` directories are skipped AUTOMATICALLY (disable with
`--no-skip-test`). Extra folders are skipped with `--skip-dirs`, a
comma-separated list of case-insensitive **substrings** matched against every
directory name, e.g. ``--skip-dirs valid,preview,fog,nighttime``. Can be repeated.

Usage:
    # stratified sample: 15 images per image-holding directory
    .venv/bin/python dev_scripts/score_fire_images.py <dir> [<dir2> ...] --per-dir 15

    # every image under every root (skip any 'test' dirs automatically)
    .venv/bin/python dev_scripts/score_fire_images.py <dir> --per-dir 0 --sample 0

    # more skips + copy everything scoring >= 0.35 for a look
    .venv/bin/python dev_scripts/score_fire_images.py "fire-model-training/to extract" \
        --per-dir 0 --sample 0 --skip-dirs valid,preview --out fire-model-training/scores_x \
        --copy-hits fire-model-training/hits_x --min-score 0.35

Read-only against the images; writes only under --out / --copy-hits.
"""
import argparse
import csv
import os
import random
import shutil
import sys
import time

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THRESHOLDS = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.35, 0.30, 0.20]
AUTO_SKIP = ("test", "tests", "testing")

_T0 = time.time()


def step(msg):
    """Print one execution step, prefixed with the elapsed wall-clock time."""
    print("[step %6.1fs] %s" % (time.time() - _T0, msg), flush=True)


def fmt_eta(seconds):
    """Compact human ETA (e.g. '3m20s')."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return "%ds" % seconds
    if seconds < 3600:
        return "%dm%02ds" % (seconds // 60, seconds % 60)
    return "%dh%02dm" % (seconds // 3600, (seconds % 3600) // 60)


def load_fire_model(model_dir):
    """Reuse the PRODUCTION decoder from firewatch/firewatch.py."""
    repo = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.insert(0, os.path.join(repo, "firewatch"))
    from firewatch import FireModel  # noqa: E402
    return FireModel(model_dir)


def raw_class_max(model, pil):
    """(max fire score, max smoke score) from the model's raw output tensor.

    Mirrors FireModel.detect's decoding (letterbox + optional sigmoid) but keeps
    only the per-class maxima, so there is no threshold/NMS and no 8400-anchor
    NMS cost per image.
    """
    blob, _scale, _px, _py = model._letterbox(pil, model.width)
    out = np.asarray(model._compiled([blob])[model._output])
    data = out[0] if out.ndim == 3 else out
    if (data.ndim == 2 and data.shape[0] == 4 + model.nc
            and data.shape[1] > data.shape[0]):
        data = data.T
    labels = [str(name).lower() for name in model.labels]
    fire_i = labels.index("fire") if "fire" in labels else 0
    smoke_i = labels.index("smoke") if "smoke" in labels else None
    if data.shape[1] == 4 + model.nc:               # raw per-class output
        sc = data[:, 4:4 + model.nc].astype(np.float32)
        if np.nanmax(sc) > 1.0:
            sc = 1.0 / (1.0 + np.exp(-sc))
        fire = float(sc[:, fire_i].max())
        smoke = float(sc[:, smoke_i].max()) if smoke_i is not None else 0.0
    elif data.shape[1] == 6:                        # YOLO26 end-to-end NMS
        conf = data[:, 4].astype(np.float32)
        cls = data[:, 5].astype(np.int64)
        fsel = cls == fire_i
        fire = float(conf[fsel].max()) if fsel.any() else 0.0
        if smoke_i is not None:
            ssel = cls == smoke_i
            smoke = float(conf[ssel].max()) if ssel.any() else 0.0
        else:
            smoke = 0.0
    else:
        raise RuntimeError("unexpected model output width %d" % data.shape[1])
    return fire, smoke


def _is_skipped(name, skip):
    """True when a directory name matches any (substring) skip pattern."""
    low = name.lower()
    return any(p in low for p in skip)


def _norm_skip(extra, no_skip_test):
    """The skip-substring set: auto test dirs (unless disabled) + --skip-dirs.

    `extra` is the list of raw --skip-dirs values (argparse appends each flag);
    every value is comma-split, so both `--skip-dirs a,b` and repeated
    `--skip-dirs a --skip-dirs b` work.
    """
    skip = set() if no_skip_test else {p.lower() for p in AUTO_SKIP}
    for entry in extra:
        for part in entry.split(","):
            part = part.strip().lower()
            if part:
                skip.add(part)
    return skip


def discover(roots, skip, stats):
    """[(image_path, group)] under every root, at any depth, minus skipped dirs.

    `group` is the directory of the image RELATIVE to its root - so a nested
    hierarchy keeps a readable, collision-free label in the CSV/summary.
    Skipped directory names are recorded in `stats["skipped"]`.
    """
    items = []
    for root in roots:
        if os.path.isfile(root):
            if os.path.splitext(root)[1].lower() in IMG_EXTS:
                items.append((root, os.path.basename(os.path.dirname(root))))
            else:
                step("  ! not an image, skipped: %s" % root)
            continue
        if not os.path.isdir(root):
            step("  ! not a path, skipped: %s" % root)
            continue
        step("  walking %s ..." % root)
        root_hits = 0
        base = root.rstrip(os.sep) or root
        for dirpath, dirs, names in os.walk(base):
            keep = [d for d in dirs if not _is_skipped(d, skip)]
            for d in dirs:
                if d not in keep:
                    stats["skipped"].add(d)
            dirs[:] = sorted(keep)
            imgs = sorted(n for n in names
                          if os.path.splitext(n)[1].lower() in IMG_EXTS)
            if not imgs:
                continue
            rel = os.path.relpath(dirpath, base)
            stats["image_dirs"].add(rel)
            for n in imgs:
                items.append((os.path.join(dirpath, n), rel))
            root_hits += len(imgs)
        step("  %s -> %d image(s)" % (root, root_hits))
    return items


def gather(roots, per_dir, sample, seed, skip):
    """Stratified (`per_dir` per image-holding dir) or random (`sample`) pick."""
    stats = {"skipped": set(), "image_dirs": set()}
    step("discovering images under %d root(s) ..." % len(roots))
    items = discover(roots, skip, stats)
    step("found %d image(s) across %d image-holding dir(s)"
         % (len(items), len(stats["image_dirs"])))
    if stats["skipped"]:
        names = ", ".join(sorted(stats["skipped"]))
        step("skipped %d dir name(s): %s" % (len(stats["skipped"]), names))
    rng = random.Random(seed)
    if per_dir and per_dir > 0:
        groups = {}
        for path, rel in items:
            groups.setdefault(rel, []).append(path)
        step("sampling: %d per-dir across %d dir(s) (seed %d)"
             % (per_dir, len(groups), seed))
        picked = []
        for rel in sorted(groups):
            files = sorted(groups[rel])
            rng.shuffle(files)
            picked.extend((p, rel) for p in files[:per_dir])
        step("sample picked: %d image(s)" % len(picked))
        return picked
    if sample and sample < len(items):
        step("sampling: %d at random across the tree (seed %d)" % (sample, seed))
        rng.shuffle(items)
        return sorted(items[:sample], key=lambda t: t[0])
    step("sample: none - scoring every discovered image")
    return items


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+",
                    help="one or more image dirs (recursed) or single images")
    ap.add_argument("--model-dir", default="models/fire")
    ap.add_argument("--per-dir", type=int, dest="per_dir", default=15,
                    help="sample this many per image-holding dir at any depth; "
                         "0 disables [15]")
    ap.add_argument("--sample", type=int, default=0,
                    help="else sample N images at random across all roots "
                         "(0 = all) [0]")
    ap.add_argument("--skip-dirs", dest="skip_dirs", action="append", default=[],
                    help="comma-separated dir-name substrings to skip "
                         "(case-insensitive); repeatable. test/tests/testing are "
                         "skipped automatically")
    ap.add_argument("--no-skip-test", dest="no_skip_test", action="store_true",
                    help="do NOT auto-skip test/tests/testing directories")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="fire-model-training/fire_scores",
                    help="output dir for the CSV + summary [fire-model-training/fire_scores]")
    ap.add_argument("--top", type=int, default=25,
                    help="how many worst offenders to print [25]")
    ap.add_argument("--min-score", type=float, dest="min_score", default=0.5,
                    help="threshold labelled 'hit' in the summary [0.5]")
    ap.add_argument("--copy-hits", dest="copy_hits",
                    help="copy images with fire >= --min-score here (review)")
    args = ap.parse_args()

    step("start")
    step("roots=%s" % ", ".join(args.images))
    step("options: per_dir=%s sample=%s min_score=%s seed=%s top=%s"
         % (args.per_dir, args.sample, args.min_score, args.seed, args.top))
    step("skip-dirs given: %s" % (args.skip_dirs or "-",))
    step("no-skip-test=%s" % args.no_skip_test)

    missing = [p for p in args.images if not os.path.exists(p)]
    if missing:
        sys.exit("path(s) not found: %s" % ", ".join(missing))

    skip = _norm_skip(args.skip_dirs, args.no_skip_test)
    step("skip patterns in effect: %s" % (", ".join(sorted(skip)) or "-"))

    files = gather(args.images, args.per_dir, args.sample, args.seed, skip)
    if not files:
        sys.exit("no images found under %s (check --skip-dirs: %s)"
                 % (", ".join(args.images), sorted(skip)))

    step("loading fire model from %s ..." % args.model_dir)
    model = load_fire_model(args.model_dir)
    step("model ready: classes=%s input=%dx%d" % (model.labels, model.width,
                                                  model.height))

    total = len(files)
    step("scoring %d image(s) (progress every 100) ..." % total)
    rows = []
    t_score = time.time()
    for n, (path, group) in enumerate(files, 1):
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - keep scanning
            step("  ! skip %s (%s)" % (path, exc))
            continue
        fire, smoke = raw_class_max(model, pil)
        rows.append({"path": path, "file": os.path.basename(path),
                     "group": group, "fire": fire, "smoke": smoke})
        if n % 100 == 0 or n == total:
            done = len(rows)
            rate = done / max(1e-6, time.time() - t_score)
            eta = fmt_eta((total - n) / rate) if rate else "-"
            hits = sum(1 for r in rows if r["fire"] >= args.min_score)
            step("  scored %d/%d (%.0f%%) - hits >= %.2f: %d - %.1f img/s - ETA %s"
                 % (n, total, 100.0 * n / total, args.min_score, hits, rate, eta))
    step("scoring finished: %d image(s) in %.1fs" % (len(rows),
                                                     time.time() - t_score))

    step("sorting by fire score (descending) ...")
    rows.sort(key=lambda r: r["fire"], reverse=True)
    fires = np.array([r["fire"] for r in rows], dtype=np.float32)

    # ---- outputs ------------------------------------------------------------
    csv_path = None
    if args.out:
        step("writing per-image CSV -> %s/fire_scores.csv" % args.out)
        os.makedirs(args.out, exist_ok=True)
        csv_path = os.path.join(args.out, "fire_scores.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["path", "file", "group", "fire", "smoke"])
            w.writeheader()
            w.writerows(rows)
        step("wrote %d row(s) to %s" % (len(rows), csv_path))

    if args.copy_hits:
        step("copying hits (fire >= %.2f) -> %s ..." % (args.min_score, args.copy_hits))
        os.makedirs(args.copy_hits, exist_ok=True)
        copied = 0
        for r in rows:
            if r["fire"] >= args.min_score:
                shutil.copy2(r["path"], os.path.join(args.copy_hits, r["file"]))
                copied += 1
        step("copied %d hit(s)" % copied)

    step("building summary ...")
    lines = ["Fire model scoring - %s" % ", ".join(args.images),
             "=" * 64,
             "images scored : %d" % len(rows),
             "dirs skipped  : %s" % (", ".join(sorted(skip)) if skip else "-"),
             "fire  min/mean/max : %.4f / %.4f / %.4f"
             % (float(fires.min()), float(fires.mean()), float(fires.max())),
             "", "Images at or above a fire threshold:"]
    for t in THRESHOLDS:
        k = int((fires >= t).sum())
        lines.append("  >= %.2f : %5d / %d  (%.1f%%)"
                     % (t, k, len(rows), 100.0 * k / len(rows)))
    lines.append("")
    lines.append("Worst %d offenders (fire):" % min(args.top, len(rows)))
    for r in rows[:args.top]:
        lines.append("  %.4f  %-40s %s" % (r["fire"], r["group"], r["file"]))
    report = "\n".join(lines)
    print("\n" + report)
    if csv_path:
        step("writing SUMMARY.txt -> %s" % args.out)
        with open(os.path.join(args.out, "SUMMARY.txt"), "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
    step("done: %d scored, %d >= %.2f, max fire %.4f (total %.1fs)"
         % (len(rows),
            int((fires >= args.min_score).sum()), args.min_score,
            float(fires.max()), time.time() - _T0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
