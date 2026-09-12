#!/usr/bin/env python3
"""score_fire_images.py - run the PRODUCTION fire model over a folder (or a
sample) of images and review the fire confidence.

Bulk sibling of [`analyze_fire_alert_frames.py`](analyze_fire_alert_frames.py):
that one prints raw+bonus boxes (+ a COCO cross-check) for a handful of frames;
THIS one scores a whole tree and summarises the distribution - the tool for
"does the model fire on this negative set?" (e.g. the Stanford Dogs download).

It reuses the production decoder (`FireModel` from `firewatch/firewatch.py`,
`models/fire` OpenVINO IR) and reads the model's raw class scores directly, so
the reported `fire` value is the true max fire confidence (no threshold / NMS
bias). Writes a per-image CSV + a summary, optionally copying the flagged images
for visual review.

Usage:
    # stratified sample: 15 images from EACH subdirectory (e.g. each dog breed)
    .venv/bin/python dev_scripts/score_fire_images.py \
        fire-model-training/stanford-dogs-dataset/images/Images --per-dir 15 \
        --out fire-model-training/fire_scores

    # every image, no sampling
    .venv/bin/python dev_scripts/score_fire_images.py <dir> --sample 0

    # also copy everything scoring >= 0.5 for a look
    .venv/bin/python dev_scripts/score_fire_images.py <dir> --per-dir 15 \
        --copy-hits fire-model-training/dog_fire_hits --min-score 0.5

Read-only against the images; writes only under --out / --copy-hits.
"""
import argparse
import csv
import os
import random
import shutil
import sys

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THRESHOLDS = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.35, 0.30, 0.20]


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


def gather(root, per_dir, sample, seed):
    """Image paths under `root`: `per_dir` per immediate subdir, else `sample`."""
    if os.path.isfile(root):
        return [root]
    subdirs = [d for d in sorted(os.listdir(root))
               if os.path.isdir(os.path.join(root, d))]
    rng = random.Random(seed)
    if per_dir and subdirs:
        picked = []
        for d in subdirs:
            files = sorted(f for f in os.listdir(os.path.join(root, d))
                           if os.path.splitext(f)[1].lower() in IMG_EXTS)
            rng.shuffle(files)
            picked.extend(os.path.join(root, d, f) for f in files[:per_dir])
        return picked
    files = []
    for dirpath, _dirs, names in os.walk(root):
        for n in sorted(names):
            if os.path.splitext(n)[1].lower() in IMG_EXTS:
                files.append(os.path.join(dirpath, n))
    files.sort()
    if sample and sample < len(files):
        rng.shuffle(files)
        files = sorted(files[:sample])
    return files


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", help="image dir (recursed) or a single image")
    ap.add_argument("--model-dir", default="models/fire")
    ap.add_argument("--per-dir", type=int, dest="per_dir", default=15,
                    help="sample this many per immediate subdir (e.g. per dog "
                         "breed); 0 disables [15]")
    ap.add_argument("--sample", type=int, default=0,
                    help="else sample N images at random across the tree "
                         "(0 = all) [0]")
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

    if not os.path.exists(args.images):
        sys.exit("path not found: %s" % args.images)

    files = gather(args.images, args.per_dir, args.sample, args.seed)
    if not files:
        sys.exit("no images found under %s" % args.images)

    model = load_fire_model(args.model_dir)
    print("model %s classes=%s" % (args.model_dir, model.labels))
    print("scoring %d image(s) from %s ..." % (len(files), args.images))

    rows = []
    for n, path in enumerate(files, 1):
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - keep scanning
            print("  ! skip %s (%s)" % (path, exc))
            continue
        fire, smoke = raw_class_max(model, pil)
        rows.append({"path": path, "file": os.path.basename(path),
                     "group": os.path.basename(os.path.dirname(path)),
                     "fire": fire, "smoke": smoke})
        if n % 250 == 0 or n == len(files):
            hits = sum(1 for r in rows if r["fire"] >= args.min_score)
            print("  %d/%d scored, %d >= %.2f" % (n, len(files), hits, args.min_score))

    rows.sort(key=lambda r: r["fire"], reverse=True)
    fires = np.array([r["fire"] for r in rows], dtype=np.float32)

    # ---- outputs ------------------------------------------------------------
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        csv_path = os.path.join(args.out, "fire_scores.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["path", "file", "group", "fire", "smoke"])
            w.writeheader()
            w.writerows(rows)
        print("\nsaved: %s" % csv_path)

    if args.copy_hits:
        os.makedirs(args.copy_hits, exist_ok=True)
        copied = 0
        for r in rows:
            if r["fire"] >= args.min_score:
                shutil.copy2(r["path"], os.path.join(args.copy_hits, r["file"]))
                copied += 1
        print("copied %d hit(s) -> %s" % (copied, args.copy_hits))

    # ---- summary ------------------------------------------------------------
    lines = ["Fire model scoring - %s" % args.images,
             "=" * 64,
             "images scored : %d" % len(rows),
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
        lines.append("  %.4f  %-12s %s" % (r["fire"], r["group"], r["file"]))
    report = "\n".join(lines)
    print("\n" + report)
    if args.out:
        with open(os.path.join(args.out, "SUMMARY.txt"), "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
