#!/usr/bin/env python3
"""build_fire_negatives_eval.py - stage curated negative images as an empty-label eval split.

User-curated "background" images (no fire/smoke, e.g. the organised
fire-model-training/default-other/ set) are copied into a self-contained eval split so a
detector can be audited purely for fire/smoke false positives.

Output layout (git-ignored under fire-model-training/*):
    <eval-dir>/
      images/      <- copied image files
      labels/      <- one EMPTY .txt per image (background => no GT boxes)
      data.yaml    <- names fire/other/smoke, val -> images dir

Usage:
    python dev_scripts/build_fire_negatives_eval.py --src fire-model-training/default-other
        [--eval-dir fire-model-training/eval/negatives] [--names fire,other,smoke]

Then run a pure FP audit:
    .venv/bin/python dev_scripts/test_fire_model.py models/fire/best.pt \
        fire-model-training/eval/negatives/data.yaml --out fire-model-training/eval/negatives/results
"""
import argparse
import os
import shutil
import sys

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="dir of curated negative images (e.g. fire-model-training/default-other)")
    ap.add_argument("--eval-dir", default="fire-model-training/eval/negatives")
    ap.add_argument("--names", default="fire,other,smoke",
                    help="comma class names (dataset/model order 0=fire 1=other 2=smoke)")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    if not os.path.isdir(src):
        sys.exit(f"source dir not found: {src}")

    img_dir = os.path.join(args.eval_dir, "images")
    lbl_dir = os.path.join(args.eval_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(src)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS)
    n_img = n_lbl = 0
    for f in files:
        dst_img = os.path.join(img_dir, f)
        if not os.path.exists(dst_img):
            shutil.copy2(os.path.join(src, f), dst_img)
        n_img += 1
        # empty label file => background/negative (no GT boxes)
        lbl = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
        if not os.path.exists(lbl):
            open(lbl, "w", encoding="utf-8").close()
        n_lbl += 1

    names = [p.strip() for p in args.names.split(",") if p.strip()]
    if not names:
        sys.exit("--names empty")
    nc = len(names)

    abs_img = os.path.abspath(img_dir)
    yaml_path = os.path.join(args.eval_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write("path: %s\n" % os.path.abspath(args.eval_dir))
        fh.write("train: %s\n" % abs_img)
        fh.write("val: %s\n" % abs_img)
        fh.write("test: %s\n" % abs_img)
        fh.write("nc: %d\n" % nc)
        fh.write("names:\n")
        for i, n in enumerate(names):
            fh.write("  %d: %s\n" % (i, n))

    note = os.path.join(args.eval_dir, "build_note.txt")
    with open(note, "w", encoding="utf-8") as fh:
        fh.write("negative eval split: %d curated background images (no GT boxes)\n" % n_img)
        fh.write("source: %s\nnames: %s\n" % (src, names))

    print(f"negatives split ready: {n_img} images, {n_lbl} empty labels -> {args.eval_dir}")
    print("data.yaml ->", yaml_path)


if __name__ == "__main__":
    main()
