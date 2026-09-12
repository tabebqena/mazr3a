#!/usr/bin/env python3
"""prep_fire_large_dataset.py - build a self-contained train/val layout for fine-tuning the
fire model on the LARGE local datasets.

Inputs (all under the git-ignored fire-model-training/):
    --ready      Roboflow export root, e.g. fire-model-training/ready_fire_smoke_dataset.yolov8
                 (uses its train/{images,labels}; 12,799 imgs, fire/other/smoke at idx 0/1/2)
    --negatives  curated pure-background dir, e.g. fire-model-training/default-other (430 imgs, no labels)
                 -> added to TRAIN as empty-label background samples (FP reduction)

Output (default fire-model-training/large_finetune/):
    train/images/ train/labels/   # labeled ready split + empty-label ready + negatives
    val/images/   val/labels/     # held-out labeled subset (mAP / early stopping)
    data.yaml                     # names fire/other/smoke, relative train/val, abs path
    prep_report.txt               # counts + class coverage
    <out>.zip (optional --zip)    # single archive for Colab upload

Class contract: the export class ids are ALREADY fire(0)/other(1)/smoke(2), matching
models/fire/best.pt, so labels are copied with NO remap; only data.yaml names are set.

Deterministic split: images are grouped by "source key" (video/clip prefix, so frames from the
same video never straddle train/val), groups are shuffled with --seed, and groups are assigned to
val until ~--val-frac of the LABELED images are held out. Empty-label/background images only go to
train.

Usage:
    .venv/bin/python dev_scripts/prep_fire_large_dataset.py \
        --ready fire-model-training/ready_fire_smoke_dataset.yolov8 \
        --negatives fire-model-training/default-other \
        --out fire-model-training/large_finetune \
        --val-frac 0.05 --seed 0 --zip
"""
import argparse
import os
import random
import re
import shutil
import zipfile
from collections import Counter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = ["fire", "other", "smoke"]


def img2label(img_dir, img_name):
    return os.path.join(img_dir.replace("images", "labels"), os.path.splitext(img_name)[0] + ".txt")


def read_classes(label_path):
    """Return sorted set of class ids in a label file (empty file -> ())."""
    if not os.path.isfile(label_path):
        return ()
    cls = []
    with open(label_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if parts:
                try:
                    cls.append(int(float(parts[0])))
                except ValueError:
                    pass
    return tuple(sorted(set(cls)))


def source_key(name):
    """Group frames of the same video/clip: strip trailing '_f<digits>_jpg.rf.<hash>'."""
    m = re.match(r"^(.*?)(?:_f\d+)?_jpg\.rf\..*$", os.path.splitext(name)[0])
    if m:
        base = m.group(1)
        if base:
            return base
    return os.path.splitext(name)[0]


def copy_if_missing(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ready", required=True, help="ready_fire_smoke_dataset export root")
    ap.add_argument("--negatives", action="append", default=None,
                    help="curated background dir; REPEATABLE to mix several "
                         "(default: fire-model-training/default-other). Each is "
                         "walked RECURSIVELY and added to train as empty labels. "
                         "Use --no-negatives to skip.")
    ap.add_argument("--out", default="fire-model-training/large_finetune")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-negatives", action="store_true", help="do not add --negatives images")
    ap.add_argument("--zip", nargs="?", const="", default=None,
                    help="also write a Colab-upload zip (default: <out>_colab.zip); no value => default path")
    args = ap.parse_args()

    ready = os.path.abspath(args.ready)
    img_dir = os.path.join(ready, "train", "images")
    lbl_dir = os.path.join(ready, "train", "labels")
    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        raise SystemExit(f"ready train split not found under {ready} (need train/images + train/labels)")

    out = os.path.abspath(args.out)
    train_img, train_lbl = os.path.join(out, "train", "images"), os.path.join(out, "train", "labels")
    val_img, val_lbl = os.path.join(out, "val", "images"), os.path.join(out, "val", "labels")
    for d in (train_img, train_lbl, val_img, val_lbl):
        os.makedirs(d, exist_ok=True)

    imgs = sorted(f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS)
    rng = random.Random(args.seed)

    # 1. Classify ready images: labeled (>=1 box) vs background (empty/missing label)
    labeled = []   # list of (name, classes)
    bg_ready = []  # names of empty-label ready images
    for n in imgs:
        c = read_classes(os.path.join(lbl_dir, os.path.splitext(n)[0] + ".txt"))
        if c:
            labeled.append((n, c))
        else:
            bg_ready.append(n)

    # 2. Leakage-aware split of LABELED images by source key (video/clip), seeded
    groups = {}
    for n, c in labeled:
        groups.setdefault(source_key(n), []).append((n, c))
    group_keys = sorted(groups)
    rng.shuffle(group_keys)

    n_labeled = len(labeled)
    target_val = int(round(n_labeled * args.val_frac))
    val_names, train_labeled = {}, {}
    val_count = 0
    for key in group_keys:
        g = groups[key]
        if val_count < target_val:
            for n, c in g:
                val_names[n] = c
            val_count += len(g)
        else:
            for n, c in g:
                train_labeled[n] = c

    # Optional tiny rebalance so a giant late group does not blow past the target too far:
    # if val is far above target and the last added group was big, that's accepted (leakage-free > exact split).

    # 3. Copy labeled train images+labels
    n_train_lab = 0
    train_classes = Counter()
    for n in sorted(train_labeled):
        copy_if_missing(os.path.join(img_dir, n), os.path.join(train_img, n))
        src_l = os.path.join(lbl_dir, os.path.splitext(n)[0] + ".txt")
        copy_if_missing(src_l, os.path.join(train_lbl, os.path.splitext(n)[0] + ".txt"))
        for c in train_labeled[n]:
            train_classes[c] += 1
        n_train_lab += 1

    # 4. Copy empty-label ready images to train as background (empty .txt)
    n_bg = 0
    for n in bg_ready:
        copy_if_missing(os.path.join(img_dir, n), os.path.join(train_img, n))
        lbl = os.path.join(train_lbl, os.path.splitext(n)[0] + ".txt")
        if not os.path.exists(lbl):
            open(lbl, "w", encoding="utf-8").close()
        n_bg += 1

    # 5. Add curated negatives to train as background (empty .txt).
    #    Several dirs may be given (--negatives is repeatable) and each is walked
    #    RECURSIVELY, so nested negative collections need no staging copy.
    n_neg = 0
    neg_report = []
    if not args.no_negatives:
        neg_dirs = args.negatives or ["fire-model-training/default-other"]
        for raw in neg_dirs:
            neg_src = os.path.abspath(raw)
            if not os.path.isdir(neg_src):
                raise SystemExit(
                    f"--negatives dir not found: {neg_src} (pass --no-negatives to skip)")
            n_src = 0
            for dirpath, _dirs, names in os.walk(neg_src):
                for f in sorted(names):
                    if os.path.splitext(f)[1].lower() not in IMG_EXTS:
                        continue
                    dst = os.path.join(train_img, f)
                    if os.path.exists(dst):        # basename clash across sources
                        stem, ext = os.path.splitext(f)
                        k = 1
                        while os.path.exists(os.path.join(
                                train_img, f"{stem}__neg{k}{ext}")):
                            k += 1
                        dst = os.path.join(train_img, f"{stem}__neg{k}{ext}")
                    copy_if_missing(os.path.join(dirpath, f), dst)
                    lbl = os.path.join(
                        train_lbl,
                        os.path.splitext(os.path.basename(dst))[0] + ".txt")
                    if not os.path.exists(lbl):
                        open(lbl, "w", encoding="utf-8").close()
                    n_src += 1
            n_neg += n_src
            neg_report.append((raw, n_src))

    # 6. Copy val images+labels
    n_val = 0
    val_classes = Counter()
    for n in sorted(val_names):
        copy_if_missing(os.path.join(img_dir, n), os.path.join(val_img, n))
        copy_if_missing(os.path.join(lbl_dir, os.path.splitext(n)[0] + ".txt"),
                        os.path.join(val_lbl, os.path.splitext(n)[0] + ".txt"))
        for c in val_names[n]:
            val_classes[c] += 1
        n_val += 1

    # 7. data.yaml
    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write("# Fine-tune HF YOLO26-S best.pt on ready_fire_smoke + default-other negatives.\n")
        fh.write("# Class order matches the checkpoint: fire=0, other=1, smoke=2 (no remap; names only).\n")
        fh.write(f"path: {out}\n")
        fh.write("train: train/images\n")
        fh.write("val: val/images\n")
        fh.write(f"nc: {len(NAMES)}\n")
        fh.write("names:\n")
        for i, n in enumerate(NAMES):
            fh.write(f"  {i}: {n}\n")

    # 8. report (count ACTUAL on-disk files - default-other negatives are a subset of the
    # ready empty-label set, so inputs overlap and must not simply be summed)
    ready_img_names = set(imgs)
    n_neg_new = sum(1 for f in sorted(os.listdir(train_img))
                    if f not in ready_img_names and os.path.splitext(f)[1].lower() in IMG_EXTS)
    n_train_actual = len(os.listdir(train_img))
    n_val_actual = len(os.listdir(val_img))
    n_train_empty = sum(1 for f in os.listdir(train_lbl)
                        if os.path.getsize(os.path.join(train_lbl, f)) == 0)
    report = []
    report.append(f"Large fire/smoke fine-tune prep ({os.path.basename(out)})")
    report.append("=" * 60)
    report.append(f"source ready      : {ready} (train split)")
    report.append(f"source negatives  : "
                  f"{'(none - skipped)' if args.no_negatives else ''}")
    for raw_src, count in neg_report:
        report.append(f"    - {raw_src} : {count} img(s)")
    report.append(f"negatives added   : {n_neg}")
    report.append(f"seed / val-frac   : {args.seed} / {args.val_frac}")
    report.append("")
    report.append(f"ready total imgs   : {len(imgs)}")
    report.append(f"  labeled (>=1 box): {n_labeled}")
    report.append(f"  empty-label      : {len(bg_ready)}")
    report.append("")
    report.append(f"TRAIN images (actual files) = {n_train_actual}")
    report.append(f"  of which empty-label/background .txt = {n_train_empty}")
    report.append(f"  negatives new to ready (only if outside train split): {n_neg_new}")
    report.append(f"  per-class boxes (train labeled): " + ", ".join(f"{NAMES[k]}={v}" for k, v in sorted(train_classes.items())))
    report.append(f"VAL images (actual files, all labeled) = {n_val_actual}")
    report.append(f"  per-class boxes (val): " + ", ".join(f"{NAMES[k]}={v}" for k, v in sorted(val_classes.items())))
    report.append("")
    report.append("data.yaml -> " + yaml_path)
    missing_val = [NAMES[i] for i in range(len(NAMES)) if val_classes.get(i, 0) == 0]
    report.append("WARNING: val has no boxes for class(es): " + (", ".join(missing_val) if missing_val else "none"))
    report_text = "\n".join(report)
    with open(os.path.join(out, "prep_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report_text + "\n")
    print(report_text)

    # 9. optional zip
    if args.zip is not None:
        zpath = args.zip if args.zip else out + "_colab.zip"
        zpath = os.path.abspath(zpath)
        # Zip top-level = content of <out> (train/, val/, data.yaml, prep_report.txt)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for root, _dirs, files in os.walk(out):
                for f in sorted(files):
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, out)
                    zf.write(full, rel)
        print(f"\nColab zip -> {zpath}")


if __name__ == "__main__":
    main()
