#!/usr/bin/env python3
"""build_fire_eval_subset.py - produce a self-contained validation dataset for a YOLOv8 export.

Generic helper shared by the fire-model benchmarks (plans/fire-model-dataset-test.md,
plans/fire-model-abonia-benchmark.md). The Roboflow fire exports used here and the
checkpoint `models/fire/best.pt` all share the index order  fire=0 / other(default)=1 /
smoke=2  (verified by loading best.pt), so labels are copied UNCHANGED - no remap.

Images WITHOUT a usable label (no .txt file, or an empty .txt with zero boxes) are
ignored by default because they cannot be scored - pass --include-unlabeled to keep
them as extra negatives.

If --limit N is given and the split is larger than N, a class-stratified subset is
materialised (copies images + labels) so heavy CPU runs stay bounded.

Usage:
    python dev_scripts/build_fire_eval_subset.py <dataset_root> --split train \
        [--eval-dir DIR] [--limit N] [--seed 0] [--names fire,other,smoke] \
        [--include-unlabeled]

Writes into --eval-dir (default <dataset_root>_eval/eval):
    data.yaml        (absolute image paths; class names from the dataset data.yaml,
                      overridden by --names when supplied)
    build_note.txt
"""
import argparse
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(img_dir):
    return [f for f in sorted(os.listdir(img_dir))
            if os.path.splitext(f)[1].lower() in IMG_EXTS]


def load_names(dataset_root):
    """Parse the simple `names:` block of data.yaml (dict or list form)."""
    path = os.path.join(dataset_root, "data.yaml")
    names = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                m = __import__("re").match(r"^(-?\d+):\s*(.+)$", line)
                if m:
                    names[int(m.group(1))] = m.group(2).strip().strip("'\"")
                    continue
                m = __import__("re").match(r"^-\s*(.+)$", line)
                if m:
                    names[len(names)] = m.group(1).strip().strip("'\"")
    return names or None


def label_meta(label_path):
    """Return (class_id_set, has_valid_box). A label is 'usable' iff it has >=1 box line."""
    cids = set()
    valid = False
    if os.path.isfile(label_path):
        with open(label_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 5:
                    valid = True
                    try:
                        cids.add(int(float(parts[0])))
                    except ValueError:
                        pass
    return cids, valid


def materialise(chosen, img_dir, lbl_dir, sub):
    """Copy a chosen image list (+ their label files) under sub/{images,labels}."""
    s_img, s_lbl = os.path.join(sub, "images"), os.path.join(sub, "labels")
    os.makedirs(s_img, exist_ok=True)
    os.makedirs(s_lbl, exist_ok=True)
    for f in chosen:
        shutil.copy2(os.path.join(img_dir, f), os.path.join(s_img, f))
        stem = os.path.splitext(f)[0]
        src_l = os.path.join(lbl_dir, stem + ".txt")
        if os.path.isfile(src_l):
            shutil.copy2(src_l, os.path.join(s_lbl, stem + ".txt"))
    return s_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("--split", default="train", choices=("train", "valid", "test"))
    ap.add_argument("--eval-dir", default=None)
    ap.add_argument("--limit", type=int, default=0, help="max images (0 = whole split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--names", default=None,
                    help="comma list of class names to write into data.yaml "
                         "(overrides the dataset data.yaml)")
    ap.add_argument("--include-unlabeled", action="store_true",
                    help="keep images with no .txt or an empty label as negatives")
    args = ap.parse_args()

    root = os.path.abspath(args.dataset_root)
    img_dir = os.path.join(root, args.split, "images")
    lbl_dir = os.path.join(root, args.split, "labels")
    if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
        sys.exit(f"split '{args.split}' not present: need {img_dir} and {lbl_dir}")

    eval_dir = os.path.abspath(args.eval_dir or os.path.join(
        os.path.dirname(root), os.path.basename(root) + "_eval", "eval"))
    os.makedirs(eval_dir, exist_ok=True)

    names = load_names(root)
    if args.names:
        parts = [p.strip() for p in args.names.split(",") if p.strip()]
        names = {i: n for i, n in enumerate(parts)}
    if not names:
        names = {i: f"class{i}" for i in range(3)}
    nc = max(names) + 1

    imgs = image_files(img_dir)

    # By default drop images that carry no usable label (cannot be scored).
    kept, dropped = [], 0
    for f in imgs:
        _, valid = label_meta(os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt"))
        if valid or args.include_unlabeled:
            kept.append(f)
        else:
            dropped += 1
    imgs = kept
    random.seed(args.seed)

    need_copy = False
    chosen = []
    if args.limit > 0 and len(imgs) > args.limit:
        # Stratified sampling across class-presence signatures.
        by_class = defaultdict(list)
        for f in imgs:
            cids, _ = label_meta(os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt"))
            by_class[frozenset(cids)].append(f)
        chosen = set()
        targets = sorted(by_class, key=lambda k: (-len(k), -len(by_class[k])))
        quota = max(1, args.limit // max(1, len(targets)))
        for k in targets:
            random.shuffle(by_class[k])
            for f in by_class[k][:quota]:
                chosen.add(f)
        pool = [f for k in targets for f in by_class[k] if f not in chosen]
        random.shuffle(pool)
        for f in pool:
            if len(chosen) >= args.limit:
                break
            chosen.add(f)
        chosen = sorted(chosen)
        need_copy = True
        note = (f"subset: {len(chosen)}/{len(imgs)} labeled images sampled "
                f"(limit={args.limit}) from split '{args.split}'")
    elif dropped and not args.include_unlabeled:
        # Whole split requested but unlabeled images were dropped -> copy the kept set.
        chosen = sorted(imgs)
        need_copy = True
        note = (f"labeled-only split '{args.split}': {len(chosen)} images copied "
                f"({dropped} unlabeled dropped)")
    else:
        note = f"whole split '{args.split}': {len(imgs)} images referenced in place (no copies)"

    if need_copy:
        sub = os.path.join(eval_dir, args.split)
        val_img = materialise(chosen, img_dir, lbl_dir, sub)
    else:
        val_img = img_dir
    if dropped:
        note += f"; {dropped} unlabeled image(s) ignored"

    # train key: point to real train split if present, else same as val
    train_img = os.path.join(root, "train", "images")
    if not os.path.isdir(train_img):
        train_img = val_img

    yaml_path = os.path.join(eval_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write("path: %s\n" % root)
        fh.write("train: %s\n" % train_img)
        fh.write("val: %s\n" % val_img)
        fh.write("test: %s\n" % val_img)
        fh.write("nc: %d\n" % nc)
        fh.write("names:\n")
        for i in range(nc):
            fh.write("  %d: %s\n" % (i, names.get(i, f"class{i}")))

    with open(os.path.join(eval_dir, "build_note.txt"), "w", encoding="utf-8") as fh:
        fh.write(note + "\nnames: %s\nval: %s\n" % (names, val_img))

    print(note)
    print("data.yaml ->", yaml_path)
    print("names:", names)


if __name__ == "__main__":
    main()
