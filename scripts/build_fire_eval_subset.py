#!/usr/bin/env python3
"""build_fire_eval_subset.py - produce a validation data.yaml for a YOLOv8 dataset.

Generic helper shared by the fire-model benchmarks. Because the Abonia dataset class
indices (Fire=0, default=1, smoke=2) already align with the checkpoint `best.pt`
(fire=0, other=1, smoke=2), NO label remap is required here: the builder just writes a
working `data.yaml` whose `val`/`test` keys point at the requested split's images dir.

If `--limit N` is given and the split is larger than N, a stratified subset is created
(copies images + labels) so heavy runs stay bounded on CPU.

Usage:
    python scripts/build_fire_eval_subset.py <dataset_root> --split test \
        [--eval-dir DIR] [--limit N] [--seed 0]

Writes into --eval-dir (default <dataset_root>_eval/eval):
    data.yaml   (absolute image paths; names taken from the dataset data.yaml)
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


def classes_of(label_path):
    """Set of class indices present in a YOLO label file."""
    out = set()
    if os.path.isfile(label_path):
        with open(label_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if parts:
                    try:
                        out.add(int(float(parts[0])))
                    except ValueError:
                        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("--split", default="test", choices=("train", "valid", "test"))
    ap.add_argument("--eval-dir", default=None)
    ap.add_argument("--limit", type=int, default=0, help="max images (0 = whole split)")
    ap.add_argument("--seed", type=int, default=0)
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
    if not names:
        names = {i: f"class{i}" for i in range(3)}
    nc = max(names) + 1

    imgs = image_files(img_dir)
    random.seed(args.seed)

    if args.limit > 0 and len(imgs) > args.limit:
        # Stratified sampling across classes (and some negatives for realism).
        by_class = defaultdict(list)
        for f in imgs:
            cls = classes_of(os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt"))
            by_class[frozenset(cls)].append(f)
        chosen = set()
        targets = sorted(by_class, key=lambda k: (-len(k), -len(by_class[k])))
        quota = max(1, args.limit // max(1, len(targets)))
        for k in targets:
            random.shuffle(by_class[k])
            for f in by_class[k][:quota]:
                chosen.add(f)
        # top-up to reach --limit
        pool = [f for k in targets for f in by_class[k] if f not in chosen]
        random.shuffle(pool)
        for f in pool:
            if len(chosen) >= args.limit:
                break
            chosen.add(f)
        chosen = sorted(chosen)
        # materialise subset dir so val sees self-contained images+labels
        sub = os.path.join(eval_dir, args.split)
        s_img, s_lbl = os.path.join(sub, "images"), os.path.join(sub, "labels")
        os.makedirs(s_img, exist_ok=True)
        os.makedirs(s_lbl, exist_ok=True)
        for f in chosen:
            shutil.copy2(os.path.join(img_dir, f), os.path.join(s_img, f))
            stem = os.path.splitext(f)[0]
            src_l = os.path.join(lbl_dir, stem + ".txt")
            if os.path.isfile(src_l):
                shutil.copy2(src_l, os.path.join(s_lbl, stem + ".txt"))
        val_img = s_img
        note = (f"subset: {len(chosen)}/{len(imgs)} images sampled (limit={args.limit}) "
                f"from split '{args.split}'")
    else:
        val_img = img_dir
        note = f"whole split '{args.split}': {len(imgs)} images referenced in place (no copies)"

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
