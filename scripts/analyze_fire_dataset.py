#!/usr/bin/env python3
"""analyze_fire_dataset.py - audit a YOLOv8 (Roboflow-style) fire/smoke dataset.

Generic helper shared by the fire-model benchmarks (plans/fire-model-dataset-test.md,
plans/fire-model-abonia-benchmark.md). It does NOT touch ultralytics - just walks the
export layout and reports ground-truth facts.

Layout expected (each split dir contains images/ and labels/):
    <dataset_root>/
      data.yaml          # optional, read for class names
      train/{images,labels}/
      valid/{images,labels}/
      test/{images,labels}/

Usage:
    python scripts/analyze_fire_dataset.py <dataset_root> [--out DIR] [--preview N]

Writes into --out (default: dataset_root/../<name>_eval):
    dataset_summary.txt   human-readable audit
    class_map.json        {"names": {idx: name}, "nc": N} read from data.yaml (if any)
    preview/GT_*.jpg      GT-box annotated samples (class 0/1/2 -> red/gray/yellow)
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

from PIL import Image, ImageDraw

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = {0: (220, 40, 40), 1: (130, 130, 130), 2: (240, 190, 40)}  # fire/other/smoke-ish
DEFAULT_NAMES = {0: "Fire", 1: "default/other", 2: "smoke"}


def load_yaml_names(dataset_root):
    """Read names list/dict from data.yaml with a tiny parser (no pyyaml dep needed)."""
    path = os.path.join(dataset_root, "data.yaml")
    if not os.path.isfile(path):
        return None, path
    names = {}
    nc = None
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("names") and ":" in line:
                continue
            m = re.match(r"^(-?\d+):\s*(.+)$", line)          # dict form: 0: Fire
            if m:
                names[int(m.group(1))] = m.group(2).strip().strip("'\"")
                continue
            m = re.match(r"^-\s*(.+)$", line)                 # list form: - Fire
            if m:
                names[len(names)] = m.group(1).strip().strip("'\"")
            m = re.match(r"^nc:\s*(\d+)", line)
            if m:
                nc = int(m.group(1))
    if not names:
        return None, path
    if nc is None:
        nc = max(names) + 1
    return {"names": names, "nc": nc}, path


def split_layout(dataset_root):
    """Return {split: (images_dir, labels_dir)} for any present train/valid/test dirs."""
    out = {}
    for split in ("train", "valid", "test"):
        img = os.path.join(dataset_root, split, "images")
        lbl = os.path.join(dataset_root, split, "labels")
        if os.path.isdir(img) and os.path.isdir(lbl):
            out[split] = (img, lbl)
    return out


def image_files(img_dir):
    return [f for f in sorted(os.listdir(img_dir)) if os.path.splitext(f)[1].lower() in IMG_EXTS]


def label_stem(img_name):
    return os.path.splitext(img_name)[0]


def read_boxes(label_path):
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    boxes.append(tuple(float(p) for p in parts[:5]))
                except ValueError:
                    pass
    return boxes


def prefix_group(name):
    m = re.match(r"([A-Za-z_]+?)(?:_|\d)", name)
    return m.group(1) if m else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("--out", default=None, help="eval output dir (default <root>_eval)")
    ap.add_argument("--preview", type=int, default=14, help="max GT preview images")
    args = ap.parse_args()

    root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(root):
        sys.exit(f"dataset_root not found: {root}")
    out = args.out or os.path.join(os.path.dirname(root), os.path.basename(root) + "_eval")
    os.makedirs(out, exist_ok=True)
    prev_dir = os.path.join(out, "preview")
    os.makedirs(prev_dir, exist_ok=True)

    yaml_info, yaml_path = load_yaml_names(root)
    splits = split_layout(root)
    if not splits:
        sys.exit(f"no train/valid/test images+labels dirs found under {root}")

    lines = []
    lines.append("Dataset audit: %s" % root)
    lines.append("=" * 70)
    lines.append("data.yaml: %s" % (yaml_path if os.path.isfile(yaml_path) else "NOT FOUND"))
    if yaml_info:
        lines.append("names: %s" % json.dumps(yaml_info["names"]))
        lines.append("nc: %s" % yaml_info["nc"])

    total_imgs = total_labels = total_boxes = 0
    per_split = {}
    cls_boxes = Counter()
    cls_images = Counter()
    dims = Counter()
    orphan_imgs = Counter()
    orphan_lbls = Counter()
    prefix_imgs = Counter()
    all_items = []  # (split, img_path, label_path, classes_present)

    for split, (img_dir, lbl_dir) in splits.items():
        imgs = image_files(img_dir)
        boxes_split = 0
        lbl_present = 0
        prefixes = Counter()
        for img_name in imgs:
            stem = label_stem(img_name)
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            boxes = read_boxes(lbl_path)
            classes = Counter(int(b[0]) for b in boxes)
            total_imgs += 1
            total_boxes += len(boxes)
            boxes_split += len(boxes)
            prefixes[prefix_group(img_name)] += 1
            prefix_imgs[prefix_group(img_name)] += 1
            try:
                with Image.open(os.path.join(img_dir, img_name)) as im:
                    dims[im.size] += 1
            except Exception:
                pass
            if boxes:
                lbl_present += 1
            else:
                # non-empty label file but zero boxes? count as annotated-but-empty
                pass
            for c in classes:
                cls_boxes[c] += classes[c]
                cls_images[c] += 1
            if os.path.isfile(lbl_path):
                total_labels += 1
            else:
                orphan_imgs[split] += 1
            all_items.append((split, os.path.join(img_dir, img_name),
                              lbl_path, set(classes)))
        # orphan labels (no matching image)
        for lbl_name in sorted(os.listdir(lbl_dir)):
            if not lbl_name.endswith(".txt"):
                continue
            if not os.path.isfile(os.path.join(img_dir, label_stem(lbl_name) + ".jpg")) \
               and not os.path.isfile(os.path.join(img_dir, label_stem(lbl_name) + ".png")) \
               and not os.path.isfile(os.path.join(img_dir, label_stem(lbl_name) + ".jpeg")):
                orphan_lbls[split] += 1
        per_split[split] = (len(imgs), lbl_present, boxes_split)

    lines.append("")
    lines.append("Splits (images, with>=1 box, boxes):")
    for split, (ni, nl, nb) in per_split.items():
        lines.append("  %-6s images=%4d labeled=%4d boxes=%5d" % (split, ni, nl, nb))
    total_labeled = sum(v[1] for v in per_split.values())
    lines.append("")
    lines.append("Totals: images=%d labels-with-boxes=%d boxes=%d"
                 % (total_imgs, total_labeled, total_boxes))
    lines.append("Unlabeled images (no .txt file or empty .txt / 0 boxes): %d"
                 % (total_imgs - total_labeled))
    lines.append("")
    lines.append("Per-class boxes (index: count): %s" % dict(sorted(cls_boxes.items())))
    lines.append("Per-class images (index: count): %s" % dict(sorted(cls_images.items())))
    lines.append("")
    lines.append("Image dimensions (size: count): %s"
                 % dict(sorted(dims.items(), key=lambda kv: -kv[1])))
    lines.append("Source-prefix images: %s"
                 % dict(sorted(prefix_imgs.items(), key=lambda kv: -kv[1])))
    lines.append("Orphan images (no label file): %s" % dict(orphan_imgs))
    lines.append("Orphan labels (no image file): %s" % dict(orphan_lbls))

    summary_path = os.path.join(out, "dataset_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # class_map.json: persist whatever names we found (or fallback), for downstream steps.
    names = (yaml_info["names"] if yaml_info and yaml_info["names"]
             else {i: DEFAULT_NAMES.get(i, f"class_{i}")
                   for i in range(max(max(cls_boxes, default=-1) + 1, 1))})
    with open(os.path.join(out, "class_map.json"), "w", encoding="utf-8") as fh:
        json.dump({"names": {str(k): v for k, v in sorted(names.items())},
                   "nc": len(names),
                   "source_yaml": os.path.basename(yaml_path) if os.path.isfile(yaml_path) else None},
                  fh, indent=2)
    print("\nwrote", summary_path, "and class_map.json in", out)

    # ---- GT preview montage ----
    present = sorted(cls_images)
    by_class = defaultdict(list)
    for split, img_path, lbl_path, classes in all_items:
        for c in classes:
            by_class[c].append((img_path, lbl_path))
    chosen = []
    seen = set()
    for c in present:  # cover every class
        for item in by_class[c]:
            if item[0] in seen:
                continue
            chosen.append(item)
            seen.add(item[0])
            break
    # top up with any images (negatives preferred for realism)
    for split, img_path, lbl_path, classes in all_items:
        if len(chosen) >= args.preview:
            break
        if img_path in seen:
            continue
        chosen.append((img_path, lbl_path))
        seen.add(img_path)
    chosen = chosen[:args.preview]

    n_written = 0
    for img_path, lbl_path in chosen:
        try:
            im = Image.open(img_path).convert("RGB")
        except Exception as e:
            print("skip preview", img_path, e)
            continue
        draw = ImageDraw.Draw(im)
        W, H = im.size
        for box in read_boxes(lbl_path):
            c, cx, cy, bw, bh = box[:5]
            ci = int(c)
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            col = COLORS.get(ci, (0, 0, 200))
            draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
            label = "%s(%d)" % (names.get(ci, "?"), ci)
            draw.text((x1 + 2, y1 + 2), label, fill=col)
        stem = os.path.basename(img_path)
        dst = os.path.join(prev_dir, "GT_" + stem)
        try:
            im.save(dst, quality=85)
            n_written += 1
        except Exception as e:
            print("save fail", dst, e)
    print("preview: wrote %d annotated images to %s" % (n_written, prev_dir))


if __name__ == "__main__":
    main()
