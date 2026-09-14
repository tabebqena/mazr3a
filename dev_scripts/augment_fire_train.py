#!/usr/bin/env python3
"""augment_fire_train.py - de-correlate the TRAIN split with random augmentation.

The v3 policy: dedup_fire_scratch.py runs ONLY the isolation passes (test/val self-dedup,
train-vs-test, train-vs-val, vs prev-test) and SKIPS the train-side near-dup scan
(``--no-train-self`` and no ``--prev-train-index``), so no positive train signal is thrown away.
This script then re-renders EVERY kept train image once with a deterministic random transform, so
byte-identical / near-identical sequential frames become genuinely distinct training samples while
the boxes stay correct.

Only the TRAIN split is augmented; test and val stay byte-original for honest scoring.

AUGMENTATION (deterministic, seeded per stem; params recorded in <report>/augmentation_report.csv)
  * horizontal flip (left<->right) prob 0.5 - NO upside-down flip
  * rotation uniform(-15 deg, +15 deg), expand=True, black fill (boxes re-derived)
  * brightness x uniform(0.90, 1.10)
  * contrast   x uniform(0.90, 1.10)
  * hue shift  uniform(-0.015, +0.015) + saturation uniform(-0.1, +0.1)   (colour noise)
  * additive Gaussian noise sigma ~5/255

USAGE
-----
    python augment_fire_train.py --clean /content/clean_yolo --report /content/clean_yolo_report
"""
import argparse
import csv
import hashlib
import os
import random
import sys

try:
    from PIL import Image, ImageEnhance
except ImportError:
    sys.exit("augment_fire_train: Pillow is required (pip install pillow)")

# Modern Pillow enum names (Pillow >= 9.1, which the Colab/ultralytics env always has).
FLIP_LR = Image.Transpose.FLIP_LEFT_RIGHT
BICUBIC = Image.Resampling.BICUBIC

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REPORT_COLS = ["split", "stem", "rot_deg", "flip", "bright_factor", "contrast_factor",
               "hue_delta", "sat_delta", "noise_sigma"]


def die(msg):
    sys.exit("augment_fire_train: " + msg)


def stable_seed(stem):
    """Deterministic 32-bit seed from the stem (so a re-run re-renders identically)."""
    return int(hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8], 16)


def read_boxes(lbl_path):
    """Return list of (cls, cx, cy, w, h) normalized YOLO boxes; [] when empty/missing."""
    rows = []
    if not lbl_path or not os.path.isfile(lbl_path):
        return rows
    with open(lbl_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                rows.append(tuple(float(x) for x in parts))
            except ValueError:
                continue
    return rows


def write_boxes(lbl_path, boxes):
    with open(lbl_path, "w", encoding="utf-8") as fh:
        for c, cx, cy, w, h in boxes:
            fh.write("%d %.6f %.6f %.6f %.6f\n" % (int(c), cx, cy, w, h))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def flip_box(box):
    c, cx, cy, w, h = box
    return (c, 1.0 - cx, cy, w, h)


def rotate_boxes(boxes, w, h, deg_ccw, new_w, new_h):
    """Re-derive axis-aligned YOLO boxes after a PIL rotate(deg_ccw, expand=True).

    PIL rotate is counter-clockwise positive. For each box we rotate its four pixel corners
    about the original centre, translate to the expanded canvas, then take the bounding box
    of the rotated corners and re-normalise to the new canvas size.
    """
    import math
    th = math.radians(deg_ccw)
    cos_t, sin_t = math.cos(th), math.sin(th)
    cx0, cy0 = w / 2.0, h / 2.0
    nx0, ny0 = new_w / 2.0, new_h / 2.0
    out = []
    for c, bcx, bcy, bw, bh in boxes:
        x1 = (bcx - bw / 2.0) * w
        y1 = (bcy - bh / 2.0) * h
        x2 = (bcx + bw / 2.0) * w
        y2 = (bcy + bh / 2.0) * h
        rx, ry = [], []
        for px, py in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
            dx, dy = px - cx0, py - cy0
            rx.append(cos_t * dx - sin_t * dy + nx0)
            ry.append(sin_t * dx + cos_t * dy + ny0)
        nx1, nx2 = clamp(min(rx), 0.0, new_w), clamp(max(rx), 0.0, new_w)
        ny1, ny2 = clamp(min(ry), 0.0, new_h), clamp(max(ry), 0.0, new_h)
        if nx2 <= nx1 or ny2 <= ny1:
            continue
        out.append((c,
                    (nx1 + nx2) / 2.0 / new_w,
                    (ny1 + ny2) / 2.0 / new_h,
                    (nx2 - nx1) / new_w,
                    (ny2 - ny1) / new_h))
    return out


def shift_hsv(img, hue_delta, sat_delta):
    """Shift HSV hue/saturation (colour noise). Hue scale 0-255 = 0-360 deg."""
    hsv = img.convert("HSV")
    h, s, v = hsv.split()
    h = h.point(lambda p: (p + int(round(hue_delta * 255))) % 256)
    s = s.point(lambda p: clamp(p + int(round(sat_delta * 255)), 0, 255))
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def add_noise(img, sigma, seed):
    """Additive Gaussian noise via numpy (no-op fallback if numpy is unavailable)."""
    try:
        import numpy as np
    except ImportError:
        return img
    arr = np.asarray(img).astype(np.float32)
    noise = np.random.default_rng(seed).normal(0.0, sigma, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", required=True, help="CLEAN pool produced by dedup_fire_scratch.py")
    ap.add_argument("--report", default=None, help="report dir (default: <clean>_report)")
    ap.add_argument("--seed", type=int, default=0, help="unused (seeds derive from stems)")
    args = ap.parse_args()

    clean = os.path.abspath(args.clean)
    report = os.path.abspath(args.report or (clean + "_report"))
    os.makedirs(report, exist_ok=True)

    img_dir = os.path.join(clean, "train", "images")
    lbl_dir = os.path.join(clean, "train", "labels")
    if not os.path.isdir(img_dir):
        die("no train/images under --clean (%s)" % clean)

    names = sorted(n for n in os.listdir(img_dir) if os.path.splitext(n)[1].lower() in IMG_EXTS)
    rows = []
    n = 0
    for name in names:
        stem = os.path.splitext(name)[0]
        img_path = os.path.join(img_dir, name)
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        seed = stable_seed(stem)
        rng = random.Random(seed)

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w0, h0 = im.size
            boxes = read_boxes(lbl_path)

            # 1) horizontal flip only (no upside-down), boxes mirrored
            if rng.random() < 0.5:
                im = im.transpose(FLIP_LR)
                boxes = [flip_box(b) for b in boxes]
                flip = 1
            else:
                flip = 0

            # 2) rotation +-15 deg, expand + black fill, boxes re-derived
            deg = rng.uniform(-15.0, 15.0)
            im = im.rotate(deg, expand=True, fillcolor=(0, 0, 0), resample=BICUBIC)
            nw, nh = im.size
            boxes = rotate_boxes(boxes, w0, h0, deg, nw, nh) if boxes else []

            # 3) brightness +-10 %
            bright = rng.uniform(0.90, 1.10)
            im = ImageEnhance.Brightness(im).enhance(bright)

            # 4) contrast +-10 %
            contrast = rng.uniform(0.90, 1.10)
            im = ImageEnhance.Contrast(im).enhance(contrast)

            # 5) colour noise (hue + saturation shift)
            hue = rng.uniform(-0.015, 0.015)
            sat = rng.uniform(-0.1, 0.1)
            im = shift_hsv(im, hue, sat)

            # 6) additive Gaussian noise
            im = add_noise(im, 5.0, seed)

            ext = os.path.splitext(name)[1].lower()
            im.save(os.path.join(img_dir, stem + ext), quality=95)
            write_boxes(lbl_path, boxes)

        rows.append(["train", stem, round(deg, 4), flip, round(bright, 4),
                     round(contrast, 4), round(hue, 4), round(sat, 4), 5.0])
        n += 1
        if n % 2000 == 0:
            print("  ... augmented %d/%d train images" % (n, len(names)), flush=True)

    rep_path = os.path.join(report, "augmentation_report.csv")
    with open(rep_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(REPORT_COLS)
        w.writerows(rows)

    text = "\n".join([
        "augment_fire_train.py - train augmentation report",
        "=" * 64,
        "augmented train images: %d" % n,
        "report -> %s" % rep_path,
    ])
    with open(os.path.join(report, "augmentation_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
