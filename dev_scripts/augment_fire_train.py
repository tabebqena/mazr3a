#!/usr/bin/env python3
"""augment_fire_train.py - re-render TRAIN-side duplicates into distinct, correctly-labelled
training images instead of dropping them.

dedup_fire_scratch.py still removes train near-duplicates (exact md5 OR dHash Hamming <= N) and
records each removal reason in <report>/per_image.csv. This script reads that report and, for
every TRAIN image removed ONLY for a duplication reason (``train-self`` or
``train-vs-prev-train``), re-renders it with a deterministic random transform and copies the
augmented image + a label whose geometry has been re-derived to match the transform.

The point: the FireViewer corpus is mostly sequential video frames (near-duplicates). Aggressive
dedup collapses those to a small core and throws away most of the smoke/fire signal. Instead of
deleting them we de-correlate them with random augmentation, so each duplicate becomes a fresh,
distinct training sample.

WHAT GETS AUGMENTED (and what does not)
--------------------------------------
  * re-added  : TRAIN images removed for ``train-self`` / ``train-vs-prev-train`` AND whose label
                has >= 1 box (positive). Covers exact-md5 AND dHash duplicates - the transform is
                seeded per stem, so byte-identical images get different transforms.
  * skipped   : background duplicates (empty label) - the dedup balance cap (``--max-bg-share``)
                stays authoritative for the negative fraction.
  * NEVER     : anything removed for ``train-vs-test``, ``train-vs-val`` or ``train-vs-prev-test``,
                and every test/val removal. Strict train/test isolation is untouched.

AUGMENTATION (deterministic, seeded per stem; params recorded in <report>/augmentation_report.csv)
  * horizontal flip (left<->right) prob 0.5 - NO upside-down flip
  * rotation uniform(-15 deg, +15 deg), expand=True, black fill (boxes re-derived)
  * brightness x uniform(0.90, 1.10)
  * contrast   x uniform(0.90, 1.10)
  * hue shift  uniform(-0.015, +0.015) + saturation uniform(-0.1, +0.1)   (colour noise)
  * additive Gaussian noise sigma ~5/255

USAGE
-----
    python augment_fire_train.py --pool /content/raw_yolo --clean /content/clean_yolo \
        --report /content/clean_yolo_report --seed 0
"""
import argparse
import csv
import glob
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
# only these dedup reasons are "duplication" and eligible for re-render; everything else is
# an isolation/balance decision that must stay untouched.
AUG_REASONS = {"train-self", "train-vs-prev-train"}

REPORT_COLS = ["split", "stem", "status", "reason", "orig_stem", "rot_deg", "flip",
               "bright_factor", "contrast_factor", "hue_delta", "sat_delta", "noise_sigma"]


def die(msg):
    sys.exit("augment_fire_train: " + msg)


def stable_seed(stem):
    """Deterministic 32-bit seed from the stem (so a re-run re-renders identically)."""
    return int(hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8], 16)


def find_image(pool_images, stem):
    """Return the image path for a stem in the pool's train/images dir (any supported ext)."""
    for ext in IMG_EXTS:
        p = os.path.join(pool_images, stem + ext)
        if os.path.isfile(p):
            return p
    hits = glob.glob(os.path.join(pool_images, stem + ".*"))
    hits = [h for h in hits if os.path.splitext(h)[1].lower() in IMG_EXTS]
    return hits[0] if hits else None


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
    ap.add_argument("--pool", required=True, help="RAW merged pool (prep_fire_scratch_dataset.py)")
    ap.add_argument("--clean", required=True, help="CLEAN pool produced by dedup_fire_scratch.py")
    ap.add_argument("--report", required=True, help="dedup report dir holding per_image.csv")
    ap.add_argument("--seed", type=int, default=0, help="unused (seeds derive from stems)")
    args = ap.parse_args()

    pool = os.path.abspath(args.pool)
    clean = os.path.abspath(args.clean)
    report = os.path.abspath(args.report)

    per_image = os.path.join(report, "per_image.csv")
    if not os.path.isfile(per_image):
        die("per_image.csv not found under --report (%s) - run dedup first" % report)

    out_img = os.path.join(clean, "train", "images")
    out_lbl = os.path.join(clean, "train", "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    pool_img = os.path.join(pool, "train", "images")
    pool_lbl = os.path.join(pool, "train", "labels")

    rows = []
    n_kept = n_bg = n_missing = 0
    n_scan = 0
    with open(per_image, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] != "train":
                continue
            if r["status"] != "removed" or r["reason"] not in AUG_REASONS:
                continue
            n_scan += 1
            if n_scan % 1000 == 0:
                print("  ... scanned %d eligible duplicates (re-rendered %d)"
                      % (n_scan, n_kept), flush=True)
            stem = r["stem"]
            img_src = find_image(pool_img, stem)
            lbl_src = os.path.join(pool_lbl, stem + ".txt")
            boxes = read_boxes(lbl_src)
            if not boxes:
                n_bg += 1
                continue
            if not img_src:
                n_missing += 1
                continue

            seed = stable_seed(stem)
            rng = random.Random(seed)

            with Image.open(img_src) as im:
                im = im.convert("RGB")
                w0, h0 = im.size

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
                boxes = rotate_boxes(boxes, w0, h0, deg, nw, nh)

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

                ext = os.path.splitext(img_src)[1].lower()
                new_stem = "aug__%s" % stem
                im.save(os.path.join(out_img, new_stem + ext), quality=95)
                write_boxes(os.path.join(out_lbl, new_stem + ".txt"), boxes)
                n_kept += 1

            rows.append([r["split"], new_stem, "augmented", r["reason"], stem,
                         round(deg, 4), flip, round(bright, 4), round(contrast, 4),
                         round(hue, 4), round(sat, 4), 5.0])

    rep_path = os.path.join(report, "augmentation_report.csv")
    with open(rep_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(REPORT_COLS)
        w.writerows(rows)

    text = "\n".join([
        "augment_fire_train.py - duplicate re-render report",
        "=" * 64,
        "eligible train duplicates scanned: %d" % n_scan,
        "re-rendered (positive train dups): %d" % n_kept,
        "skipped (background dups, cap-owned): %d" % n_bg,
        "skipped (source image missing): %d" % n_missing,
        "report -> %s" % rep_path,
    ])
    with open(os.path.join(report, "augmentation_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
