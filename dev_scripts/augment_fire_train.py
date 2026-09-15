#!/usr/bin/env python3
"""augment_fire_train.py - de-correlate the TRAIN split with random augmentation (resumable + reusable).

The v3 policy: dedup_fire_scratch.py runs ONLY the isolation passes (test/val self-dedup,
train-vs-test, train-vs-val, vs prev-test) and SKIPS the train-side near-dup scan
(``--no-train-self`` and no ``--prev-train-index``), so no positive train signal is thrown away.
This script then re-renders EVERY kept train image once with ONE deterministic random operation
(chosen uniformly at random per image), so byte-identical / near-identical sequential frames become
genuinely distinct training samples while the boxes stay correct.

Only the TRAIN split is augmented; test and val stay byte-original for honest scoring.

AUGMENTATION (deterministic, seeded per stem; exactly ONE operation per image, chosen uniformly
at random, and the applied operation is recorded in <report>/augmentation_report.csv as ``op``):
  * flip       - horizontal flip (left<->right), NO upside-down flip (boxes mirrored)
  * rotate     - rotation uniform(-15 deg, +15 deg), expand=True, black fill (boxes re-derived)
  * brightness - x uniform(0.90, 1.10)
  * contrast   - x uniform(0.90, 1.10)
  * hue        - hue shift uniform(-0.015, +0.015) + saturation uniform(-0.1, +0.1)
  * noise      - additive Gaussian noise sigma ~5/255

RESUMABLE + REUSABLE (2026-09-15)
--------------------------------
Every finished image is journalled ONE LINE AT A TIME (append + flush) to
``<report>/augmentation_state.jsonl``, so a kill mid-run loses at most the in-flight image.
Each line records the stem, the chosen ``op`` + its parameters, and two md5s:
  * ``src_md5`` - the image bytes BEFORE augmentation (the original);
  * ``dst_md5`` - the image bytes AFTER augmentation (what is now on disk).
On a re-run the journal is loaded and used in BOTH ways:

  * RESUME  - an image whose CURRENT on-disk md5 already equals ``dst_md5`` is skipped, so a
              re-run after an interrupt only re-renders the images that were not finished.
  * REUSE   - a stem already in the journal keeps its recorded ``op`` + parameters (they are NOT
              re-rolled), so a re-run applies exactly the same augmentation decision; a brand-new
              stem gets a fresh deterministic op from ``stable_seed(stem)``.

``augmentation_report.csv`` and ``augmentation_summary.txt`` are re-derived from the journal, so
they reflect the true final state even after a resume. There is deliberately NO ``--fresh``:
augmentation rewrites the source image in place, so re-applying it would double-augment; to start
over, re-run the dedup step (which re-copies the original bytes) and delete ``augmentation_state.jsonl``.

USAGE
-----
    python augment_fire_train.py --clean /content/clean_yolo --report /content/clean_yolo_report

    # resume after a crash (default behaviour):
    python augment_fire_train.py --clean /content/clean_yolo --report /content/clean_yolo_report
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter
from io import BytesIO

try:
    from PIL import Image, ImageEnhance
except ImportError:
    sys.exit("augment_fire_train: Pillow is required (pip install pillow)")

# Modern Pillow enum names (Pillow >= 9.1, which the Colab/ultralytics env always has).
FLIP_LR = Image.Transpose.FLIP_LEFT_RIGHT
BICUBIC = Image.Resampling.BICUBIC

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REPORT_COLS = ["split", "stem", "op", "rot_deg", "flip", "bright_factor", "contrast_factor",
               "hue_delta", "sat_delta", "noise_sigma"]

OPS = ("flip", "rotate", "brightness", "contrast", "hue", "noise")

# The per-image parameters recorded in the journal (everything else defaults for the report).
OP_PARAMS = ("rot_deg", "flip", "bright", "contrast", "hue", "sat", "noise_sigma")

DEFAULT_PARAMS = {"rot_deg": 0.0, "flip": 0, "bright": 1.0, "contrast": 1.0,
                  "hue": 0.0, "sat": 0.0, "noise_sigma": 0.0}

# ``format`` PIL expects for each extension (save to a .part temp must spell it out explicitly).
EXT_FORMAT = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP", ".webp": "WEBP"}


def die(msg):
    sys.exit("augment_fire_train: " + msg)


def stable_seed(stem):
    """Deterministic 32-bit seed from the stem (so a re-run re-renders identically)."""
    return int(hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8], 16)


def md5_bytes(data):
    return hashlib.md5(data).hexdigest()


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
    """Atomically write YOLO boxes (tmp + os.replace)."""
    tmp = lbl_path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        for c, cx, cy, w, h in boxes:
            fh.write("%d %.6f %.6f %.6f %.6f\n" % (int(c), cx, cy, w, h))
    os.replace(tmp, lbl_path)


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


# ---------------------------------------------------------------------------
# Journal (resumable + reusable state)
# ---------------------------------------------------------------------------

def load_journal(path):
    """Return {stem: record} from the append-only journal.

    Malformed / truncated tail lines are ignored, so a kill mid-append only loses the last
    partially-written line (which is simply re-augmented on the next run).
    """
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except ValueError:
                continue
            stem = o.get("stem")
            if stem:
                out[stem] = o
    return out


def save_journal(records, path):
    """Atomically rewrite the journal compactly (one line per stem, sorted)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for stem in sorted(records):
            fh.write(json.dumps(records[stem]) + "\n")
    os.replace(tmp, path)


def roll_params(op, rng):
    """Fresh default + op parameters for a NEW stem (exactly ONE op, chosen by the caller)."""
    p = dict(DEFAULT_PARAMS)
    if op == "flip":
        p["flip"] = 1
    elif op == "rotate":
        p["rot_deg"] = rng.uniform(-15.0, 15.0)
    elif op == "brightness":
        p["bright"] = rng.uniform(0.90, 1.10)
    elif op == "contrast":
        p["contrast"] = rng.uniform(0.90, 1.10)
    elif op == "hue":
        p["hue"] = rng.uniform(-0.015, 0.015)
        p["sat"] = rng.uniform(-0.1, 0.1)
    else:  # noise
        p["noise_sigma"] = 5.0
    return p


def render_op(im, w0, h0, boxes, op, p, seed):
    """Apply the single chosen operation to a converted-RGB image; return (im, boxes)."""
    if op == "flip":
        im = im.transpose(FLIP_LR)
        boxes = [flip_box(b) for b in boxes]
    elif op == "rotate":
        im = im.rotate(p["rot_deg"], expand=True, fillcolor=(0, 0, 0), resample=BICUBIC)
        nw, nh = im.size
        boxes = rotate_boxes(boxes, w0, h0, p["rot_deg"], nw, nh) if boxes else []
    elif op == "brightness":
        im = ImageEnhance.Brightness(im).enhance(p["bright"])
    elif op == "contrast":
        im = ImageEnhance.Contrast(im).enhance(p["contrast"])
    elif op == "hue":
        im = shift_hsv(im, p["hue"], p["sat"])
    else:  # noise
        im = add_noise(im, p["noise_sigma"], seed)
    return im, boxes


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

    state_path = os.path.join(report, "augmentation_state.jsonl")
    journal = load_journal(state_path)

    names = sorted(n for n in os.listdir(img_dir) if os.path.splitext(n)[1].lower() in IMG_EXTS)
    known = sum(1 for n in names if os.path.splitext(n)[0] in journal)
    print("augment_fire_train: %d train images (%d already journalled)" % (len(names), known),
          flush=True)

    op_counts = Counter()
    done = skipped = 0

    with open(state_path, "a", encoding="utf-8") as jfh:
        for idx, name in enumerate(names, 1):
            stem = os.path.splitext(name)[0]
            img_path = os.path.join(img_dir, name)
            lbl_path = os.path.join(lbl_dir, stem + ".txt")

            with open(img_path, "rb") as fh:
                data = fh.read()
            cur_md5 = md5_bytes(data)

            rec = journal.get(stem)
            if rec is not None and rec.get("op") in OPS and rec.get("dst_md5"):
                # RESUME: the on-disk image already IS the recorded augmented output -> skip.
                if cur_md5 == rec["dst_md5"]:
                    skipped += 1
                    op_counts[rec["op"]] += 1
                    continue
                # REUSE: a known stem keeps its recorded op + parameters (never re-rolled).
                op = rec["op"]
                params = {k: rec.get(k, DEFAULT_PARAMS[k]) for k in OP_PARAMS}
            else:
                # new stem: roll a fresh deterministic op from stable_seed(stem)
                seed = stable_seed(stem)
                rng = random.Random(seed)
                op = rng.choice(OPS)
                params = roll_params(op, rng)

            src_md5 = cur_md5
            with Image.open(BytesIO(data)) as im:
                im = im.convert("RGB")
                w0, h0 = im.size
                boxes = read_boxes(lbl_path)
                im, boxes = render_op(im, w0, h0, boxes, op, params, stable_seed(stem))

                fmt = EXT_FORMAT.get(os.path.splitext(name)[1].lower(), "JPEG")
                buf = BytesIO()
                im.save(buf, format=fmt, quality=95)
                out = buf.getvalue()

            dst_md5 = md5_bytes(out)
            tmp_img = img_path + ".part"
            with open(tmp_img, "wb") as fh:
                fh.write(out)
            os.replace(tmp_img, img_path)
            write_boxes(lbl_path, boxes)

            rec = {
                "stem": stem, "op": op,
                "rot_deg": round(params["rot_deg"], 4),
                "flip": params["flip"],
                "bright": round(params["bright"], 4),
                "contrast": round(params["contrast"], 4),
                "hue": round(params["hue"], 4),
                "sat": round(params["sat"], 4),
                "noise_sigma": params["noise_sigma"],
                "src_md5": src_md5, "dst_md5": dst_md5,
            }
            journal[stem] = rec
            jfh.write(json.dumps(rec) + "\n")
            jfh.flush()
            done += 1
            op_counts[op] += 1
            if idx % 2000 == 0:
                print("  ... augmented %d/%d train images (skipped %d)"
                      % (idx, len(names), skipped), flush=True)

    # compact rewrite (drops duplicate lines an interrupted run may have appended)
    save_journal(journal, state_path)

    # derive the final CSV + summary from the journal so they reflect the true state after a resume
    rows = []
    for stem in sorted(journal):
        r = journal[stem]
        rows.append(["train", stem, r.get("op", ""),
                     r.get("rot_deg", 0.0), r.get("flip", 0),
                     r.get("bright", 1.0), r.get("contrast", 1.0),
                     r.get("hue", 0.0), r.get("sat", 0.0), r.get("noise_sigma", 0.0)])

    rep_path = os.path.join(report, "augmentation_report.csv")
    with open(rep_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(REPORT_COLS)
        w.writerows(rows)

    op_lines = ["  %-10s %d" % (op, op_counts.get(op, 0)) for op in OPS]
    text = "\n".join([
        "augment_fire_train.py - train augmentation report",
        "=" * 64,
        "augmented train images: %d (skipped %d already done)" % (done, skipped),
        "operations applied (exactly one per image):",
    ] + op_lines + [
        "state -> %s" % state_path,
        "report -> %s" % rep_path,
    ])
    with open(os.path.join(report, "augmentation_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
