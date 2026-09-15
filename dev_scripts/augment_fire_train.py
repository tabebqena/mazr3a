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

NON-DESTRUCTIVE + RESUMABLE + REUSABLE (2026-09-15)
---------------------------------------------------
The ORIGINAL train images (dedup output) live in ``<clean>/train/images`` and are NEVER
overwritten. Each augmented image is written to the runtime store ``<clean>/train/images_aug``
(and its re-derived boxes to ``<clean>/train/labels_aug``) via ``*.part`` + ``os.replace``; only
AFTER the augmented file is fully written is the original image + label REMOVED, so disk usage
never doubles (originals shrink as the augmented store grows). ``<clean>/data.yaml`` is updated to
``train: train/images_aug`` at the end.

Every finished image is journalled ONE LINE AT A TIME (append + flush) to
``<report>/augmentation_state.jsonl``, so a kill mid-run loses at most the in-flight image. Each
line records the stem, the chosen ``op`` + its parameters, and two md5s for audit
(``src_md5`` = original bytes, ``dst_md5`` = augmented bytes).

  * RESUME  - the completion marker is the PRESENCE of the augmented file in the runtime store
              (``train/images_aug/<stem>.<ext>``). A re-run skips any image whose augmented output
              already exists (tidy-ing away a stale original left by a crash between write and
              remove); images whose original is still present are re-rendered. No md5 comparison
              is used for the skip decision.
  * REUSE   - a stem already in the journal keeps its recorded ``op`` + parameters (they are NOT
              re-rolled), so a re-run applies exactly the same augmentation decision; a brand-new
              stem gets a fresh deterministic op from ``stable_seed(stem)``.

``augmentation_report.csv`` and ``augmentation_summary.txt`` are re-derived from the journal, so
they reflect the true final state even after a resume. There is deliberately NO ``--fresh``:
augmentation consumes the originals, so "starting over" = re-run the dedup step (which re-copies
the original bytes) and delete ``<clean>/train/images_aug`` + ``augmentation_state.jsonl``.

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
from concurrent.futures import ThreadPoolExecutor
from functools import partial
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


def _acquire_lock(path, label):
    """Take an advisory exclusive lock held for the whole process lifetime.

    Prevents two concurrent invocations from racing on the same clean pool (e.g. the
    notebook cell re-run by mistake, or a manual augment while dedup is still copying).
    Uses ``fcntl.flock`` on Linux/macOS (what Colab runs) and ``msvcrt.locking`` on
    Windows. The OS releases the lock when the process exits, so a second invocation
    fails fast with a clear message instead of silently corrupting the output.
    """
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fh = open(path, "w", encoding="utf-8")
    try:
        try:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            # Windows fallback (never exercised on Linux/Colab; msvcrt has no type stubs).
            import msvcrt
            locking = getattr(msvcrt, "locking", None)
            if locking is None:
                raise RuntimeError("no flock and no msvcrt.locking available")
            locking(fh.fileno(), getattr(msvcrt, "LK_NBLCK", 1), 1)
    except OSError:
        fh.close()
        die("another process already holds the lock %s (is a previous run still active?)"
            % path)
    return fh


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


def rewrite_data_yaml(clean):
    """Point ``train:`` at the augmented store (idempotent)."""
    yaml_path = os.path.join(clean, "data.yaml")
    if not os.path.isfile(yaml_path):
        return
    lines = []
    for ln in open(yaml_path, encoding="utf-8"):
        if ln.startswith("train:"):
            ln = "train: train/images_aug\n"
        lines.append(ln)
    tmp = yaml_path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    os.replace(tmp, yaml_path)


def augment_one(name, img_dir, lbl_dir, aug_img_dir, aug_lbl_dir, journal_snapshot):
    """Process ONE train image end-to-end; returns (status, rec, op).

    ``status`` is 'skip' (augmented file already in the store) or 'done'. Every stem and
    destination path is unique to this worker, so no lock is needed here: ``journal_snapshot``
    is read-only (the REUSE source), and the worker writes only its own ``*.part`` + dst files.
    """
    stem = os.path.splitext(name)[0]
    ext = os.path.splitext(name)[1].lower()
    src_img = os.path.join(img_dir, stem + ext)
    src_lbl = os.path.join(lbl_dir, stem + ".txt")
    dst_img = os.path.join(aug_img_dir, stem + ext)
    dst_lbl = os.path.join(aug_lbl_dir, stem + ".txt")

    rec = journal_snapshot.get(stem)

    # RESUME: the augmented output already exists -> skip; tidy away a stale original
    # left by a crash between the atomic write and the original-removal below.
    if os.path.isfile(dst_img):
        for p in (src_img, src_lbl):
            try:
                os.remove(p)
            except OSError:
                pass
        return "skip", rec, (rec.get("op") if rec else None)

    with open(src_img, "rb") as fh:
        data = fh.read()
    src_md5 = md5_bytes(data)

    # REUSE: a known stem keeps its recorded op + parameters; a new stem rolls fresh.
    if rec is not None and rec.get("op") in OPS:
        op = rec["op"]
        params = {k: rec.get(k, DEFAULT_PARAMS[k]) for k in OP_PARAMS}
    else:
        seed = stable_seed(stem)
        rng = random.Random(seed)
        op = rng.choice(OPS)
        params = roll_params(op, rng)

    with Image.open(BytesIO(data)) as im:
        im = im.convert("RGB")
        w0, h0 = im.size
        boxes = read_boxes(src_lbl)
        im, boxes = render_op(im, w0, h0, boxes, op, params, stable_seed(stem))

        fmt = EXT_FORMAT.get(ext, "JPEG")
        buf = BytesIO()
        im.save(buf, format=fmt, quality=95)
        out = buf.getvalue()

    dst_md5 = md5_bytes(out)

    # atomic write of the augmented image + re-derived label, THEN remove the originals
    # (order matters: only delete the original once its replacement is safely on disk)
    tmp_img = dst_img + ".part"
    with open(tmp_img, "wb") as fh:
        fh.write(out)
    os.replace(tmp_img, dst_img)
    write_boxes(dst_lbl, boxes)
    for p in (src_img, src_lbl):
        try:
            os.remove(p)
        except OSError:
            pass

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
    return "done", rec, op


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", required=True, help="CLEAN pool produced by dedup_fire_scratch.py")
    ap.add_argument("--report", default=None, help="report dir (default: <clean>_report)")
    ap.add_argument("--seed", type=int, default=0, help="unused (seeds derive from stems)")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker threads for augmentation (default: min(16, 2*cpu_count); "
                         "1 = single-threaded)")
    args = ap.parse_args()

    if args.workers is None or args.workers <= 0:
        args.workers = min(16, max(2, (os.cpu_count() or 4) * 2))

    clean = os.path.abspath(args.clean)
    report = os.path.abspath(args.report or (clean + "_report"))
    os.makedirs(report, exist_ok=True)

    # Cross-process guard: the same lock file dedup_fire_scratch.py uses, so a concurrent
    # dataset build can't race this one. Held until process exit (OS releases).
    _lock_fh = _acquire_lock(os.path.join(report, "build.lock"), "augment_fire_train")

    img_dir = os.path.join(clean, "train", "images")           # originals (dedup output)
    lbl_dir = os.path.join(clean, "train", "labels")
    aug_img_dir = os.path.join(clean, "train", "images_aug")   # augmented runtime store
    aug_lbl_dir = os.path.join(clean, "train", "labels_aug")
    if not os.path.isdir(img_dir):
        die("no train/images under --clean (%s)" % clean)
    os.makedirs(aug_img_dir, exist_ok=True)
    os.makedirs(aug_lbl_dir, exist_ok=True)

    state_path = os.path.join(report, "augmentation_state.jsonl")
    journal = load_journal(state_path)

    names = sorted(n for n in os.listdir(img_dir) if os.path.splitext(n)[1].lower() in IMG_EXTS)
    known = sum(1 for n in names if os.path.splitext(n)[0] in journal)
    print("augment_fire_train: %d original train images pending (%d with recorded op)"
          % (len(names), known), flush=True)

    # REUSE reads an immutable snapshot: workers never see each other's new records, and the
    # MAIN thread is the single writer of the journal + counters, so no lock is needed.
    journal_snapshot = dict(journal)
    worker = partial(augment_one, img_dir=img_dir, lbl_dir=lbl_dir,
                     aug_img_dir=aug_img_dir, aug_lbl_dir=aug_lbl_dir,
                     journal_snapshot=journal_snapshot)

    op_counts = Counter()
    done = skipped = 0

    ex = ThreadPoolExecutor(max_workers=args.workers)
    try:
        with open(state_path, "a", encoding="utf-8") as jfh:
            for idx, (status, rec, op) in enumerate(ex.map(worker, names), 1):
                if status == "skip":
                    skipped += 1
                    if op in OPS:
                        op_counts[op] += 1
                else:
                    journal[rec["stem"]] = rec
                    jfh.write(json.dumps(rec) + "\n")
                    jfh.flush()
                    done += 1
                    op_counts[rec["op"]] += 1
                if idx % 2000 == 0:
                    print("  ... augmented %d/%d train images (skipped %d)"
                          % (idx, len(names), skipped), flush=True)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    # compact rewrite (drops duplicate lines an interrupted run may have appended)
    save_journal(journal, state_path)
    rewrite_data_yaml(clean)

    # final op distribution is derived from the journal so it reflects the WHOLE dataset,
    # not just the images touched by this particular run (a resume skips most of them)
    op_counts = Counter(r.get("op") for r in journal.values())

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
        "augmented train images: %d (skipped %d already in store)" % (done, skipped),
        "operations applied (exactly one per image):",
    ] + op_lines + [
        "store -> %s" % aug_img_dir,
        "state -> %s" % state_path,
        "report -> %s" % rep_path,
    ])
    with open(os.path.join(report, "augmentation_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
