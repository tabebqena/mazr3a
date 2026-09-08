#!/usr/bin/env python3
"""score_gmail_screenshots.py - analyse Telegram fire-alert screenshots.

The user downloaded (from Telegram) the alert photos that `firewatch.py`
(`firewatch/firewatch.py`) sent for every fire/smoke alarm. Each alert photo is
the CCTV frame with firewatch's RED box + label text overlaid (e.g. "fire 0.73")
- that text is the confidence the checkpoint that fired the alert had at that
moment. The screenshots are full 1080x2400 Android captures (media/Gmail/*.jpg).

Layout (measured): each screenshot shows ONE alert's CCTV frame as a full-width
~16:9 image occupying roughly x 0-1080, y ~830-1440. The burned-in CCTV
date/time sits ~13 px below the frame's top edge at x ~8 (its top is ~y 842-848
across the set), so the frame top can be anchored to the date line. V4 is scored
on that CCTV-frame crop (NOT the whole screenshot, per user request) because the
comparison is "old-printed score vs what the ACTIVE model says about this frame".

For every screenshot this tool emits one row:
  file          : screenshot file name
  alert_ts      : burned-in CCTV date/time of the alert (OCR)
  printed_score : the red "fire/smoke <conf>" text read from the image (filled
                  from --printed-scores CSV, e.g. a vision-model reading; my
                  local OCR could only auto-read it on a few - too small)
  v4_fire       : top-1 fire confidence of the ACTIVE model (models/fire, v4)
                  re-run on the CCTV-frame crop
  v4_smoke      : top-1 smoke confidence of the same re-run
  verdict       : empty column for the user to fill TRUE / FALSE per alert

Outputs (written next to the screenshots so the user can review/edit):
  media/Gmail/fire_alerts_scores.csv
  media/Gmail/fire_alerts_scores.md      (human review table)

Dependencies (throwaway venv is fine):
  pip install openvino pillow numpy rapidocr_onnxruntime
  + system `tesseract` (tiny red label, best-effort) + opencv optional.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

SRC = os.path.join("media", "Gmail")
MODEL_DIR = "models/fire"
SCORE_THRESHOLD = 0.5  # firewatch production gate (config/firewatch.conf)

# Frame geometry on the 1080x2400 screenshot (see module docstring).
FRAME_TOP_FALLBACK = 830      # used when the CCTV clock could not be read
FRAME_TOP_INSET = 13          # ~pixels from frame top edge down to clock text
FRAME_ASPECT = 9.0 / 16.0     # source detect frame is 640x360 (16:9)

# Matches firewatch's overlay text: "<label> <conf>" e.g. "fire 0.73".
LABEL_RE = re.compile(
    r"(fire|smoke|tra|snoke|srnoke|fir|fite|smok|tir|ture|fve|fue|seoke|smoke|fure|tmoke)"
    r"\s*[^0-9]{0,4}([01]\.\d{1,2})",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"(\d{2}[/-]\d{2}[/-]\d{4})\s*(\d{2}):(\d{2}):(\d{2})"
)


def red_mask(img, thr=60):
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    s = np.clip(a[..., 0] - np.maximum(a[..., 1], a[..., 2]), 0, 255)
    ys, xs = np.where(s > thr)
    if not len(xs):
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _tesseract(img, psms=(7, 6, 11)):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        img.save(tf.name)
        path = tf.name
    try:
        out = []
        for psm in psms:
            r = subprocess.run(
                ["tesseract", path, "stdout", "--psm", str(psm)],
                capture_output=True, text=True,
            )
            out.append(r.stdout.strip().replace("\n", " "))
    finally:
        os.unlink(path)
    return out


def _scale(img, up):
    return img.resize((img.width * up, img.height * up), Image.LANCZOS)


def read_overlay_label(im):
    """Best-effort OCR of firewatch's red '<label> <conf>' text on the photo.

    Returns (word, conf) e.g. ("fire", 0.73) or None. Usually too small/low-
    contrast to read - the reliable printed scores come from --printed-scores.
    """
    bb = red_mask(im)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    W, H = im.size
    found = {}

    def try_ocr(crop, up):
        txts = _tesseract(_scale(crop, up))
        for t in txts:
            for m in LABEL_RE.finditer(t):
                w = m.group(1).lower()
                c = float(m.group(2))
                found[c] = w
        if not found:
            for t in txts:
                m = re.search(r"([01]\.\d{1,2})", t)
                if m:
                    found.setdefault(float(m.group(1)), "")

    cands = [
        (max(0, x0 - 8), max(0, y0 - 8), min(W, x1 + 8), min(H, y1 + 8)),
        (max(0, x0 - 40), max(0, y0 - 45), min(W, x1 + 40), min(H, y0 + 60)),
        (max(0, x0 - 8), max(0, y0 - 45), min(W, min(x0 + 260, x1 + 8)),
         min(H, y0 + 10)),
    ]
    for cx0, cy0, cx1, cy1 in cands:
        if cx1 - cx0 < 12 or cy1 - cy0 < 6:
            continue
        try_ocr(im.crop((cx0, cy0, cx1, cy1)), 5)
    if not found:
        return None
    wordy = {c: w for c, w in found.items() if w}
    if wordy:
        c = max(wordy)
        return (wordy[c], c)
    c = max(found)
    return ("", c)


def read_cctv_clock(im):
    """OCR the burned-in CCTV 'dd/mm/yyyy hh:mm:ss' near the frame's top-left.

    Returns (timestamp_str, clock_top_y_in_full_image) or (None, None). The
    clock sits ~y 845 on the 1080x2400 screenshot, just below the frame top.
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        return None, None
    crop_y0 = 800
    crop_y1 = 960
    up = 2  # upscale factor used below; coordinates come back scaled by `up`
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        _scale(im.crop((0, crop_y0, 420, crop_y1)), up).save(tf.name)
        path = tf.name
    try:
        res, _ = RapidOCR()(path)
    finally:
        os.unlink(path)
    if not res:
        return None, None
    best = None
    for item in res:
        txt = str(item[1])
        m = DATE_RE.search(txt)
        if m:
            d = m.group(1).replace("-", "/")
            ys = [pt[1] for pt in item[0]]
            y_full = crop_y0 + int(min(ys)) // up  # back out the upscale + crop
            stamp = f"{d} {m.group(2)}:{m.group(3)}:{m.group(4)}"
            if best is None or y_full < best[1]:
                best = (stamp, y_full)
    return (best[0], best[1]) if best else (None, None)


def frame_crop(im, clock_top=None):
    """Return the CCTV-frame region as a PIL crop box (x0,y0,x1,y1).

    Frame is full-width (0..W) and ~16:9; its top is anchored just above the
    burned-in clock (or FRAME_TOP_FALLBACK when the clock is unreadable).
    """
    W, H = im.size
    top = (clock_top - FRAME_TOP_INSET) if clock_top is not None \
        else FRAME_TOP_FALLBACK
    height = int(round(W * FRAME_ASPECT)) + 3  # small tolerance for rounding
    y0 = max(0, top)
    y1 = min(H, y0 + height)
    return (0, y0, W, y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--threshold", type=float, default=SCORE_THRESHOLD)
    ap.add_argument("--printed-scores", default=None,
                    help="optional CSV (columns image_name, fire_confidence) to "
                         "fill the printed-score column, e.g. a vision-model "
                         "reading of the tiny red text on each alert")
    args = ap.parse_args()

    # firewatch/ is not a package: put the firewatch dir on sys.path so
    # `from firewatch import FireModel` resolves to firewatch/firewatch.py
    # (the same production OpenVINO decoder the firewatch service uses).
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo, "firewatch"))
    from firewatch import FireModel  # noqa: E402

    model = FireModel(args.model_dir)
    files = sorted(
        f for f in os.listdir(args.src)
        if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
    )
    rows = []
    auto_reads = {}
    for fname in files:
        im = Image.open(os.path.join(args.src, fname)).convert("RGB")
        label = read_overlay_label(im)
        ts, clock_top = read_cctv_clock(im)
        crop = im.crop(frame_crop(im, clock_top))
        dets = model.detect(crop, score_thresh=0.01)
        fire = max((d["score"] for d in dets if d["label"].lower() == "fire"),
                   default=0.0)
        smoke = max((d["score"] for d in dets if d["label"].lower() == "smoke"),
                    default=0.0)
        if label:
            auto_reads[fname] = f"{label[0]} {label[1]:.2f}".strip()
        rows.append({
            "file": fname,
            "alert_ts": ts or "",
            "printed_score": "",  # filled from --printed-scores
            "v4_fire": f"{fire:.3f}",
            "v4_smoke": f"{smoke:.3f}",
            "verdict": "",        # user fills TRUE / FALSE
        })
        print(f"{fname}: ts={ts or ''!r} frame={crop} v4_fire={fire:.3f} "
              f"v4_smoke={smoke:.3f}")

    # Optional: merge an external "printed score" reading (e.g. a vision model).
    printed_src = None
    agree, differ, missing = [], [], []
    if args.printed_scores and os.path.isfile(args.printed_scores):
        filled = {}
        with open(args.printed_scores, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                name = (r.get("image_name") or "").strip()
                conf = (r.get("fire_confidence") or "").strip()
                if name and conf:
                    filled[name] = conf
        for row in rows:
            if row["file"] in filled:
                row["printed_score"] = filled[row["file"]]
            else:
                missing.append(row["file"])
        agree = sorted(f for f in auto_reads if filled.get(f) == auto_reads[f])
        differ = sorted(f for f in auto_reads
                        if f in filled and filled[f] != auto_reads[f])
        printed_src = f"{os.path.basename(args.printed_scores)} ({len(filled)} rows)"
    elif args.printed_scores:
        print(f"WARN: --printed-scores file not found: {args.printed_scores}")

    cols = ["file", "alert_ts", "printed_score", "v4_fire", "v4_smoke",
            "verdict"]
    csv_path = os.path.join(args.src, "fire_alerts_scores.csv")
    md_path = os.path.join(args.src, "fire_alerts_scores.md")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Markdown review table - TRUE/FALSE is the column the user fills in.
    if printed_src:
        score_step = ("1. `printed score` was filled from the provided "
                      "reading.\n")
        printed_note = (f"- `printed score` source: {printed_src} (a vision "
                        f"reading of the red text on each alert).\n"
                        + (f"- Cross-check vs local OCR: **agrees exactly on** "
                           f"{', '.join(agree)}.\n" if agree else "")
                        + (f"- Cross-check vs local OCR: **disagrees on** "
                           f"{', '.join(differ)} - worth a quick visual check "
                           f"of those.\n" if differ else "")
                        + "- Local OCR experiments this session gave different "
                          "values for 084522 (~0.58), 084543 (~0.63) and "
                          "084620 (~0.57/0.67) than the reading above - if you "
                          "want certainty on those three, give them a quick "
                          "visual check before marking.\n"
                        + (f"- No external score found for: {', '.join(missing)}"
                           f"\n" if missing else ""))
    else:
        score_step = ("1. (optional) type the score printed on the alert photo "
                      "into `printed score`\n")
        auto_note = (", ".join(f"{k} = {v}" for k, v in auto_reads.items())
                     if auto_reads else "none")
        printed_note = (f"- OCR could only auto-read the tiny printed score on: "
                        f"{auto_note} - the rest are blank to fill in.\n")

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Fire alert screenshots review (media/Gmail)\n\n"
                 "One row per Telegram fire-alert screenshot. Open each image "
                 "(`media/Gmail/`), then:\n"
                 + score_step
                 + "2. fill `TRUE/FALSE` with **TRUE** (real fire) or **FALSE** "
                 "(false positive).\n\n")
        fh.write("Notes:\n"
                 "- `alert time` = burned-in CCTV date/time on the alert frame "
                 "(read by OCR).\n"
                 "- `active-model fire/smoke` = the ACTIVE model (`models/fire`, "
                 "v4) re-run on the **CCTV frame area of the screenshot** "
                 "(crop x0..1080, y top-of-frame..+610, anchored to the burned-in "
                 "clock), not the whole screenshot. These alerts were sent "
                 "2026-09-06/07 by an earlier checkpoint - v4 is the comparison "
                 "you want for 'was this a model error that still persists'.\n"
                 + printed_note + "\n")
        fh.write("| # | file | alert time (CCTV) | printed score | "
                 "active-model fire | smoke | TRUE/FALSE |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(rows, 1):
            fh.write(f"| {i} | {r['file']} | {r['alert_ts']} | "
                     f"{r['printed_score']} | {r['v4_fire']} | "
                     f"{r['v4_smoke']} |  |\n")
    print(f"\nWrote {csv_path} and {md_path} ({len(rows)} screenshots)"
          f"{'; printed scores merged from ' + printed_src if printed_src else ''}")


if __name__ == "__main__":
    sys.exit(main())
