#!/usr/bin/env python3
"""score_gmail_screenshots.py - analyse Telegram fire-alert screenshots.

The user downloaded (from Telegram) the alert photos that `firewatch.py`
(`firewatch/firewatch.py`) sent for every fire/smoke alarm. Each alert photo is
the CCTV frame with firewatch's RED box + label text overlaid (e.g. "fire 0.73")
- that text is the confidence the model that fired the alert had at that moment.
The screenshots are full 1080x2400 Android captures (media/Gmail/*.jpg), so each
holds one alert's CCTV frame in the middle (the burned-in CCTV date/time sits at
~y 845 and the alert frame spans roughly y 835-1440).

For every screenshot this tool emits one row:
  file        : screenshot file name
  alert_ts    : burned-in CCTV date/time of the alert (read with OCR)
  label_read  : best-effort read of the overlaid red "fire/smoke <conf>" text
                ("" when the tiny text could not be read reliably)
  v4_fire     : top-1 fire confidence of the ACTIVE model (models/fire, v4) when
                re-run on the screenshot - a re-evaluation that may legitimately
                differ from the on-image value if that alert came from an older
                checkpoint (pre-v4 alerts on 2026-09-06/07 often score < 0.5 now)
  v4_smoke    : top-1 smoke confidence of the same re-run
  v4_alerts   : "yes" when v4_fire >= SCORE_THRESHOLD (0.5) else "no"
  verdict     : empty column for the user to fill TRUE / FALSE per alert

Outputs (written next to the screenshots so the user can review/edit):
  media/Gmail/fire_alerts_scores.csv
  media/Gmail/fire_alerts_scores.md      (human review table)

Dependencies (throwaway venv is fine):
  pip install openvino pillow numpy opencv-python-headless rapidocr_onnxruntime
  + system `tesseract` (used for the tiny red label; rapidocr for the date/time).
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

# Matches firewatch's overlay text: "<label> <conf>" e.g. "fire 0.73".
LABEL_RE = re.compile(
    r"(fire|smoke|tra|snoke|srnoke|fir|fite|smok|tir|ture|fve|fue|seoke|smoke|fure|tmoke)"
    r"\s*[^0-9]{0,4}([01]\.\d{1,2})",
    re.IGNORECASE,
)
DEC_RE = re.compile(r"([01]\.\d{1,2})")
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
    """Run tesseract on a PIL image; return list of stdout strings per psm."""
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

    Returns (word, conf) e.g. ("fire", 0.73) or None. Multiple crops/engines are
    tried; only matches that name an actual fire/smoke-like word are kept so a
    stray number in the CCTV burn-in is not mistaken for the confidence.
    """
    bb = red_mask(im)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    W, H = im.size
    found = {}  # conf -> word

    def try_ocr(crop, up):
        txts = _tesseract(_scale(crop, up))
        for t in txts:
            for m in LABEL_RE.finditer(t):
                w = m.group(1).lower()
                c = float(m.group(2))
                found[c] = found.get(c) and w or w
        # word-less decimal fallback only when nothing wordy yet
        if not found:
            for t in txts:
                for m in DEC_RE.finditer(t):
                    c = float(m.group(1))
                    found.setdefault(c, "")

    # candidate crops: the red content (box+text), its top strip (label zone),
    # and generous band around the red top-left (label sits just above the box).
    cands = [
        (max(0, x0 - 8), max(0, y0 - 8), min(W, x1 + 8), min(H, y1 + 8)),
        (max(0, x0 - 40), max(0, y0 - 45), min(W, x1 + 40), min(H, y0 + 60)),
        (max(0, x0 - 8), max(0, y0 - 45), min(W, min(x0 + 260, x1 + 8)), min(H, y0 + 10)),
    ]
    for cx0, cy0, cx1, cy1 in cands:
        if cx1 - cx0 < 12 or cy1 - cy0 < 6:
            continue
        try_ocr(im.crop((cx0, cy0, cx1, cy1)), 5)
    if not found:
        return None
    # prefer a word-bearing match; else highest conf
    wordy = {c: w for c, w in found.items() if w}
    if wordy:
        c = max(wordy)
        return (wordy[c], c)
    c = max(found)
    return ("", c)


def read_cctv_time(im):
    """OCR the burned-in CCTV 'dd/mm/yyyy hh:mm:ss' timestamp (top-left of frame)."""
    # Whole-image pass only needs the small top-left area; but rapidocr on the
    # full image is cheap enough and found it in ~25/28. Crop top-left too.
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        _scale(im.crop((0, 800, 420, 960)), 2).save(tf.name)
        path = tf.name
    try:
        res, _ = RapidOCR()(path)
    finally:
        os.unlink(path)
    if not res:
        return ""
    for item in res:
        t = str(item[1])
        m = DATE_RE.search(t)
        if m:
            d = m.group(1).replace("-", "/")
            return f"{d} {m.group(2)}:{m.group(3)}:{m.group(4)}"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--threshold", type=float, default=SCORE_THRESHOLD)
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
        ts = read_cctv_time(im)
        dets = model.detect(im, score_thresh=0.01)
        fire = max((d["score"] for d in dets if d["label"].lower() == "fire"),
                   default=0.0)
        smoke = max((d["score"] for d in dets if d["label"].lower() == "smoke"),
                    default=0.0)
        if label:
            auto_reads[fname] = f"{label[0]} {label[1]:.2f}".strip()
        rows.append({
            "file": fname,
            "alert_ts": ts,
            "printed_score": "",  # user types what they see on the alert
            "v4_fire": f"{fire:.3f}",
            "v4_smoke": f"{smoke:.3f}",
            "verdict": "",        # user fills TRUE / FALSE
        })
        print(f"{fname}: ts={ts!r} v4_fire={fire:.3f} v4_smoke={smoke:.3f}")

    cols = ["file", "alert_ts", "printed_score", "v4_fire", "v4_smoke",
            "verdict"]
    csv_path = os.path.join(args.src, "fire_alerts_scores.csv")
    md_path = os.path.join(args.src, "fire_alerts_scores.md")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    # Markdown review table - TRUE/FALSE is the column the user fills in.
    auto_note = (", ".join(f"{k} = {v}" for k, v in auto_reads.items())
                 if auto_reads else "none")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Fire alert screenshots review (media/Gmail)\n\n"
                 "One row per Telegram fire-alert screenshot. Open each image "
                 "(`media/Gmail/`), then:\n"
                 "1. (optional) type the score printed on the alert photo into "
                 "`printed score`\n"
                 "2. fill `TRUE/FALSE` with **TRUE** (real fire) or **FALSE** "
                 "(false positive).\n\n")
        fh.write("Notes:\n"
                 "- `alert time` = burned-in CCTV date/time on the alert frame "
                 "(read by OCR).\n"
                 "- `active-model fire/smoke` = top confidence of the ACTIVE "
                 "model (`models/fire`, v4) when re-run on the screenshot now. "
                 "These alerts were sent 2026-09-06/07, i.e. by an earlier "
                 "checkpoint, and v4 currently scores fire < 0.50 for every one "
                 "of them (the 0.50 gate in `config/firewatch.conf`) - so v4 "
                 "would **not** alert on any of these frames today. Your "
                 "TRUE/FALSE verdict is the ground truth.\n"
                 f"- OCR could only auto-read the tiny printed score on: "
                 f"{auto_note}.\n\n")
        fh.write("| # | file | alert time (CCTV) | printed score | "
                 "active-model fire | smoke | TRUE/FALSE |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(rows, 1):
            fh.write(f"| {i} | {r['file']} | {r['alert_ts']} | "
                     f"{r['printed_score']} | {r['v4_fire']} | "
                     f"{r['v4_smoke']} |  |\n")
    print(f"\nWrote {csv_path} and {md_path} ({len(rows)} screenshots)")


if __name__ == "__main__":
    sys.exit(main())
