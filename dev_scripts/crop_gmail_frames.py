#!/usr/bin/env python3
"""crop_gmail_frames.py - crop each Telegram fire-alert screenshot to its CCTV frame.

The alert screenshots in media/Gmail/ (1080x2400) each contain ONE firewatch
alert photo: the CCTV frame shown roughly full-width at y ~830-1440, with the
burned-in CCTV date/time ~13 px below the frame's top edge. This helper crops
each screenshot to that CCTV-frame region so the user can CONFIRM the crop is
right before the ACTIVE model is re-scored on the frames.

Originals are NEVER modified. Outputs (all new, next to the screenshots):
  media/Gmail/cropped_frames/  <name>.jpg  - the CCTV frame crop (~1080x611)
  media/Gmail/crop_overlays/   <name>.jpg  - original with the crop box drawn in
                                              green + the clock anchor marked

The crop geometry is shared with score_gmail_screenshots.py (read_cctv_clock /
frame_crop), so what you confirm here is exactly what V4 is scored on later.

Usage:  python3 dev_scripts/crop_gmail_frames.py [--src media/Gmail]
Needs:  pillow (+ rapidocr_onnxruntime only for the clock OCR used as anchor).
"""
import argparse
import os
import sys

from PIL import Image, ImageDraw

from score_gmail_screenshots import frame_crop, read_cctv_clock  # noqa: E402

SRC = os.path.join("media", "Gmail")
CROPS_DIR = "cropped_frames"
OVERLAYS_DIR = "crop_overlays"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    args = ap.parse_args()

    crops_dir = os.path.join(args.src, CROPS_DIR)
    overlays_dir = os.path.join(args.src, OVERLAYS_DIR)
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(args.src)
        if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"Cropping {len(files)} screenshots from {args.src}")
    print(f"  frames -> {crops_dir}")
    print(f"  overlays -> {overlays_dir}")
    for fname in files:
        im = Image.open(os.path.join(args.src, fname)).convert("RGB")
        ts, clock_top = read_cctv_clock(im)
        box = frame_crop(im, clock_top)
        crop = im.crop(box)
        crop.save(os.path.join(crops_dir, fname), quality=92)

        # overlay: green crop box + red dot at the clock anchor
        ov = im.copy()
        d = ImageDraw.Draw(ov)
        x0, y0, x1, y1 = box
        d.rectangle([x0, y0, x1, y1], outline=(0, 220, 0), width=4)
        if clock_top is not None:
            r = 8
            d.ellipse([x0 + 6, clock_top - r, x0 + 6 + 2 * r, clock_top + r],
                      fill=(255, 0, 0))
        ov.save(os.path.join(overlays_dir, fname), quality=92)
        anchor = f"clock@{clock_top}" if clock_top is not None else "clock?fallback"
        print(f"{fname}: {box} (top={y0}, h={y1-y0}) {anchor} "
              f"ts={ts or ''}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
