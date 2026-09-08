#!/usr/bin/env python3
"""compare_gmail_models.py - v1 vs v4 on the same fire-alert frames (diagnosis).

WHY THIS EXISTS: the fire-alert review (media/Gmail/fire_alerts_scores.csv) showed the
ACTIVE v4 re-scoring <0.50 fire on every one of the 11 TRUE-fire frames, which was read
as "v4 would miss all real fires". That conclusion is WRONG, and this script proves why:

  1. firewatch paints a RED box + "fire <conf>" text on the alert photo it sends to
     Telegram. The screenshot review re-scores that OVERLAID image - input the deployed
     model never sees (it detects on the RAW frame, then draws the box on the copy).
  2. The red overlay sitting on/near the fire suppresses the fire class. Inpainting the
     pure-red overlay out (cv2.inpaint, keeping orange fire pixels) lifts v4's fire score
     on most TRUE fires from ~0.05-0.27 to ~0.47-0.67 (>=0.50 on ~6/11). So the "v4 is
     silent" headline was an artefact of scoring overlaid screenshots.
  3. v1 (extracted from git at 32ede8e = parent of the v4 promotion 1336b1b) is scored
     the same way for a like-for-like model-vs-model comparison on IDENTICAL pixels
     (both raw-crop and de-overlaid): v4's fire confidence is still lower than v1's on
     these scenes (~half), and v4 re-labels palm-leaf motion as SMOKE (0.45-0.75), which
     persists even with the overlay removed.

Outputs:
  media/Gmail/model_compare_v1_v4.csv  - per-frame top fire/other/smoke for v1 and v4,
                                         on the raw crop and on the de-overlaid crop
  stdout                                  - summary + class-confusion detail

Usage (needs the dev venv: openvino pillow numpy opencv-python-headless):
  .venv/bin/python dev_scripts/compare_gmail_models.py
"""
import csv
import os
import sys

import cv2
import numpy as np

V4_DIR = "models/fire"
V1_DIR = "media/_v1ir"      # extracted v1 IR (git 32ede8e) - gitignored media/
CROPS = os.path.join("media", "Gmail", "cropped_frames")
REVIEW_CSV = os.path.join("media", "Gmail", "fire_alerts_scores.csv")
OUT_CSV = os.path.join("media", "Gmail", "model_compare_v1_v4.csv")
THR = 0.01


def load_model(model_dir):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo, "firewatch"))
    from firewatch import FireModel  # noqa: PLC0415 - repo layout, not a package
    return FireModel(model_dir)


def de_overlay(img_bgr):
    """Inpaint the pure-red firewatch box+text, keeping orange/white fire pixels.

    firewatch draws with outline='red' / fill='red' i.e. ~(255,0,0). Fire is orange
    (R high, G mid) or white-hot (all high) - a tight pure-red threshold + slight
    dilate separates the thin overlay from the fire. Returns (clean_bgr, red_area_frac).
    """
    b, g, r = cv2.split(img_bgr.astype(np.int16))
    mask = (((r - g) > 110) & ((r - b) > 110) & (r > 140)).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    area = float(mask.mean()) / 255.0
    if area <= 0:
        return img_bgr, 0.0
    return cv2.inpaint(img_bgr, mask, 5, cv2.INPAINT_TELEA), area


def top_by_label(model, pil_img):
    dets = model.detect(pil_img, score_thresh=THR)
    top = {}
    for d in dets:
        lab = d["label"].lower()
        if d["score"] > top.get(lab, 0.0):
            top[lab] = d["score"]
    return top, dets


def main():
    from PIL import Image  # noqa: PLC0415

    review = {}
    with open(REVIEW_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.reader(fh):
            if r and r[0]:
                review[r[0]] = (r[5] if len(r) > 5 else "",
                                r[6] if len(r) > 6 else "")

    files = sorted(f for f in os.listdir(CROPS) if f.lower().endswith(".jpg"))
    m1 = load_model(V1_DIR)   # v1 (the checkpoint that raised these alerts)
    m4 = load_model(V4_DIR)   # v4 (current ACTIVE)
    print(f"scoring {len(files)} frames with v1 and v4 "
          f"(raw crop + de-overlaid crop)\n")

    out = []
    for fname in files:
        img_bgr = cv2.imread(os.path.join(CROPS, fname))
        if img_bgr is None:  # skip unreadable frame
            continue
        img_clean, ovr_frac = de_overlay(img_bgr)
        verdict, note = review.get(fname, ("", ""))
        t1o, d1o = top_by_label(m1, Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
        t4o, d4o = top_by_label(m4, Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
        t1c, d1c = top_by_label(m1, Image.fromarray(cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)))
        t4c, d4c = top_by_label(m4, Image.fromarray(cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)))

        def fmt(t):
            return (f"{t.get('fire', 0.0):.3f}", f"{t.get('other', 0.0):.3f}",
                    f"{t.get('smoke', 0.0):.3f}")

        o = {
            "file": fname, "verdict": verdict, "note": note,
            "overlay_frac": f"{ovr_frac:.3f}",
            "v1_fire_ovr": fmt(t1o)[0], "v1_other_ovr": fmt(t1o)[1],
            "v1_smoke_ovr": fmt(t1o)[2],
            "v1_fire_clean": fmt(t1c)[0], "v1_other_clean": fmt(t1c)[1],
            "v1_smoke_clean": fmt(t1c)[2],
            "v4_fire_ovr": fmt(t4o)[0], "v4_other_ovr": fmt(t4o)[1],
            "v4_smoke_ovr": fmt(t4o)[2],
            "v4_fire_clean": fmt(t4c)[0], "v4_other_clean": fmt(t4c)[1],
            "v4_smoke_clean": fmt(t4c)[2],
            "v4_clean_dets": "; ".join(
                f"{d['label']} {d['score']:.2f}"
                for d in sorted(d4c, key=lambda x: x["score"], reverse=True)[:4]),
        }
        out.append(o)

    keys = ["file", "verdict", "note", "overlay_frac",
            "v1_fire_ovr", "v1_other_ovr", "v1_smoke_ovr",
            "v1_fire_clean", "v1_other_clean", "v1_smoke_clean",
            "v4_fire_ovr", "v4_other_ovr", "v4_smoke_ovr",
            "v4_fire_clean", "v4_other_clean", "v4_smoke_clean", "v4_clean_dets"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out)

    # ---- summary ----------------------------------------------------------
    def grp(cond):
        return [o for o in out if cond(o)]

    for name, rows in (
        ("TRUE fires (11)", grp(lambda o: o["verdict"] == "TRUE")),
        ("FALSE (17)", grp(lambda o: o["verdict"] == "FALSE")),
        ("FALSE palm-leaf (9)", grp(lambda o: "palm" in o["note"])),
    ):
        def avg(k):
            return sum(float(o[k]) for o in rows) / len(rows) if rows else float("nan")

        def n_ge(k, v=0.5):
            return sum(1 for o in rows if float(o[k]) >= v)

        print(f"{name:<20} n={len(rows):>2}")
        print(f"  v1 fire  overlay {avg('v1_fire_ovr'):.3f} -> clean {avg('v1_fire_clean'):.3f}"
              f"   (clean>=0.5 x{n_ge('v1_fire_clean')})")
        print(f"  v4 fire  overlay {avg('v4_fire_ovr'):.3f} -> clean {avg('v4_fire_clean'):.3f}"
              f"   (clean>=0.5 x{n_ge('v4_fire_clean')})")
        print(f"  v4 smoke overlay {avg('v4_smoke_ovr'):.3f} -> clean {avg('v4_smoke_clean'):.3f}"
              f"   (clean>=0.5 x{n_ge('v4_smoke_clean')})")

    print("\n-- TRUE fires: v4 fire with overlay vs de-overlaid (firewatch red box removed) --")
    for o in [x for x in out if x["verdict"] == "TRUE"]:
        mark = " <== reaches gate" if float(o["v4_fire_clean"]) >= 0.5 else ""
        print(f"  {o['file'][-12:-4]}  v4 fire {o['v4_fire_ovr']} -> {o['v4_fire_clean']}"
              f"  (v1 clean {o['v1_fire_clean']}){mark}")
    print(f"\nwrote {len(out)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
