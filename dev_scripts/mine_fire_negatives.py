#!/usr/bin/env python3
"""mine_fire_negatives.py - mine HARD NEGATIVES for the fire model from the local
camera-clip corpus.

The fire model mis-labels some subjects as ``fire`` (cam01 dogs, sunlit objects,
IR fur). Those frames are the *hard negatives* that fix the model - but they are
rare and scattered, so this tool finds them by running the PRODUCTION model over
sampled frames of the local clips (``camera-clips/<cam>/*.mp4``) and keeping every
frame whose ``fire`` score lands in the misclassification band
(``--min-score`` .. ``--max-score``, default 0.35 .. 0.90).

Output is a ready-to-use YOLO *negative* set (see
[`build_fire_negatives_eval.py`](build_fire_negatives_eval.py)) plus provenance:

    <out>/images/<cam>_<clip>_<ms>ms_conf<score>.jpg
    <out>/labels/<same>.txt        # EMPTY = background (no fire/smoke boxes)
    <out>/data.yaml                # negative eval split (names fire/other/smoke)
    <out>/manifest.csv             # cam, clip, offset_s, fire_conf, boxes, coco, status
    <out>/SUMMARY.txt              # per-camera counts + score histogram

The candidates are found WITH the buggy model, so the band may contain a real
(rare) fire: the manifest/contact-sheet is for human triage BEFORE training.

Usage:
    .venv/bin/python dev_scripts/mine_fire_negatives.py \
        --clips-dir camera-clips --out fire-model-training/dog_negatives \
        --every-s 2 --min-score 0.35 --max-score 0.90
    # smoke test:
    .venv/bin/python dev_scripts/mine_fire_negatives.py --cams cam08 --limit-clips 8

RUN-WIDTH NOTE: 6,458 clips x ~5 samples @ ~0.15 s/frame is ~30-60 min on CPU.
Use --cams / --limit-clips to slice, and --every-s to trade yield for time.
Read-only against the clips; the ONLY writes are under --out.
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image


def dhash(img, size=8):
    """64-bit difference hash as a bool array (scale/brightness robust)."""
    small = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.int16)
    return (arr[:, 1:] > arr[:, :-1]).flatten()


def hamming_min(bits, kept_matrix):
    """Min Hamming distance from `bits` to every kept row (inf when empty)."""
    if kept_matrix.shape[0] == 0:
        return 10 ** 9
    return int(np.min(np.sum(kept_matrix != bits, axis=1)))


def load_fire_model(model_dir):
    """Reuse the PRODUCTION decoder from firewatch/firewatch.py."""
    repo = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.insert(0, os.path.join(repo, "firewatch"))
    from firewatch import FireModel  # noqa: E402
    return FireModel(model_dir)


def load_coco(coco_path):
    """Optional ultralytics ONNX COCO model for an animal cross-check."""
    if not coco_path:
        return None
    from ultralytics import YOLO  # noqa: E402
    return YOLO(coco_path, task="detect")


def coco_labels(model, img_path, conf):
    """Comma-joined COCO labels above `conf` (for triage only)."""
    res = model.predict(img_path, conf=conf, verbose=False)[0]
    names = model.names
    found = []
    if res.boxes is not None:
        for cls, c in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            found.append("%s:%.2f" % (names.get(int(cls), int(cls)), float(c)))
    return ",".join(sorted(set(found)))


def clip_list(clips_dir, cams):
    """[(camera, clip_path)] across the selected cameras, sorted."""
    out = []
    for cam in cams:
        d = os.path.join(clips_dir, cam)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".mp4", ".mkv", ".avi")):
                out.append((cam, os.path.join(d, f)))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-dir", default="camera-clips")
    ap.add_argument("--out", default="fire-model-training/dog_negatives")
    ap.add_argument("--model-dir", default="models/fire")
    ap.add_argument("--cams", help="comma-separated cameras (default: all dirs)")
    ap.add_argument("--every-s", type=float, dest="every_s", default=2.0,
                    help="sample one frame every N seconds of each clip [2.0]")
    ap.add_argument("--min-score", type=float, dest="min_score", default=0.35,
                    help="keep frames with fire score >= this [0.35]")
    ap.add_argument("--max-score", type=float, dest="max_score", default=0.90,
                    help="... and <= this (band = model 'sees' fire) [0.90]")
    ap.add_argument("--max-per-clip", type=int, dest="max_per_clip", default=3,
                    help="cap kept frames per clip [3]")
    ap.add_argument("--hamming", type=int, default=6,
                    help="drop candidates within this dHash distance of a kept "
                         "one (0 = keep all) [6]")
    ap.add_argument("--limit-clips", type=int, dest="limit_clips",
                    help="only scan the first N clips (smoke test)")
    ap.add_argument("--coco", default=None,
                    help="optional COCO onnx (e.g. models/coco/yolo11s.onnx) to tag "
                         "animal presence in candidates")
    ap.add_argument("--coco-conf", type=float, dest="coco_conf", default=0.25)
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality [95]")
    args = ap.parse_args()

    import cv2  # noqa: E402 - lazy so --help stays light

    frames_csv_out = os.path.join(args.out, "manifest.csv")
    img_dir = os.path.join(args.out, "images")
    lbl_dir = os.path.join(args.out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    cams = ([c.strip() for c in args.cams.split(",") if c.strip()]
            if args.cams else sorted(
                d for d in os.listdir(args.clips_dir)
                if os.path.isdir(os.path.join(args.clips_dir, d))))
    clips = clip_list(args.clips_dir, cams)
    if args.limit_clips:
        clips = clips[:args.limit_clips]
    if not clips:
        sys.exit("no clips found under %s for cameras %s" % (args.clips_dir, cams))

    model = load_fire_model(args.model_dir)
    coco = load_coco(args.coco)

    rows = []
    kept_matrix = np.zeros((0, 64), dtype=bool)
    per_cam_kept, per_cam_dup, per_cam_scan = {}, {}, {}
    buckets = {">0.35-0.45": 0, ">0.45-0.55": 0, ">0.55-0.65": 0,
               ">0.65-0.75": 0, ">0.75-0.90": 0}

    print("scanning %d clip(s) from %s ..." % (len(clips), args.clips_dir))
    for n, (cam, clip_path) in enumerate(clips, 1):
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            print("  ! cannot open %s" % clip_path)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps * args.every_s)))
        per_cam_scan[cam] = per_cam_scan.get(cam, 0) + 1
        kept_here = 0
        stem = os.path.splitext(os.path.basename(clip_path))[0]
        idx = 0
        while idx < total and kept_here < args.max_per_clip:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            dets = model.detect(pil, score_thresh=args.min_score,
                                allowed={"fire"})
            best = max((d["score"] for d in dets), default=0.0)
            if best >= args.min_score and best <= args.max_score:
                bits = dhash(pil)
                dist = hamming_min(bits, kept_matrix)
                offset = idx / fps if fps else 0.0
                name = "%s_%s_%05dms_conf%.2f.jpg" % (
                    cam, stem, int(round(offset * 1000)), best)
                if args.hamming and dist <= args.hamming:
                    per_cam_dup[cam] = per_cam_dup.get(cam, 0) + 1
                    rows.append([name, cam, stem, "%.3f" % offset,
                                 "%.4f" % best, len(dets), "", "dup"])
                else:
                    dest = os.path.join(img_dir, name)
                    pil.save(dest, quality=args.quality)
                    open(os.path.join(
                        lbl_dir, os.path.splitext(name)[0] + ".txt"), "w").close()
                    kept_matrix = np.vstack([kept_matrix, bits[None, :]])
                    per_cam_kept[cam] = per_cam_kept.get(cam, 0) + 1
                    kept_here += 1
                    coco_txt = ""
                    if coco is not None:
                        try:
                            coco_txt = coco_labels(coco, dest, args.coco_conf)
                        except Exception:  # noqa: BLE001 - triage aid only
                            coco_txt = ""
                    rows.append([name, cam, stem, "%.3f" % offset,
                                 "%.4f" % best, len(dets), coco_txt, "kept"])
                    for key, upper in ((">0.75-0.90", 0.90), (">0.65-0.75", 0.75),
                                       (">0.55-0.65", 0.65), (">0.45-0.55", 0.55),
                                       (">0.35-0.45", 0.45)):
                        if best <= upper:
                            buckets[key] += 1
                            break
            idx += step
        cap.release()
        if n % 200 == 0 or n == len(clips):
            print("  %d/%d clips scanned, %d candidate(s) kept"
                  % (n, len(clips), int(kept_matrix.shape[0])))

    # ---- manifest -----------------------------------------------------------
    with open(frames_csv_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "camera", "clip", "offset_s", "fire_conf",
                    "fire_boxes", "coco", "status"])
        w.writerows(rows)

    # ---- data.yaml (standalone negative eval split) -------------------------
    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("# fire/smoke NEGATIVE (background) split - empty labels.\n")
        fh.write("path: %s\n" % os.path.abspath(args.out))
        fh.write("train: %s\n" % os.path.abspath(img_dir))
        fh.write("val: %s\n" % os.path.abspath(img_dir))
        fh.write("test: %s\n" % os.path.abspath(img_dir))
        fh.write("nc: 3\nnames:\n  0: fire\n  1: other\n  2: smoke\n")

    # ---- summary ------------------------------------------------------------
    kept = int(kept_matrix.shape[0])
    dup = sum(per_cam_dup.values())
    lines = ["Fire hard-negative mining - %s" % args.clips_dir,
             "=" * 60,
             "clips scanned     : %d" % sum(per_cam_scan.values()),
             "cameras           : %s" % ", ".join(sorted(per_cam_scan)),
             "sample interval   : every %.1f s" % args.every_s,
             "score band        : %.2f .. %.2f" % (args.min_score, args.max_score),
             "candidates kept   : %d" % kept,
             "dropped as near-dup: %d (hamming <= %d)" % (dup, args.hamming),
             ""]
    lines.append("Per camera (kept / near-dup / clips):")
    for cam in sorted(per_cam_scan):
        lines.append("  %-7s %5d / %5d / %d" % (
            cam, per_cam_kept.get(cam, 0), per_cam_dup.get(cam, 0),
            per_cam_scan.get(cam, 0)))
    lines.append("")
    lines.append("Score buckets (kept):")
    for key in (">0.35-0.45", ">0.45-0.55", ">0.55-0.65", ">0.65-0.75", ">0.75-0.90"):
        lines.append("  %-10s %5d" % (key, buckets[key]))
    lines.append("")
    lines.append("Output: %s" % os.path.abspath(args.out))
    lines.append("  images/ labels/ data.yaml manifest.csv")
    lines.append("REVIEW the manifest (and the images) - mining uses the model "
                 "being fixed, so a rare real fire may be in the band.")
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(args.out, "SUMMARY.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
