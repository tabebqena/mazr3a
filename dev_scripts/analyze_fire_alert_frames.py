#!/usr/bin/env python3
"""analyze_fire_alert_frames.py - re-score stored firewatch alert frames locally.

READ-ONLY diagnostic / analysis helper (no production code change). Given one or
more evidence JPEGs saved by firewatch (the ORIGINAL detect frames it scored, or
its ``*_annotated.jpg`` twins), this:

  1. runs the PRODUCTION fire model (``models/fire`` OpenVINO IR, reusing the
     identical decoder from ``firewatch/firewatch.py`` -> ``FireModel``) and
     prints every fire/smoke box + score, plus the derived motion-bonus view
     (``eff = min(1, raw + MOTION_BONUS)``) so a false-positive alert can be
     reproduced exactly as the watcher saw it; and
  2. optionally runs a COCO detector (``models/coco/*.onnx``) and reports any
     non-fire objects (dog/cat/horse/sheep/cow/person/...) in the SAME frame -
     the cross-check that reveals what the fire model actually locked onto.

Used to investigate the "dogs are confident fire" false positives; see
``plans/firewatch-dog-false-positives.md``.

Usage:
    .venv/bin/python dev_scripts/analyze_fire_alert_frames.py IMG [IMG ...]
        [--fire-model models/fire] [--coco models/coco/yolo11s.onnx]
        [--conf 0.35] [--coco-conf 0.25] [--bonus 0.15]
"""
import argparse
import os
import sys


def _score_fire(fire_dir, images, conf, bonus):
    """Run the production FireModel on each image; print raw + bonus view."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))), "firewatch"))
    from firewatch import FireModel  # noqa: E402  (production decoder, reused)

    model = FireModel(fire_dir)
    print("\n=== FIRE MODEL ({}) conf>={:.2f} ===".format(fire_dir, conf))
    for path in images:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        dets = model.detect(img, score_thresh=conf, allowed={"fire", "smoke"})
        print("\n{}  ({}x{})".format(os.path.basename(path), img.width, img.height))
        if not dets:
            print("  no fire/smoke above {:.2f}".format(conf))
            continue
        for d in sorted(dets, key=lambda x: x["score"], reverse=True):
            raw = d["score"]
            eff = min(1.0, raw + bonus)
            x1, y1, x2, y2 = d["box"]
            print("  {:<5} raw={:.3f}  eff(+{:.2f} if near motion)={:.3f}  "
                  "box=({:.0f},{:.0f},{:.0f},{:.0f}) {:.0f}x{:.0f}px".format(
                      d["label"], raw, bonus, eff, x1, y1, x2, y2,
                      x2 - x1, y2 - y1))


def _score_coco(coco_path, images, conf):
    """Run a COCO detector (ultralytics ONNX) and report non-fire objects."""
    from ultralytics import YOLO  # noqa: E402

    model = YOLO(coco_path, task="detect")
    names = model.names
    print("\n=== COCO MODEL ({}) conf>={:.2f} ===".format(coco_path, conf))
    for path in images:
        res = model.predict(path, conf=conf, verbose=False)[0]
        print("\n{}".format(os.path.basename(path)))
        found = []
        if res.boxes is not None:
            for cls, c, xy in zip(res.boxes.cls.tolist(),
                                  res.boxes.conf.tolist(),
                                  res.boxes.xyxy.tolist()):
                found.append((names.get(int(cls), str(int(cls))), float(c),
                              [round(v) for v in xy]))
        if not found:
            print("  nothing above {:.2f}".format(conf))
        for label, c, xy in sorted(found, key=lambda t: t[1], reverse=True):
            print("  {:<12} {:.2f}  box={}".format(label, c, xy))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--fire-model", default="models/fire")
    ap.add_argument("--coco", default="models/coco/yolo11s.onnx",
                    help="COCO ONNX for the non-fire cross-check ('' = skip)")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="fire-model raw detect floor (SCORE_FLOOR default)")
    ap.add_argument("--coco-conf", type=float, default=0.25)
    ap.add_argument("--bonus", type=float, default=0.15,
                    help="MOTION_BONUS used in the derived eff column")
    args = ap.parse_args()

    missing = [p for p in args.images if not os.path.isfile(p)]
    if missing:
        sys.exit("missing image(s): " + ", ".join(missing))

    _score_fire(args.fire_model, args.images, args.conf, args.bonus)
    if args.coco:
        if os.path.isfile(args.coco):
            _score_coco(args.coco, args.images, args.coco_conf)
        else:
            print("\n[coco] skipped - not found: {}".format(args.coco))


if __name__ == "__main__":
    main()
