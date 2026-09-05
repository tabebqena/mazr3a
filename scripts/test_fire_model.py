#!/usr/bin/env python3
"""test_fire_model.py - benchmark a fire/smoke checkpoint on a YOLOv8 dataset locally.

Shared helper for the fire-model benchmarks (plans/fire-model-dataset-test.md,
plans/fire-model-abonia-benchmark.md). Runs on CPU with ultralytics + PyTorch.

Class-index contract (verified for best.pt and the Abonia fire-8 dataset):
    index 0 = fire, index 1 = other/default (ignored in production), index 2 = smoke.
The two index spaces align, so this harness does NO remapping - it only reports.

Two complementary outputs:
  (a) ultralytics model.val()  -> mAP@50 / mAP@50-95 / P / R (all classes).
  (b) deploy-oriented pass at --conf over every image -> per-image CSV,
      image-level fire/smoke detection-rate, and annotated JPGs (GT + predictions).

Usage:
    python scripts/test_fire_model.py <model.pt> <data.yaml> [--images-dir DIR]
        [--conf 0.5] [--imgsz 640] [--out DIR] [--annotate]
"""
import argparse
import csv
import json
import os
import sys
import time

from PIL import Image, ImageDraw

GT_COLORS = {0: (255, 70, 70), 1: (150, 150, 150), 2: (255, 200, 40)}   # fire/other/smoke
PRED_COLORS = {0: (220, 20, 20), 1: (110, 110, 110), 2: (230, 170, 0)}  # bold-ish
# Production semantics: firewatch tracks fire(0) [+ optional smoke(2)]; class 1 = "other" is ignored.


def yaml_val_images(data_path):
    """Return absolute images dir of the 'val:' key in a tiny yaml parser."""
    base = os.path.dirname(os.path.abspath(data_path))
    val = None
    with open(data_path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("val:") and ":" in line:
                val = line.split(":", 1)[1].strip().strip("'\"")
                break
    if not val:
        sys.exit(f"no 'val:' key found in {data_path}")
    if not os.path.isabs(val):
        # try relative to data.yaml dir first, then to any 'path:' root
        cand = os.path.join(base, val)
        if os.path.isdir(cand):
            return cand
        with open(data_path, encoding="utf-8") as fh:
            for raw in fh:
                if raw.strip().startswith("path:"):
                    root = raw.split(":", 1)[1].strip().strip("'\"")
                    cand = os.path.join(root, val)
                    if os.path.isdir(cand):
                        return cand
        sys.exit(f"cannot resolve val images: {val}")
    return val


def read_boxes(label_path):
    boxes = []
    if os.path.isfile(label_path):
        with open(label_path, encoding="utf-8") as fh:
            for line in fh:
                p = line.split()
                if len(p) >= 5:
                    try:
                        boxes.append([int(float(p[0]))] + [float(x) for x in p[1:5]])
                    except ValueError:
                        pass
    return boxes


def xyxy(box, W, H):
    _, cx, cy, bw, bh = box
    return [(cx - bw / 2) * W, (cy - bh / 2) * H,
            (cx + bw / 2) * W, (cy + bh / 2) * H]


def annotate(img_path, gt_boxes, preds, names):
    im = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(im)
    W, H = im.size
    for box in gt_boxes:
        c = box[0]
        col = GT_COLORS.get(c, (0, 0, 220))
        d.rectangle(xyxy(box, W, H), outline=col, width=2)
        d.text((xyxy(box, W, H)[0] + 2, xyxy(box, W, H)[1] - 10),
               "GT %s(%d)" % (names.get(c, "?"), c), fill=col)
    for c, conf, x1, y1, x2, y2 in preds:
        col = PRED_COLORS.get(c, (0, 0, 220))
        d.rectangle([x1, y1, x2, y2], outline=col, width=4)
        d.text((x1 + 2, y1 + 2), "%s %.2f" % (names.get(c, "?"), conf), fill=col)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_pt")
    ap.add_argument("data_yaml")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=None)
    ap.add_argument("--annotate", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO  # imported here so --help stays light

    model = YOLO(args.model_pt)
    m_names = model.names
    print("model.names:", m_names)
    if {m_names.get(0), m_names.get(1), m_names.get(2)} != {None}:
        print("  -> index contract: 0=%s 1=%s 2=%s (1 is the catch-all 'other')"
              % (m_names.get(0, "?"), m_names.get(1, "?"), m_names.get(2, "?")))

    images_dir = args.images_dir or yaml_val_images(args.data_yaml)
    if not os.path.isdir(images_dir):
        sys.exit(f"images dir not found: {images_dir}")
    out = os.path.abspath(args.out or os.path.join(
        os.path.dirname(args.data_yaml), "results"))
    os.makedirs(out, exist_ok=True)
    if args.annotate:
        os.makedirs(os.path.join(out, "annotated"), exist_ok=True)

    # ---- (a) ultralytics val metrics ----
    print("\n[1/2] model.val() over", images_dir)
    t0 = time.time()
    res = model.val(data=args.data_yaml, imgsz=args.imgsz, device="cpu",
                    verbose=False, conf=0.001,
                    project=os.path.join(out, "runs"), name="val")
    box = res.box
    nc = len(m_names)
    metrics = {
        "mAP50": float(box.map50) if hasattr(box, "map50") else None,
        "mAP50_95": float(box.map) if hasattr(box, "map") else None,
        "precision": float(box.mp) if hasattr(box, "mp") else None,
        "recall": float(box.mr) if hasattr(box, "mr") else None,
        "per_class": {},
    }
    print("val time %.1fs" % (time.time() - t0))
    print("Overall: mAP@50=%.4f mAP@50-95=%.4f P=%.4f R=%.4f"
          % (metrics["mAP50"], metrics["mAP50_95"], metrics["precision"], metrics["recall"]))
    # ap50/ap/p/r are per-CLASS arrays aligned to box.ap_class_index (classes that had GT),
    # NOT to 0..nc-1 - map them back to model class indices and names.
    apci = [int(v) for v in box.ap_class_index] if hasattr(box, "ap_class_index") else list(range(nc))
    seq = {k: [] for k in ("ap50", "ap", "p", "r")}
    for attr in seq:
        v = getattr(box, attr, None)
        seq[attr] = [float(x) for x in v] if v is not None else []
    for k, ci in enumerate(apci):
        nm = m_names.get(ci, str(ci))
        p50 = seq["ap50"][k] if k < len(seq["ap50"]) else None
        p = seq["ap"][k] if k < len(seq["ap"]) else None
        pr = seq["p"][k] if k < len(seq["p"]) else None
        rc = seq["r"][k] if k < len(seq["r"]) else None
        metrics["per_class"][nm] = {"index": ci, "mAP50": p50, "mAP50_95": p,
                                    "precision": pr, "recall": rc}
        print("  class %d %-10s mAP@50=%.4f mAP@50-95=%.4f P=%.4f R=%.4f"
              % (ci, nm, p50 if p50 is not None else -1, p if p is not None else -1,
                 pr if pr is not None else -1, rc if rc is not None else -1))
    for ci in range(nc):
        if ci not in apci:
            nm = m_names.get(ci, str(ci))
            metrics["per_class"].setdefault(nm, {"index": ci, "mAP50": None, "mAP50_95": None,
                                                 "precision": None, "recall": None})
            print("  class %d %-10s (no ground truth in split)" % (ci, nm))

    # ---- (b) deploy-oriented per-image pass ----
    print("\n[2/2] per-image predict (conf=%.2f)" % args.conf)
    csv_path = os.path.join(out, "per_image.csv")
    det_csv = os.path.join(out, "detection_rate.csv")
    rows = []
    stat = {"gt_fire_imgs": 0, "gt_smoke_imgs": 0, "gt_other_imgs": 0,
            "det_fire_of_gt_fire": 0, "det_smoke_of_gt_smoke": 0,
            "no_gt_fire_imgs": 0, "no_gt_smoke_imgs": 0,
            "fp_fire_imgs": 0, "fp_smoke_imgs": 0,
            "clean_imgs": 0, "any_pred_on_clean": 0}
    lbl_dir = os.path.join(os.path.dirname(images_dir), "labels")
    files = sorted(f for f in os.listdir(images_dir)
                   if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    t0 = time.time()
    for k, fname in enumerate(files, 1):
        img_path = os.path.join(images_dir, fname)
        stem = os.path.splitext(fname)[0]
        gt = read_boxes(os.path.join(lbl_dir, stem + ".txt"))
        g_fire = sum(1 for b in gt if b[0] == 0)
        g_smoke = sum(1 for b in gt if b[0] == 2)
        g_other = sum(1 for b in gt if b[0] == 1)

        r = model.predict(img_path, conf=args.conf, imgsz=args.imgsz,
                          device="cpu", verbose=False)[0]
        preds = []
        p_fire = p_smoke = p_other = 0
        mc_fire = mc_smoke = 0.0
        if len(r.boxes):
            for cls, conf, xy in zip(r.boxes.cls.tolist(),
                                     r.boxes.conf.tolist(),
                                     r.boxes.xyxy.tolist()):
                ci = int(cls)
                preds.append((ci, float(conf)) + tuple(xy))
                if ci == 0:
                    p_fire += 1
                    mc_fire = max(mc_fire, float(conf))
                elif ci == 2:
                    p_smoke += 1
                    mc_smoke = max(mc_smoke, float(conf))
                else:
                    p_other += 1

        rows.append([fname, g_fire, g_smoke, g_other, p_fire, p_smoke, p_other,
                     round(mc_fire, 3), round(mc_smoke, 3),
                     int(g_fire > 0), int(g_smoke > 0),
                     int(p_fire > 0), int(p_smoke > 0)])
        if g_fire:
            stat["gt_fire_imgs"] += 1
            if p_fire:
                stat["det_fire_of_gt_fire"] += 1
        else:
            stat["no_gt_fire_imgs"] += 1
            if p_fire:
                stat["fp_fire_imgs"] += 1
        if g_smoke:
            stat["gt_smoke_imgs"] += 1
            if p_smoke:
                stat["det_smoke_of_gt_smoke"] += 1
        else:
            stat["no_gt_smoke_imgs"] += 1
            if p_smoke:
                stat["fp_smoke_imgs"] += 1
        if g_other:
            stat["gt_other_imgs"] += 1
        if not gt:
            stat["clean_imgs"] += 1
            if p_fire or p_smoke or p_other:
                stat["any_pred_on_clean"] += 1

        if args.annotate:
            im = annotate(img_path, gt, preds, m_names)
            im.save(os.path.join(out, "annotated", fname), quality=85)
        if k % 20 == 0:
            print("  %d/%d (%.1fs)" % (k, len(files), time.time() - t0))

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "gt_fire", "gt_smoke", "gt_other",
                    "pred_fire", "pred_smoke", "pred_other",
                    "max_conf_fire", "max_conf_smoke",
                    "gt_has_fire", "gt_has_smoke", "det_has_fire", "det_has_smoke"])
        w.writerows(rows)

    dr = {}
    dr["images"] = len(files)
    dr["conf"] = args.conf
    dr["fire_images"] = stat["gt_fire_imgs"]
    dr["fire_detected_images"] = stat["det_fire_of_gt_fire"]
    dr["fire_image_recall"] = (stat["det_fire_of_gt_fire"] / stat["gt_fire_imgs"]
                               if stat["gt_fire_imgs"] else None)
    dr["smoke_images"] = stat["gt_smoke_imgs"]
    dr["smoke_detected_images"] = stat["det_smoke_of_gt_smoke"]
    dr["smoke_image_recall"] = (stat["det_smoke_of_gt_smoke"] / stat["gt_smoke_imgs"]
                                if stat["gt_smoke_imgs"] else None)
    dr["fp_fire_images_on_nonfire"] = (stat["fp_fire_imgs"], stat["no_gt_fire_imgs"])
    dr["fp_smoke_images_on_nonsmoke"] = (stat["fp_smoke_imgs"], stat["no_gt_smoke_imgs"])
    dr["clean_images"] = stat["clean_imgs"]
    dr["any_pred_on_clean_images"] = stat["any_pred_on_clean"]

    with open(det_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for k, v in dr.items():
            w.writerow([k, v])

    print("\nDetection-rate summary (image-level, conf=%.2f):" % args.conf)
    for k, v in dr.items():
        print("  %-28s %s" % (k, v))

    metrics["detection_rate"] = dr
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    print("\nwrote:", csv_path, det_csv, os.path.join(out, "metrics.json"))
    print("total %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
