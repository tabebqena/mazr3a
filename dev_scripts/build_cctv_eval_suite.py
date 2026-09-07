#!/usr/bin/env python3
"""build_cctv_eval_suite.py - build the Phase-5 CCTV Smoke & Fire compare suite.

Phase-5 tooling for the clean evaluation / retraining workflow
(plans/fire-model-clean-eval-workflow.md).

WHY
---
"4_CCTV_Emergency" (Kaggle simuletic/cctv-smoke-and-fire-emergency-detection-dataset,
100 % synthetic, high-angle CCTV) is structured as ~48 events (24 fire + 24 smoke,
`fire_detected_varN_*` / `smoke_detected_varN_*`) x ~5 near-identical frames each -
the SAME event filmed from a few camera angles (user: "nearly identical, same event,
with few angle changes"). Scoring every frame would count each event ~5x and inflate
image-level recall, so the eval suite is built at EVENT level: one representative
frame per `varN` family.

The export is FLAT (images/+labels/ sibling - no train/valid/test split dirs), and its
native class order is 0=fire, 1=smoke (NOT the model contract fire/other/smoke; its
shipped data.yaml is also broken: `nc: 1` yet two names). remap_dataset_classes.py
(split-dir-only) therefore does not apply - this helper copies the KEPT representative
images verbatim and rewrites each label row 1->2 (smoke -> model smoke; fire stays 0;
class 1 'other' gets no GT in this set).

USAGE
-----
  # Event-level dedup + remap to model order + write the clean flat suite:
  .venv/bin/python dev_scripts/build_cctv_eval_suite.py \
      --src fire-model-training/4_CCTV_Emergency \
      --out fire-model-training/dedup/cctv_clean_eval \
      --names fire,other,smoke

  # representative: 'medoid' (frame closest to the family's other frames, default)
  #                 'first' (lexicographically first labelled frame of the family)

SAFETY / READ-ONLY CONTRACT
---------------------------
Never modifies the source tree (images/labels opened read-only). Outputs (images,
labels, data.yaml, report) are written only under --out, which must be OUTSIDE the
source root; refuses to run otherwise.

OUTPUTS
-------
  <out>/images/ + <out>/labels/      flat suite (one representative per event family)
  <out>/data.yaml                    path + train/val = images, names = model order
  <out>/build_report.txt             family -> representative + box counts before/after
"""
import argparse
import os
import shutil
import sys
from collections import Counter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REMAP = {0: 0, 1: 2}  # native fire0 stays; native smoke1 -> model smoke2


def is_image(name):
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def ensure_outside(src_root, dst):
    src_root = os.path.abspath(src_root)
    dst = os.path.abspath(dst)
    try:
        if os.path.commonpath([src_root, dst]) == src_root:
            sys.exit(
                "--out (%s) is inside the read-only source %s - refusing.\n"
                "This tool NEVER modifies a source dataset; it only writes a NEW "
                "clean suite to a NEW location outside it." % (dst, src_root))
    except ValueError:
        pass  # different drive/prefix (e.g. Windows)


def family_of(name):
    """'fire_detected_var10_img3.png' -> 'fire_detected_var10' (or None)."""
    base = name
    for ext in IMG_EXTS:
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    parts = base.rsplit("_", 1)  # drop imgN
    if len(parts) == 2 and parts[1].startswith("img"):
        return parts[0]
    return None


def read_boxes(label_path):
    boxes = []
    if os.path.isfile(label_path):
        with open(label_path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    p = ln.split()
                    try:
                        boxes.append([int(float(p[0]))] + [float(x) for x in p[1:5]])
                    except ValueError:
                        pass
    return boxes


def _lanczos():
    """Pillow's LANCZOS resampling constant (API-safe across versions)."""
    img = __import__("PIL.Image", fromlist=["Image"])
    res = getattr(getattr(img, "Resampling", None), "LANCZOS", None)
    return res if res is not None else getattr(img, "LANCZOS", 1)


def _gray32(path):
    from PIL import Image
    return Image.open(path).convert("L").resize((32, 32), _lanczos())


def medoid_index(names, imgdir):
    """Index (into names) of the labelled frame with min summed 32x32 MAE to the
    other labelled family frames; ties -> lexicographically smallest name."""
    thumbs = []
    for n in names:
        try:
            thumbs.append((n, _gray32(os.path.join(imgdir, n))))
        except Exception as e:  # noqa: BLE001 - unreadable frame -> exclude
            print("  ! unreadable, excluding from representative pick:", n, e)
            thumbs.append((n, None))
    # MAE via raw pixel access; images are tiny (32x32) so pure Python is fine.
    def mae(a, b):
        if a is None or b is None:
            return None
        pa, pb = a.load(), b.load()
        s = 0
        for y in range(32):
            for x in range(32):
                s += abs(pa[x, y] - pb[x, y])
        return s / (32 * 32 * 255.0)

    # precompute pairwise; None-pairs count as excluded (not best)
    n = len(names)
    if n == 1:
        return 0
    sums = [0.0] * n
    anyv = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = mae(thumbs[i][1], thumbs[j][1])
            if v is not None:
                sums[i] += v
                anyv[i] = True
    best = min(i for i in range(n) if anyv[i])
    for i in range(n):
        if anyv[i] and sums[i] < sums[best]:
            best = i
    return best


def rewrite_label(src, dst, cmap):
    """Copy a label file remapping leading class index; empty stays empty."""
    lines = [ln for ln in open(src, encoding="utf-8") if ln.strip()]
    counts = Counter()
    if not lines:
        open(dst, "w").close()
        return counts
    out = []
    for ln in lines:
        p = ln.split()
        old = int(float(p[0]))
        new = cmap.get(old)
        if new is None:
            raise SystemExit(
                "label row has native class %d which is NOT in --map %r (file %s) - "
                "map every class present in the source before running." % (old, cmap, src))
        out.append("%d %s\n" % (new, " ".join(p[1:])))
        counts[new] += 1
    with open(dst, "w", encoding="utf-8") as fh:
        fh.writelines(out)
    return counts


def write_data_yaml(out_root, names):
    lines = [
        "# generated by dev_scripts/build_cctv_eval_suite.py - event-level clean "
        "CCTV suite (class remap applied to the model contract)",
        "path: %s" % out_root.replace("\\", "/"),
        "train: images",
        "val: images",
        "nc: %d" % len(names),
        "names:",
    ]
    for i, nm in enumerate(names):
        lines.append("  %d: %s" % (i, nm))
    with open(os.path.join(out_root, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Phase-5: build the event-level CCTV Smoke & Fire compare suite "
                    "(one representative per varN family; native fire0/smoke1 -> fire/other/smoke).")
    ap.add_argument("--src", default="fire-model-training/4_CCTV_Emergency",
                    help="source root with flat images/+labels/ (read-only)")
    ap.add_argument("--out", default="fire-model-training/dedup/cctv_clean_eval",
                    help="NEW clean-suite root (must be outside --src)")
    ap.add_argument("--names", default="fire,other,smoke",
                    help="comma-separated model-order class names (default fire,other,smoke)")
    ap.add_argument("--representative", choices=("medoid", "first"), default="medoid",
                    help="how to pick one frame per event family (default medoid)")
    args = ap.parse_args()

    src_root = os.path.abspath(args.src)
    if not os.path.isdir(src_root):
        sys.exit("source not found: %s" % src_root)
    imgdir = os.path.join(src_root, "images")
    lbldir = os.path.join(src_root, "labels")
    if not os.path.isdir(imgdir):
        sys.exit("no images/ dir under source: %s" % imgdir)
    ensure_outside(src_root, args.out)
    out_root = os.path.abspath(args.out)
    if os.path.exists(out_root) and any(os.scandir(out_root)):
        sys.exit("--out already exists and is not empty: %s" % out_root)

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    print("read-only sources (never modified): %s" % src_root)

    # group labelled images by event family
    fam = {}  # family key -> sorted labelled image names
    no_lbl = []
    for name in sorted(os.listdir(imgdir)):
        if not is_image(name):
            continue
        key = family_of(name)
        stem = os.path.splitext(name)[0]
        has_lbl = os.path.isfile(os.path.join(lbldir, stem + ".txt"))
        if key is None:
            print("  ! image not in a varN family, skipping:", name)
            continue
        fam.setdefault(key, []).append(name)
        if not has_lbl:
            no_lbl.append(name)
    if no_lbl:
        print("  note: frames without a label are excluded from representative picks:")
        for n in no_lbl:
            print("    -", n)

    os.makedirs(os.path.join(out_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "labels"), exist_ok=True)

    report = []
    report.append("CCTV Smoke & Fire - event-level clean suite build")
    report.append("=" * 70)
    report.append("source (read-only)      : %s" % src_root)
    report.append("representative per event: %s" % args.representative)
    report.append("class remap             : native fire0->0, smoke1->2")
    n_fam = n_img = 0
    boxes_before = Counter()
    boxes_after = Counter()
    for key in sorted(fam):
        cands = [n for n in fam[key]
                 if os.path.isfile(os.path.join(lbldir, os.path.splitext(n)[0] + ".txt"))]
        if not cands:
            sys.exit("family %s has NO labelled frame - cannot build suite." % key)
        pick = (sorted(cands)[0] if args.representative == "first"
                else cands[medoid_index(cands, imgdir)])
        stem = os.path.splitext(pick)[0]
        src_img = os.path.join(imgdir, pick)
        src_lbl = os.path.join(lbldir, stem + ".txt")
        dst_img = os.path.join(out_root, "images", pick)
        dst_lbl = os.path.join(out_root, "labels", stem + ".txt")
        shutil.copy2(src_img, dst_img)
        n_img += 1
        bb = Counter(b[0] for b in read_boxes(src_lbl))
        boxes_before.update(bb)
        ab = rewrite_label(src_lbl, dst_lbl, REMAP)
        boxes_after.update(ab)
        n_fam += 1
        report.append("%-34s <- %s   (native %s -> model %s)"
                      % (key, pick,
                         ", ".join("%d:%d" % (k, v) for k, v in sorted(bb.items())),
                         ", ".join("%d:%d" % (k, v) for k, v in sorted(ab.items()))))

    write_data_yaml(out_root, names)
    report.append("-" * 70)
    report.append("events (families)       : %d" % n_fam)
    report.append("representative images   : %d" % n_img)
    report.append("boxes BEFORE (native)   : %s" % dict(boxes_before))
    report.append("boxes AFTER (model)     : %s" % dict(boxes_after))
    report.append("data.yaml names         : %s" % ", ".join(names))
    with open(os.path.join(out_root, "build_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    print("\n".join(report))
    print("\nwritten:", out_root)
    print("NOTE: the 234-frame dHash-near-dup view and all 240 angle frames remain "
          "archived under the source (4_CCTV_Emergency/); this suite is the event-level "
          "(48) view used for the Phase-5 compare matrix.")


if __name__ == "__main__":
    main()
