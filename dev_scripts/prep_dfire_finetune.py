#!/usr/bin/env python3
"""prep_dfire_finetune.py - build the Phase-4 fine-tune set from the CLEAN model-order datasets.

Phase-4 tooling for the clean evaluation / retraining workflow
(plans/fire-model-clean-eval-workflow.md). After Phase 3 confirmed the retrain
gate for D-Fire, this merges the two CLEAN, MODEL-ORDER sources into one
self-contained train/val layout for fine-tuning models/fire/best.pt:

    --dfire   fire-model-training/dedup/dfire_model_order    (D-Fire clean, model order)
              -> train: 6,869 imgs (4,643 labeled fire/smoke + 2,226 background)
              -> test : 2,164 imgs (HELD OUT - never enters this training split)
    --abonia  fire-model-training/dedup/abonia_dedup         (Abonia clean, model order)
              -> train: 182 imgs (fire 49 / other 31 / smoke 122) - folded into TRAIN only

Both sources are already in the model index contract fire(0)/other(1)/smoke(2)
(D-Fire labels were remapped 1->0,0->2 by remap_dataset_classes.py; Abonia was
never reordered - its Roboflow export is Fire(0)/default(1)/smoke(2)), so labels
are copied VERBATIM - only the data.yaml names are pinned to fire/other/smoke.

Class-index facts:
  * D-Fire has NO class-1 ("other") boxes - only fire(0) and smoke(2) + empties.
  * Abonia contributes the only class-1 (other) supervision in the mix (31 boxes),
    keeping the 3-class head alive during fine-tune. This is a key reason to
    include Abonia even at only 2.6 % of the train count.

Split policy:
  * The held-out D-Fire TEST split is NOT copied into this set - final eval uses
    the clean D-Fire test via fire-model-training/dedup/dfire_clean_eval/data.yaml.
  * A small VAL split (default ~5 % of LABELED images) is carved out of the
    D-Fire train, group-aware by source clip so frames of the same fire video do
    not straddle train/val. Background/empty D-Fire images and ALL Abonia images
    go to TRAIN only (empty labels = FP-reduction negatives, as in
    prep_fire_large_dataset.py).

Output (default fire-model-training/dedup/dfire_finetune/):
    train/images/ train/labels/   # D-Fire labeled (minus val) + D-Fire bg + Abonia
    val/images/   val/labels/     # carved labeled D-Fire (early-stopping / mAP)
    data.yaml                     # names fire/other/smoke, nc=3, abs path
    prep_report.txt               # counts + class coverage
    <out>_colab.zip (--zip)       # single archive for Colab upload

Usage:
    .venv/bin/python dev_scripts/prep_dfire_finetune.py \
        --dfire fire-model-training/dedup/dfire_model_order \
        --abonia fire-model-training/dedup/abonia_dedup \
        --out fire-model-training/dedup/dfire_finetune \
        --val-frac 0.05 --seed 0 --zip
"""
import argparse
import os
import random
import re
import shutil
import zipfile
from collections import Counter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = ["fire", "other", "smoke"]
AOFT_RE = re.compile(r"^AoF(\d+)")
AOF_BLOCK = 1000  # treat AoF frames in 1000-frame blocks as one source clip


def read_classes(label_path):
    if not os.path.isfile(label_path):
        return ()
    cls = []
    with open(label_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if parts:
                try:
                    cls.append(int(float(parts[0])))
                except ValueError:
                    pass
    return tuple(sorted(set(cls)))


def source_key(name):
    """Group frames of the same source clip.

    D-Fire 'AoF<num>.jpg' are sequential fire-video frames -> block by 1000 so the
    same scene never straddles train/val. WEB/PublicDataset/Abonia names are
    independent images -> each is its own group (random pick).
    """
    m = AOFT_RE.match(name)
    if m:
        return "AoF%04d" % (int(m.group(1)) // AOF_BLOCK)
    return os.path.splitext(name)[0]


def copy_pair(img_src, lbl_src, img_dst, lbl_dst, force_empty=False):
    if os.path.exists(img_dst):
        raise SystemExit("filename collision in output: %s" % img_dst)
    shutil.copy2(img_src, img_dst)
    if force_empty or not lbl_src or not os.path.isfile(lbl_src):
        open(lbl_dst, "w").close()  # background negative
        return 0
    shutil.copy2(lbl_src, lbl_dst)
    n = 0
    with open(lbl_dst, encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                n += 1
    return n


def collect_split(root, split):
    """Return {name: classes} for a Roboflow split (images+labels)."""
    imdir = os.path.join(root, split, "images")
    lbldir = os.path.join(root, split, "labels")
    out = {}
    if not os.path.isdir(imdir):
        raise SystemExit("missing %s split images in %s" % (split, root))
    for f in sorted(os.listdir(imdir)):
        if os.path.splitext(f)[1].lower() not in IMG_EXTS:
            continue
        lbl = os.path.join(lbldir, os.path.splitext(f)[0] + ".txt") \
            if os.path.isdir(lbldir) else None
        out[f] = read_classes(lbl)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dfire", required=True,
                    help="clean model-order D-Fire root (dfire_model_order)")
    ap.add_argument("--abonia", default=None,
                    help="clean model-order Abonia root (abonia_dedup); fold its train "
                         "into TRAIN only. Use '' or omit to skip.")
    ap.add_argument("--out", default="fire-model-training/dedup/dfire_finetune")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zip", nargs="?", const="", default=None,
                    help="also write a Colab-upload zip (default: <out>_colab.zip)")
    args = ap.parse_args()

    d_fire = os.path.abspath(args.dfire)
    a_bonia = os.path.abspath(args.abonia) if args.abonia else None
    out = os.path.abspath(args.out)

    for d in (d_fire,):
        if not os.path.isdir(d):
            raise SystemExit("dfire root not found: %s" % d)
    if a_bonia and not os.path.isdir(a_bonia):
        raise SystemExit("abonia root not found: %s" % a_bonia)
    if os.path.exists(out) and any(os.scandir(out)):
        raise SystemExit("--out already exists and is not empty: %s" % out)

    # ---- source images (read-only) ----
    d_train = collect_split(d_fire, "train")
    d_test = collect_split(d_fire, "test")
    a_train = collect_split(a_bonia, "train") if a_bonia else {}
    print("read-only sources (never modified): %s%s"
          % (d_fire, " + " + a_bonia if a_bonia else ""))

    # ---- split D-Fire LABELED train into train/val, group-aware by clip ----
    d_labeled = [(n, c) for n, c in d_train.items() if c]
    d_bg = [n for n, c in d_train.items() if not c]
    groups = {}
    for n, c in d_labeled:
        groups.setdefault(source_key(n), []).append((n, c))
    gkeys = sorted(groups)
    rng = random.Random(args.seed)
    rng.shuffle(gkeys)
    target_val = int(round(len(d_labeled) * args.val_frac))
    val_set, val_groups = {}, []
    val_n = 0
    for k in gkeys:
        g = groups[k]
        if val_n < target_val:
            for n, c in g:
                val_set[n] = c
            val_n += len(g)
            val_groups.append(k)
        else:
            break  # remaining groups stay in train (we only assign the head groups)

    # ---- copy to output ----
    ti, tl = os.path.join(out, "train", "images"), os.path.join(out, "train", "labels")
    vi, vl = os.path.join(out, "val", "images"), os.path.join(out, "val", "labels")
    for d in (ti, tl, vi, vl):
        os.makedirs(d, exist_ok=True)
    d_img = os.path.join(d_fire, "train", "images")
    d_lbl = os.path.join(d_fire, "train", "labels")

    train_cls = Counter()
    val_cls = Counter()
    n_train_lab = n_train_bg = 0
    for n in sorted(d_train):
        stem = os.path.splitext(n)[0]
        is_val = n in val_set
        dst_img = os.path.join(vi, n) if is_val else os.path.join(ti, n)
        dst_lbl = os.path.join(vl, stem + ".txt") if is_val else os.path.join(tl, stem + ".txt")
        nb = copy_pair(os.path.join(d_img, n),
                       os.path.join(d_lbl, stem + ".txt") if os.path.isfile(os.path.join(d_lbl, stem + ".txt")) else None,
                       dst_img, dst_lbl, force_empty=(not is_val and not d_train[n]))
        if is_val:
            for c in d_train[n]:
                val_cls[c] += 1
        else:
            if d_train[n]:
                n_train_lab += 1
                for c in d_train[n]:
                    train_cls[c] += 1
            else:
                n_train_bg += 1

    # Abonia -> train only
    n_abonia = 0
    if a_bonia and a_train:
        a_img = os.path.join(a_bonia, "train", "images")
        a_lbl = os.path.join(a_bonia, "train", "labels")
        for n in sorted(a_train):
            stem = os.path.splitext(n)[0]
            nb = copy_pair(os.path.join(a_img, n),
                           os.path.join(a_lbl, stem + ".txt"),
                           os.path.join(ti, n), os.path.join(tl, stem + ".txt"))
            for c in a_train[n]:
                train_cls[c] += 1
            n_abonia += 1

    # ---- data.yaml ----
    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write("# Phase-4 fine-tune set: clean D-Fire (model order) + clean Abonia.\n")
        fh.write("# Class contract matches models/fire/best.pt: fire=0, other=1, smoke=2 (no remap; names only).\n")
        fh.write(f"path: {out}\n")
        fh.write("train: train/images\n")
        fh.write("val: val/images\n")
        fh.write(f"nc: {len(NAMES)}\n")
        fh.write("names:\n")
        for i, n in enumerate(NAMES):
            fh.write(f"  {i}: {n}\n")

    # ---- report ----
    rep = []
    rep.append(f"Phase-4 fine-tune prep ({os.path.basename(out)})")
    rep.append("=" * 64)
    rep.append(f"dfire source   : {d_fire}  (train {len(d_train)} / test {len(d_test)} held out)")
    rep.append(f"abonia source  : {a_bonia or '(none)'}  (train {len(a_train)} -> TRAIN only)")
    rep.append(f"seed / val-frac: {args.seed} / {args.val_frac}")
    rep.append(f"D-Fire labeled train : {len(d_labeled)}   D-Fire background : {len(d_bg)}")
    rep.append(f"val clip groups used : {sorted(val_groups)} ({val_n} labeled imgs)")
    rep.append("")
    n_train_total = len(os.listdir(ti))
    n_val_total = len(os.listdir(vi))
    rep.append(f"TRAIN images (on disk) = {n_train_total}  "
               f"(D-Fire labeled {n_train_lab} + D-Fire bg {n_train_bg} + Abonia {n_abonia})")
    rep.append(f"  per-class boxes (train) : " + ", ".join(
        f"{NAMES[k]}={v}" for k, v in sorted(train_cls.items())))
    rep.append(f"VAL images (on disk, labeled) = {n_val_total}")
    rep.append(f"  per-class boxes (val) : " + ", ".join(
        f"{NAMES[k]}={v}" for k, v in sorted(val_cls.items())))
    rep.append("")
    missing = [NAMES[i] for i in range(len(NAMES)) if train_cls.get(i, 0) == 0]
    rep.append("WARNING: no train boxes for class(es): " + (", ".join(missing) if missing else "none"))
    rep.append("NOTE: D-Fire test (2,164 imgs) is NOT in this set - final eval uses "
               "dfire_clean_eval/data.yaml (held-out).")
    rep.append("data.yaml -> " + yaml_path)
    text = "\n".join(rep)
    with open(os.path.join(out, "prep_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)

    # ---- optional Colab zip ----
    if args.zip is not None:
        zpath = os.path.join(os.path.dirname(out), os.path.basename(out) + "_colab.zip") \
            if args.zip == "" else os.path.abspath(args.zip)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for d in ("train", "val"):
                for sub in ("images", "labels"):
                    src = os.path.join(out, d, sub)
                    for f in sorted(os.listdir(src)):
                        z.write(os.path.join(src, f), os.path.join(d, sub, f))
            z.write(yaml_path, "data.yaml")
        print("\ncolab zip -> " + zpath)


if __name__ == "__main__":
    main()
