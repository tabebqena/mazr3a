#!/usr/bin/env python3
"""remap_dataset_classes.py - copy a YOLOv8/Roboflow dataset, rewriting label class indices.

Phase-3 tooling for the clean evaluation / retraining workflow
(plans/fire-model-clean-eval-workflow.md).

WHY
---
Dedup and per-set storage always preserve the NATIVE class order of each source
set (dedup_images.py copies labels verbatim - never remaps). But the ACTIVE
fire model (models/fire/) and the eval harness (dev_scripts/test_fire_model.py)
use the fixed index contract fire(0)/other(1)/smoke(2). D-Fire is the outlier:
its shipped labels are 0=smoke, 1=fire (the inverse). To evaluate v1 on (clean)
D-Fire - and later to fine-tune on it - the labels must be remapped to the model
contract without ever touching the source set:

    D-Fire 1 (fire)  -> model 0 (fire)
    D-Fire 0 (smoke) -> model 2 (smoke)

This helper copies images + labels from a SOURCE dataset root into a NEW root,
rewriting each label's leading class index through an explicit old:new map, and
writes a data.yaml with the target names. Source stays byte-for-byte untouched.

USAGE
-----
  # Remap clean D-Fire test split to the model contract (Phase 3 clean-eval).
  # --test-as-val writes BOTH val: and test: -> test/images so the output
  # data.yaml is directly usable by dev_scripts/test_fire_model.py (D-Fire has
  # no 'valid' split; the held-out test split plays the val role, as in Phase 2):
  .venv/bin/python dev_scripts/remap_dataset_classes.py \
      --src fire-model-training/dedup/dfire_dedup \
      --out fire-model-training/dedup/dfire_clean_eval \
      --map "1:0,0:2" --names "fire,other,smoke" --splits test --test-as-val

  # Remap the whole clean set (train + test) for a Phase-4 fine-tune zip:
  .venv/bin/python dev_scripts/remap_dataset_classes.py \
      --src fire-model-training/dedup/dfire_dedup \
      --out fire-model-training/dedup/dfire_model_order \
      --map "1:0,0:2" --names "fire,other,smoke"

SAFETY / READ-ONLY CONTRACT
---------------------------
Like the other dev_scripts in this workflow this tool NEVER modifies the source
tree. Sources are opened read-only; every output (images/labels/data.yaml/
report) is written under --out, which must be a NEW path OUTSIDE the source
root. It refuses to run when --out resolves inside --src.

OUTPUTS
-------
  <out>/{train,valid,test}/{images,labels}   (splits actually present in --splits)
  <out>/data.yaml                            path + names + split keys
  <out>/remap_report.txt                     per-split box counts before/after
"""
import argparse
import os
import shutil
import sys
from collections import Counter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
_NAMES_DICT_RE = None  # unused; we accept names on the CLI


def _is_img(name):
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def ensure_outside(src_root, dst):
    src_root = os.path.abspath(src_root)
    dst = os.path.abspath(dst)
    try:
        if os.path.commonpath([src_root, dst]) == src_root:
            sys.exit(
                "--out (%s) is inside the read-only source %s - refusing.\n"
                "This tool NEVER modifies a source dataset; it only writes a "
                "remapped COPY to a NEW location outside it."
                % (dst, src_root))
    except ValueError:
        pass  # different drive/prefix (e.g. Windows)


def split_images_dirs(root):
    return {sp: os.path.join(root, sp, "images")
            for sp in SPLITS if os.path.isdir(os.path.join(root, sp, "images"))}


def labels_dir_for(imdir):
    parent = os.path.dirname(imdir)
    cand = os.path.join(parent, "labels")
    return cand if os.path.isdir(cand) else None


def parse_map(spec):
    """'1:0,0:2' -> {1: 0, 0: 2} (int keys/values)."""
    out = {}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            sys.exit("bad --map token %r (want old:new)" % tok)
        a, b = tok.split(":", 1)
        out[int(a)] = int(b)
    if not out:
        sys.exit("--map is empty")
    return out


def rewrite_label(src, dst, cmap):
    """Copy a label file, mapping each row's leading class index. Empty stays empty."""
    with open(src, encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    if not lines:
        open(dst, "w").close()
        return 0
    out = []
    counts = Counter()
    for ln in lines:
        parts = ln.split()
        old = int(float(parts[0]))
        new = cmap.get(old)
        if new is None:
            raise SystemExit(
                "label row has class %d which is NOT in --map %r (file %s) - "
                "map every class present in the source before running."
                % (old, cmap, src))
        out.append("%d %s\n" % (new, " ".join(parts[1:])))
        counts[new] += 1
    with open(dst, "w", encoding="utf-8") as fh:
        fh.writelines(out)
    return sum(counts.values())


def write_data_yaml(out_root, splits_present, names, test_as_val=False):
    lines = [
        "# generated by dev_scripts/remap_dataset_classes.py (class remap applied)",
        "path: %s" % out_root.replace("\\", "/"),
    ]
    for sp in SPLITS:
        if sp in splits_present:
            lines.append("%s: %s/images" % (sp, sp))
    if test_as_val and "test" in splits_present and "valid" not in splits_present:
        # D-Fire has no valid split: held-out test plays the val role for
        # test_fire_model.py (which reads the 'val:' key only), Phase 3 clean-eval.
        lines.append("val: test/images")
    if names:
        lines.append("nc: %d" % len(names))
        lines.append("names:")
        for i, nm in enumerate(names):
            lines.append("  %d: %s" % (i, nm))
    with open(os.path.join(out_root, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Copy a YOLOv8/Roboflow dataset root, remapping label class indices "
                    "(e.g. D-Fire 0=smoke,1=fire -> model 0=fire,1=other,2=smoke).")
    ap.add_argument("--src", required=True, help="source dataset root (read-only)")
    ap.add_argument("--out", required=True,
                    help="NEW destination root (must be outside --src)")
    ap.add_argument("--map", required=True,
                    help="class remap as old:new pairs, e.g. '1:0,0:2'")
    ap.add_argument("--names", default=None,
                    help="comma-separated target names (e.g. fire,other,smoke) for data.yaml")
    ap.add_argument("--splits", default=None,
                    help="comma subset of train,valid,test to process (default: all present)")
    ap.add_argument("--test-as-val", action="store_true",
                    help="D-Fire has no 'valid' split; write BOTH val: and test: -> test/images "
                         "so the output data.yaml is directly usable by dev_scripts/test_fire_model.py "
                         "(Phase 3 clean-eval).")
    args = ap.parse_args()

    src_root = os.path.abspath(args.src)
    if not os.path.isdir(src_root):
        sys.exit("source not found: %s" % src_root)
    ensure_outside(src_root, args.out)
    out_root = os.path.abspath(args.out)
    if os.path.exists(out_root) and any(os.scandir(out_root)):
        sys.exit("--out already exists and is not empty: %s" % out_root)

    cmap = parse_map(args.map)
    names = [n.strip() for n in args.names.split(",")] if args.names else None
    wanted = set(args.splits.split(",")) if args.splits else None

    sdirs = split_images_dirs(src_root)
    if not sdirs:
        sys.exit("no train/valid/test split dirs found under source: %s" % src_root)
    if wanted:
        missing = sorted(wanted - set(sdirs))
        if missing:
            sys.exit("requested --splits %s but source lacks %s" %
                     (sorted(wanted), missing))
        sdirs = {sp: d for sp, d in sdirs.items() if sp in wanted}

    print("read-only sources (never modified): %s" % src_root)
    print("class remap:", {str(k): str(v) for k, v in sorted(cmap.items())})
    total_before = Counter()
    total_after = Counter()
    n_img_total = n_lbl_total = 0
    for sp, imdir in sdirs.items():
        lbldir = labels_dir_for(imdir)
        dst_im = os.path.join(out_root, sp, "images")
        dst_lb = os.path.join(out_root, sp, "labels")
        os.makedirs(dst_im, exist_ok=True)
        os.makedirs(dst_lb, exist_ok=True)
        n_img = n_lbl = 0
        boxes_before = Counter()
        boxes_after = Counter()
        for name in sorted(os.listdir(imdir)):
            if not _is_img(name):
                continue
            shutil.copy2(os.path.join(imdir, name),
                         os.path.join(dst_im, name))
            n_img += 1
            stem = os.path.splitext(name)[0]
            lbl_src = os.path.join(lbldir, stem + ".txt") if lbldir else None
            lbl_dst = os.path.join(dst_lb, stem + ".txt")
            if lbl_src and os.path.isfile(lbl_src):
                # count source boxes per class (informational)
                with open(lbl_src, encoding="utf-8") as fh:
                    for ln in fh:
                        if ln.strip():
                            boxes_before[int(float(ln.split()[0]))] += 1
                n = rewrite_label(lbl_src, lbl_dst, cmap)
                n_lbl += 1
                # n is box count after remap (rewrite_label tallies new classes)
                with open(lbl_dst, encoding="utf-8") as fh:
                    for ln in fh:
                        if ln.strip():
                            boxes_after[int(float(ln.split()[0]))] += 1
            else:
                open(lbl_dst, "w").close()  # no source label -> empty (kept aligned)
                n_lbl += 1
        print("  [%s] images %d / label files %d / boxes %d -> %d"
              % (sp, n_img, n_lbl, sum(boxes_before.values()),
                 sum(boxes_after.values())))
        total_before.update(boxes_before)
        total_after.update(boxes_after)
        n_img_total += n_img
        n_lbl_total += n_lbl

    write_data_yaml(out_root, set(sdirs), names, args.test_as_val)
    rep = []
    rep.append("Class remap report: %s -> %s" % (src_root, out_root))
    rep.append("map: " + ", ".join("%d->%d" % (k, v)
                                   for k, v in sorted(cmap.items())))
    rep.append("images copied  : %d" % n_img_total)
    rep.append("label files    : %d" % n_lbl_total)
    rep.append("boxes BEFORE by native class: " +
               ", ".join("%d:%d" % (k, total_before[k])
                         for k in sorted(total_before)))
    rep.append("boxes AFTER  by target class: " +
               ", ".join("%d:%d" % (k, total_after[k])
                         for k in sorted(total_after)))
    rep.append("data.yaml names: %s" % (", ".join(names) if names else "(none)"))
    with open(os.path.join(out_root, "remap_report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    print("\nwritten:", out_root)


if __name__ == "__main__":
    main()
