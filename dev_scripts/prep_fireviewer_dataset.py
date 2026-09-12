#!/usr/bin/env python3
"""prep_fireviewer_dataset.py - export the FireViewer fire/smoke corpus to a YOLO dataset.

WHY
---
`model-training/datasets--fireviewer--fire-smoke-detection-corpus-v1/` is a *HuggingFace
datasets cache* (blobs/ + snapshots/ + refs/), NOT an image folder: the JPEGs live inside
Parquet shards (`data/{train,validation,test}/*.parquet`) and the boxes live in the
per-source JSONL manifests (`manifests/<source>/*.jsonl`). Ultralytics cannot train on
that layout, so this tool streams the shards and writes a normal YOLO dataset:

    <out>/train/{images,labels}/      <- from the corpus "train" split
    <out>/val/{images,labels}/        <- from the corpus "validation" split
    <out>/test/{images,labels}/       <- from the corpus "test" split (held out)
    <out>/data.yaml                   <- nc=3, names fire/other/smoke
    <out>/prep_report.txt             <- per-split/per-source/class counts + licenses
    <out>_colab.zip                   <- optional single archive for Colab (--zip)

CLASS CONTRACT (must match models/fire/best.pt so it can FINE-TUNE, not retrain)
-------------------------------------------------------------------------------
    flame_visible -> fire  (index 0)
    smoke_visible -> smoke (index 2)
    (no annotation) -> empty .txt   = background negative (firewatch ignores 'other'=1)
The corpus has NO class-1 ("other") boxes; the head is still built with nc=3 so the
existing checkpoint's 3-class head and its `other` supervision (from Abonia/ready) stay
intact. `class_id` in the manifests is NOT consistent across sources (fasdd uses 0 for
smoke, alarmod uses 1 for flame) - we therefore map by `class_name` ONLY.

ROW SCHEMA (pyarrow) - each row is self-describing, no manifest join needed:
    image: struct<bytes: binary, path: string>   # JPEG bytes; path = images/<xx>/<sha>.jpg
    sha256, width, height, split, split_group, negative, source_name, license
    annotations_json: '[{"class_name":"smoke_visible","bbox_xywh":[x,y,w,h]}, ...]'
`bbox_xywh` is ABSOLUTE PIXELS, top-left origin -> converted to normalized YOLO cx cy w h.

SPLIT / LEAKAGE POLICY
----------------------
The corpus ships its own leak-free `split_group` (a clip/sequence/event id). We copy the
splits AS-IS: train -> train, validation -> val, test -> test. `--limit N` samples WHOLE
`split_group`s (never partial), so frames of one clip never straddle a split boundary.
The `test` split is exported but must stay held out (register it as a new suite, do not
train on it).

DEPENDENCY
----------
Needs `pyarrow` in the venv used to run it:
    .venv/bin/python -m pip install pyarrow

USAGE
-----
    # Full export (all 102,257 imgs, ~31 GB on disk) + Colab zip:
    .venv/bin/python dev_scripts/prep_fireviewer_dataset.py \
        --out model-training/fireviewer_v1_yolo --zip

    # Quick smoke test: 300 imgs per split, train+val only:
    .venv/bin/python dev_scripts/prep_fireviewer_dataset.py \
        --out /tmp/fv_smoke --limit 300 --splits train,validation

    # A smaller, fire-balanced training pool: cap each split at 8,000 imgs, drop GPL-3.0
    # (alarmod), keep only the surveillance/smoke-heavy sources:
    .venv/bin/python dev_scripts/prep_fireviewer_dataset.py \
        --out model-training/fireviewer_subset --limit 8000 \
        --exclude-sources alarmod
"""
import argparse
import glob
import json
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict

NAMES = ["fire", "other", "smoke"]
CLASS_BY_NAME = {"flame_visible": 0, "smoke_visible": 2}
# corpus split dir -> output split dir (YOLO key is 'val', not 'validation')
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def die(msg):
    sys.exit("prep_fireviewer_dataset: " + msg)


def require_pyarrow():
    try:
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError:
        die("pyarrow is required: .venv/bin/python -m pip install pyarrow")
    import pyarrow.parquet as pq
    return pq


def resolve_snapshot(corpus):
    """Return the snapshot dir (refs/main revision, else the newest snapshot)."""
    ref = os.path.join(corpus, "refs", "main")
    rev = None
    if os.path.isfile(ref):
        with open(ref, encoding="utf-8") as fh:
            rev = fh.read().strip()
    if rev:
        snap = os.path.join(corpus, "snapshots", rev)
        if os.path.isdir(snap):
            return snap
    snaps = sorted(glob.glob(os.path.join(corpus, "snapshots", "*")))
    snaps = [s for s in snaps if os.path.isdir(os.path.join(s, "data"))]
    if not snaps:
        die("no snapshot with data/ under %s (is this the HF datasets cache?)" % corpus)
    return snaps[0]


def split_parquet(snap, corpus_split):
    d = os.path.join(snap, "data", corpus_split)
    if not os.path.isdir(d):
        return []
    return sorted(glob.glob(os.path.join(d, "*.parquet")))


def choose_groups(pq, files, limit, seed, include=None, exclude=None):
    """Pick whole split_group ids until >= limit rows are covered. None => keep all.

    Source filtering happens BEFORE sampling, so `--sources pyro-sdis --limit N` selects
    N rows worth of pyro-sdis groups (not N random groups that may all be filtered out).
    The cap is SOFT: whole groups are kept, so a split of a few very large sequential
    groups (e.g. pyro-sdis) can overshoot --limit substantially.
    """
    if not limit:
        return None
    sizes = Counter()
    for f in files:
        tbl = pq.ParquetFile(f).read(columns=["split_group", "source_name"])
        groups = tbl.column("split_group").to_pylist()
        srcs = tbl.column("source_name").to_pylist()
        for g, src in zip(groups, srcs):
            if include and src not in include:
                continue
            if exclude and src in exclude:
                continue
            sizes[g] += 1
    groups = sorted(sizes)
    random.Random(seed).shuffle(groups)
    keep, n = set(), 0
    for g in groups:
        if n >= limit:
            break
        keep.add(g)
        n += sizes[g]
    return keep


def norm_boxes(ann_json, width, height):
    """annotations_json -> list of (cls, cx, cy, w, h) normalized; skips bad boxes."""
    rows = []
    if not ann_json:
        return rows
    try:
        anns = json.loads(ann_json)
    except (ValueError, TypeError):
        return rows
    if not isinstance(anns, list):
        return rows
    for a in anns:
        if not isinstance(a, dict):
            continue
        name = a.get("class_name")
        if not isinstance(name, str):
            continue
        cls = CLASS_BY_NAME.get(name)
        if cls is None:
            continue
        bb = a.get("bbox_xywh")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        try:
            x, y, w, h = (float(v) for v in bb)
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0 or width <= 0 or height <= 0:
            continue
        # clamp to the frame, then normalize (cx cy w h)
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(width), x + w), min(float(height), y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        nw, nh = (x1 - x0) / width, (y1 - y0) / height
        cx, cy = (x0 + (x1 - x0) / 2.0) / width, (y0 + (y1 - y0) / 2.0) / height
        rows.append((cls, cx, cy, nw, nh))
    return rows


def write_split(pq, files, split_name, out_split, limit, seed, include, exclude, stats):
    img_dir = os.path.join(out_split, "images")
    lbl_dir = os.path.join(out_split, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    keep = choose_groups(pq, files, limit, seed, include, exclude)
    n_img = n_neg = 0
    boxes_here = Counter()
    int_keys = sys.intern("split_group")
    for f in files:
        pf = pq.ParquetFile(f)
        cols = ["image", "sha256", "width", "height", "negative",
                "annotations_json", "source_name", "split_group", "license"]
        for batch in pf.iter_batches(batch_size=256, columns=cols):
            for row in batch.to_pylist():
                key = row.get(int_keys)
                if keep is not None and key not in keep:
                    continue
                src = row.get("source_name")
                if include and src not in include:
                    continue
                if exclude and src in exclude:
                    continue
                blob = row.get("image") or {}
                b = blob.get("bytes") if isinstance(blob, dict) else None
                stem = row.get("sha256") or os.path.splitext(
                    os.path.basename(blob.get("path", "") if isinstance(blob, dict) else ""))[0]
                if not b or not stem:
                    stats["skipped_no_image"] += 1
                    continue
                with open(os.path.join(img_dir, stem + ".jpg"), "wb") as fh:
                    fh.write(b)
                boxes = norm_boxes(row.get("annotations_json"), row.get("width"), row.get("height"))
                if boxes:
                    with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as fh:
                        for c, cx, cy, w, h in boxes:
                            fh.write("%d %.6f %.6f %.6f %.6f\n" % (c, cx, cy, w, h))
                            stats["boxes"][NAMES[c]] += 1
                            boxes_here[NAMES[c]] += 1
                else:
                    open(os.path.join(lbl_dir, stem + ".txt"), "w").close()
                    n_neg += 1
                n_img += 1
                stats["sources"][src] += 1
                stats["licenses"][row.get("license") or "-"] += 1
    stats["per_split"][split_name] = (n_img, n_neg, dict(boxes_here))
    print("  [%s] images=%d  background=%d  boxes=%s"
          % (split_name, n_img, n_neg,
             ", ".join("%s=%d" % (k, v) for k, v in sorted(boxes_here.items())) or "-"))


def write_data_yaml(out, splits, sources=None, limit=0, seed=0):
    path = os.path.join(out, "data.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# FireViewer Fire and Smoke Detection Corpus v1 -> YOLO export\n")
        fh.write("# generated by dev_scripts/prep_fireviewer_dataset.py\n")
        if sources:
            fh.write("# sources included: %s\n" % ", ".join(sorted(sources)))
        if limit:
            fh.write("# per-split cap %d imgs (WHOLE split_groups, seed %d)\n" % (limit, seed))
        fh.write("# class contract: fire=0 / other=1 / smoke=2 (matches models/fire/best.pt)\n")
        fh.write("path: %s\n" % out.replace("\\", "/"))
        for sp in ("train", "val", "test"):
            if sp in splits:
                fh.write("%s: %s/images\n" % (sp, sp))
        fh.write("nc: %d\n" % len(NAMES))
        fh.write("names:\n")
        for i, n in enumerate(NAMES):
            fh.write("  %d: %s\n" % (i, n))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus",
                    default="model-training/datasets--fireviewer--fire-smoke-detection-corpus-v1",
                    help="HF datasets cache root of the FireViewer corpus")
    ap.add_argument("--out", required=True, help="NEW output YOLO dataset root")
    ap.add_argument("--splits", default="train,validation,test",
                    help="comma list of corpus splits to export (train,validation,test)")
    ap.add_argument("--limit", type=int, default=0,
                    help="soft max images PER SPLIT (0=all); samples WHOLE split_groups "
                         "so a split of a few huge sequential clips can overshoot")
    ap.add_argument("--seed", type=int, default=0, help="seed for --limit group sampling")
    ap.add_argument("--sources", default=None,
                    help="comma list of source_name to INCLUDE (default: all)")
    ap.add_argument("--exclude-sources", default=None,
                    help="comma list of source_name to EXCLUDE (e.g. alarmod = GPL-3.0)")
    ap.add_argument("--zip", nargs="?", const="", default=None,
                    help="also write a Colab zip (default: <out>_colab.zip)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing into a non-empty --out")
    args = ap.parse_args()

    pq = require_pyarrow()

    corpus = os.path.abspath(args.corpus)
    if not os.path.isdir(corpus):
        die("corpus not found: %s" % corpus)
    snap = resolve_snapshot(corpus)
    out = os.path.abspath(args.out)
    if os.path.exists(out) and any(os.scandir(out)) and not args.overwrite:
        die("--out already exists and is not empty: %s (use --overwrite)" % out)
    os.makedirs(out, exist_ok=True)

    include = set(x.strip() for x in args.sources.split(",")) if args.sources else None
    exclude = set(x.strip() for x in args.exclude_sources.split(",")) if args.exclude_sources else set()
    want = [s.strip() for s in args.splits.split(",") if s.strip()]
    bad = [s for s in want if s not in SPLIT_MAP]
    if bad:
        die("bad --splits %s (want subset of %s)" % (bad, sorted(SPLIT_MAP)))

    print("corpus  :", corpus)
    print("snapshot:", snap)
    print("out     :", out)
    if args.limit:
        print("limit   : %d images/split (whole split_groups, seed %d)" % (args.limit, args.seed))
    if exclude or include:
        print("sources : include=%s exclude=%s" % (sorted(include) if include else "all",
                                                    sorted(exclude) if exclude else "-"))

    stats = {"boxes": Counter(), "sources": Counter(), "licenses": Counter(),
             "per_split": {}, "skipped_no_image": 0}
    done = []
    for corpus_split in want:
        spl = SPLIT_MAP[corpus_split]
        files = split_parquet(snap, corpus_split)
        if not files:
            print("  [%s] no parquet shards -> skipped" % spl)
            continue
        print("  [%s] %d parquet shard(s) <- data/%s" % (spl, len(files), corpus_split))
        write_split(pq, files, spl, os.path.join(out, spl), args.limit, args.seed,
                    include, exclude, stats)
        done.append(spl)

    yaml_path = write_data_yaml(out, set(done), include, args.limit, args.seed)

    rep = []
    rep.append("FireViewer corpus v1 -> YOLO export")
    rep.append("=" * 64)
    rep.append("corpus   : %s" % corpus)
    rep.append("snapshot : %s" % snap)
    rep.append("splits   : %s" % ", ".join(done))
    rep.append("limit    : %s" % (args.limit or "none (all)"))
    rep.append("")
    total = 0
    for spl in ("train", "val", "test"):
        if spl in stats["per_split"]:
            n, neg, box_here = stats["per_split"][spl]
            total += n
            rep.append("%-5s images=%-7d background=%-7d boxes=%s"
                       % (spl, n, neg,
                          ", ".join("%s=%d" % (k, v) for k, v in sorted(box_here.items())) or "-"))
    rep.append("TOTAL images=%d" % total)
    rep.append("boxes     : " + (", ".join("%s=%d" % (k, v)
                                            for k, v in sorted(stats["boxes"].items())) or "-"))
    rep.append("sources   : " + ", ".join("%s=%d" % (k, v)
                                          for k, v in stats["sources"].most_common()))
    rep.append("licenses  : " + ", ".join("%s=%d" % (k, v)
                                          for k, v in stats["licenses"].most_common()))
    if stats["skipped_no_image"]:
        rep.append("skipped (no image bytes): %d" % stats["skipped_no_image"])
    rep.append("")
    rep.append("data.yaml -> %s" % yaml_path)
    rep.append("HELD OUT: the 'test' split must NOT be trained on - register it as a new "
               "eval suite (dev_scripts/compare_fire_models.py SUITES).")
    text = "\n".join(rep)
    with open(os.path.join(out, "prep_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\n" + text)

    if args.zip is not None:
        zpath = args.zip if args.zip else out + "_colab.zip"
        zpath = os.path.abspath(zpath)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for root, _dirs, files in os.walk(out):
                for fn in sorted(files):
                    full = os.path.join(root, fn)
                    z.write(full, os.path.relpath(full, out))
        print("\ncolab zip -> %s" % zpath)


if __name__ == "__main__":
    main()
