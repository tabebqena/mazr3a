#!/usr/bin/env python3
"""build_fire_aug_benchmark2000.py - materialise a 2000-image, category-stratified fire/smoke
benchmark whose pixels no model has ever seen: held-out test splits only, plus a seeded
rotation (up to +/-15 deg) and brightness (+/-10%) re-render that turns reused negatives into
unseen variants.

This is the 2000-image companion to build_fire_benchmark200.py (200-image). Same four
image-level categories, same source contract, but:

  * 500 images per category (default) = 2000 total;
  * every copied image is optionally re-rendered:
      - rotation   with prob --rot-prob, angle = uniform(-ROT_DEG, +ROT_DEG) (black fill,
        expand=True);
      - brightness with prob --bright-prob, factor = uniform(1 - BRIGHT_PCT/100,
        1 + BRIGHT_PCT/100);
    the two transforms are independent, so ~75% of images get at least one, and the exact
    angle/factor are recorded per row in manifest.csv (columns rot_deg / bright_factor);
  * negatives (especially ours/negatives/places) are included in the 'other' category
    alongside empty-label test images, spread round-robin so no single domain dominates.

Positives come ONLY from test splits, so they are held out from every model we hold:
  - D-Fire test      (v4/v5/scratch trained on D-Fire train)
  - FireViewer test  (v6/scratch trained on FireViewer train/val)
  - CCTV Emergency   (eval-only, never trained by anyone)

Output layout:

    <out>/<category>/<source>__<original-stem>.jpg
    <out>/manifest.csv

Usage:
    .venv/bin/python dev_scripts/fire/build_fire_aug_benchmark2000.py \
        [--out model-training/eval/fire-benchmark-aug2000] [--per-category 500] \
        [--seed 0] [--rot-prob 0.5] [--rot-deg 15] [--bright-prob 0.5] [--bright-pct 10] \
        [--jpeg-quality 95] [--force]
"""
import argparse
import csv
import os
import random
import shutil
import sys
import time

from PIL import Image, ImageEnhance, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CATEGORIES = ["fire_smoke", "fire_only", "smoke_only", "other"]

# Label-bearing test sources.  `map` is the source's native class-id -> role
# (only 'fire'/'smoke' matter for categorising; everything else is ignored).
LABEL_SOURCES = [
    {
        "key": "dfire_test",
        "title": "D-Fire test (held out from V4/V5/scratch D-Fire train)",
        "images": "model-training/sources/dfire/dfire/test/images",
        "labels": "model-training/sources/dfire/dfire/test/labels",
        "map": {0: "smoke", 1: "fire"},
        "domain": "varied internet / CCTV fire+smoke",
    },
    {
        "key": "fireviewer_test",
        "title": "FireViewer corpus test (held out from V6/scratch train)",
        "images": ("model-training/sources/fireviewer/"
                   "fireviewer_v1_yolo/test/images"),
        "labels": ("model-training/sources/fireviewer/"
                   "fireviewer_v1_yolo/test/labels"),
        "map": {0: "fire", 1: "other", 2: "smoke"},
        "domain": "aerial / forest / tower wildfire",
    },
    {
        "key": "cctv_emergency",
        "title": "CCTV Smoke & Fire Emergency, Simuletic (eval-only)",
        "images": "model-training/sources/cctv_emergency/images",
        "labels": "model-training/sources/cctv_emergency/labels",
        "map": {0: "fire", 1: "smoke"},
        "domain": "synthetic high-angle CCTV",
    },
]

# No-label sources -> always category 'other' (no fire/smoke boxes by construction).
# places/dogs/climate/default-other may have entered some training runs by identity, so the
# augmentation step re-renders them into unseen pixels before scoring.
NEG_SOURCES = [
    {"key": "eval_negatives", "title": "Curated no-fire backgrounds",
     "images": "model-training/eval/negatives/images", "domain": "web/stock backgrounds"},
    {"key": "fn_places", "title": "Hard negatives: places",
     "images": "model-training/ours/negatives/places", "domain": "deployment-hard no-fire"},
    {"key": "fn_dogs", "title": "Hard negatives: dogs",
     "images": "model-training/ours/negatives/dogs", "domain": "deployment-hard no-fire"},
    {"key": "fn_climate", "title": "Hard negatives: climate",
     "images": "model-training/ours/negatives/climate", "domain": "deployment-hard no-fire"},
    {"key": "fn_default_other", "title": "Hard negatives: default-other",
     "images": "model-training/ours/negatives/default-other", "domain": "deployment-hard no-fire"},
]

_T0 = time.time()


def step(msg):
    print("[step %6.1fs] %s" % (time.time() - _T0, msg), flush=True)


def abs_repo(rel):
    return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)


def image_files(d):
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in IMG_EXTS:
            out.append(p)
    return out


def scan_label(label_path, cmap):
    """Return (n_fire, n_smoke) box counts for one YOLO label file."""
    n_fire = n_smoke = 0
    if not os.path.isfile(label_path):
        return n_fire, n_smoke
    with open(label_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                continue
            role = cmap.get(cid)
            if role == "fire":
                n_fire += 1
            elif role == "smoke":
                n_smoke += 1
    return n_fire, n_smoke


def categorize(n_fire, n_smoke):
    if n_fire and n_smoke:
        return "fire_smoke"
    if n_fire:
        return "fire_only"
    if n_smoke:
        return "smoke_only"
    return "other"


def collect_candidates():
    """{category: {source_key: [candidate-dict, ...]}} plus a source-title map."""
    by_cat = {c: {} for c in CATEGORIES}
    titles = {}

    for src in LABEL_SOURCES:
        key = src["key"]
        titles[key] = src["title"]
        idir, ldir = abs_repo(src["images"]), abs_repo(src["labels"])
        imgs = image_files(idir)
        step("  %-16s %d image(s) in %s" % (key, len(imgs), src["images"]))
        for p in imgs:
            stem = os.path.splitext(os.path.basename(p))[0]
            nf, ns = scan_label(os.path.join(ldir, stem + ".txt"), src["map"])
            cat = categorize(nf, ns)
            by_cat[cat].setdefault(key, []).append({
                "category": cat, "source": key, "gt_fire": int(nf > 0),
                "gt_smoke": int(ns > 0), "n_fire": nf, "n_smoke": ns,
                "orig_path": p, "domain": src["domain"], "source_title": src["title"],
            })

    for src in NEG_SOURCES:
        key = src["key"]
        titles[key] = src["title"]
        imgs = image_files(abs_repo(src["images"]))
        step("  %-16s %d image(s) in %s" % (key, len(imgs), src["images"]))
        for p in imgs:
            by_cat["other"].setdefault(key, []).append({
                "category": "other", "source": key, "gt_fire": 0, "gt_smoke": 0,
                "n_fire": 0, "n_smoke": 0, "orig_path": p,
                "domain": src["domain"], "source_title": src["title"],
            })
    return by_cat, titles


def round_robin_pick(cands_by_source, target, rng):
    """Evenly spread `target` picks across sources (sorted key order)."""
    pools = {}
    for k, lst in cands_by_source.items():
        lst = list(lst)
        rng.shuffle(lst)
        pools[k] = lst
    order = sorted(pools)
    idx = {k: 0 for k in order}
    picked = []
    while len(picked) < target:
        progressed = False
        for k in order:
            if len(picked) >= target:
                break
            if idx[k] < len(pools[k]):
                picked.append(pools[k][idx[k]])
                idx[k] += 1
                progressed = True
        if not progressed:
            break
    return picked


def render_image(src, dst, rng, args):
    """Copy src -> dst, applying a seeded random rotation/brightness re-render.

    Returns (rot_deg, bright_factor) actually applied.
    """
    rot_deg = 0.0
    bright_factor = 1.0

    im = Image.open(src)
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")

    if rng.random() < args.rot_prob:
        rot_deg = rng.uniform(-args.rot_deg, args.rot_deg)
    if rot_deg != 0.0:
        im = im.rotate(rot_deg, resample=Image.Resampling.BICUBIC, expand=True,
                       fillcolor=(0, 0, 0))

    if rng.random() < args.bright_prob:
        # uniform in [-bright_pct/100, +bright_pct/100] -> factor in [0.9, 1.1] at 10%.
        delta = rng.uniform(-args.bright_pct / 100.0, args.bright_pct / 100.0)
        bright_factor = 1.0 + delta
    if bright_factor != 1.0:
        im = ImageEnhance.Brightness(im).enhance(bright_factor)

    im.save(dst, "JPEG", quality=args.jpeg_quality)
    return round(rot_deg, 3), round(bright_factor, 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="model-training/eval/fire-benchmark-aug2000")
    ap.add_argument("--per-category", type=int, default=500, dest="per_category")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rot-prob", type=float, default=0.5,
                    help="probability an image is rotated [0.5]")
    ap.add_argument("--rot-deg", type=float, default=15.0,
                    help="max absolute rotation angle in degrees [15.0]")
    ap.add_argument("--bright-prob", type=float, default=0.5,
                    help="probability an image is brightness-adjusted [0.5]")
    ap.add_argument("--bright-pct", type=float, default=10.0,
                    help="max absolute brightness delta in percent [10.0]")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--force", action="store_true", help="wipe --out first")
    args = ap.parse_args()

    out = abs_repo(args.out)
    if os.path.isdir(out) and args.force:
        step("--force: removing %s" % out)
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    step("scanning held-out test + negative sources ...")
    by_cat, titles = collect_candidates()

    step("availability per category:")
    for c in CATEGORIES:
        tot = sum(len(v) for v in by_cat[c].values())
        per_src = ", ".join("%s=%d" % (k, len(v)) for k, v in sorted(by_cat[c].items()))
        step("  %-11s total=%-6d %s" % (c, tot, per_src))

    rng = random.Random(args.seed)
    manifest = []
    for c in CATEGORIES:
        picked = round_robin_pick(by_cat[c], args.per_category, rng)
        if len(picked) < args.per_category:
            step("  ! %s: only %d available (< %d requested)"
                 % (c, len(picked), args.per_category))
        cat_dir = os.path.join(out, c)
        os.makedirs(cat_dir, exist_ok=True)
        for cand in picked:
            stem = os.path.splitext(os.path.basename(cand["orig_path"]))[0]
            dst = os.path.join(cat_dir, "%s__%s.jpg" % (cand["source"], stem))
            rot_deg, bright_factor = render_image(cand["orig_path"], dst, rng, args)
            manifest.append({
                "category": c,
                "source": cand["source"],
                "source_title": cand["source_title"],
                "domain": cand["domain"],
                "gt_fire": cand["gt_fire"],
                "gt_smoke": cand["gt_smoke"],
                "n_fire": cand["n_fire"],
                "n_smoke": cand["n_smoke"],
                "rot_deg": rot_deg,
                "bright_factor": bright_factor,
                "orig_path": os.path.relpath(cand["orig_path"], ROOT),
                "bench_path": os.path.relpath(dst, ROOT),
            })

    man_path = os.path.join(out, "manifest.csv")
    with open(man_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "category", "source", "source_title", "domain", "gt_fire", "gt_smoke",
            "n_fire", "n_smoke", "rot_deg", "bright_factor", "orig_path", "bench_path"])
        w.writeheader()
        w.writerows(manifest)

    step("wrote %d rows -> %s" % (len(manifest), man_path))
    step("per-category / per-source:")
    for c in CATEGORIES:
        rows = [m for m in manifest if m["category"] == c]
        per_src = {}
        for m in rows:
            per_src[m["source"]] = per_src.get(m["source"], 0) + 1
        n_rot = sum(1 for m in rows if m["rot_deg"] != 0.0)
        n_bri = sum(1 for m in rows if m["bright_factor"] != 1.0)
        step("  %-11s %-3d  (rotated=%d, brightened=%d)  %s"
             % (c, len(rows), n_rot, n_bri,
                ", ".join("%s=%d" % kv for kv in sorted(per_src.items()))))
    step("total images: %d" % len(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
