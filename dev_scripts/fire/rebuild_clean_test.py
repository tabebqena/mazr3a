#!/usr/bin/env python3
"""rebuild_clean_test.py - rebuild the SAME clean_yolo pool locally that the Colab notebook
builds, so the held-out test split can be reproduced WITHOUT downloading it from the runtime.

It mirrors the notebook's Cell 2 SOURCES + Cell 5 (prep) + Cell 6 (dedup), but resolves every
source to a LOCAL path:
  * fireviewer  -> model-training/sources/fireviewer/fireviewer_v1_yolo  (exclude alarmod)
  * abonia      -> model-training/sources/abonia/abonia_repo/datasets/fire-8
  * cctv_emergency / negatives / domain_test  -> unpacked from the SAME bundle the notebook
    uploads (model-training/runs/fire_scratch_colab/colab_upload.zip, built by
    pack_fire_scratch_colab.sh), so the flattening is byte-for-byte what Colab uses.

The dedup parameters default to the notebook values (hamming 8, train-scope group, max-bg-share
0.60). For the FIRST run the previous-run fingerprint index is empty, so the result is fully
deterministic. For a LATER run, pass --prev-train-index / --prev-test-index pointing at the
copies of the Drive fingerprint dirs.

Usage:
    .venv/bin/python dev_scripts/fire/rebuild_clean_test.py \
        --out model-training/runs/clean_yolo_local
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MT = os.path.join(ROOT, "model-training")
BUNDLE = os.path.join(MT, "runs", "fire_scratch_colab", "colab_upload.zip")
CACHE = os.path.join(MT, "runs", "scratch_cache")

CLASSES = ["fire"]
CLASS_MAP = {
    "fire": "fire", "Fire": "fire", "flame_visible": "fire",
    "smoke": None, "smoke_visible": None, "other": None, "default": None,
}


def run(cmd):
    print("  + " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def unpack_bundle(cache):
    """Unpack colab_upload.zip (building it first if absent) into cache/bundle/."""
    if not os.path.isfile(BUNDLE):
        print("bundle missing - running pack_fire_scratch_colab.sh ...", flush=True)
        run([os.path.join(ROOT, "dev_scripts", "pack_fire_scratch_colab.sh")])
    bdir = os.path.join(cache, "bundle")
    if os.path.isdir(bdir):
        return bdir
    os.makedirs(bdir, exist_ok=True)
    with zipfile.ZipFile(BUNDLE) as z:
        z.extractall(bdir)
    return bdir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(MT, "runs", "clean_yolo_local"),
                    help="clean YOLO output root (its test/ is the benchmark)")
    ap.add_argument("--hamming", type=int, default=8)
    ap.add_argument("--train-scope", choices=("group", "global"), default="group")
    ap.add_argument("--max-bg-share", type=float, default=0.60)
    ap.add_argument("--prev-train-index", default=None,
                    help="dir of *.jsonl fingerprints of previously-trained images (empty for run 1)")
    ap.add_argument("--prev-test-index", default=None,
                    help="dir of *.jsonl fingerprints of previously-held-out test/val images")
    ap.add_argument("--run-name", default="local-rebuild")
    ap.add_argument("--keep-raw", action="store_true", help="keep the raw pool after dedup")
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    bundle = unpack_bundle(CACHE)

    sources = [
        {"id": "fireviewer", "type": "yolo_dir",
         "path": os.path.join(MT, "sources", "fireviewer", "fireviewer_v1_yolo"),
         "role": "train", "exclude_sources": ["alarmod"],
         "class_map": {"fire": "fire", "smoke": None, "other": None}},
        {"id": "abonia", "type": "yolo_dir",
         "path": os.path.join(MT, "sources", "abonia", "abonia_repo", "datasets", "fire-8"),
         "role": "train",
         "class_map": {"Fire": "fire", "default": None, "smoke": None}},
        {"id": "cctv_emergency", "type": "yolo_dir",
         "path": os.path.join(bundle, "cctv_emergency"), "role": "train",
         "class_map": {"fire": "fire", "smoke": None}},
        {"id": "negatives", "type": "yolo_dir",
         "path": os.path.join(bundle, "negatives"), "role": "negatives"},
        {"id": "cctv_test", "type": "yolo_dir",
         "path": os.path.join(bundle, "domain_test"), "role": "test"},
    ]
    cfg = {"classes": CLASSES, "class_map": CLASS_MAP, "sources": sources}
    cfg_path = os.path.join(CACHE, "sources.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    raw = os.path.join(CACHE, "raw")
    out = os.path.abspath(args.out)
    prep = os.path.join(ROOT, "dev_scripts", "prep_fire_scratch_dataset.py")
    dedup = os.path.join(ROOT, "dev_scripts", "dedup_fire_scratch.py")

    print("\n=== 1/2 prep (download/merge local sources) ===")
    if os.path.isfile(os.path.join(out, "data.yaml")):
        print("clean pool already exists ->", out, "(delete it to rebuild)")
    else:
        if os.path.isdir(raw):
            shutil.rmtree(raw)
        run([sys.executable, prep, "--config", cfg_path, "--out", raw,
             "--cache", os.path.join(CACHE, "src"), "--scripts",
             os.path.join(ROOT, "dev_scripts"), "--force"])
        print(open(os.path.join(raw, "prep_report.txt"), encoding="utf-8").read())

        print("\n=== 2/2 dedup (self + cross + balance) ===")
        cmd = [sys.executable, dedup, "--pool", raw, "--out", out,
               "--hamming", str(args.hamming), "--train-scope", args.train_scope,
               "--max-bg-share", str(args.max_bg_share), "--run-name", args.run_name,
               "--report", out + "_report", "--skip-broken",
               "--broken-out", os.path.join(CACHE, "broken")]
        if args.prev_train_index:
            cmd += ["--prev-train-index", args.prev_train_index]
        if args.prev_test_index:
            cmd += ["--prev-test-index", args.prev_test_index]
        run(cmd)
        print(open(out + "_report/summary.txt", encoding="utf-8").read())
        if not args.keep_raw:
            shutil.rmtree(raw, ignore_errors=True)

    print("\nDONE. Held-out test split (identical to Colab's clean_yolo/test):")
    print("  %s" % os.path.join(out, "test"))
    print("Score a model against it locally with:")
    print("  .venv/bin/python dev_scripts/fire/test_fire_model.py <ckpt> "
          "%s --conf 0.5" % os.path.join(out, "data.yaml"))


if __name__ == "__main__":
    main()
