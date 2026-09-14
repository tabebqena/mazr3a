#!/usr/bin/env python3
"""prep_fire_scratch_dataset.py - download + merge external fire/smoke sources into ONE raw
YOLO tree (the dedup passes run afterwards in dev_scripts/dedup_fire_scratch.py).

Runs on the Colab machine. Sources are declared in a JSON config (``--config``); each source is
downloaded (never re-uploaded), normalised to a per-source YOLO tree, its labels remapped onto
the TARGET class order (``--classes`` + the source's ``class_map``) and merged into
``<out>/train|val|test``.

SOURCE TYPES
------------
  huggingface_fireviewer  FireViewer Fire & Smoke Detection Corpus v1 (parquet). Downloaded with
                          ``huggingface_hub.snapshot_download(..., allow_patterns=['data/**'])``
                          and converted by ``dev_scripts/prep_fireviewer_dataset.py`` (bundled).
                          Fields: ``repo``, ``splits`` ("train,validation,test"), ``exclude_sources``,
                          ``limit``, ``sample_mode``.
  zip_url                 A Roboflow-style YOLO export zip (train/valid/test) or a flat
                          images/labels zip. Fields: ``url``, ``download`` ("wget"|"gdown"),
                          ``layout`` ("roboflow"|"flat"), ``class_map``, ``role``.
  yolo_dir                An existing local YOLO tree (uploaded / Drive) - used for the held-out
                          CCTV domain test (role=test) and for domain negatives (role=negatives).

ROLES
-----
  train       source splits merge train->train, valid->val, test->test (flat -> train)
  test        EVERY image goes to the held-out test split (never trained)   <- CCTV images
  negatives   EVERY image goes to train with an EMPTY label (pure background)

CLASS REMAP
-----------
  ``class_map`` maps ``source_class_name -> target_class_name | null``. ``null`` drops that box;
  an image whose boxes are all dropped becomes a background negative (empty .txt). Source names
  not in the map are dropped (with a warning) unless they already equal a target class.

OUTPUTS
-------
  <out>/data.yaml          merged, target class order (train: train/images, val: val/images)
  <out>/train|val|test/    merged images + labels (stems prefixed "<source_id>__")
  <out>/manifest.csv       split,stem,source_id,source_name,split_group,n_boxes,class_ids
  <out>/sources.json       resolved config echo (provenance)
  <out>/prep_report.txt    per-source + merged counts

IDEMPOTENT: if <out>/data.yaml exists the run is a no-op unless ``--force``.

USAGE
-----
    python prep_fire_scratch_dataset.py --config /content/sources.json \
        --out /content/raw_yolo --cache /content/src_cache \
        --scripts /content/upload/scripts
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_MAP = {"train": "train", "valid": "val", "val": "val", "validation": "val", "test": "test"}
MANIFEST_COLS = ["split", "stem", "source_id", "source_name", "split_group",
                 "n_boxes", "class_ids"]


def die(msg):
    sys.exit("prep_fire_scratch_dataset: " + msg)


def is_image(name):
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def source_key(name):
    """Clip-prefix of a Roboflow-style frame name (strip '_f<digits>_jpg.rf.<hash>')."""
    m = re.match(r"^(.*?)(?:_f\d+)?_jpg\.rf\..*$", os.path.splitext(name)[0])
    if m and m.group(1):
        return m.group(1)
    return os.path.splitext(name)[0]


def read_yaml_names(data_yaml):
    """Return the class-name list from a YOLO data.yaml (index or list form), else [].

    Handles both
        names: ['fire', 'smoke']
    and
        names:
          0: fire
          1: smoke
    """
    if not os.path.isfile(data_yaml):
        return []
    try:
        text = open(data_yaml, encoding="utf-8").read()
    except OSError:
        return []
    # index map form: "names:" on its own line followed by "0: fire", "1: smoke", ...
    m = re.search(r"(?:^|\n)\s*names\s*:\s*\n((?:[ \t]*\d+\s*:.*(?:\n|$))+)", text)
    if m:
        idx = {}
        for line in m.group(1).splitlines():
            mm = re.match(r"\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if mm:
                idx[int(mm.group(1))] = mm.group(2).strip().strip("'\"")
        if idx:
            return [idx[i] for i in sorted(idx)]
    # block-sequence form (Roboflow style):
    #   names:
    #   - Fire
    #   - default
    #   - smoke
    m = re.search(r"(?:^|\n)\s*names\s*:\s*\n((?:[ \t]*-[ \t]*[^\n]+(?:\n|$))+)", text)
    if m:
        vals = [ln.strip().lstrip("-").strip().strip("'\"") for ln in m.group(1).splitlines()]
        vals = [v for v in vals if v]
        if vals:
            return vals
    # inline list form: "names: ['fire', 'smoke']" (YOLO files use single quotes, not valid JSON)
    m = re.search(r"(?:^|\n)\s*names\s*:\s*(\[[^\]]*\])", text)
    if m:
        inner = m.group(1)[1:-1]           # strip the surrounding [ ]
        vals = [v.strip().strip("'\"") for v in inner.split(",")]
        vals = [v for v in vals if v]
        if vals:
            return vals
    return []


def read_manifest_map(manifest_csv):
    """{stem: (split_group, source_name)} from a prep_fireviewer_dataset.py manifest."""
    out = {}
    if not os.path.isfile(manifest_csv):
        return out
    with open(manifest_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("stem"):
                out[r["stem"]] = (r.get("split_group") or "", r.get("source_name") or "")
    return out


def run(cmd, cwd=None):
    print("  + " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


# --------------------------------------------------------------------------- #
# downloads
# --------------------------------------------------------------------------- #
def download_hf(repo, cache, splits):
    """snapshot_download the parquet shards we need (idempotent)."""
    if glob.glob(os.path.join(cache, "data", "**", "*.parquet"), recursive=True):
        print("  HF cache already populated ->", cache)
        return
    from huggingface_hub import snapshot_download
    patterns = ["data/%s/*" % s for s in splits] if splits else ["data/**"]
    snapshot_download(repo, repo_type="dataset", local_dir=cache,
                      allow_patterns=patterns)
    if not glob.glob(os.path.join(cache, "data", "**", "*.parquet"), recursive=True):
        die("HF download produced no parquet under %s (repo=%s)" % (cache, repo))


def download_zip(url, download, cache, out_zip):
    os.makedirs(cache, exist_ok=True)
    if os.path.isfile(out_zip) and os.path.getsize(out_zip) > 0:
        print("  zip already present ->", out_zip)
        return
    if download == "gdown":
        import gdown  # noqa: F401
        cmd = [sys.executable, "-m", "gdown", "--output", out_zip, url]
    else:
        cmd = ["wget", "-q", "-O", out_zip, url]
    run(cmd)


# --------------------------------------------------------------------------- #
# source enumeration
# --------------------------------------------------------------------------- #
def source_records(src_root):
    """Yield (split, img_path, lbl_path, stem) for a normalised YOLO tree.

    Accepts Roboflow split trees (train/valid/test with images/ + labels/) or a flat
    images/ + labels/ layout. Labels default to a MISSING path (background) when absent.
    """
    records = []
    split_dirs = [sp for sp in ("train", "valid", "val", "test")
                  if os.path.isdir(os.path.join(src_root, sp, "images"))]
    if split_dirs:
        for sp in split_dirs:
            imdir = os.path.join(src_root, sp, "images")
            lbldir = os.path.join(src_root, sp, "labels")
            for name in sorted(os.listdir(imdir)):
                if not is_image(name):
                    continue
                stem = os.path.splitext(name)[0]
                lbl = os.path.join(lbldir, stem + ".txt")
                records.append((SPLIT_MAP[sp], os.path.join(imdir, name),
                                lbl if os.path.isfile(lbl) else None, stem))
        return records
    imdir = os.path.join(src_root, "images")
    if not os.path.isdir(imdir):
        # bare directory of images (no images/ subdir)
        imdir = src_root
    lbldir = os.path.join(src_root, "labels")
    for name in sorted(os.listdir(imdir)):
        if not is_image(name):
            continue
        stem = os.path.splitext(name)[0]
        lbl = os.path.join(lbldir, stem + ".txt")
        records.append((None, os.path.join(imdir, name),
                        lbl if os.path.isfile(lbl) else None, stem))
    return records


def remap_boxes(lbl_path, src_names, class_map, index_map, classes, drop_counter, src_id):
    """Return list of (target_class, x, y, w, h) rows after remap/drop.

    ``index_map`` (optional) maps a SOURCE CLASS INDEX -> target class name and takes
    precedence over name-based ``class_map``; use it for sources whose data.yaml names are
    missing/broken (e.g. D-Fire's 0=smoke/1=fire, ready_fire_smoke's ['0','1','2']).
    """
    target_index = {name: i for i, name in enumerate(classes)}
    rows = []
    if not lbl_path or not os.path.isfile(lbl_path):
        return rows
    with open(lbl_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                si = int(float(parts[0]))
                vals = parts[1:]
            except ValueError:
                continue
            if index_map and si in index_map:
                tname = index_map[si]        # may be None to drop the box
            else:
                sname = src_names[si] if 0 <= si < len(src_names) else ""
                if sname in class_map:
                    tname = class_map[sname]
                elif sname in target_index:
                    tname = sname
                else:
                    tname = None
                    if sname:
                        drop_counter[(src_id, sname)] += 1
            if tname is None:
                continue
            if tname not in target_index:
                die("map maps %r -> %r which is not in --classes %s"
                    % (si, tname, classes))
            rows.append((target_index[tname],) + tuple(vals))
    return rows


# --------------------------------------------------------------------------- #
# per-source staging
# --------------------------------------------------------------------------- #
def stage_source(src, cache, scripts_dir):
    """Return (src_root, names, manifest_map) for a source, downloading if needed."""
    typ = src.get("type")
    src_id = src["id"]
    if typ == "huggingface_fireviewer":
        repo = src.get("repo")
        if not repo:
            die("source %r: missing 'repo'" % src_id)
        repo_dir = os.path.join(cache, "hf_" + src_id)
        splits = [s.strip() for s in src.get("splits", "train,validation,test").split(",") if s.strip()]
        download_hf(repo, repo_dir, splits)
        per_src = os.path.join(cache, "conv_" + src_id)
        if not os.path.isfile(os.path.join(per_src, "data.yaml")):
            conv = os.path.join(scripts_dir, "prep_fireviewer_dataset.py")
            if not os.path.isfile(conv):
                die("converter not found: %s (pass --scripts)" % conv)
            cmd = [sys.executable, conv, "--corpus", repo_dir, "--out", per_src,
                   "--splits", ",".join(splits), "--overwrite"]
            if src.get("exclude_sources"):
                cmd += ["--exclude-sources", ",".join(src["exclude_sources"])]
            if src.get("sources"):
                cmd += ["--sources", ",".join(src["sources"])]
            if src.get("limit"):
                cmd += ["--limit", str(src["limit"]),
                        "--sample-mode", src.get("sample_mode", "group")]
            run(cmd)
        # the parquet cache is ~25 GB and redundant once the YOLO export exists
        if src.get("free_cache", True):
            shutil.rmtree(repo_dir, ignore_errors=True)
            print("  freed parquet cache ->", repo_dir, flush=True)
        return per_src, read_yaml_names(os.path.join(per_src, "data.yaml")), \
            read_manifest_map(os.path.join(per_src, "manifest.csv"))

    if typ == "zip_url":
        url = src.get("url")
        if not url:
            die("source %r: missing 'url'" % src_id)
        stage = os.path.join(cache, "zip_" + src_id)
        os.makedirs(stage, exist_ok=True)
        out_zip = os.path.join(stage, "src.zip")
        download_zip(url, src.get("download", "wget"), cache, out_zip)
        extract_dir = os.path.join(stage, "extracted")
        if not os.path.isdir(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(out_zip) as zf:
                zf.extractall(extract_dir)
        root = extract_dir
        # Roboflow zips nest under a single top dir; if so, step into it.
        if src.get("layout", "roboflow") == "roboflow":
            entries = [os.path.join(root, d) for d in sorted(os.listdir(root))
                       if os.path.isdir(os.path.join(root, d))]
            if len(entries) == 1 and os.path.isdir(os.path.join(entries[0], "train", "images")):
                root = entries[0]
        return root, read_yaml_names(os.path.join(root, "data.yaml")), {}

    if typ == "github_repo":
        # sparse clone pulls ONLY the dataset subpath (e.g. Abonia's datasets/fire-8),
        # not the repo's 390 MB of demos/weights/runs.
        repo = src.get("repo")                 # "owner/name"
        branch = src.get("branch", "main")
        subpath = src.get("subpath")           # e.g. "datasets/fire-8"
        if not repo or not subpath:
            die("source %r: github_repo needs 'repo' and 'subpath'" % src_id)
        url = "https://github.com/%s.git" % repo.strip("/")
        clone_dir = os.path.join(cache, "git_" + src_id)
        if not os.path.isdir(os.path.join(clone_dir, ".git")):
            run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                 "--branch", branch, url, clone_dir])
            run(["git", "-C", clone_dir, "sparse-checkout", "set", subpath])
        root = os.path.join(clone_dir, subpath.strip("/"))
        if not os.path.isdir(root):
            die("source %r: subpath %r not found after sparse clone" % (src_id, subpath))
        return root, read_yaml_names(os.path.join(root, "data.yaml")), {}

    if typ == "yolo_dir":
        path = src.get("path")
        if not path or not os.path.isdir(path):
            die("source %r: yolo_dir path not found: %s" % (src_id, path))
        return path, read_yaml_names(os.path.join(path, "data.yaml")), \
            read_manifest_map(os.path.join(path, "manifest.csv"))

    die("source %r: unknown type %r" % (src_id, typ))


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def merge_source(src, src_root, src_names, src_manifest, out, classes, cfg_map,
                 stats, drop_counter):
    """Copy/remap one source into the merged pool. Returns manifest rows."""
    src_id = src["id"]
    role = src.get("role", "train")
    class_map = dict(cfg_map)
    class_map.update(src.get("class_map") or {})
    index_map = {int(k): v for k, v in (src.get("index_map") or {}).items()}
    exclude = set(src.get("exclude_sources") or [])
    include = set(src.get("sources") or [])
    mf_rows = []
    out_img = {s: os.path.join(out, s, "images") for s in ("train", "val", "test")}
    out_lbl = {s: os.path.join(out, s, "labels") for s in ("train", "val", "test")}
    for d in list(out_img.values()) + list(out_lbl.values()):
        os.makedirs(d, exist_ok=True)

    n = n_bg = 0
    for sp, img, lbl, stem in source_records(src_root):
        # filter by the source's own manifest source_name (e.g. drop alarmod from fireviewer)
        rec_name = src_manifest.get(stem, ("", ""))[1]
        if exclude and rec_name in exclude:
            continue
        if include and rec_name and rec_name not in include:
            continue
        if role == "test":
            tgt = "test"
        elif role == "negatives":
            tgt = "train"
        else:
            tgt = sp if sp else "train"

        if role == "negatives":
            rows = []
        else:
            rows = remap_boxes(lbl, src_names, class_map, index_map, classes,
                               drop_counter, src_id)

        new_stem = "%s__%s" % (src_id, stem)
        dst_img = os.path.join(out_img[tgt], new_stem + os.path.splitext(img)[1].lower())
        if not os.path.exists(dst_img):
            shutil.copy2(img, dst_img)
        dst_lbl = os.path.join(out_lbl[tgt], new_stem + ".txt")
        with open(dst_lbl, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write("%d %s %s %s %s\n" % r)
        if not rows:
            n_bg += 1
        group = src_manifest.get(stem, ("", ""))[0]
        src_name = src_manifest.get(stem, ("", src.get("source_name", src_id)))[1]
        if not group:
            group = source_key(stem)
        mf_rows.append([tgt, new_stem, src_id, src_name, group,
                        len(rows), ",".join(str(r[0]) for r in rows)])
        n += 1

    stats["per_source"][src_id] = {"images": n, "background": n_bg}

    # downloaded sources are disposable once merged; keep only the bundled yolo_dir trees
    if src.get("type") in ("huggingface_fireviewer", "zip_url", "github_repo"):
        shutil.rmtree(src_root, ignore_errors=True)
        print("  freed source tree ->", src_root, flush=True)
    return mf_rows


def write_data_yaml(out, classes):
    path = os.path.join(out, "data.yaml")
    val_has = os.path.isdir(os.path.join(out, "val", "images")) and \
        any(is_image(f) for f in os.listdir(os.path.join(out, "val", "images")))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# merged raw pool - dedup with dev_scripts/dedup_fire_scratch.py\n")
        fh.write("path: %s\n" % out.replace("\\", "/"))
        fh.write("train: train/images\n")
        fh.write("val: %s\n" % ("val/images" if val_has else "test/images"))
        fh.write("nc: %d\n" % len(classes))
        fh.write("names:\n")
        for i, n in enumerate(classes):
            fh.write("  %d: %s\n" % (i, n))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="sources JSON (see docstring)")
    ap.add_argument("--out", required=True, help="NEW merged YOLO dataset root")
    ap.add_argument("--cache", default="/content/src_cache",
                    help="download/conversion cache dir")
    ap.add_argument("--scripts", default="/content/upload/scripts",
                    help="dir holding prep_fireviewer_dataset.py")
    ap.add_argument("--force", action="store_true", help="re-merge even if --out exists")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    classes = cfg.get("classes") or ["fire"]
    cfg_map = cfg.get("class_map") or {}
    sources = cfg.get("sources") or []
    if not sources:
        die("config has no 'sources'")

    out = os.path.abspath(args.out)
    if os.path.isfile(os.path.join(out, "data.yaml")) and not args.force:
        print("merged pool already exists ->", out, "(use --force to rebuild)")
        print(open(os.path.join(out, "data.yaml"), encoding="utf-8").read())
        return
    os.makedirs(args.cache, exist_ok=True)

    stats = {"per_source": {}, "boxes": Counter()}
    drop_counter = Counter()
    all_rows = []

    for src in sources:
        src_id = src["id"]
        print("\n=== source %s (type=%s role=%s) ==="
              % (src_id, src.get("type"), src.get("role", "train")), flush=True)
        src_root, names, manifest = stage_source(src, args.cache, args.scripts)
        print("  names=%s  images=%d" % (names, len(source_records(src_root))), flush=True)
        rows = merge_source(src, src_root, names, manifest, out, classes, cfg_map,
                            stats, drop_counter)
        all_rows.extend(rows)

    for split in ("train", "val", "test"):
        idir = os.path.join(out, split, "images")
        n = sum(1 for f in os.listdir(idir) if is_image(f)) if os.path.isdir(idir) else 0
        stats["per_split_" + split] = n
        # class-id histogram (from labels)
        ldir = os.path.join(out, split, "labels")
        if os.path.isdir(ldir):
            for f in os.listdir(ldir):
                for line in open(os.path.join(ldir, f), encoding="utf-8"):
                    parts = line.split()
                    if parts:
                        stats["boxes"][int(float(parts[0]))] += 1

    yaml_path = write_data_yaml(out, classes)
    with open(os.path.join(out, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(MANIFEST_COLS)
        w.writerows(all_rows)
    with open(os.path.join(out, "sources.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)

    rep = []
    rep.append("Scratch fire training - raw merged pool")
    rep.append("=" * 64)
    rep.append("classes  : %s" % classes)
    rep.append("out      : %s" % out)
    rep.append("")
    rep.append("per split: train=%d val=%d test=%d"
               % (stats["per_split_train"], stats["per_split_val"], stats["per_split_test"]))
    rep.append("boxes    : " + (", ".join("%s=%d" % (classes[k], v)
                                           for k, v in sorted(stats["boxes"].items())) or "-"))
    rep.append("per source:")
    for sid, s in stats["per_source"].items():
        rep.append("  - %-14s images=%d background=%d" % (sid, s["images"], s["background"]))
    if drop_counter:
        rep.append("")
        rep.append("dropped boxes (class_map/unknown):")
        for (sid, nm), c in drop_counter.most_common():
            rep.append("  - %s:%-12s %d" % (sid, nm, c))
    rep.append("")
    rep.append("data.yaml  -> %s" % yaml_path)
    rep.append("manifest   -> %s/manifest.csv" % out)
    rep.append("NEXT: run dedup_fire_scratch.py --pool %s --out <clean>" % out)
    text = "\n".join(rep)
    with open(os.path.join(out, "prep_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
