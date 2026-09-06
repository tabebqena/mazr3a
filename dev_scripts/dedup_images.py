#!/usr/bin/env python3
"""dedup_images.py - fingerprint + dedup a candidate image set against reference set(s).

Phase-1 tooling for the clean evaluation / retraining workflow
(plans/fire-model-clean-eval-workflow.md).

WHY
---
We must remove the "unknown overlap with the model's own training set" problem:
every external eval/train set (Abonia, D-Fire, ...) is deduplicated against the
*real* 8,939-image training pool before it is tested or retrained on. Internet
fire images are re-published re-encoded, so an exact hash alone misses most
duplicates -> each image is fingerprinted with BOTH:

  * md5   - byte-identical duplicate test (exact re-publish / copy).
  * dHash - a 64-bit perceptual hash; two images are treated as near-duplicates
            when their Hamming distance <= --hamming (default 10). dHash is
            scale / brightness-robust and needs no extra dependency
            (imagehash is NOT installed; this script only uses Pillow + an
            optional numpy speed-up with a pure-Python fallback).

SOURCES
-------
A "set" (reference OR candidate) is any of the YOLOv8/Roboflow-style layouts:

  <root>/data.yaml            # optional (read for class names when copying)
  <root>/{train,valid,test}/{images,labels}     # per-split Roboflow export
  <root>/{images,labels}                          # flat images+labels pair
  <root>/*.jpg  ...                              # a bare images dir
  <root>/some.zip                                # a Roboflow-export .zip
  <path>.json                                   # a cache/index written by --cache

DECISION (per candidate image, in scan order)
---------------------------------------------
  removed/ref-md5     md5 present in a reference set (byte identical)
  removed/ref-phash   Hamming(own dHash, ref dHash) <= --hamming
  removed/int-md5     --internal: md5 duplicates an earlier KEPT candidate image
  removed/int-phash   --internal: dHash near-duplicates an earlier KEPT image
  kept/unique         otherwise

Reference matches take precedence over internal ones; an image removed because
it collides with the training pool never counts as the "keeper" of a cluster.

OUTPUTS
-------
Reports (always written to --report-dir, default fire-model-training/dedup/
overlap_reports):
  <tag>_per_image.csv   one row per candidate (status/reason/match/hamming)
  <tag>_kept.txt        kept image paths (relative to the candidate root)
  <tag>_removed.txt     removed image paths
  <tag>_summary.txt     counts + reasons + hamming histogram + match samples

Clean copy (only when --out is given): kept images + their label .txt files are
copied verbatim (class order per set is preserved - labels are never remapped)
into <out>/, either preserving the split structure (Roboflow-style, default) or
flattened into <out>/{images,labels} with --flat. A data.yaml is written so the
result can feed dev_scripts/test_fire_model.py / analyze_fire_dataset.py.

USAGE
-----
  # 1) index the real training pool once (cache it so later runs are fast):
  python dev_scripts/dedup_images.py --candidate X \
      --ref "fire-model-training/1_SalahALHaismawi/dataset/Fire Detection.v1i.yolov8.zip" \
      --ref-label 8939 --cache fire-model-training/dedup/overlap_reports/_index_8939.json

  # 2) internal dedup of a candidate (video-frame near-dups), report only:
  python dev_scripts/dedup_images.py --candidate .../fire-8 --tag abonia --internal

  # 3) dedup against the cached training pool and write the clean set:
  python dev_scripts/dedup_images.py --candidate .../fire-8 --tag abonia --out \
      fire-model-training/dedup/abonia_dedup --flat \
      --cache fire-model-training/dedup/overlap_reports/_index_8939.json

Note: this module doubles as the shared fingerprint/matching library used by
dev_scripts/analyze_overlap.py (imported by adding this dir to sys.path).
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from collections import Counter
from io import BytesIO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")

_LANCZOS = None


def _lanzos():
    """Pillow's LANCZOS resampling constant (API-safe across versions)."""
    global _LANCZOS
    if _LANCZOS is None:
        img = __import__("PIL.Image", fromlist=["Image"])
        res = getattr(getattr(img, "Resampling", None), "LANCZOS", None)
        _LANCZOS = res if res is not None else getattr(img, "LANCZOS", 1)
    return _LANCZOS


class _NpOps:
    """Thin numpy wrapper; ok=False -> callers use the pure-Python fallback."""

    def __init__(self):
        self.ok = False
        try:
            import numpy as np
            if hasattr(np, "bitwise_count"):  # numpy >= 2.0
                self._np = np
                self.ok = True
            else:
                self._np = None
        except Exception:
            self._np = None

    def array(self, vals):
        if not self.ok or self._np is None:  # type: ignore[truthy-function]
            raise RuntimeError("numpy unavailable")
        return self._np.array(vals, dtype=self._np.uint64)

    def best(self, arr, phash, limit):
        """Return (index, hamming) of closest hash in arr within limit, else None."""
        if arr is None or len(arr) == 0:
            return None
        if not self.ok or self._np is None:  # type: ignore[truthy-function]
            raise RuntimeError("numpy unavailable")
        d = self._np.bitwise_count(self._np.bitwise_xor(arr, self._np.uint64(int(phash))))
        j = int(self._np.argmin(d))
        hd = int(d[j])
        return (j, hd) if hd <= limit else None


# --------------------------------------------------------------------------- #
# fingerprinting
# --------------------------------------------------------------------------- #
def is_image_name(name):
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def md5_of(data):
    return hashlib.md5(data).hexdigest()


def dhash_of(data, hash_size=8):
    """64-bit (8x8) difference-hash of image bytes, or None if undecodable.

    dHash compares adjacent horizontal pixel pairs of a hash_size+1 x hash_size
    grayscale thumbnail. Robust to resizing / mild re-encode, which is what
    internet fire images get when re-published.
    """
    try:
        from PIL import Image
        im = Image.open(BytesIO(data)).convert("L")
        im = im.resize((hash_size + 1, hash_size), _lanzos())
        # mode 'L' => one grayscale byte per pixel, row-major
        vals = list(im.tobytes())
    except Exception:
        return None
    if len(vals) != (hash_size + 1) * hash_size:
        return None
    bits = 0
    for r in range(hash_size):
        row = r * (hash_size + 1)
        for c in range(hash_size):
            bits = (bits << 1) | (1 if vals[row + c] > vals[row + c + 1] else 0)
    return bits


def fingerprint(data, hash_size=8):
    """Return (md5, dhash_int); dhash may be None for undecodable images."""
    return md5_of(data), dhash_of(data, hash_size)


def hamming(a, b):
    return (int(a) ^ int(b)).bit_count()


# --------------------------------------------------------------------------- #
# hamming neighbour search structures
# --------------------------------------------------------------------------- #
class RefIndex:
    """Reference pool: md5 -> displays, plus a vectorised dHash set."""

    def __init__(self):
        self.md5_map = {}          # md5 -> [display, ...]
        self._ph = []              # [(display, phash), ...]
        self._arr = None
        self._disp = None
        self._np = _NpOps()
        self.count = 0

    def add(self, display, md5, phash):
        self.count += 1
        if md5:
            self.md5_map.setdefault(md5, []).append(display)
        if phash is not None:
            self._ph.append((display, int(phash)))

    def finalize(self):
        if self._ph:
            if self._np.ok:
                try:
                    self._arr = self._np.array([p for _, p in self._ph])
                    self._disp = [d for d, _ in self._ph]
                except Exception:
                    self._arr = None
                    self._disp = None
            if self._arr is None:  # pure-Python fallback also needs the pairs
                self._disp = [d for d, _ in self._ph]

    def has_md5(self, md5):
        v = self.md5_map.get(md5)
        return v[0] if v else None

    def find_neighbor(self, phash, hamming_lim):
        """Return (display, hamming) of best ref match within limit, else None."""
        if phash is None or not self._ph:
            return None
        if self._np.ok and self._arr is not None:
            hit = self._np.best(self._arr, phash, hamming_lim)
            if hit:
                j, hd = hit
                return (self._disp[j], hd)  # type: ignore[index]
            return None
        best = None  # pure-Python fallback
        for disp, q in self._ph:
            hd = hamming(phash, q)
            if hd <= hamming_lim and (best is None or hd < best[1]):
                best = (disp, hd)
                if hd == 0:
                    break
        return best


class KeptSet:
    """Running pool of KEPT candidate images, used for --internal dedup.

    Exact md5 hits are O(1); dHash hits are checked against a numpy array that
    is rebuilt in chunks (REBUILD inserts) so scanning 10k+ video frames stays
    fast; a small "dirty tail" (not yet baked) is scanned in pure Python.
    """

    REBUILD = 512

    def __init__(self):
        self.md5_owner = {}        # md5 -> display of the kept image
        self._ph = []              # [(display, phash), ...]
        self._arr = None
        self._baked = 0            # how many entries are in _arr
        self._np = _NpOps()

    def _rebuild(self):
        if self._ph and self._np.ok:
            try:
                self._arr = self._np.array([p for _, p in self._ph])
                self._baked = len(self._ph)
                return
            except Exception:
                self._arr = None
                self._baked = 0
        self._arr = None
        self._baked = 0

    def owner_md5(self, md5):
        return self.md5_owner.get(md5)

    def add(self, display, md5, phash):
        if md5 and md5 not in self.md5_owner:
            self.md5_owner[md5] = display
        if phash is not None:
            self._ph.append((display, int(phash)))
            if len(self._ph) - self._baked >= self.REBUILD:
                self._rebuild()

    def find_neighbor(self, phash, hamming_lim):
        """Return (display, hamming) of a kept image within limit, else None."""
        if phash is None or not self._ph:
            return None
        if self._arr is None or len(self._ph) - self._baked >= self.REBUILD:
            self._rebuild()
        if self._np.ok and self._arr is not None and self._baked:
            hit = self._np.best(self._arr, phash, hamming_lim)
            if hit:
                j, hd = hit
                return (self._ph[j][0], hd)
        # dirty tail not baked into the numpy array (or pure-Python mode)
        start = self._baked if (self._np.ok and self._arr is not None) else 0
        best = None
        for k in range(start, len(self._ph)):
            hd = hamming(phash, self._ph[k][1])
            if hd <= hamming_lim and (best is None or hd < best[1]):
                best = (self._ph[k][0], hd)
        return best

    def finalize(self):
        self._rebuild()


# --------------------------------------------------------------------------- #
# source enumeration
# --------------------------------------------------------------------------- #
def _split_images_dirs(root):
    """Return {split: images_dir} for splits that exist under root."""
    return {sp: os.path.join(root, sp, "images")
            for sp in SPLITS if os.path.isdir(os.path.join(root, sp, "images"))}


def _flat_images_dirs(root):
    """Return [images_dir] candidates for a flat layout or bare images dir."""
    cands = []
    im = os.path.join(root, "images")
    if os.path.isdir(im):
        cands.append(im)
    if os.path.isdir(root) and any(is_image_name(f) for f in os.listdir(root)):
        cands.append(root)  # root itself is an images dir
    return cands


def image_dirs_in(root):
    """All images dirs that should be fingerprinted for a directory source."""
    split_dirs = _split_images_dirs(root)
    if split_dirs:
        return list(split_dirs.values())
    return _flat_images_dirs(root)


def _label_dir_for_images(imdir):
    """Best-effort labels dir next to an images dir; None if absent."""
    if os.path.basename(imdir) == "images":
        parent = os.path.dirname(imdir)
        cand = os.path.join(parent, "labels")
        return cand if os.path.isdir(cand) else None
    for cand in (os.path.join(imdir, "labels"),
                 os.path.join(os.path.dirname(imdir),
                              os.path.basename(imdir) + "_labels")):
        if os.path.isdir(cand):
            return cand
    return None


def list_candidate(root):
    """Return candidate records: {split, rel, img, lbl, stem}.

    rel is relative to 'root' (e.g. train/images/a.jpg) and keeps image + label
    aligned when copying. split is None for flat layouts.
    """
    recs = []
    split_dirs = _split_images_dirs(root)
    if split_dirs:
        for sp in sorted(split_dirs):
            imdir = split_dirs[sp]
            lbldir = _label_dir_for_images(imdir)
            for name in sorted(os.listdir(imdir)):
                if not is_image_name(name):
                    continue
                lbl = None
                if lbldir:
                    cand = os.path.join(lbldir, os.path.splitext(name)[0] + ".txt")
                    if os.path.isfile(cand):
                        lbl = cand
                recs.append({"split": sp, "rel": "%s/images/%s" % (sp, name),
                             "img": os.path.join(imdir, name), "lbl": lbl,
                             "stem": os.path.splitext(name)[0]})
        return recs
    # flat: <root>/images + <root>/labels, or root itself as the images dir
    flat = _flat_images_dirs(root)
    if not flat:
        return recs
    imdir = flat[0]
    lbldir = _label_dir_for_images(imdir)
    for name in sorted(os.listdir(imdir)):
        if not is_image_name(name):
            continue
        lbl = None
        if lbldir:
            cand = os.path.join(lbldir, os.path.splitext(name)[0] + ".txt")
            if os.path.isfile(cand):
                lbl = cand
        recs.append({"split": None, "rel": name, "img": os.path.join(imdir, name),
                     "lbl": lbl, "stem": os.path.splitext(name)[0]})
    return recs


def iter_dir_images(root):
    """Yield abs paths of every image in a directory reference source."""
    for imdir in image_dirs_in(root):
        for name in sorted(os.listdir(imdir)):
            p = os.path.join(imdir, name)
            if os.path.isfile(p) and is_image_name(name):
                yield p


def _scan_src(path, tag):
    """Fingerprint one reference source; yields (display, md5, phash)."""
    if os.path.isdir(path):
        for p in iter_dir_images(path):
            try:
                with open(p, "rb") as fh:
                    data = fh.read()
            except Exception:
                continue
            rel = os.path.relpath(p, path)
            m, ph = fingerprint(data)
            yield "%s:%s" % (tag, rel), m, ph
        return
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith("/") or not is_image_name(name):
                    continue
                try:
                    data = zf.read(name)
                except Exception:
                    continue
                m, ph = fingerprint(data)
                yield "%s:%s" % (tag, name), m, ph
        return
    raise SystemExit("reference source not found (or not a dir/zip): %s" % path)


def _json_cache_items(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["items"]


def build_ref_index(ref_specs, cache_path=None):
    """ref_specs: [(path, label), ...]. Returns (RefIndex, items)."""
    idx = RefIndex()
    items = []
    if cache_path and os.path.isfile(cache_path):
        print("loading reference index cache:", cache_path)
        for it in _json_cache_items(cache_path):
            idx.add(it["d"], it.get("m"), it.get("p"))
        idx.finalize()
        return idx, items
    for path, tag in ref_specs:
        n = 0
        for disp, m, ph in _scan_src(path, tag):
            idx.add(disp, m, ph)
            items.append({"d": disp, "m": m, "p": ph})
            n += 1
        print("indexed %d images from %s (%s)" % (n, path, tag))
    idx.finalize()
    if cache_path and ref_specs:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump({"version": 1, "items": items}, fh)
        print("wrote reference index cache:", cache_path)
    return idx, items


# --------------------------------------------------------------------------- #
# classify
# --------------------------------------------------------------------------- #
def classify(cands, ref, do_internal, hamming_lim, hash_size=8):
    """Mark every candidate kept/removed. Returns a list of row dicts."""
    rows = []
    kept = KeptSet()
    for c in cands:
        try:
            with open(c["img"], "rb") as fh:
                data = fh.read()
        except Exception as e:
            rows.append({**c, "md5": None, "phash": None,
                         "status": "error", "reason": "unreadable: %s" % e,
                         "match": "", "hamming": ""})
            continue
        m, ph = fingerprint(data, hash_size)
        reason = ""
        match = ""
        hd = ""
        if m and ref is not None and ref.has_md5(m):
            reason, match = "ref-md5", ref.has_md5(m)
        elif ph is not None and ref is not None:
            nb = ref.find_neighbor(ph, hamming_lim)
            if nb:
                reason, match, hd = "ref-phash", nb[0], nb[1]
        if not reason and do_internal:
            if m and kept.owner_md5(m):
                reason, match = "int-md5", kept.owner_md5(m)
            elif ph is not None:
                nb = kept.find_neighbor(ph, hamming_lim)
                if nb:
                    reason, match, hd = "int-phash", nb[0], nb[1]
        if reason:
            rows.append({**c, "md5": m, "phash": ph,
                         "status": "removed", "reason": reason,
                         "match": match, "hamming": hd})
        else:
            kept.add(c["rel"], m, ph)
            rows.append({**c, "md5": m, "phash": ph,
                         "status": "kept", "reason": "unique", "match": "", "hamming": ""})
    return rows


# --------------------------------------------------------------------------- #
# class-name parser (tiny, mirrors dev_scripts/analyze_fire_dataset.py)
# --------------------------------------------------------------------------- #
_DICT_RE = re.compile(r"^(-?\d+):\s*(.+)$")
_LIST_RE = re.compile(r"^-\s*(.+)$")


def _read_names_yaml(path):
    names = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("names"):
                continue
            if line.startswith("path:"):
                continue
            m = _DICT_RE.match(line)
            if m:
                names[int(m.group(1))] = m.group(2).strip().strip("'\"")
                continue
            m = _LIST_RE.match(line)
            if m:
                names[len(names)] = m.group(1).strip().strip("'\"")
    return names if names else None


def candidate_names(candidate_root):
    """Best-effort {idx: name} from a candidate data.yaml, else None."""
    cands = [os.path.join(candidate_root, "data.yaml"),
             os.path.join(os.path.dirname(candidate_root.rstrip("/\\")), "data.yaml")]
    for path in cands:
        if os.path.isfile(path):
            names = _read_names_yaml(path)
            if names:
                return names
    return None


# --------------------------------------------------------------------------- #
# reports + copy
# --------------------------------------------------------------------------- #
def write_reports(rows, tag, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    csv_path = os.path.join(report_dir, "%s_per_image.csv" % tag)
    kept_path = os.path.join(report_dir, "%s_kept.txt" % tag)
    rem_path = os.path.join(report_dir, "%s_removed.txt" % tag)
    sum_path = os.path.join(report_dir, "%s_summary.txt" % tag)

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "image", "md5", "dhash", "status", "reason",
                    "match", "hamming"])
        for r in rows:
            w.writerow([r.get("split") or "all", r["rel"], r["md5"] or "",
                        "" if r["phash"] is None else "%016x" % r["phash"],
                        r["status"], r["reason"], r["match"], r["hamming"]])

    kept = [r for r in rows if r["status"] == "kept"]
    removed = [r for r in rows if r["status"] == "removed"]
    with open(kept_path, "w", encoding="utf-8") as fh:
        fh.writelines(r["rel"] + "\n" for r in kept)
    with open(rem_path, "w", encoding="utf-8") as fh:
        fh.writelines(r["rel"] + "\n" for r in removed)

    reasons = Counter(r["reason"] for r in removed)
    ham_hist = Counter(r["hamming"] for r in removed
                       if r["hamming"] not in ("", None))
    err = [r for r in rows if r["status"] == "error"]
    lines = []
    lines.append("Dedup report: %s" % tag)
    lines.append("=" * 78)
    lines.append("candidate images scanned : %d" % len(rows))
    lines.append("kept                      : %d" % len(kept))
    lines.append("removed                   : %d" % len(removed))
    lines.append("  " + "\n  ".join("%s: %d" % kv for kv in reasons.most_common()))
    if ham_hist:
        lines.append("hamming histogram (phash removals):")
        for hd in sorted(ham_hist):
            lines.append("  hamming %3d : %d" % (hd, ham_hist[hd]))
    if err:
        lines.append("ERRORS (could not read/decrypt): %d" % len(err))
        for r in err[:20]:
            lines.append("  " + r["rel"] + "  " + r["reason"])
    lines.append("")
    lines.append("match samples (first 10 removed):")
    for r in removed[:10]:
        lines.append("  %-60s %-14s %s" % (r["rel"], r["reason"], r["match"]))
    lines.append("")
    lines.append("written: %s" % csv_path)
    with open(sum_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return kept, removed, "\n".join(lines)


def _write_data_yaml(out_root, layout_splits, names):
    """layout_splits: {split: rel_images_dir}; paths relative to out_root."""
    lines = ["# generated by dev_scripts/dedup_images.py - labels copied verbatim",
             "# (class order per source set preserved - no remapping)",
             "path: %s" % out_root.replace("\\", "/")]
    for sp in SPLITS:
        if sp in layout_splits:
            lines.append("%s: %s" % (sp, layout_splits[sp].replace("\\", "/")))
    if names:
        lines.append("nc: %d" % len(names))
        lines.append("names:")
        for i in sorted(names):
            lines.append("  %d: %s" % (i, names[i]))
    with open(os.path.join(out_root, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def copy_kept(rows, out_root, flat):
    """Copy kept images + aligned labels into out_root.

    Preserves split structure by default; --flat merges into images/labels/.
    Labels are byte-copied so per-set class indexes are preserved.
    Returns (n_images, n_labels).
    """
    kept = [r for r in rows if r["status"] == "kept"]
    if not out_root or not kept:
        return 0, 0
    flat = bool(flat) or all((r.get("split") or None) is None for r in kept)
    seen = {}
    n_img = n_lbl = 0
    for r in kept:
        fname = os.path.basename(r["img"])
        if flat:
            dst_img = os.path.join(out_root, "images", fname)
            if dst_img in seen:
                raise SystemExit(
                    "flat mode filename collision for %s (also produced by %s) - "
                    "re-run without --flat to keep splits separate"
                    % (dst_img, seen[dst_img]))
            seen[dst_img] = r["rel"]
            dst_lbl = os.path.join(out_root, "labels",
                                   os.path.basename(r["stem"]) + ".txt")
        else:
            dst_img = os.path.join(out_root, r["split"], "images", fname)
            dst_lbl = os.path.join(out_root, r["split"], "labels",
                                   os.path.basename(r["stem"]) + ".txt")
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        shutil.copy2(r["img"], dst_img)
        n_img += 1
        if r["lbl"]:
            os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
            shutil.copy2(r["lbl"], dst_lbl)
            n_lbl += 1
    return n_img, n_lbl


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _default_report_dir():
    return ("fire-model-training/dedup/overlap_reports"
            if os.path.isdir("fire-model-training") else "./dedup_overlap_reports")


def main():
    ap = argparse.ArgumentParser(
        description="Dedup a candidate eval/train image set against reference set(s) "
                    "(md5 exact + 64-bit dHash perceptual near-dup).")
    ap.add_argument("--candidate", required=True,
                    help="dataset root / images dir to clean (Roboflow split or flat)")
    ap.add_argument("--tag", default=None,
                    help="report name (default: basename of --candidate)")
    ap.add_argument("--ref", action="append", default=[],
                    help="reference set source (dir/zip). Repeatable.")
    ap.add_argument("--ref-label", action="append", default=[],
                    help="short label per --ref (default: basename without ext)")
    ap.add_argument("--internal", action="store_true",
                    help="also drop internal near-dups of the candidate (video frames)")
    ap.add_argument("--hamming", type=int, default=10,
                    help="dHash Hamming threshold for 'near-dup' (default 10)")
    ap.add_argument("--hash-size", type=int, default=8,
                    help="dHash size, bits = size^2 (default 8 -> 64-bit)")
    ap.add_argument("--out", default=None,
                    help="if set, copy KEPT images+labels here (clean set dir)")
    ap.add_argument("--flat", action="store_true",
                    help="with --out: merge splits into <out>/{images,labels}")
    ap.add_argument("--report-dir", default=None,
                    help="where CSV/lists/summary go "
                         "(default fire-model-training/dedup/overlap_reports)")
    ap.add_argument("--cache", default=None,
                    help="persist/reuse the reference fingerprint index as JSON")
    args = ap.parse_args()

    cand_root = os.path.abspath(args.candidate)
    if not os.path.isdir(cand_root):
        sys.exit("candidate not found: %s" % cand_root)
    cands = list_candidate(cand_root)
    if not cands:
        sys.exit("no images found under candidate: %s" % cand_root)
    tag = args.tag or os.path.basename(cand_root.rstrip("/\\"))

    # reference index (built from --ref specs, or reused from --cache)
    ref = None
    if args.cache and os.path.isfile(args.cache):
        ref, _ = build_ref_index([], args.cache)
    elif args.ref:
        labels = list(args.ref_label)
        specs = []
        for i, path in enumerate(args.ref):
            lab = labels[i] if i < len(labels) else \
                os.path.splitext(os.path.basename(path.rstrip("/\\")))[0]
            specs.append((os.path.abspath(path), lab))
        ref, _ = build_ref_index(specs, args.cache)
    else:
        print("no reference set given (--ref/--cache); only --internal or plain keep")
    if ref is not None:
        print("reference pool: %d images (md5+phash)" % ref.count)

    print("candidate: %d images from %s" % (len(cands), cand_root))
    rows = classify(cands, ref, args.internal, args.hamming, args.hash_size)

    report_dir = os.path.abspath(args.report_dir or _default_report_dir())
    kept, removed, summary = write_reports(rows, tag, report_dir)
    print(summary)

    if args.out:
        out_root = os.path.abspath(args.out)
        os.makedirs(out_root, exist_ok=True)
        flat_mode = args.flat or all((r.get("split") or None) is None for r in rows)
        n_img, n_lbl = copy_kept(rows, out_root, flat_mode)
        layout = {}
        if flat_mode:
            layout = {"val": "images"}
        else:
            for r in kept:
                if r["split"] and r["split"] not in layout:
                    layout[r["split"]] = "%s/images" % r["split"]
        _write_data_yaml(out_root, layout, candidate_names(cand_root))
        print("clean set: copied %d images + %d labels into %s%s"
              % (n_img, n_lbl, out_root, " (flat)" if flat_mode else ""))
    print("report dir:", report_dir)
    print("done - kept %d / removed %d" % (len(kept), len(removed)))


if __name__ == "__main__":
    main()
