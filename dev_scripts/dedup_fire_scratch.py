#!/usr/bin/env python3
"""dedup_fire_scratch.py - the dedup passes for the scratch fire training set.

Reads the merged pool written by dev_scripts/prep_fire_scratch_dataset.py
(train/val/test + manifest.csv) and writes a CLEAN copy, dropping images in this FIXED order:

  Intra-run (every run, the user's four honest-validation requirements):
    1. test self         (group)  - a test image near-dup of an earlier KEPT image OF THE SAME CLIP is removed
    2. val self          (group)  - same for val
    3. train self        (group|global) - a train image near-dup of an earlier KEPT train image is removed
    4. test vs prev-train/prev-test  - never score an image that was already trained on / already scored
    5. val  vs prev-train/prev-test  - same for early-stop honesty
    6. train vs prev-train/prev-test - never re-memorise a previously trained POSITIVE image, and
                                       never leak a previously held-out test image into training.
                                       Background-only use is NOT a blocker: pixels trained as
                                       background stay re-usable as a positive class later.
    7. train vs test     (global)  - honest final score (test never leaks into train)
    8. train vs val      (global)  - honest early-stop

Near-dup = 64-bit dHash Hamming distance <= --hamming (default 8) OR byte-identical md5.
md5 is checked O(1); dHash uses a vectorised numpy pool (uint64 XOR + bitwise_count) with a
small pure-Python "dirty tail", exactly like dev_scripts/dedup_images.py. Nothing is ever
deleted: kept images + labels are COPIED into --out.

LIGHTWEIGHT PERSISTENT FINGERPRINT INDEX (never re-download old datasets)
------------------------------------------------------------------------
Datasets are too large to keep on Drive, so only their FINGERPRINTS are persisted:
one JSONL line per image holding {md5, dhash, stem, run} (~100 bytes/image). Pass
``--prev-train-index`` / ``--prev-test-index`` (directories of ``*.jsonl``) to dedup the new
set against every previous run's train/test pool WITHOUT re-downloading a single image.

This run's kept-train fingerprints are written to ``<report>/run_train_index.jsonl`` and its
kept test+val fingerprints to ``<report>/run_test_index.jsonl``; the notebook copies those two
files into the Drive index directories ONLY AFTER a successful training, so the persistent
index accumulates exactly the runs that were actually trained.

USAGE
-----
    python dedup_fire_scratch.py --pool /content/raw_yolo --out /content/clean_yolo \
        --hamming 8 --train-scope group \
        --prev-train-index /content/drive/MyDrive/.../fingerprints/train \
        --prev-test-index  /content/drive/MyDrive/.../fingerprints/test
"""
import argparse
import csv
import hashlib
import json
import math
import os
import time
import shutil
import sys
from collections import Counter
from io import BytesIO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_LANCZOS = None


def _lanzos():
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
        self._mod = None
        try:
            import numpy as np
            if hasattr(np, "bitwise_count"):  # numpy >= 2.0
                self._mod = np
                self.ok = True
        except Exception:
            self._mod = None

    def array(self, vals):
        m = self._mod
        if m is None:
            raise RuntimeError("numpy>=2 unavailable")
        return m.array(vals, dtype=m.uint64)

    def best(self, arr, phash, limit):
        """Return (index, hamming) of the closest hash in arr within limit, else None."""
        m = self._mod
        if m is None or arr is None or len(arr) == 0:
            return None
        d = m.bitwise_count(m.bitwise_xor(arr, m.uint64(int(phash))))
        j = int(m.argmin(d))
        hd = int(d[j])
        return (j, hd) if hd <= limit else None


def dhash_of(data, hash_size=8):
    """64-bit (8x8) difference-hash of image bytes, or None if undecodable."""
    try:
        from PIL import Image
        im = Image.open(BytesIO(data)).convert("L")
        im = im.resize((hash_size + 1, hash_size), _lanzos())
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


def hamming(a, b):
    return (int(a) ^ int(b)).bit_count()


class KeptSet:
    """Running pool of KEPT images for a self-dedup pass (adapted from dedup_images.py)."""

    REBUILD = 512

    def __init__(self):
        self.md5 = set()
        self._ph = []               # list of dhash ints
        self._arr = None
        self._baked = 0
        self._np = _NpOps()

    def _rebuild(self):
        if self._ph and self._np.ok:
            try:
                self._arr = self._np.array(self._ph)
                self._baked = len(self._ph)
                return
            except Exception:
                self._arr = None
                self._baked = 0
        self._arr = None
        self._baked = 0

    def add(self, md5, phash):
        if md5:
            self.md5.add(md5)
        if phash is not None:
            self._ph.append(int(phash))
            if len(self._ph) - self._baked >= self.REBUILD:
                self._rebuild()

    def near(self, md5, phash, limit):
        """True if md5 exact OR dHash within `limit` of a kept image."""
        if md5 and md5 in self.md5:
            return True
        if phash is None or not self._ph:
            return False
        if self._arr is None or len(self._ph) - self._baked >= self.REBUILD:
            self._rebuild()
        if self._np.ok and self._arr is not None and self._baked:
            if self._np.best(self._arr, phash, limit):
                return True
        start = self._baked if (self._np.ok and self._arr is not None) else 0
        for k in range(start, len(self._ph)):
            if hamming(phash, self._ph[k]) <= limit:
                return True
        return False


class FixedPool:
    """Immutable pool of reference hashes (previous-run index or cross-split kept set)."""

    def __init__(self, md5s, hashes):
        self.md5 = set(md5s)
        self._ph = [int(h) for h in hashes]
        self._np = _NpOps()
        self._arr = None
        if self._np.ok and self._ph:
            self._arr = self._np.array(self._ph)

    @classmethod
    def from_rows(cls, rows):
        return cls([r["md5"] for r in rows if r["md5"]],
                   [r["dhash"] for r in rows if r["dhash"] is not None])

    def near(self, md5, phash, limit):
        if md5 and md5 in self.md5:
            return True
        if phash is None or not self._ph:
            return False
        if self._np.ok and self._arr is not None:
            if self._np.best(self._arr, phash, limit):
                return True
        for h in self._ph:
            if hamming(phash, h) <= limit:
                return True
        return False


def is_image(name):
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def load_pool(pool):
    """Return rows = [{split, stem, group, img, lbl, md5, dhash, kept, reason}]."""
    manifest = {}
    mf = os.path.join(pool, "manifest.csv")
    if os.path.isfile(mf):
        with open(mf, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                manifest[r["stem"]] = r
    rows = []
    for split in ("train", "val", "test"):
        imdir = os.path.join(pool, split, "images")
        lbldir = os.path.join(pool, split, "labels")
        if not os.path.isdir(imdir):
            continue
        for name in sorted(os.listdir(imdir)):
            if not is_image(name):
                continue
            stem = os.path.splitext(name)[0]
            m = manifest.get(stem, {})
            cls = set()
            lp = os.path.join(lbldir, stem + ".txt")
            if os.path.isfile(lp):
                for line in open(lp, encoding="utf-8"):
                    parts = line.split()
                    if parts:
                        try:
                            cls.add(int(float(parts[0])))
                        except ValueError:
                            pass
            rows.append({
                "split": split,
                "stem": stem,
                "group": m.get("split_group") or stem,
                "img": os.path.join(imdir, name),
                "lbl": lp,
                "cls": cls,
                "md5": None, "dhash": None,
                "kept": True, "reason": "",
            })
    # deterministic scan order; train groups become contiguous for the group-scoped self-dedup
    rows.sort(key=lambda r: (r["split"], r["group"], r["stem"]))
    return rows


def fingerprint(rows, skip_broken, broken_out):
    """Compute md5 + dhash for every row; quarantine unreadable images."""
    broken = 0
    for i, r in enumerate(rows):
        try:
            with open(r["img"], "rb") as fh:
                data = fh.read()
            r["md5"] = hashlib.md5(data).hexdigest()
            r["dhash"] = dhash_of(data)
        except Exception:
            broken += 1
            r["kept"] = False
            r["reason"] = "unreadable"
            if not skip_broken:
                sys.exit("unreadable image (set --skip-broken to quarantine): %s" % r["img"])
            if broken_out:
                os.makedirs(os.path.join(broken_out, "images"), exist_ok=True)
                try:
                    shutil.move(r["img"], os.path.join(broken_out, "images",
                                r["stem"] + os.path.splitext(r["img"])[1]))
                except OSError:
                    pass
        if i and i % 5000 == 0:
            print("  fingerprinted %d/%d images" % (i, len(rows)), flush=True)
    print("  fingerprinted %d images (%d unreadable)" % (len(rows), broken), flush=True)


def self_dedup(rows, split, hamming_lim, group_bounded):
    """Remove images of `split` that near-dup an earlier KEPT image of the same split.

    Exact md5 duplicates are ALWAYS removed globally (byte-identical is byte-identical,
    whatever clip it came from); dHash near-dups are group-scoped when group_bounded.
    """
    seen_md5 = set()               # global: byte-identical images are always duplicates
    pool = KeptSet()
    cur_group = None
    removed = 0
    total = sum(1 for r in rows if r["split"] == split)
    done = 0
    t0 = time.time()
    print("  %s self-dedup: %d images (scope=%s)"
          % (split, total, "group" if group_bounded else "global"), flush=True)
    for r in rows:
        if r["split"] != split or not r["kept"]:
            continue
        done += 1
        if group_bounded and r["group"] != cur_group:
            pool = KeptSet()          # new clip/group: a fresh dHash pool (frames compared within)
            cur_group = r["group"]
        if r["md5"] and r["md5"] in seen_md5:
            r["kept"] = False
            r["reason"] = "%s-self" % split
            removed += 1
        elif pool.near(r["md5"], r["dhash"], hamming_lim):
            r["kept"] = False
            r["reason"] = "%s-self" % split
            removed += 1
        else:
            seen_md5.add(r["md5"])
            pool.add(r["md5"], r["dhash"])
        if done % 5000 == 0:
            el = time.time() - t0
            rate = done / el if el else 0
            eta = (total - done) / rate / 60 if rate else 0
            print("    %d/%d (%.0f%%) removed %d | %.0f img/s | ETA %.1f min"
                  % (done, total, 100.0 * done / max(1, total), removed, rate, eta), flush=True)
    return removed


def mark_vs_pool(rows, split, pool, reason, limit):
    """Remove images of `split` that near-dup any entry of an immutable reference pool."""
    total = sum(1 for r in rows if r["split"] == split)
    ref_n = len(pool._ph) if hasattr(pool, "_ph") else 0
    print("  %s: %d images vs %d reference hashes" % (reason, total, ref_n), flush=True)
    removed = done = 0
    t0 = time.time()
    for r in rows:
        if r["split"] != split or not r["kept"]:
            continue
        done += 1
        if pool.near(r["md5"], r["dhash"], limit):
            r["kept"] = False
            r["reason"] = reason
            removed += 1
        if done % 5000 == 0:
            el = time.time() - t0
            rate = done / el if el else 0
            eta = (total - done) / rate / 60 if rate else 0
            print("    %d/%d (%.0f%%) removed %d | %.0f img/s | ETA %.1f min"
                  % (done, total, 100.0 * done / max(1, total), removed, rate, eta), flush=True)
    return removed


def load_index(index_dir, positives_only=False):
    """FixedPool from every *.jsonl in index_dir, or None if the dir is empty/missing.

    ``positives_only=True`` keeps ONLY images that were trained as a POSITIVE class
    (at least one box); background-only usage is ignored so a later run can still train
    those same pixels as a real class (e.g. train smoke after a fire-only run - the
    smoke boxes that were dropped to background must stay re-usable).

    Each line: {"md5": "...", "d": <int>, "stem": "...", "run": "...",
                "classes": [..], "pos": 0|1}
    """
    if not index_dir or not os.path.isdir(index_dir):
        return None
    md5s, hashes = set(), []
    for f in sorted(os.listdir(index_dir)):
        if not f.endswith(".jsonl"):
            continue
        with open(os.path.join(index_dir, f), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except ValueError:
                    continue
                if positives_only and o.get("pos") == 0:
                    continue
                if o.get("md5"):
                    md5s.add(o["md5"])
                if o.get("d") is not None:
                    hashes.append(int(o["d"]))
    if not md5s and not hashes:
        return None
    return FixedPool(md5s, hashes)


def save_index(path, rows, split_filter, run_name):
    """Write one JSONL file {md5,d,stem,run,classes,pos} for kept rows of the given splits."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            if r["split"] not in split_filter or not r["kept"]:
                continue
            fh.write(json.dumps({"md5": r["md5"], "d": r["dhash"], "stem": r["stem"],
                                 "run": run_name, "classes": sorted(r["cls"]),
                                 "pos": 1 if r["cls"] else 0}) + "\n")


def balance_background(rows, max_bg_share):
    """Cap the background fraction of KEPT train images (fire vs not-fire balance).

    Returns the number of kept background train images moved out (reason 'balance-bg-cap').
    Solves ``bg_new / n_new <= max_bg_share`` for the minimum number to remove:
        remove = ceil((n_bg - max_bg_share * n) / (1 - max_bg_share)).
    Background = an image with no positive box (empty label); in a fire-only run that
    includes pure background AND smoke-only images.
    """
    train = [r for r in rows if r["split"] == "train" and r["kept"]]
    n = len(train)
    if n == 0:
        return 0
    n_bg = sum(1 for r in train if not r["cls"])
    share = n_bg / n
    if share <= max_bg_share:
        return 0
    excess = math.ceil((n_bg - max_bg_share * n) / (1.0 - max_bg_share))
    removed = 0
    for r in rows:
        if r["split"] != "train" or not r["kept"] or r["cls"]:
            continue
        if removed >= excess:
            break
        r["kept"] = False
        r["reason"] = "balance-bg-cap"
        removed += 1
    return removed


def write_bg_extra(rows, bg_extra_dir):
    """Copy background images moved out by the cap so they stay reversible (not deleted)."""
    if not bg_extra_dir:
        return 0
    n = 0
    for r in rows:
        if r["reason"] != "balance-bg-cap":
            continue
        os.makedirs(os.path.join(bg_extra_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(bg_extra_dir, "labels"), exist_ok=True)
        ext = os.path.splitext(r["img"])[1].lower()
        shutil.copy2(r["img"], os.path.join(bg_extra_dir, "images", r["stem"] + ext))
        lbl = os.path.join(bg_extra_dir, "labels", r["stem"] + ".txt")
        if os.path.isfile(r["lbl"]):
            shutil.copy2(r["lbl"], lbl)
        else:
            open(lbl, "w", encoding="utf-8").close()
        n += 1
    return n


def write_clean(rows, out, pool):
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(out, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out, split, "labels"), exist_ok=True)
    kept_rows = []
    for r in rows:
        if not r["kept"]:
            continue
        ext = os.path.splitext(r["img"])[1].lower()
        dst = os.path.join(out, r["split"], "images", r["stem"] + ext)
        if not os.path.exists(dst):
            shutil.copy2(r["img"], dst)
        lbl = r["lbl"]
        if os.path.isfile(lbl):
            shutil.copy2(lbl, os.path.join(out, r["split"], "labels", r["stem"] + ".txt"))
        else:
            open(os.path.join(out, r["split"], "labels", r["stem"] + ".txt"), "w").close()
        kept_rows.append(r)
    return kept_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="merged pool from prep_fire_scratch_dataset.py")
    ap.add_argument("--out", required=True, help="NEW clean YOLO dataset root")
    ap.add_argument("--hamming", type=int, default=8, help="dHash Hamming limit (default 8)")
    ap.add_argument("--train-scope", choices=("group", "global"), default="group",
                    help="train self-dedup scope: within split_group (clip) or global")
    ap.add_argument("--report", default=None, help="report dir (default: <out>_report)")
    ap.add_argument("--skip-broken", action="store_true", help="quarantine unreadable images")
    ap.add_argument("--broken-out", default=None, help="quarantine dir for unreadable images")
    ap.add_argument("--prev-train-index", default=None,
                    help="dir of *.jsonl fingerprints of EVERY previously-trained image")
    ap.add_argument("--prev-test-index", default=None,
                    help="dir of *.jsonl fingerprints of EVERY previously-held-out test/val image")
    ap.add_argument("--run-name", default="scratch", help="tag stored in the index files")
    ap.add_argument("--max-bg-share", type=float, default=0.60,
                    help="cap the background (non-fire) fraction of TRAIN images [0.60]. "
                         "Higher = more negatives / fewer false positives; lower = more fire "
                         "samples / better recall.")
    ap.add_argument("--bg-extra", default=None,
                    help="dir that receives background images moved out by the cap "
                         "(default: <out>_bg_extra; reversible, nothing is deleted)")
    ap.add_argument("--no-test-self", action="store_true")
    ap.add_argument("--no-val-self", action="store_true")
    ap.add_argument("--no-train-self", action="store_true")
    ap.add_argument("--no-train-vs-test", action="store_true")
    ap.add_argument("--no-train-vs-val", action="store_true")
    args = ap.parse_args()

    pool = os.path.abspath(args.pool)
    out = os.path.abspath(args.out)
    if os.path.isfile(os.path.join(out, "data.yaml")):
        print("clean pool already exists ->", out, "(remove it or use a new --out to re-run)")
        return
    report_dir = os.path.abspath(args.report or (out + "_report"))
    os.makedirs(report_dir, exist_ok=True)

    rows = load_pool(pool)
    by_split = Counter(r["split"] for r in rows)
    print("pool images:", dict(by_split), flush=True)
    print("fingerprinting ...", flush=True)
    fingerprint(rows, args.skip_broken, args.broken_out)

    counts = {}
    # test/val self-dedup is GROUP-scoped (within a clip) like train: video-frame corpora
    # collapse to ~1 image per clip under GLOBAL scope, which would starve early stopping.
    if not args.no_test_self:
        counts["test-self"] = self_dedup(rows, "test", args.hamming, True)
    if not args.no_val_self:
        counts["val-self"] = self_dedup(rows, "val", args.hamming, True)
    if not args.no_train_self:
        counts["train-self"] = self_dedup(rows, "train", args.hamming,
                                          args.train_scope == "group")

    prev_train = load_index(args.prev_train_index, positives_only=True)
    prev_test = load_index(args.prev_test_index, positives_only=False)
    if prev_train:
        counts["test-vs-prev-train"] = mark_vs_pool(rows, "test", prev_train, "test-vs-prev-train", args.hamming)
        counts["val-vs-prev-train"] = mark_vs_pool(rows, "val", prev_train, "val-vs-prev-train", args.hamming)
        counts["train-vs-prev-train"] = mark_vs_pool(rows, "train", prev_train, "train-vs-prev-train", args.hamming)
    if prev_test:
        counts["test-vs-prev-test"] = mark_vs_pool(rows, "test", prev_test, "test-vs-prev-test", args.hamming)
        counts["val-vs-prev-test"] = mark_vs_pool(rows, "val", prev_test, "val-vs-prev-test", args.hamming)
        counts["train-vs-prev-test"] = mark_vs_pool(rows, "train", prev_test, "train-vs-prev-test", args.hamming)

    if not args.no_train_vs_test:
        cur_test = FixedPool.from_rows([r for r in rows if r["split"] == "test" and r["kept"]])
        counts["train-vs-test"] = mark_vs_pool(rows, "train", cur_test, "train-vs-test", args.hamming)
    if not args.no_train_vs_val:
        cur_val = FixedPool.from_rows([r for r in rows if r["split"] == "val" and r["kept"]])
        counts["train-vs-val"] = mark_vs_pool(rows, "train", cur_val, "train-vs-val", args.hamming)

    # fire vs not-fire balance (runs LAST, after every dedup pass has settled the set)
    counts["balance-bg-cap"] = balance_background(rows, args.max_bg_share)
    bg_extra = args.bg_extra or (out + "_bg_extra")
    if counts["balance-bg-cap"]:
        n_extra = write_bg_extra(rows, bg_extra)
        print("background cap: moved %d -> %s" % (n_extra, bg_extra), flush=True)

    kept_rows = write_clean(rows, out, pool)

    # data.yaml for the clean pool (val fallback to test when val is empty)
    kept_split = Counter(r["split"] for r in kept_rows)
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("# CLEAN deduplicated pool (dedup_fire_scratch.py)\n")
        fh.write("path: %s\n" % out.replace("\\", "/"))
        fh.write("train: train/images\n")
        fh.write("val: %s\n" % ("val/images" if kept_split["val"] else "test/images"))
        fh.write("test: test/images\n")
    src_yaml = os.path.join(pool, "data.yaml")
    if os.path.isfile(src_yaml):
        tail = [ln for ln in open(src_yaml, encoding="utf-8")
                if ln.startswith(("nc:", "names:", "  "))]
        with open(os.path.join(out, "data.yaml"), "a", encoding="utf-8") as fh:
            fh.writelines(tail)

    # this run's lightweight fingerprints (the notebook promotes them to Drive after training)
    save_index(os.path.join(report_dir, "run_train_index.jsonl"), rows, ("train",), args.run_name)
    save_index(os.path.join(report_dir, "run_test_index.jsonl"), rows, ("test", "val"), args.run_name)

    with open(os.path.join(out, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "stem", "split_group", "kept", "reason"])
        for r in rows:
            w.writerow([r["split"], r["stem"], r["group"],
                        "1" if r["kept"] else "0", r["reason"]])
    with open(os.path.join(report_dir, "per_image.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "stem", "status", "reason"])
        for r in rows:
            w.writerow([r["split"], r["stem"], "kept" if r["kept"] else "removed", r["reason"]])

    rep = ["Dedup report (hamming <= %d, train scope %s)" % (args.hamming, args.train_scope),
           "=" * 64]
    for split in ("train", "val", "test"):
        n = by_split[split]
        k = sum(1 for r in rows if r["split"] == split and r["kept"])
        rep.append("%-5s kept %d / %d (removed %d)" % (split, k, n, n - k))
    kept_train = [r for r in rows if r["split"] == "train" and r["kept"]]
    n_tr = len(kept_train)
    n_bg = sum(1 for r in kept_train if not r["cls"])
    rep.append("")
    rep.append("train class balance (fire vs not-fire):")
    rep.append("  fire-positive = %d (%.0f%%)" % (n_tr - n_bg, 100 * (n_tr - n_bg) / max(1, n_tr)))
    rep.append("  background    = %d (%.0f%%)   cap = %.0f%%"
               % (n_bg, 100 * n_bg / max(1, n_tr), 100 * args.max_bg_share))
    if n_tr and (n_tr - n_bg) / n_tr < 0.10:
        rep.append("  WARNING: fire-positive share is under 10%% - recall will likely suffer; "
                   "lower --max-bg-share or add more fire sources.")
    rep.append("")
    rep.append("removals by pass:")
    for k, v in counts.items():
        rep.append("  %-20s %d" % (k, v))
    rep.append("")
    rep.append("this-run indexes (promote to Drive AFTER a successful train):")
    rep.append("  %s/run_train_index.jsonl" % report_dir)
    rep.append("  %s/run_test_index.jsonl" % report_dir)
    rep.append("clean -> %s" % out)
    text = "\n".join(rep)
    with open(os.path.join(report_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
