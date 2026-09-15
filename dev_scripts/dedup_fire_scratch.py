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
deleted from the pool: kept images + labels are COPIED into --out.

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

RESUMABLE STAGES + PROGRESS (2026-09-14)
----------------------------------------
The old version was one long in-memory job: a mid-copy failure (e.g. ENOSPC) meant re-running
EVERYTHING from scratch, including fingerprinting all images again. This version splits the run
into four idempotent stages that checkpoint to ``<report>/`` so a re-run resumes where it left
off instead of recomputing expensive work:

    1. fingerprint  - md5 + dHash for every pool image, APPENDED incrementally to
                      ``<report>/fingerprints.jsonl`` (partial lines are ignored on resume).
                      SLOW + safely resumable -> intra-stage resume.
    2. mark         - all dedup/balance decisions, computed in-memory and written ONCE to
                      ``<report>/decisions.jsonl`` when the stage finishes. It is FAST,
                      deterministic and replaying half-done passes would be error-prone, so it is
                      NOT intra-stage resumable: an interrupt restarts this stage from scratch
                      (only the completed-stage result is replayed on later runs).
    3. copy         - bg-extra + clean tree, written one file at a time via ``*.part`` +
                      ``os.replace`` (atomic), skipping destinations that already exist with the
                      correct size. Interrupting a 100k-file copy just resumes the remaining files.
                      SLOW + safely resumable -> intra-stage resume.
    4. report       - data.yaml / manifest.csv / per_image.csv / run indexes / summary.txt
                      (FAST + idempotent; always rewritten, never resumed).

Run progress is printed to stdout AND appended to ``<report>/dedup.log`` every ``--progress-every``
files (default 1000) or every ~15 s, whichever comes first, with img/s + ETA.

USAGE
-----
    python dedup_fire_scratch.py --pool /content/raw_yolo --out /content/clean_yolo \
        --hamming 8 --train-scope group \
        --prev-train-index /content/drive/MyDrive/.../fingerprints/train \
        --prev-test-index  /content/drive/MyDrive/.../fingerprints/test

    # resume after a crash (default behaviour):
    python dedup_fire_scratch.py --pool /content/raw_yolo --out /content/clean_yolo

    # ignore all checkpoints and recompute fingerprint + mark from scratch:
    python dedup_fire_scratch.py --pool /content/raw_yolo --out /content/clean_yolo --fresh
"""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

STATE_VERSION = 1

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


def _acquire_lock(path, label):
    """Take an advisory exclusive lock held for the whole process lifetime.

    Prevents two concurrent invocations (e.g. the notebook cell re-run by mistake, or a
    manual augment while dedup is still copying) from racing on the same output. Uses
    ``fcntl.flock`` on Linux/macOS (what Colab runs) and ``msvcrt.locking`` on Windows.
    The OS releases the lock when the process exits, so a second invocation fails fast
    with a clear message instead of silently corrupting the output.
    """
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fh = open(path, "w", encoding="utf-8")
    try:
        try:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            # Windows fallback (never exercised on Linux/Colab; msvcrt has no type stubs).
            import msvcrt
            locking = getattr(msvcrt, "locking", None)
            if locking is None:
                raise RuntimeError("no flock and no msvcrt.locking available")
            locking(fh.fileno(), getattr(msvcrt, "LK_NBLCK", 1), 1)
    except OSError:
        fh.close()
        sys.exit("%s: another process already holds the lock %s "
                 "(is a previous run still active?)" % (label, path))
    return fh


# ---------------------------------------------------------------------------
# Logging / progress helpers
# ---------------------------------------------------------------------------

class Logger:
    """Tee every progress line to stdout and an append-only log file."""

    def __init__(self, path):
        self.path = path
        if path:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            self.fh = open(path, "a", encoding="utf-8")
        else:
            self.fh = None

    def __call__(self, msg):
        line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        print(line, flush=True)
        if self.fh is not None:
            self.fh.write(line + "\n")
            self.fh.flush()

    def close(self):
        if self.fh is not None:
            self.fh.close()


class Progress:
    """Emit a progress line every `every` ticks or `interval` seconds, then a final line."""

    def __init__(self, total, label, every=1000, interval=15.0, log=None, suffix=None):
        self.total = max(1, total)
        self.label = label
        self.every = max(1, every)
        self.interval = interval
        self.log = log or (lambda m: print(m, flush=True))
        self.suffix = suffix or (lambda: "")
        self.done = 0
        self.t0 = time.time()
        self.last = self.t0

    def _emit(self, final=False):
        now = time.time()
        el = now - self.t0
        rate = self.done / el if el else 0
        eta = (self.total - self.done) / rate if rate else 0
        pct = 100.0 * self.done / self.total
        state = "done" if final else "%.0f%%" % pct
        msg = "    %s %d/%d (%s) | %.0f img/s | ETA %.1f min | %s" % (
            self.label, self.done, self.total, state, rate, eta / 60.0, self.suffix())
        self.log(msg)
        self.last = now

    def tick(self, n=1):
        self.done += n
        now = time.time()
        if (self.done % self.every == 0) or (now - self.last >= self.interval):
            self._emit()

    def finish(self):
        self._emit(final=True)


# ---------------------------------------------------------------------------
# Pool / checkpoint load + save
# ---------------------------------------------------------------------------

def _fp_key(split, stem):
    return split + "\t" + stem


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


def load_fingerprints(fp_path):
    """Return {key: {"md5":..., "dhash":..., "unreadable":bool}} from fingerprints.jsonl.

    Malformed / truncated tail lines are ignored, so a kill mid-append only loses the last
    partially-written line (which is simply recomputed on the next run).
    """
    out = {}
    if not os.path.isfile(fp_path):
        return out
    with open(fp_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                key = _fp_key(o["split"], o["stem"])
                out[key] = {
                    "md5": o.get("md5"),
                    "dhash": o.get("dhash"),
                    "unreadable": bool(o.get("u")),
                }
            except (ValueError, KeyError):
                continue
    return out


def apply_fingerprints(rows, fp_cache):
    """Merge cached md5/dhash (and unreadable quarantine flag) into the in-memory rows."""
    for r in rows:
        hit = fp_cache.get(_fp_key(r["split"], r["stem"]))
        if hit is None:
            continue
        r["md5"] = hit["md5"]
        r["dhash"] = hit["dhash"]
        if hit["unreadable"]:
            r["kept"] = False
            r["reason"] = "unreadable"


def load_decisions(path):
    """Return {key: (kept, reason)} from decisions.jsonl checkpoint."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                out[_fp_key(o["split"], o["stem"])] = (bool(o.get("kept")), o.get("reason", ""))
            except (ValueError, KeyError):
                continue
    return out


def apply_decisions(rows, dec):
    for r in rows:
        hit = dec.get(_fp_key(r["split"], r["stem"]))
        if hit is not None:
            r["kept"], r["reason"] = hit


def save_decisions(rows, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"split": r["split"], "stem": r["stem"],
                                 "kept": 1 if r["kept"] else 0, "reason": r["reason"]}) + "\n")
    os.replace(tmp, path)


def load_state(path):
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                s = json.load(fh)
            if isinstance(s, dict):
                return s
        except (ValueError, OSError):
            pass
    return {"version": STATE_VERSION}


def save_state(state, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Stage 1: fingerprint
# ---------------------------------------------------------------------------

def fingerprint_stage(rows, args, state, report_dir, log, progress_every):
    fp_path = os.path.join(report_dir, "fingerprints.jsonl")
    cache = load_fingerprints(fp_path)
    apply_fingerprints(rows, cache)

    missing = [r for r in rows if _fp_key(r["split"], r["stem"]) not in cache]
    if not missing:
        log("stage fingerprint: all %d images already cached (%s)" % (len(rows), fp_path))
        state["fingerprint_done"] = True
        return

    log("stage fingerprint: %d/%d images still need hashing" % (len(missing), len(rows)))
    # Warm the lazily-cached PIL LANCZOS enum once so the worker threads don't race to
    # initialise it (the race would be benign, but a single pre-load is cleaner).
    _lanzos()

    broken = 0
    prog = Progress(len(missing), "fingerprint", progress_every, log=log,
                    suffix=lambda: "broken=%d" % broken)

    # Each worker hashes ONLY its own image and returns the result; the MAIN thread is the
    # single writer of fingerprints.jsonl, so the append-only checkpoint never has two
    # threads touching it at once (the same crash-consistency guarantee as before).
    def work(r):
        try:
            with open(r["img"], "rb") as img_fh:
                data = img_fh.read()
            return r, hashlib.md5(data).hexdigest(), dhash_of(data), 0
        except Exception:
            return r, None, None, 1

    ex = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = [ex.submit(work, r) for r in missing]
        with open(fp_path, "a", encoding="utf-8") as fh:
            for fut in as_completed(futures):
                r, md5, dhash, u = fut.result()
                if u:
                    broken += 1
                    r["kept"] = False
                    r["reason"] = "unreadable"
                    if not args.skip_broken:
                        log("fatal: unreadable image (set --skip-broken to quarantine): %s"
                            % r["img"])
                        ex.shutdown(wait=False, cancel_futures=True)
                        sys.exit(1)
                    if args.broken_out:
                        os.makedirs(os.path.join(args.broken_out, "images"), exist_ok=True)
                        try:
                            shutil.move(r["img"], os.path.join(
                                args.broken_out, "images",
                                r["stem"] + os.path.splitext(r["img"])[1]))
                        except OSError:
                            pass
                else:
                    r["md5"], r["dhash"] = md5, dhash
                fh.write(json.dumps({"split": r["split"], "stem": r["stem"],
                                     "md5": r["md5"], "dhash": r["dhash"], "u": u}) + "\n")
                fh.flush()
                prog.tick()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    prog.finish()
    log("stage fingerprint: complete (%d unreadable)" % broken)
    state["fingerprint_done"] = True


# ---------------------------------------------------------------------------
# Stage 2: dedup / balance marking
# ---------------------------------------------------------------------------

def self_dedup(rows, split, hamming_lim, group_bounded, log, progress_every):
    """Remove images of `split` that near-dup an earlier KEPT image of the same split.

    Exact md5 duplicates are ALWAYS removed globally (byte-identical is byte-identical,
    whatever clip it came from); dHash near-dups are group-scoped when group_bounded.
    """
    seen_md5 = set()               # global: byte-identical images are always duplicates
    pool = KeptSet()
    cur_group = None
    removed = 0
    total = sum(1 for r in rows if r["split"] == split and r["kept"])
    log("  %s self-dedup: %d images (scope=%s)"
        % (split, total, "group" if group_bounded else "global"))
    prog = Progress(total, "  %s self" % split, progress_every, log=log,
                    suffix=lambda: "removed %d" % removed)
    for r in rows:
        if r["split"] != split or not r["kept"]:
            continue
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
        prog.tick()
    prog.finish()
    return removed


def mark_vs_pool(rows, split, pool, reason, limit, log, progress_every):
    """Remove images of `split` that near-dup any entry of an immutable reference pool."""
    total = sum(1 for r in rows if r["split"] == split and r["kept"])
    ref_n = len(pool._ph) if hasattr(pool, "_ph") else 0
    log("  %s: %d images vs %d reference hashes" % (reason, total, ref_n))
    removed = 0
    prog = Progress(total, "  " + reason, progress_every, log=log,
                    suffix=lambda: "removed %d" % removed)
    for r in rows:
        if r["split"] != split or not r["kept"]:
            continue
        if pool.near(r["md5"], r["dhash"], limit):
            r["kept"] = False
            r["reason"] = reason
            removed += 1
        prog.tick()
    prog.finish()
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


def mark_stage(rows, args, state, report_dir, log, progress_every):
    """Run all dedup + balance passes in-memory and persist the final decisions.

    Deliberately NOT intra-stage resumable: the passes are pure in-memory computation
    (fast relative to fingerprinting/copying) and re-running them from scratch is fully
    deterministic, so per-pass checkpointing would add replay risk for little benefit.
    The stage IS resumable as a whole - once it completes, ``decisions.jsonl`` holds the
    final result and a later run just replays it onto the freshly-loaded rows.
    """
    dec_path = os.path.join(report_dir, "decisions.jsonl")
    state_path = os.path.join(report_dir, "state.json")

    if state.get("mark_done"):
        # Replay the completed decisions onto freshly-loaded rows so copy/report see them.
        apply_decisions(rows, load_decisions(dec_path))
        log("stage mark: already done, replaying recorded decisions and skipping")
        return

    log("stage mark: running all passes (not intra-stage resumable; "
        "an interrupt restarts this stage from scratch)")
    prev_train = load_index(args.prev_train_index, positives_only=True)
    prev_test = load_index(args.prev_test_index, positives_only=False)
    counts = {}

    def run(name, fn):
        counts[name] = fn()
        log("stage mark: pass %s -> removed %d" % (name, counts[name]))

    if not args.no_test_self:
        run("test-self", lambda: self_dedup(rows, "test", args.hamming, True,
                                            log, progress_every))
    if not args.no_val_self:
        run("val-self", lambda: self_dedup(rows, "val", args.hamming, True,
                                           log, progress_every))
    if not args.no_train_self:
        run("train-self", lambda: self_dedup(rows, "train", args.hamming,
                                             args.train_scope == "group", log, progress_every))
    if prev_train:
        run("test-vs-prev-train", lambda: mark_vs_pool(
            rows, "test", prev_train, "test-vs-prev-train", args.hamming, log, progress_every))
        run("val-vs-prev-train", lambda: mark_vs_pool(
            rows, "val", prev_train, "val-vs-prev-train", args.hamming, log, progress_every))
        run("train-vs-prev-train", lambda: mark_vs_pool(
            rows, "train", prev_train, "train-vs-prev-train", args.hamming, log, progress_every))
    if prev_test:
        run("test-vs-prev-test", lambda: mark_vs_pool(
            rows, "test", prev_test, "test-vs-prev-test", args.hamming, log, progress_every))
        run("val-vs-prev-test", lambda: mark_vs_pool(
            rows, "val", prev_test, "val-vs-prev-test", args.hamming, log, progress_every))
        run("train-vs-prev-test", lambda: mark_vs_pool(
            rows, "train", prev_test, "train-vs-prev-test", args.hamming, log, progress_every))
    if not args.no_train_vs_test:
        cur_test = FixedPool.from_rows([r for r in rows if r["split"] == "test" and r["kept"]])
        run("train-vs-test", lambda: mark_vs_pool(
            rows, "train", cur_test, "train-vs-test", args.hamming, log, progress_every))
    if not args.no_train_vs_val:
        cur_val = FixedPool.from_rows([r for r in rows if r["split"] == "val" and r["kept"]])
        run("train-vs-val", lambda: mark_vs_pool(
            rows, "train", cur_val, "train-vs-val", args.hamming, log, progress_every))
    # positive-vs-background balance runs LAST, after every dedup pass has settled the set
    run("balance-bg-cap", lambda: balance_background(rows, args.max_bg_share))

    save_decisions(rows, dec_path)
    state["counts"] = counts
    state["mark_done"] = True
    save_state(state, state_path)
    log("stage mark: complete (%d passes)" % len(counts))


# ---------------------------------------------------------------------------
# Stage 3: copy bg-extra + clean tree
# ---------------------------------------------------------------------------

def _copy_image(src, dst, log, copied, skipped, missing):
    """Atomically copy one image unless dst already exists with the right size."""
    if not os.path.isfile(src):
        missing += 1
        return copied, skipped, missing
    if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
        skipped += 1
        return copied, skipped, missing
    part = dst + ".part"
    shutil.copy2(src, part)
    os.replace(part, dst)
    copied += 1
    return copied, skipped, missing


def _copy_label(src, dst, skipped, missing):
    """Copy a label file, or create an empty one when the source has no boxes."""
    if os.path.isfile(src):
        if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
            skipped += 1
            return skipped, missing
        part = dst + ".part"
        shutil.copy2(src, part)
        os.replace(part, dst)
    else:
        if not os.path.isfile(dst):
            open(dst, "w", encoding="utf-8").close()
        else:
            skipped += 1
    return skipped, missing


def _copy_rows(rows, dst_dir_for, label, log, progress_every, workers):
    """Copy every row's image + label concurrently; each (stem, dst) is unique -> race-free.

    Workers copy their own (image, label) pair; the MAIN thread only aggregates counts and
    advances the progress line. ``*.part`` temp names are per-destination, so no two workers
    collide, and the size-check skip preserves the resumable semantics.
    """
    copied = skipped = missing = 0
    prog = Progress(len(rows), label, progress_every, log=log,
                    suffix=lambda: "copied %d, skipped %d" % (copied, skipped))
    if not rows:
        prog.finish()
        return copied, skipped, missing

    def work(r):
        ext = os.path.splitext(r["img"])[1].lower()
        d = dst_dir_for(r)
        c = s = m = 0
        c, s, m = _copy_image(r["img"], os.path.join(d, "images", r["stem"] + ext),
                              None, c, s, m)
        s, m = _copy_label(r["lbl"], os.path.join(d, "labels", r["stem"] + ".txt"), s, m)
        return c, s, m

    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        for c, s, m in ex.map(work, rows):
            copied += c
            skipped += s
            missing += m
            prog.tick()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    prog.finish()
    return copied, skipped, missing


def copy_bg_extra(rows, bg_dir, log, progress_every, workers):
    """Copy background images moved out by the cap so they stay reversible (not deleted)."""
    bg_rows = [r for r in rows if r["reason"] == "balance-bg-cap"]
    if not bg_rows:
        return 0
    os.makedirs(os.path.join(bg_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(bg_dir, "labels"), exist_ok=True)
    log("stage copy: bg-extra %d images -> %s" % (len(bg_rows), bg_dir))
    copied, skipped, missing = _copy_rows(
        bg_rows, lambda r: bg_dir, "  bg-extra", log, progress_every, workers)
    log("stage copy: bg-extra done (copied %d, skipped %d, missing %d)"
        % (copied, skipped, missing))
    return len(bg_rows)


def copy_clean(rows, out, log, progress_every, workers):
    """Copy every kept image + label into the clean YOLO tree (resumable per file)."""
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(out, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out, split, "labels"), exist_ok=True)
    kept = [r for r in rows if r["kept"]]
    log("stage copy: clean tree %d kept images -> %s" % (len(kept), out))
    copied, skipped, missing = _copy_rows(
        kept, lambda r: os.path.join(out, r["split"]), "  clean", log, progress_every, workers)
    log("stage copy: clean tree done (copied %d, skipped %d, missing %d)"
        % (copied, skipped, missing))
    return len(kept)


def copy_stage(rows, args, state, log, progress_every):
    # Always run the copy loop: it is idempotent (skips files whose destination already
    # exists with the correct size), so an interrupted copy resumes by copying only the
    # missing files. This also self-heals a partially-written tree even if ``copy_done``
    # was already recorded.
    bg_dir = args.bg_extra or (args.out + "_bg_extra")
    copy_bg_extra(rows, bg_dir, log, progress_every, args.workers)
    copy_clean(rows, args.out, log, progress_every, args.workers)
    state["copy_done"] = True
    save_state(state, os.path.join(args.report, "state.json"))


# ---------------------------------------------------------------------------
# Stage 4: final report / dataset metadata (idempotent)
# ---------------------------------------------------------------------------

def save_index(path, rows, split_filter, run_name):
    """Write one JSONL file {md5,d,stem,run,classes,pos} for kept rows of the given splits."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            if r["split"] not in split_filter or not r["kept"]:
                continue
            fh.write(json.dumps({"md5": r["md5"], "d": r["dhash"], "stem": r["stem"],
                                 "run": run_name, "classes": sorted(r["cls"]),
                                 "pos": 1 if r["cls"] else 0}) + "\n")


def balance_background(rows, max_bg_share):
    """Cap the background fraction of KEPT train images (positive vs background balance).

    Returns the number of kept background train images moved out (reason 'balance-bg-cap').
    Solves ``bg_new / n_new <= max_bg_share`` for the minimum number to remove:
        remove = ceil((n_bg - max_bg_share * n) / (1 - max_bg_share)).
    Background = an image with NO box of any class (empty label). Any image with a
    fire/other/smoke box counts as a positive and is never capped. (In the old fire-only
    round smoke boxes were dropped to background, so this also covered smoke-only images;
    in the fire/other/smoke round smoke is a positive and stays protected.)
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


def report_stage(rows, args, state, by_split, log):
    out = args.out
    report_dir = args.report
    kept_rows = [r for r in rows if r["kept"]]
    kept_split = Counter(r["split"] for r in kept_rows)

    log("stage report: writing data.yaml / manifest / indexes / summary")

    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("# CLEAN deduplicated pool (dedup_fire_scratch.py)\n")
        fh.write("path: %s\n" % out.replace("\\", "/"))
        fh.write("train: train/images\n")
        fh.write("val: %s\n" % ("val/images" if kept_split["val"] else "test/images"))
        fh.write("test: test/images\n")
    src_yaml = os.path.join(args.pool, "data.yaml")
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

    counts = state.get("counts", {})
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
    rep.append("train class balance (positive vs background):")
    rep.append("  positive (fire/smoke/other) = %d (%.0f%%)" % (n_tr - n_bg, 100 * (n_tr - n_bg) / max(1, n_tr)))
    rep.append("  background                   = %d (%.0f%%)   cap = %.0f%%"
               % (n_bg, 100 * n_bg / max(1, n_tr), 100 * args.max_bg_share))
    if n_tr and (n_tr - n_bg) / n_tr < 0.10:
        rep.append("  WARNING: positive share is under 10%% - recall will likely suffer; "
                   "lower --max-bg-share or add more fire/smoke sources.")
    rep.append("")
    rep.append("removals by pass:")
    for k in sorted(counts):
        rep.append("  %-20s %d" % (k, counts[k]))
    rep.append("")
    rep.append("this-run indexes (promote to Drive AFTER a successful train):")
    rep.append("  %s/run_train_index.jsonl" % report_dir)
    rep.append("  %s/run_test_index.jsonl" % report_dir)
    rep.append("clean -> %s" % out)
    text = "\n".join(rep)
    with open(os.path.join(report_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    log(text)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

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
                    help="cap the background (empty-label) fraction of TRAIN images [0.60]. "
                         "Higher = more negatives / fewer false positives; lower = more "
                         "positive (fire/smoke) samples / better recall.")
    ap.add_argument("--bg-extra", default=None,
                    help="dir that receives background images moved out by the cap "
                         "(default: <out>_bg_extra; reversible, nothing is deleted)")
    ap.add_argument("--no-test-self", action="store_true")
    ap.add_argument("--no-val-self", action="store_true")
    ap.add_argument("--no-train-self", action="store_true")
    ap.add_argument("--no-train-vs-test", action="store_true")
    ap.add_argument("--no-train-vs-val", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing checkpoints and recompute fingerprint + mark from scratch")
    ap.add_argument("--log-file", default=None, help="log file (default: <report>/dedup.log)")
    ap.add_argument("--progress-every", type=int, default=1000,
                    help="emit a progress line every N images (default 1000)")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker threads for the fingerprint + copy stages "
                         "(default: min(16, 2*cpu_count); 1 = single-threaded)")
    args = ap.parse_args()

    if args.workers is None or args.workers <= 0:
        args.workers = min(16, max(2, (os.cpu_count() or 4) * 2))

    pool = os.path.abspath(args.pool)
    out = os.path.abspath(args.out)
    if os.path.isfile(os.path.join(out, "data.yaml")):
        print("clean pool already exists ->", out, "(remove it or use a new --out to re-run)")
        return
    report_dir = os.path.abspath(args.report or (out + "_report"))
    os.makedirs(report_dir, exist_ok=True)
    args.out = out
    args.report = report_dir
    args.pool = pool

    # Cross-process guard: hold the same lock file that augment_fire_train.py uses so a
    # concurrent dataset build can't race this one. Held until process exit (OS releases).
    _lock_fh = _acquire_lock(os.path.join(report_dir, "build.lock"), "dedup_fire_scratch")

    log = Logger(args.log_file or os.path.join(report_dir, "dedup.log"))
    log("=== dedup_fire_scratch.py start ===")
    log("command: %s" % " ".join(sys.argv))
    log("pool=%s out=%s report=%s" % (pool, out, report_dir))

    state_path = os.path.join(report_dir, "state.json")
    if args.fresh:
        for p in ("state.json", "fingerprints.jsonl", "decisions.jsonl"):
            try:
                os.remove(os.path.join(report_dir, p))
            except OSError:
                pass
        log("--fresh: cleared state/fingerprint/decision checkpoints")
    state = load_state(state_path)
    if state.get("version") != STATE_VERSION:
        state = {"version": STATE_VERSION}
        log("state version mismatch -> starting fresh checkpoints")

    rows = load_pool(pool)
    by_split = Counter(r["split"] for r in rows)
    log("pool images: %s" % dict(by_split))

    # --- stage 1: fingerprint (resumable) ---
    fingerprint_stage(rows, args, state, report_dir, log, args.progress_every)
    save_state(state, state_path)

    # --- stage 2: mark (resumable, checkpointed per pass) ---
    mark_stage(rows, args, state, report_dir, log, args.progress_every)
    save_state(state, state_path)

    # --- stage 3: copy (resumable per file) ---
    copy_stage(rows, args, state, log, args.progress_every)
    save_state(state, state_path)

    # --- stage 4: report (idempotent) ---
    report_stage(rows, args, state, by_split, log)
    save_state(state, state_path)

    log("=== dedup_fire_scratch.py complete ===")
    log.close()


if __name__ == "__main__":
    main()
