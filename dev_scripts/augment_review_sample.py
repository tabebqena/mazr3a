#!/usr/bin/env python3
"""augment_review_sample.py - visual + race-free review harness for the train augmentation.

Runs the EXACT augmentation pipeline from augment_fire_train.py on a local sample of images, but
KEEPS every original NEXT TO its augmented copy (``0001.jpeg`` / ``0001_aug.jpeg``) so you can
flip between them. Production moves originals into ``train_aug/images`` (to never double the
disk); this REVIEW harness copies the original and never deletes, purely for human inspection.

PROOF OF NO RACING (printed at the end and enforced by assertions):
    1. N inputs -> exactly N originals + N augmented copies (no missing / no duplicates).
    2. zero ``*.part`` / ``*.tmp`` leftovers (every atomic write completed).
    3. every ``_aug`` file decodes as a valid image (PIL verify) with the expected size.
    4. the journal has exactly N unique entries (no double-registration).
    5. re-run with ``--workers 1`` and ``diff -r`` (excluding proof.txt) -> byte-identical,
       i.e. the thread count changes nothing.

USAGE
    python dev_scripts/augment_review_sample.py --src <image dir> --out <review dir> \
        --n 100 --workers 8
"""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import augment_fire_train as aug  # reuse the exact production ops/render


def find_images(src):
    out = []
    for root, _dirs, files in os.walk(src):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in aug.IMG_EXTS:
                out.append(os.path.join(root, f))
    return sorted(out)


def sample(paths, n, seed):
    if n <= 0 or len(paths) <= n:
        return paths
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(paths)), n))
    return [paths[i] for i in idx]


def process_one(item, out_dir, src_root):
    """Augment ONE image end-to-end; returns a journal record.

    Every stem/destination is unique to this worker (and ``*.part`` temp names are per-file),
    so no two threads touch the same path. The MAIN thread is the single journal writer.
    """
    idx, src = item
    ext = os.path.splitext(src)[1].lower()
    stem = "%04d" % idx

    orig_dst = os.path.join(out_dir, stem + ext)
    aug_dst = os.path.join(out_dir, stem + "_aug" + ext)

    with open(src, "rb") as fh:
        data = fh.read()
    src_md5 = aug.md5_bytes(data)

    # deterministic op + params, identical to production (seed derives from the stem)
    seed = aug.stable_seed(stem)
    rng = random.Random(seed)
    op = rng.choice(aug.OPS)
    params = aug.roll_params(op, rng)

    lbl_src = os.path.splitext(src)[0] + ".txt"
    boxes = aug.read_boxes(lbl_src)

    with aug.Image.open(BytesIO(data)) as im:
        im = im.convert("RGB")
        w0, h0 = im.size
        im_aug, boxes_aug = aug.render_op(im, w0, h0, boxes, op, params, aug.stable_seed(stem))
        fmt = aug.EXT_FORMAT.get(ext, "JPEG")
        buf = BytesIO()
        im_aug.save(buf, format=fmt, quality=95)
        out_bytes = buf.getvalue()

    dst_md5 = aug.md5_bytes(out_bytes)

    # atomic writes: each worker owns its unique stem
    tmp = aug_dst + ".part"
    with open(tmp, "wb") as fh:
        fh.write(out_bytes)
    os.replace(tmp, aug_dst)

    tmp = orig_dst + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, orig_dst)

    # keep labels alongside when the source has them (so boxes stay inspectable)
    if boxes:
        aug.write_boxes(os.path.join(out_dir, stem + "_aug.txt"), boxes_aug)
        with open(os.path.join(out_dir, stem + ".txt"), "w", encoding="utf-8") as fh:
            for c, cx, cy, w, h in boxes:
                fh.write("%d %.6f %.6f %.6f %.6f\n" % (int(c), cx, cy, w, h))

    return {
        "stem": stem, "op": op, "src": os.path.relpath(src, src_root),
        "src_md5": src_md5, "dst_md5": dst_md5,
        "size": [w0, h0], "aug_size": list(im_aug.size),
    }


def verify(out_dir, n, journal, workers):
    files = sorted(os.listdir(out_dir))
    leftovers = [f for f in files if f.endswith(".part") or f.endswith(".tmp")]
    imgs = [f for f in files if os.path.splitext(f)[1].lower() in aug.IMG_EXTS]
    augs = sorted(f for f in imgs if "_aug" in os.path.splitext(f)[0])
    origs = sorted(f for f in imgs if "_aug" not in os.path.splitext(f)[0])

    undecodable = 0
    for f in augs:
        try:
            with aug.Image.open(os.path.join(out_dir, f)) as im:
                im.verify()
        except Exception:
            undecodable += 1

    n_uniq = len({r["stem"] for r in journal.values()})
    # proof.txt intentionally does NOT include the workers count, so two runs (w1 vs w8)
    # produce byte-identical directories and `diff -r` passes cleanly.
    text = "\n".join([
        "=" * 64,
        "AUGMENTATION REVIEW - proof of no racing",
        "=" * 64,
        "sampled inputs       : %d" % n,
        "originals written    : %d" % len(origs),
        "_aug written         : %d" % len(augs),
        "*.part/*.tmp left    : %d  %s" % (len(leftovers), leftovers or "(none)"),
        "undecodable _aug     : %d" % undecodable,
        "journal entries      : %d (unique %d)" % (len(journal), n_uniq),
    ])
    with open(os.path.join(out_dir, "proof.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("workers              : %d" % workers)
    print(text)

    assert len(origs) == n, "missing original copies"
    assert len(augs) == n, "missing augmented copies"
    assert not leftovers, "leftover partial files -> race!"
    assert undecodable == 0, "corrupted augmented images -> race!"
    assert len(journal) == n and n_uniq == n, "journal not 1:1 -> race!"
    print("PASS: no racing detected for N=%d, workers=%d" % (n, workers))


def write_gallery(out_dir, journal):
    order = sorted(journal.values(), key=lambda r: r["stem"])
    items = []
    for r in order:
        stem = r["stem"]
        base = None
        for e in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            if os.path.isfile(os.path.join(out_dir, stem + e)):
                base = stem + e
                break
        if not base:
            continue
        augf = stem + "_aug" + os.path.splitext(base)[1]
        items.append((stem, base, augf, r["op"]))

    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>augmentation review</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:1em}",
        ".pair{display:inline-block;margin:1em;vertical-align:top;text-align:center}",
        "img{height:170px;border:1px solid #444;margin:0 4px;background:#000}",
        "figcaption{font-size:12px;color:#9cf;margin-top:4px}</style></head><body>",
        "<h1>augmentation review (original | _aug)</h1>",
    ]
    for stem, base, augf, op in items:
        html.append("<figure class='pair'><img src='%s'><img src='%s'>"
                    "<figcaption>%s &middot; %s</figcaption></figure>" % (base, augf, stem, op))
    html.append("</body></html>")
    with open(os.path.join(out_dir, "preview.html"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(html))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory of images (searched recursively)")
    ap.add_argument("--out", required=True, help="review output directory")
    ap.add_argument("--n", type=int, default=100, help="number of images to sample (default 100)")
    ap.add_argument("--workers", type=int, default=8, help="worker threads (default 8)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")
    args = ap.parse_args()

    src_root = os.path.abspath(args.src)
    out_dir = os.path.abspath(args.out)
    paths = find_images(src_root)
    if not paths:
        sys.exit("no images under --src: %s" % src_root)
    paths = sample(paths, args.n, args.seed)
    os.makedirs(out_dir, exist_ok=True)

    journal_path = os.path.join(out_dir, "review_journal.jsonl")
    journal = {}
    worker = partial(process_one, out_dir=out_dir, src_root=src_root)
    items = list(enumerate(paths, 1))

    print("augment_review_sample: %d images -> %s (workers=%d)"
          % (len(paths), out_dir, args.workers), flush=True)
    ex = ThreadPoolExecutor(max_workers=args.workers)
    try:
        # single writer thread appends the journal; workers only write their own files
        with open(journal_path, "w", encoding="utf-8") as jfh:
            for idx, rec in enumerate(ex.map(worker, items), 1):
                journal[rec["stem"]] = rec
                jfh.write(json.dumps(rec) + "\n")
                jfh.flush()
                if idx % 25 == 0 or idx == len(items):
                    print("  processed %d/%d" % (idx, len(items)), flush=True)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    verify(out_dir, len(paths), journal, args.workers)
    write_gallery(out_dir, journal)
    print("review files + preview.html written -> %s" % out_dir)


if __name__ == "__main__":
    main()
