#!/usr/bin/env python3
"""analyze_overlap.py - report overlap of a candidate image set against reference set(s).

Phase-1 tooling for the clean evaluation / retraining workflow
(plans/fire-model-clean-eval-workflow.md). Companion to dev_scripts/dedup_images.py
(it reuses that module's fingerprint + matching library via a sys.path import).

Unlike dedup_images.py (which *produces* a clean copy), this script is purely a
diagnostic: it answers "how much of set S is already covered by the reference
pool?" and gives you files to EYEBALL the near-duplicates:

  <out>/overlap_summary.txt     counts + reasons + hamming histogram
  <out>/<tag>_per_image.csv     every candidate image + status/match (see below)
  <out>/<tag>_kept.txt          candidate images with no overlap
  <out>/<tag>_removed.txt       candidate images overlapping the reference
  <out>/matched_pairs.csv       candidate <-> reference match list (md5/phash)
  <out>/sample_pairs/           N flagged candidate images (+ ref file when the
                                reference is a directory) copied here for review

Removed reasons (same contract as dedup_images.py):
  ref-md5    candidate file byte-identical to a reference image
  ref-phash  candidate dHash within --hamming of a reference image
  int-md5 / int-phash   only when --internal is given (candidate's own dups)

USAGE
-----
  # how much of Abonia fire-8 is inside the 8,939 training pool?
  python dev_scripts/analyze_overlap.py --candidate \
      fire-model-training/2_Abonia/abonia_repo/datasets/fire-8 --tag abonia \
      --ref "fire-model-training/1_SalahALHaismawi/dataset/Fire Detection.v1i.yolov8.zip" \
      --ref-label 8939 --sample 8 \
      --cache fire-model-training/dedup/overlap_reports/_index_8939.json

  # internal near-dup rate of a set (no reference): use dedup_images.py --internal,
  # or pass --ref of the set itself.
"""
import argparse
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dedup_images as di  # noqa: E402


def _ref_specs(args):
    labels = list(args.ref_label)
    specs = []
    for i, path in enumerate(args.ref):
        lab = labels[i] if i < len(labels) else \
            os.path.splitext(os.path.basename(path.rstrip("/\\")))[0]
        specs.append((os.path.abspath(path), lab))
    return specs


def _sample_pairs(rows, specs, out, n):
    """Copy up to n flagged candidate images (and ref when it's a dir file)."""
    flagged = [r for r in rows
               if r["status"] == "removed" and r["reason"] in ("ref-md5", "ref-phash")]
    if not flagged or n <= 0:
        return 0
    sample_dir = os.path.join(out, "sample_pairs")
    os.makedirs(sample_dir, exist_ok=True)
    # map tag -> root path so directory refs can be re-opened for eyeballing
    root_by_tag = {}
    for path, lab in specs:
        if os.path.isdir(path):
            root_by_tag[lab] = path
    n_written = 0
    for r in flagged[:n]:
        stem = os.path.splitext(os.path.basename(r["img"]))[0]
        ref_disp = r["match"] or ""
        ref_base = os.path.basename(ref_disp.replace("\\", "/")) or "ref"
        hd = r["hamming"] or "md5"
        dst_cand = os.path.join(sample_dir,
                                "%03d_%s__hd%s__%s%s"
                                % (n_written, stem[:40], hd, ref_base[:60],
                                   os.path.splitext(os.path.basename(r["img"]))[1]))
        try:
            shutil.copy2(r["img"], dst_cand)
            n_written += 1
        except Exception as e:
            print("sample copy fail", r["img"], e)
            continue
        # also copy the reference file when it is a plain file on disk
        tag, _, rel = ref_disp.partition(":")
        root = root_by_tag.get(tag)
        if root and rel:
            ref_abs = os.path.join(root, rel.replace("/", os.sep))
            if os.path.isfile(ref_abs):
                dst_ref = os.path.splitext(dst_cand)[0] + "__REF" + \
                    os.path.splitext(ref_abs)[1]
                try:
                    shutil.copy2(ref_abs, dst_ref)
                except Exception as e:
                    print("ref sample copy fail", ref_abs, e)
    return n_written


def main():
    ap = argparse.ArgumentParser(
        description="Report how much of a candidate set overlaps reference set(s) "
                    "(md5 exact + 64-bit dHash near-dup) and eyeball the matches.")
    ap.add_argument("--candidate", required=True,
                    help="candidate dataset root / images dir")
    ap.add_argument("--tag", default=None,
                    help="candidate name for file names (default: basename)")
    ap.add_argument("--ref", action="append", default=[],
                    help="reference set source (dir/zip). Repeatable.")
    ap.add_argument("--ref-label", action="append", default=[],
                    help="short label per --ref (default: basename without ext)")
    ap.add_argument("--internal", action="store_true",
                    help="also report candidate's OWN near-dups (int-md5/int-phash)")
    ap.add_argument("--hamming", type=int, default=10,
                    help="dHash Hamming threshold for 'near-dup' (default 10)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: fire-model-training/dedup/"
                         "overlap_reports/<tag>_vs_<refs>)")
    ap.add_argument("--cache", default=None,
                    help="persist/reuse the reference fingerprint index as JSON")
    ap.add_argument("--sample", type=int, default=5,
                    help="copy N flagged pairs into sample_pairs/ for eyeballing (0=off)")
    args = ap.parse_args()

    cand_root = os.path.abspath(args.candidate)
    if not os.path.isdir(cand_root):
        sys.exit("candidate not found: %s" % cand_root)
    cands = di.list_candidate(cand_root)
    if not cands:
        sys.exit("no images found under candidate: %s" % cand_root)
    tag = args.tag or os.path.basename(cand_root.rstrip("/\\"))

    # reference index
    ref = None
    specs = _ref_specs(args)
    if args.cache and os.path.isfile(args.cache):
        ref, _ = di.build_ref_index([], args.cache)
    elif specs:
        ref, _ = di.build_ref_index(specs, args.cache)
    else:
        sys.exit("need at least one --ref source or an existing --cache index")
    print("reference pool: %d images" % ref.count)

    rows = di.classify(cands, ref, args.internal, args.hamming)
    ref_label = "+".join(lab for _, lab in specs) or os.path.basename(args.cache)
    out = os.path.abspath(args.out or di._default_report_dir())
    out = os.path.join(out, "%s_vs_%s" % (tag, ref_label))
    os.makedirs(out, exist_ok=True)

    # standard per-image CSV / kept / removed / summary artifacts
    kept, removed, summary = di.write_reports(rows, tag, out)

    # matched pairs CSV
    pairs_path = os.path.join(out, "matched_pairs.csv")
    with open(pairs_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate", "split", "reason", "hamming", "matched_ref",
                    "candidate_md5"])
        for r in removed:
            if r["reason"] in ("ref-md5", "ref-phash"):
                w.writerow([r["rel"], r.get("split") or "all", r["reason"],
                            r["hamming"], r["match"], r["md5"]])

    n_samp = _sample_pairs(rows, specs, out, args.sample)

    # print a concise overlap-focused summary
    ref_removed = [r for r in removed if r["reason"].startswith("ref-")]
    int_removed = [r for r in removed if r["reason"].startswith("int-")]
    lines = []
    lines.append("")
    lines.append("Overlap report: %s vs %s" % (tag, ref_label))
    lines.append("=" * 70)
    lines.append("candidate images : %d" % len(rows))
    lines.append("overlap w/ ref    : %d (%.1f%%)" % (
        len(ref_removed), 100.0 * len(ref_removed) / len(rows) if rows else 0))
    lines.append("  ref-md5 (exact) : %d" %
                 sum(1 for r in ref_removed if r["reason"] == "ref-md5"))
    lines.append("  ref-phash (near): %d" %
                 sum(1 for r in ref_removed if r["reason"] == "ref-phash"))
    if args.internal:
        lines.append("internal near-dups: %d" % len(int_removed))
    lines.append("clean/kept        : %d (%.1f%%)" % (
        len(kept), 100.0 * len(kept) / len(rows) if rows else 0))
    lines.append("")
    lines.append("sample pairs copied for eyeballing: %d -> %s" % (n_samp, out))
    print("\n".join(lines))

    print("\nfull artifacts in:", out)
    print("  per_image.csv   ", os.path.join(out, "%s_per_image.csv" % tag))
    print("  matched_pairs.csv", pairs_path)
    print("  summary         ", os.path.join(out, "%s_summary.txt" % tag))


if __name__ == "__main__":
    main()
