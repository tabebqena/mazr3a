#!/usr/bin/env python3
"""compare_pretrained_fire_models.py - score every real fire/smoke checkpoint we hold
against the 200-image, 4-category benchmark built by
[`build_fire_benchmark200.py`](build_fire_benchmark200.py), and write a matrix report.

Why a separate tool from `compare_fire_models.py`: that one runs ultralytics `val()` on
YOLO `data.yaml` suites (box mAP).  THIS one is image-level and model-agnostic: it takes
arbitrary `.pt` checkpoints (archived versions, freshly downloaded pretraineds, our ACTIVE
model), auto-detects each checkpoint's `fire`/`smoke` class **by name**, and reports
per-category detection at several thresholds - so a single-class `fire` model, a
`fire/other/smoke` model and a `Fire/default/smoke` model can be compared head-to-head.

Checkpoints with **no** fire/smoke class (e.g. the 4 COCO-base downloads) are listed as
N/A (no fire/smoke class) instead of being scored - scoring them would be 0 by construction.

Model discovery (md5-deduped):
  * ACTIVE         models/fire/best.pt                       -> tag `v4`
  * archived       models/fire/versions/*/model.pt           -> tag `v1`, `v2`, ...
  * downloads      model-training/pretrained-models/*.pt      -> tag = filename stem

Usage:
    .venv/bin/python dev_scripts/compare_pretrained_fire_models.py \
        [--manifest model-training/fire-benchmark-200/manifest.csv] \
        [--out model-training/fire-model-compare-200] \
        [--models v4 v1 abonia-yolov8s-fire-best] [--conf 0.5] \
        [--thresholds 0.25,0.35,0.5,0.7] [--imgsz 0] [--batch 16] [--force]
    .venv/bin/python dev_scripts/compare_pretrained_fire_models.py --list-models

`--imgsz 0` (default) runs each checkpoint at its own training imgsz (from the ckpt
`train_args`), which is the fairest per-model setting; pass e.g. `--imgsz 640` to force one
size for all. Outputs (git-ignored): `<out>/report.md`, `<out>/summary.csv`,
`<out>/<tag>/per_image.csv`.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSIONS_DIR = os.path.join(ROOT, "models", "fire", "versions")
ACTIVE_PT = os.path.join(ROOT, "models", "fire", "best.pt")
PRETRAINED_DIR = os.path.join(ROOT, "model-training", "pretrained-models")
DEFAULT_MANIFEST = os.path.join(ROOT, "model-training", "fire-benchmark-200", "manifest.csv")
DEFAULT_OUT = os.path.join(ROOT, "model-training", "fire-model-compare-200")

CATEGORIES = ["fire_smoke", "fire_only", "smoke_only", "other"]
FIRE_NAMES = {"fire", "flame"}
SMOKE_NAMES = {"smoke"}

_T0 = time.time()


def step(msg):
    print("[step %6.1fs] %s" % (time.time() - _T0, msg), flush=True)


# --------------------------------------------------------------------------
# model discovery / class mapping
# --------------------------------------------------------------------------
def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tag_for_version(version_id):
    return version_id.split("-", 1)[0] or version_id


def discover_models(tokens=None):
    """Ordered, md5-deduped model dicts: {tag,id,pt,md5,aliases,role}."""
    cands = []
    # ACTIVE first so it wins the tag + role over an archived copy of the same md5.
    if os.path.isfile(ACTIVE_PT):
        cands.append({"tag": "v4", "id": "ACTIVE models/fire/best.pt",
                      "pt": ACTIVE_PT, "role": "ACTIVE"})
    if os.path.isdir(VERSIONS_DIR):
        for d in sorted(os.listdir(VERSIONS_DIR)):
            pt = os.path.join(VERSIONS_DIR, d, "model.pt")
            if os.path.isfile(pt):
                cands.append({"tag": _tag_for_version(d), "id": d, "pt": pt,
                              "role": "archived"})
    if os.path.isdir(PRETRAINED_DIR):
        for f in sorted(os.listdir(PRETRAINED_DIR)):
            if f.endswith(".pt"):
                cands.append({"tag": f[:-3], "id": "pretrained/" + f,
                              "pt": os.path.join(PRETRAINED_DIR, f), "role": "pretrained"})

    for c in cands:
        c["md5"] = md5_file(c["pt"])
        c["aliases"] = []

    # md5-dedup while preserving order (first occurrence wins the tag)
    out, seen = [], {}
    for c in cands:
        if c["md5"] in seen:
            seen[c["md5"]]["aliases"].append(c["pt"])
            continue
        seen[c["md5"]] = c
        out.append(c)

    if not tokens:
        return out

    selected = []
    for tok in tokens:
        tok = tok.strip()
        if tok in ("active", "v4"):
            m = next((c for c in out if c["role"] == "ACTIVE"), None)
            if m is None:
                sys.exit("ACTIVE models/fire/best.pt not found")
            selected.append(m)
            continue
        if os.path.isfile(tok) and tok.endswith(".pt"):
            rp = os.path.realpath(tok)
            m = next((c for c in out if os.path.realpath(c["pt"]) == rp), None)
            if m is None:
                m = {"tag": os.path.basename(tok)[:-3], "id": tok, "pt": tok,
                     "role": "explicit", "md5": md5_file(tok), "aliases": []}
            selected.append(m)
            continue
        matches = [c for c in out if c["tag"] == tok or c["tag"].startswith(tok)]
        if not matches:
            sys.exit("model token '%s' matched nothing (try --list-models)" % tok)
        selected.append(matches[0])

    dedup, seen2 = [], set()
    for m in selected:
        if m["md5"] not in seen2:
            dedup.append(m)
            seen2.add(m["md5"])
    return dedup


def load_probe(path):
    """Load a checkpoint via ultralytics (handles legacy pickles) and describe it."""
    from ultralytics import YOLO
    m = YOLO(path)
    names = {int(k): str(v) for k, v in (m.names or {}).items()}
    fire_i = smoke_i = None
    for i, nm in names.items():
        low = nm.strip().lower()
        if fire_i is None and low in FIRE_NAMES:
            fire_i = i
        if smoke_i is None and low in SMOKE_NAMES:
            smoke_i = i
    imgsz = 640
    try:
        ck = m.ckpt if isinstance(m.ckpt, dict) else {}
        ta = ck.get("train_args") or {}
        if isinstance(ta, dict) and ta.get("imgsz"):
            imgsz = int(ta["imgsz"])
    except Exception:  # noqa: BLE001 - legacy ckpts may lack it
        pass
    nc = getattr(getattr(m, "model", None), "nc", len(names))
    return {"model": m, "names": names, "fire_i": fire_i, "smoke_i": smoke_i,
            "imgsz": imgsz, "nc": nc}


def score_model(model, rows, fire_i, smoke_i, imgsz, batch, conf_floor):
    """Return {bench_path: (fire_score, smoke_score, nbox)} for every manifest row."""
    paths = [os.path.join(ROOT, r["bench_path"]) for r in rows]
    out = {}
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        res = model.predict(chunk, imgsz=imgsz, conf=conf_floor, iou=0.7,
                            max_det=300, verbose=False, device="cpu")
        for p, r in zip(chunk, res):
            f = s = 0.0
            nbox = 0
            b = getattr(r, "boxes", None)
            if b is not None and len(b):
                nbox = int(len(b))
                cls = b.cls.cpu().numpy().astype(int)
                cf = b.conf.cpu().numpy()
                if fire_i is not None:
                    m = cls == fire_i
                    f = float(cf[m].max()) if m.any() else 0.0
                if smoke_i is not None:
                    m = cls == smoke_i
                    s = float(cf[m].max()) if m.any() else 0.0
            out[os.path.relpath(p, ROOT)] = (f, s, nbox)
        step("      scored %d/%d" % (min(i + batch, len(paths)), len(paths)))
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _subset(rows, cat):
    return [r for r in rows if r["category"] == cat]


def rate(rows, pred, thr):
    if not rows:
        return None
    hit = sum(1 for r in rows if pred(r, thr))
    return hit / len(rows)


def category_metrics(scored, rows, thr):
    """scored: {bench_path: (fire, smoke, nbox)}. Returns a metrics dict."""
    def P(r):
        return scored[r["bench_path"]]

    fs = _subset(rows, "fire_smoke")
    fo = _subset(rows, "fire_only")
    so = _subset(rows, "smoke_only")
    ot = _subset(rows, "other")

    fire_imgs = fs + fo
    smoke_imgs = fs + so
    return {
        "n_fire_smoke": len(fs), "n_fire_only": len(fo),
        "n_smoke_only": len(so), "n_other": len(ot),
        "fire_recall_fire_only": rate(fo, lambda r, t: P(r)[0] >= t, thr),
        "fire_recall_fire_smoke": rate(fs, lambda r, t: P(r)[0] >= t, thr),
        "smoke_recall_smoke_only": rate(so, lambda r, t: P(r)[1] >= t, thr),
        "smoke_recall_fire_smoke": rate(fs, lambda r, t: P(r)[1] >= t, thr),
        "both_recall_fire_smoke": rate(fs, lambda r, t: P(r)[0] >= t and P(r)[1] >= t, thr),
        "fire_recall_all_fire_images": rate(fire_imgs, lambda r, t: P(r)[0] >= t, thr),
        "smoke_recall_all_smoke_images": rate(smoke_imgs, lambda r, t: P(r)[1] >= t, thr),
        "fp_any_other": rate(ot, lambda r, t: P(r)[0] >= t or P(r)[1] >= t, thr),
        "fp_fire_other": rate(ot, lambda r, t: P(r)[0] >= t, thr),
        "fp_smoke_other": rate(ot, lambda r, t: P(r)[1] >= t, thr),
    }


def fmt_pct(v):
    return "—" if v is None else "%.1f%%" % (100.0 * v)


def md_table(headers, rows, bold_cols=()):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    lines = ["| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            s = str(cell)
            if i in bold_cols and s != "—":
                s = "**%s**" % s
            cells.append(s.ljust(widths[i]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="*", default=None,
                    help="tags/paths (default: all discovered, md5-deduped)")
    ap.add_argument("--conf", type=float, default=0.5, help="headline threshold [0.5]")
    ap.add_argument("--thresholds", default="0.25,0.35,0.5,0.7")
    ap.add_argument("--imgsz", type=int, default=0,
                    help="force one inference size for all models; 0 = each ckpt's own [0]")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf-floor", type=float, default=0.001,
                    help="ultralytics conf floor used to keep raw per-class maxima [0.001]")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    models = discover_models(args.models)
    if args.list_models:
        for m in models:
            print("%-26s %-8s md5=%s  %s" % (m["tag"], m["role"], m["md5"][:12],
                                             os.path.relpath(m["pt"], ROOT)))
        return
    if not models:
        sys.exit("no models found")

    man_path = args.manifest if os.path.isabs(args.manifest) \
        else os.path.join(ROOT, args.manifest)
    if not os.path.isfile(man_path):
        sys.exit("manifest not found: %s\nRun build_fire_benchmark200.py first." % man_path)
    with open(man_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    step("manifest: %d images (%s)" % (
        len(rows), ", ".join("%s=%d" % (c, sum(1 for r in rows if r["category"] == c))
                             for c in CATEGORIES)))

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    if args.conf not in thresholds:
        thresholds = sorted(set(thresholds + [args.conf]))

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    us = {}
    na = []
    step("loading + probing %d checkpoint(s) ..." % len(models))
    for m in models:
        info = load_probe(m["pt"])
        m.update(info)
        if info["fire_i"] is None and info["smoke_i"] is None:
            na.append(m)
            step("  %-26s NO fire/smoke class (nc=%s) -> N/A" % (m["tag"], info["nc"]))
        else:
            us[m["tag"]] = m
            step("  %-26s nc=%s fire_i=%s smoke_i=%s imgsz=%s"
                 % (m["tag"], info["nc"], info["fire_i"], info["smoke_i"],
                    args.imgsz or info["imgsz"]))

    scored_all = {}
    for tag, m in us.items():
        cell_dir = os.path.join(out_dir, tag)
        os.makedirs(cell_dir, exist_ok=True)
        cache = os.path.join(cell_dir, "per_image.csv")
        if os.path.isfile(cache) and not args.force:
            step("[cached] %s <- %s" % (tag, cache))
            with open(cache, newline="", encoding="utf-8") as fh:
                scored = {r["bench_path"]: (float(r["fire"]), float(r["smoke"]), int(r["nbox"]))
                          for r in csv.DictReader(fh)}
            scored_all[tag] = scored
            continue
        isz = args.imgsz or m["imgsz"]
        step("[run   ] %s  (imgsz=%d, %s)" % (tag, isz, os.path.relpath(m["pt"], ROOT)))
        scored = score_model(m["model"], rows, m["fire_i"], m["smoke_i"], isz,
                             args.batch, args.conf_floor)
        scored_all[tag] = scored
        with open(cache, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["category", "source", "bench_path", "gt_fire", "gt_smoke",
                        "fire", "smoke", "nbox"])
            for r in rows:
                f, s, nb = scored[r["bench_path"]]
                w.writerow([r["category"], r["source"], r["bench_path"], r["gt_fire"],
                            r["gt_smoke"], "%.5f" % f, "%.5f" % s, nb])

    # ---- summary.csv ----
    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "threshold"] + list(category_metrics({}, [], 0.5).keys()))
        for tag in us:
            for thr in thresholds:
                mt = category_metrics(scored_all[tag], rows, thr)
                w.writerow([tag, thr] + ["" if mt[k] is None else "%.4f" % mt[k]
                                         for k in mt])
    step("wrote %s" % summary_path)

    # ---- report ----
    L = []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L.append("# Fire model — 200-image pretrained-vs-V4 benchmark\n")
    L.append("**Generated:** %s · **manifest:** `%s` (%d imgs) · **headline conf:** %.2f · "
             "**harness:** `dev_scripts/compare_pretrained_fire_models.py`\n"
             % (now, os.path.relpath(man_path, ROOT), len(rows), args.conf))
    L.append("> Image-level comparison across the 4 benchmark categories. Each model runs at "
             "its own training imgsz (unless `--imgsz`); scores are the max class confidence "
             "on the image. See [`plans/fire-model-pretrained-200-benchmark.md`]"
             "(../plans/fire-model-pretrained-200-benchmark.md).\n")

    # models table
    L.append("\n## Models\n")
    mh = ["Model", "Role", "Checkpoint", "md5", "nc", "classes", "imgsz", "Status"]
    mrows = []
    for m in models:
        if m["tag"] in us:
            status = "scored"
            cls = ", ".join("%d:%s" % (i, n) for i, n in sorted(m["names"].items()))
            isz = args.imgsz or m["imgsz"]
        else:
            status = "N/A (no fire/smoke class)"
            cls = ("%d COCO classes" % m["nc"]) if m["nc"] and m["nc"] > 10 else \
                ", ".join("%d:%s" % (i, n) for i, n in sorted(m["names"].items()))
            isz = "—"
        tag = m["tag"] + (" *" if m["role"] == "ACTIVE" else "")
        if m["aliases"]:
            tag += " (= %s)" % ", ".join(os.path.basename(a) for a in m["aliases"])
        mrows.append([tag, m["role"], os.path.relpath(m["pt"], ROOT), m["md5"][:12],
                      m["nc"], cls, isz, status])
    L.append(md_table(mh, mrows) + "\n")

    # benchmark composition
    L.append("\n## Benchmark composition\n")
    ch = ["Category", "N"] + sorted({r["source"] for r in rows})
    crows = []
    for c in CATEGORIES:
        sub = _subset(rows, c)
        counts = collections.Counter(r["source"] for r in sub)
        crows.append([c, len(sub)] + [counts.get(s, 0) for s in sorted({r["source"] for r in rows})])
    L.append(md_table(ch, crows) + "\n")

    # headline table at args.conf
    L.append("\n## Headline (conf ≥ %.2f)\n" % args.conf)
    hh = ["Model", "fire recall\nfire-only", "fire recall\nfire+smoke",
          "smoke recall\nsmoke-only", "smoke recall\nfire+smoke",
          "both recall\nfire+smoke", "FP any\n(other)", "FP fire\n(other)", "FP smoke\n(other)"]
    hrows = []
    for tag in us:
        mt = category_metrics(scored_all[tag], rows, args.conf)
        hrows.append([tag,
                      fmt_pct(mt["fire_recall_fire_only"]),
                      fmt_pct(mt["fire_recall_fire_smoke"]),
                      fmt_pct(mt["smoke_recall_smoke_only"]),
                      fmt_pct(mt["smoke_recall_fire_smoke"]),
                      fmt_pct(mt["both_recall_fire_smoke"]),
                      fmt_pct(mt["fp_any_other"]),
                      fmt_pct(mt["fp_fire_other"]),
                      fmt_pct(mt["fp_smoke_other"])])
    L.append(md_table(hh, hrows) + "\n")
    L.append("Higher is better for every column **except** `FP …` (lower is better).\n")

    # combined recall + threshold sweep
    L.append("\n## Threshold sweep (recall over all fire / all smoke images, and FP on other)\n")
    for thr in thresholds:
        L.append("### conf ≥ %.2f\n" % thr)
        th = ["Model", "fire recall\n(all fire imgs)", "smoke recall\n(all smoke imgs)",
              "FP any\n(other)"]
        trows = []
        vals = {"f": [], "s": [], "o": []}
        for tag in us:
            mt = category_metrics(scored_all[tag], rows, thr)
            trows.append([tag, fmt_pct(mt["fire_recall_all_fire_images"]),
                          fmt_pct(mt["smoke_recall_all_smoke_images"]),
                          fmt_pct(mt["fp_any_other"])])
        L.append(md_table(th, trows) + "\n")

    # N/A models
    if na:
        L.append("\n## Not scored — no fire/smoke class\n")
        L.append("These checkpoints are COCO **base** weights (80 classes); they cannot detect "
                 "fire/smoke (COCO has only “fire hydrant”), so scoring them would be 0 by "
                 "construction:\n")
        L.append(md_table(["Model", "Checkpoint", "md5", "nc"],
                          [[m["tag"], os.path.relpath(m["pt"], ROOT), m["md5"][:12], m["nc"]]
                           for m in na]) + "\n")

    L.append("\n## Caveats\n")
    L.append("- `unknown-best`/`jaymak-best` were trained on an **unknown** `/content/data.yaml`, "
             "so benchmark/training overlap cannot be excluded; the rest are scored on held-out "
             "test splits only.\n")
    L.append("- `fire+smoke` recall needs BOTH classes ≥ conf on the same image (strict); the "
             "per-class columns are the fairer read.\n")
    L.append("- Image-level max confidence (no NMS-threshold tuning) — a domain probe, not a box "
             "mAP benchmark. For box mAP use `compare_fire_models.py`.\n")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    step("wrote report: %s" % report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
