#!/usr/bin/env python3
"""compare_fire_models.py - cross-version, cross-suite fire model comparison.

Runs every selected model version against every registered test suite using the
canonical scorer `dev_scripts/test_fire_model.py` (the exact harness that produced
every historical per-version number), then assembles a *matrix* report so that no
single test suite can silently pick a weaker model that merely overfits one benchmark
(see plans/fire-model-cross-version-eval.md).

Each cell (model x suite) is cached under the run output dir; later fine-tunes only
compute new cells unless --force.

Usage:
    .venv/bin/python dev_scripts/compare_fire_models.py \
        [--models v1 v2 v4 active] [--suites generic200 negatives430 abonia25 dfire2164] \
        [--conf 0.5] [--imgsz 640] [--force] [--keep-going] [--dry-run] [--out DIR]
    .venv/bin/python dev_scripts/compare_fire_models.py --list-models
    .venv/bin/python dev_scripts/compare_fire_models.py --list-suites

Model tokens: 'active' (models/fire/best.pt), an archived version id or unique prefix
(e.g. v1 / v4-2026-09-07-...), or an explicit path to a .pt. Default = ACTIVE + every
archived version (md5-deduped, so the active copy of an archived version is one row).
"""
import argparse
import ast
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSIONS_DIR = os.path.join(ROOT, "models", "fire", "versions")
ACTIVE_PT = os.path.join(ROOT, "models", "fire", "best.pt")
SCORER = os.path.join(ROOT, "dev_scripts", "test_fire_model.py")
DEFAULT_OUT = os.path.join(ROOT, "fire-model-training", "compare", "cross-version")
PREFERRED_PY = os.path.join(ROOT, ".venv", "bin", "python")

# --------------------------------------------------------------------------
# Test suite registry. yaml paths are repo-root relative; class order is the
# model contract fire(0)/other(1)/smoke(2) (D-Fire already remapped).
# --------------------------------------------------------------------------
SUITES = [
    {
        "key": "generic200",
        "yaml": "fire-model-training/eval/data.yaml",
        "title": "Generic internet benchmark (200 imgs, v1-era)",
        "caveat": ("Drawn from the TRAIN split of a Roboflow internet-fire pool that "
                   "overlaps v1's 8,939 base -> NOT generalization-clean. Use as a "
                   "recall/regression sanity suite (did a fine-tune break generic "
                   "web/video fire detection?). Optimistic for web/video-tuned models."),
        "skip_val": False,
    },
    {
        "key": "negatives430",
        "yaml": "fire-model-training/eval/negatives/data.yaml",
        "title": "Pure-background FP suite (430 no-fire images)",
        "caveat": ("Curated background/no-fire web+stock frames, ALL empty-label -> no "
                   "positive GT, so mAP is meaningless and val() is skipped (--skip-val). "
                   "Measures false-alarm discipline: fire/smoke predictions on non-fire "
                   "frames. A recall-maximising model that doubles FP here is NOT more "
                   "robust."),
        "skip_val": True,
    },
    {
        "key": "abonia25",
        "yaml": "fire-model-training/dedup/abonia_clean_eval/data.yaml",
        "title": "Abonia clean held-out test (25 imgs)",
        "caveat": ("Clean (internal + vs-8,939 dedup) CCTV/video 608x608 - the nearest "
                   "thing we hold to farm-CCTV. TINY N=25 -> noisy; prefer fire image "
                   "recall over box mAP."),
        "skip_val": False,
    },
    {
        "key": "dfire2164",
        "yaml": "fire-model-training/dedup/dfire_clean_eval/data.yaml",
        "title": "D-Fire clean held-out test (2,164 imgs)",
        "caveat": ("Clean (internal + vs 8,939 union Abonia-kept dedup), model-order "
                   "remap applied. Largest & cleanest numeric suite, BUT D-Fire train "
                   "built v4, so it favours D-Fire-fine-tuned models by construction."),
        "skip_val": False,
    },
    {
        "key": "cctv48",
        "yaml": "fire-model-training/dedup/cctv_clean_eval/data.yaml",
        "title": "CCTV Smoke & Fire Emergency - synthetic event-level (48 imgs)",
        "caveat": ("100% synthetic (Simuletic) high-angle CCTV, early-stage micro-ignitions "
                   "(bin fires, sidewalk paper, smoldering vegetation) - the closest "
                   "synthetic proxy we hold for the farm-CCTV ground-view deployment "
                   "domain. Event-level clean: 240 near-identical angle frames (same event "
                   "per varN family) collapsed to 1 representative each -> 48 events "
                   "(24 fire + 24 smoke). Ref-dedup vs 8,939 union Abonia-kept removed 0 "
                   "(4 ref-phash flags were coincidental 8x8-dHash collisions, verified by "
                   "low-res MAE ~ baseline). NO 'other(1)' GT (fire/smoke only, remapped "
                   "fire0/smoke2). Synthetic realism != real CCTV: read recall/FP as a "
                   "domain probe, not a numeric benchmark."),
        "skip_val": False,
    },
]

SUITE_KEYS = [s["key"] for s in SUITES]


def repo_path(rel):
    return os.path.join(ROOT, rel)


def suite_by_key(key):
    for s in SUITES:
        if s["key"] == key:
            return s
    raise KeyError(key)


# --------------------------------------------------------------------------
# model discovery
# --------------------------------------------------------------------------
def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _archived_versions():
    found = []
    if not os.path.isdir(VERSIONS_DIR):
        return found
    for d in sorted(os.listdir(VERSIONS_DIR)):
        pt = os.path.join(VERSIONS_DIR, d, "model.pt")
        if os.path.isfile(pt):
            found.append((d, pt))
    return found


def _version_tag(version_id):
    """Short stable tag (e.g. v1); falls back to the full id on collision."""
    tag = version_id.split("-", 1)[0]
    return tag or version_id


def discover_models(tokens=None):
    """Return ordered model dicts: {tag,id,pt,md5,active,aliases}.

    Dedups by md5 (an active best.pt that is a copy of an archived version becomes
    one row with the active alias noted). Order: archived ascending by tag, then
    active if it is a distinct model.
    """
    cands = []  # (tag, id_, pt, active, md5)
    seen_md5 = {}
    for version_id, pt in _archived_versions():
        md5 = md5_file(pt)
        m = {"tag": _version_tag(version_id), "id": version_id, "pt": pt,
             "md5": md5, "active": False, "aliases": []}
        cands.append(m)
        seen_md5[md5] = m
    if os.path.isfile(ACTIVE_PT):
        amd5 = md5_file(ACTIVE_PT)
        if amd5 in seen_md5:
            seen_md5[amd5]["active"] = True
            seen_md5[amd5]["aliases"].append("active(models/fire/best.pt)")
        else:
            cands.append({"tag": "active", "id": "ACTIVE", "pt": ACTIVE_PT,
                          "md5": amd5, "active": True, "aliases": []})

    if not tokens:
        return cands

    selected = []
    for tok in tokens:
        tok = tok.strip()
        if tok in ("active", "best"):
            for m in cands:
                if m["pt"] == ACTIVE_PT or m["active"]:
                    m = dict(m)
                    if m not in selected:
                        selected.append(m)
                    break
            else:
                sys.exit(f"model token '{tok}': ACTIVE models/fire/best.pt not found")
            continue
        if os.path.isfile(tok) and tok.endswith(".pt"):
            rp = os.path.realpath(tok)
            for m in cands:
                if os.path.realpath(m["pt"]) == rp:
                    m = dict(m)
                    if m not in selected:
                        selected.append(m)
                    break
            else:
                md5 = md5_file(tok)
                selected.append({"tag": os.path.basename(tok), "id": tok, "pt": tok,
                                 "md5": md5, "active": False, "aliases": []})
            continue
        matches = [m for m in cands
                   if m["id"] == tok or m["id"].startswith(tok) or m["tag"] == tok]
        if not matches:
            sys.exit(f"model token '{tok}' matched nothing "
                     f"(try --list-models; tokens are 'active' or an archived id/prefix)")
        m = matches[0]
        m = dict(m)
        if m not in selected:
            selected.append(m)
    # dedup by md5 preserving order
    out, seen = [], set()
    for m in selected:
        if m["md5"] not in seen:
            out.append(m)
            seen.add(m["md5"])
    return out


# --------------------------------------------------------------------------
# cell execution (thin wrapper around the canonical scorer)
# --------------------------------------------------------------------------
def interpreter():
    if os.path.isfile(PREFERRED_PY):
        return PREFERRED_PY
    return sys.executable


def run_cell(model, suite, conf, imgsz, cell_dir, force=False):
    """Run test_fire_model.py for one model x suite into cell_dir (if not cached)."""
    metrics_path = os.path.join(cell_dir, "metrics.json")
    if os.path.isfile(metrics_path) and not force:
        print(f"  [cached] {model['tag']:7s} x {suite['key']:<11s} <- {metrics_path}")
        return True
    os.makedirs(cell_dir, exist_ok=True)
    cmd = [interpreter(), SCORER, model["pt"], repo_path(suite["yaml"]),
           "--conf", str(conf), "--imgsz", str(imgsz), "--out", cell_dir]
    if suite.get("skip_val"):
        cmd.append("--skip-val")
    print(f"  [run   ] {model['tag']:7s} x {suite['key']:<11s} "
          f"({os.path.basename(model['pt'])})")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"    ! FAILED rc={proc.returncode} after {dt:.0f}s")
        print("    stdout tail:", "\n".join(proc.stdout.splitlines()[-8:]))
        print("    stderr tail:", "\n".join(proc.stderr.splitlines()[-8:]))
        return False
    print(f"    ok in {dt:.0f}s")
    return True


def load_cell(cell_dir):
    p = os.path.join(cell_dir, "metrics.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def per_class_mAP(metrics, index):
    pc = metrics.get("per_class") or {}
    for nm, info in pc.items():
        if isinstance(info, dict) and info.get("index") == index:
            return info.get("mAP50")
    return None


def _pair(v):
    """metrics.json stores FP tuples via default=str -> parse "(a, b)" / [a,b] / tuple."""
    if isinstance(v, (tuple, list)) and len(v) == 2:
        try:
            return int(v[0]), int(v[1])
        except (TypeError, ValueError):
            return None, None
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("(") or s.startswith("["):
            try:
                p = ast.literal_eval(s)
                return int(p[0]), int(p[1])
            except (ValueError, SyntaxError):
                return None, None
    return None, None


def extract_cell(metrics):
    if metrics is None:
        return None
    dr = metrics.get("detection_rate") or {}
    fp_fire = _pair(dr.get("fp_fire_images_on_nonfire"))
    fp_smoke = _pair(dr.get("fp_smoke_images_on_nonsmoke"))
    return {
        "mAP_all": metrics.get("mAP50"),
        "mAP_fire": per_class_mAP(metrics, 0),
        "mAP_smoke": per_class_mAP(metrics, 2),
        "rec_fire": dr.get("fire_image_recall"),
        "rec_smoke": dr.get("smoke_image_recall"),
        "fp_fire": fp_fire,
        "fp_smoke": fp_smoke,
        "images": dr.get("images"),
    }


# --------------------------------------------------------------------------
# report helpers
# --------------------------------------------------------------------------
def fmt_num(v, decimals=3):
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def fmt_pct(v):
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def fmt_pair(p):
    if p is None or p[0] is None:
        return "—"
    return f"{p[0]} / {p[1]}"


def best_marker():
    return "**"


def md_table(headers, rows, best_indexes):
    """headers: list of str; rows: list of list-of-display-str; best_indexes: set of
    (row,col) whose cell is the best value -> wrap in **bold**."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = []
    lines.append("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for ri, row in enumerate(rows):
        cells = []
        for ci, cell in enumerate(row):
            if (ri, ci) in best_indexes and cell != "—":
                cell = f"**{cell}**"
            cells.append(cell.ljust(widths[ci]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(cells, models, suites, args, run_dir):
    """cells: {(model_tag, suite_key): metrics}. Returns report text + csv files."""
    L = []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L.append("# Fire model — cross-version, cross-suite comparison\n")
    L.append(f"**Generated:** {now}  ·  **conf:** {args.conf}  ·  **imgsz:** {args.imgsz}  ·  "
             f"**harness:** `dev_scripts/test_fire_model.py` (identical for every cell)\n")
    L.append("> Each model is scored against **every** test suite, because judging a model "
             "on a single suite risks promoting a weaker model that merely fits that one "
             "benchmark. No automated winner is emitted — read the matrix + caveats.\n")

    # --- models ---
    L.append("\n## Models evaluated\n")
    mh = ["Model", "Archived version / source", "md5", "Status"]
    mrows = []
    for m in models:
        status = "ACTIVE" if m["active"] else "candidate"
        if m["aliases"]:
            status += " (= " + ", ".join(m["aliases"]) + ")"
        mrows.append([m["tag"], m["id"], m["md5"][:12], status])
    L.append(md_table(mh, mrows, set()) + "\n")

    # --- suites ---
    L.append("\n## Test suites\n")
    sh = ["Key", "Title", "Images (per cell)", "val()", "Caveat"]
    srows = []
    for s in suites:
        # images from the first model's cell for this suite
        n = None
        for m in models:
            c = extract_cell(cells.get((m["tag"], s["key"])))
            if c and c.get("images"):
                n = c["images"]
                break
        srows.append([s["key"], s["title"], str(n) if n else "—",
                      "no (FP-only)" if s.get("skip_val") else "yes", s["caveat"]])
    L.append(md_table(sh, srows, set()) + "\n")

    # --- per-suite metric tables ---
    # column defs: (header, getter, display-fn, lower_is_better)
    col_fns = [
        ("all mAP@50", lambda c: c["mAP_all"], lambda v: fmt_num(v, 3), False),
        ("fire mAP@50", lambda c: c["mAP_fire"], lambda v: fmt_num(v, 3), False),
        ("smoke mAP@50", lambda c: c["mAP_smoke"], lambda v: fmt_num(v, 3), False),
        ("fire img recall @conf", lambda c: c["rec_fire"], fmt_pct, False),
        ("smoke img recall @conf", lambda c: c["rec_smoke"], fmt_pct, False),
        ("fire FP", lambda c: c["fp_fire"][0] if c and c["fp_fire"] else None,
         lambda v: None, True),
        ("smoke FP", lambda c: c["fp_smoke"][0] if c and c["fp_smoke"] else None,
         lambda v: None, True),
    ]

    for s in suites:
        L.append(f"\n## Suite: `{s['key']}` — {s['title']}\n")
        L.append(f"`{repo_path(s['yaml'])}`\n")
        L.append(f"_{s['caveat']}_\n")
        headers = ["Model"] + [c[0] for c in col_fns]
        rows, numeric = [], {}
        for m in models:
            cell = extract_cell(cells.get((m["tag"], s["key"])))
            row = [m["tag"] + ("*" if m["active"] else "")]
            if cell is None:
                rows.append(row + ["—"] * len(col_fns))
                continue
            for name, get, disp, lower in col_fns:
                v = get(cell)
                if name.startswith("fire FP"):
                    row.append(fmt_pair(cell["fp_fire"]))
                elif name.startswith("smoke FP"):
                    row.append(fmt_pair(cell["fp_smoke"]))
                else:
                    row.append(disp(v) if v is not None else "—")
                numeric[name] = numeric.get(name, []) + [v]
            rows.append(row)
        # best per numeric column (bold), ignoring None and empty FP pairs
        best_idx = set()
        for name, get, disp, lower in col_fns:
            vals = numeric.get(name, [])
            finite = [v for v in vals if v is not None]
            if not finite:
                continue
            target = (min if lower else max)(finite)
            for ri, m in enumerate(models):
                cell = extract_cell(cells.get((m["tag"], s["key"])))
                if cell is None:
                    continue
                v = get(cell)
                if v is not None and v == target:
                    best_idx.add((ri, col_fns.index((name, get, disp, lower)) + 1))
        L.append(md_table(headers, rows, best_idx) + "\n")

    # --- headline deploy-view matrix ---
    L.append("\n## Headline deploy-view matrix\n")
    L.append("Fire image recall @conf per generalisation suite + FP discipline on the "
             "background suite. `*` = ACTIVE.\n")
    hh = ["Model"] + [f"`{s['key']}` fire rec" for s in suites if not s.get("skip_val")] \
         + ["`negatives430` fire FP rate"]
    hrows = []
    num_cols = {}
    for m in models:
        row = [m["tag"] + ("*" if m["active"] else "")]
        for s in suites:
            if s.get("skip_val"):
                continue
            c = extract_cell(cells.get((m["tag"], s["key"])))
            row.append(fmt_pct(c["rec_fire"]) if c else "—")
            key = f"{s['key']}|rec"
            num_cols.setdefault(key, []).append(c["rec_fire"] if c else None)
        cneg = extract_cell(cells.get((m["tag"], "negatives430")))
        if cneg and cneg["fp_fire"] and cneg["fp_fire"][1]:
            rate = cneg["fp_fire"][0] / cneg["fp_fire"][1]
            row.append(f"{rate * 100:.1f}%")
            num_cols.setdefault("neg|fprate", []).append(rate)
        else:
            row.append("—")
            num_cols.setdefault("neg|fprate", []).append(None)
        hrows.append(row)
    # bold best per column
    best_idx = set()
    col_keys = [f"{s['key']}|rec" for s in suites if not s.get("skip_val")] + ["neg|fprate"]
    for ci, key in enumerate(col_keys, start=1):
        vals = [v for v in num_cols.get(key, []) if v is not None]
        if not vals:
            continue
        lower = key == "neg|fprate"
        target = (min if lower else max)(vals)
        for ri, m in enumerate(models):
            v = num_cols[key][ri]
            if v is not None and v == target:
                best_idx.add((ri, ci))
    L.append(md_table(hh, hrows, best_idx) + "\n")

    # --- wins tally + guidance ---
    L.append("\n## Wins tally (per numeric column, ties counted for all) — *not* a verdict\n")
    tally = {m["tag"]: 0 for m in models}
    for s in suites:
        pass
    # recompute best per column across suite tables + headline, count models that are best
    col_best_counts = []
    # reuse suite numeric columns:
    for s in suites:
        for name, get, disp, lower in col_fns:
            vals, owners = [], []
            for m in models:
                c = extract_cell(cells.get((m["tag"], s["key"])))
                v = get(c) if c else None
                vals.append(v)
                owners.append(m["tag"])
            finite = [v for v in vals if v is not None]
            if not finite:
                continue
            target = (min if lower else max)(finite)
            for i, v in enumerate(vals):
                if v is not None and v == target:
                    tally[owners[i]] += 1
    for ci, key in enumerate(col_keys):
        vals = num_cols.get(key, [])
        finite = [v for v in vals if v is not None]
        if not finite:
            continue
        target = (min if key == "neg|fprate" else max)(finite)
        for ri, m in enumerate(models):
            if vals[ri] is not None and vals[ri] == target:
                tally[m["tag"]] += 1
    wh = ["Model", "Best-in-column count (incl. ties)"]
    wrows = [[m["tag"] + ("*" if m["active"] else ""), str(tally[m["tag"]])] for m in models]
    L.append(md_table(wh, wrows, set()) + "\n")
    L.append("The tally just flags which models excel per metric column; a model that "
             "wins **only `dfire2164`** (its own training pool) while losing "
             "`generic200`/`abonia25` or doubling negatives-FP is the *fits-the-suite* "
             "trap, not a robustness win.\n")

    L.append("\n## How to re-run\n")
    L.append("```bash\n.venv/bin/python dev_scripts/compare_fire_models.py "
             "--models " + " ".join(m["tag"] for m in models) +
             " --suites " + " ".join(s["key"] for s in suites) +
             f" --conf {args.conf} --imgsz {args.imgsz} --out {run_dir}\n```\n")
    L.append("\n---\n_Plan: plans/fire-model-cross-version-eval.md. Cell outputs (per-image "
             "CSVs, val runs) are under this directory per model/suite._\n")
    report = "\n".join(L)

    # --- CSV exports ---
    write_csvs(run_dir, models, suites, cells)

    return report


def write_csvs(run_dir, models, suites, cells):
    def matrix(metric, fmt):
        header = ["suite/model"] + [m["tag"] for m in models]
        rows = []
        for s in suites:
            row = [s["key"]]
            for m in models:
                c = extract_cell(cells.get((m["tag"], s["key"])))
                row.append(fmt(c) if c else "")
            rows.append(row)
        path = os.path.join(run_dir, f"matrix_{metric}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        return path

    paths = []
    paths.append(matrix("fire_image_recall",
                        lambda c: f"{c['rec_fire']:.4f}" if c.get("rec_fire") is not None else ""))
    paths.append(matrix("smoke_image_recall",
                        lambda c: f"{c['rec_smoke']:.4f}" if c.get("rec_smoke") is not None else ""))
    paths.append(matrix("fire_mAP50",
                        lambda c: f"{c['mAP_fire']:.4f}" if c.get("mAP_fire") is not None else ""))
    paths.append(matrix("fire_fp_count",
                        lambda c: str(c["fp_fire"][0]) if c and c["fp_fire"] and c["fp_fire"][0] is not None else ""))
    return paths


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="compare_fire_models.py - cross-version, cross-suite fire model "
                    "comparison (see plans/fire-model-cross-version-eval.md)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="model tokens: 'active', an archived id/prefix, or a .pt path "
                         "(default: ACTIVE + all archived versions)")
    ap.add_argument("--suites", nargs="*", default=None,
                    help="suite keys (default: all): " + ", ".join(SUITE_KEYS))
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=None, help="run output dir (git-ignored)")
    ap.add_argument("--force", action="store_true", help="recompute cached cells")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue other cells if one cell fails")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--list-suites", action="store_true")
    args = ap.parse_args()

    if args.list_suites:
        for s in SUITES:
            print(f"{s['key']:<12s} {s['title']}  [yaml: {s['yaml']}]")
        return

    models = discover_models(args.models)
    if args.list_models:
        for m in models:
            print(f"{m['tag']:<8s} {m['id']}  md5={m['md5'][:12]}"
                  + ("  ACTIVE" if m["active"] else "")
                  + (f"  (= {', '.join(m['aliases'])})" if m["aliases"] else ""))
        return
    if not models:
        sys.exit("no models found (no versions under models/fire/versions/ and no ACTIVE "
                 "models/fire/best.pt)")

    suites = [suite_by_key(k) for k in (args.suites or SUITE_KEYS)]
    for k in args.suites or []:
        if k not in SUITE_KEYS:
            sys.exit(f"unknown suite '{k}' (known: {', '.join(SUITE_KEYS)})")

    run_dir = args.out or (DEFAULT_OUT + "-"
                           + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print("models :", ", ".join(f"{m['tag']}({m['id']})" for m in models))
    print("suites :", ", ".join(s["key"] for s in suites))
    print("out    :", run_dir)
    total = len(models) * len(suites)
    print(f"cells  : {total}  (conf={args.conf}, imgsz={args.imgsz})\n")
    if args.dry_run:
        for m in models:
            for s in suites:
                print(f"  would-run {m['tag']:7s} x {s['key']:<11s}")
        return

    os.makedirs(run_dir, exist_ok=True)
    cells = {}
    failed = []
    for m in models:
        for s in suites:
            cell_dir = os.path.join(run_dir, m["tag"], s["key"])
            ok = run_cell(m, s, args.conf, args.imgsz, cell_dir, force=args.force)
            if not ok:
                failed.append((m["tag"], s["key"]))
                if not args.keep_going:
                    sys.exit(f"cell failed: {m['tag']} x {s['key']}")
            else:
                metrics = load_cell(cell_dir)
                if metrics is None:
                    failed.append((m["tag"], s["key"]))
                    print(f"    ! no metrics.json produced for {m['tag']} x {s['key']}")
                else:
                    cells[(m["tag"], s["key"])] = metrics

    if failed and not args.keep_going:
        sys.exit(f"{len(failed)} cell(s) failed: {failed}")

    report = build_report(cells, models, suites, args, run_dir)
    report_path = os.path.join(run_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"\nwrote report: {report_path}")
    for f in sorted(os.listdir(run_dir)):
        if f.startswith("matrix_"):
            print(f"wrote csv    : {os.path.join(run_dir, f)}")


if __name__ == "__main__":
    main()
