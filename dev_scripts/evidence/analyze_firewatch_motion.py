#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from firewatch.firewatch import _adaptive_px, _changed_mask, _region_stats

EPS = 0.005
CATEGORIES = (
    "bonus_false_fixed",
    "bonus_true_missed",
    "bonus_false_survives",
    "bonus_true_rescued",
    "candidate_motion_gain",
)


def load_labels(path):
    return {r["key"]: r["verdict"] for r in csv.DictReader(open(path))}


def load_cohorts(path):
    return {r["key"]: r["cohort"] for r in csv.DictReader(open(path))}


def load_jsonl(path):
    out = {}
    for line in open(path):
        row = json.loads(line)
        out[row["key"]] = row
    return out


def num(value):
    if isinstance(value, (bytes, bytearray)):
        if len(value) >= 4:
            return struct.unpack("<f", bytes(value[:4]))[0]
        return 0.0
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
          + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter)
    return inter / ua if ua else 0.0


def pct(value):
    return f"{100.0 * value:.1f}%"


def metrics(rows, scorer, threshold):
    tp = fp = fn = tn = 0
    for row in rows:
        positive = row["label"] == "true"
        value = scorer(row)
        if value >= threshold:
            if positive:
                tp += 1
            else:
                fp += 1
        elif positive:
            fn += 1
        else:
            tn += 1
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall, "fpr": fpr, "precision": precision, "f1": f1,
    }


def best_f1(rows, scorer):
    best = (0.0, 0.0)
    for step in range(1, 100):
        threshold = step / 100.0
        result = metrics(rows, scorer, threshold)
        if (result["f1"] > best[0] + 1e-12
                or (abs(result["f1"] - best[0]) <= 1e-12
                    and threshold > best[1])):
            best = (result["f1"], threshold)
    return best


def auc(rows, scorer):
    pairs = sorted((scorer(row), 1 if row["label"] == "true" else 0)
                   for row in rows)
    positives = sum(y for _, y in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return 0.0
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        rank_sum += avg_rank * sum(y for _, y in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives)


def average_precision(rows, scorer):
    pairs = sorted(((scorer(row), row["label"] == "true") for row in rows),
                   key=lambda item: item[0], reverse=True)
    positives = sum(1 for _, y in pairs if y)
    if not positives:
        return 0.0
    tp = fp = 0
    total = 0.0
    for _, positive in pairs:
        if positive:
            tp += 1
            total += tp / (tp + fp)
        else:
            fp += 1
    return total / positives


def load_rows(run_dir, evidence_dir, db_path, gap_max, bonus, margin,
              near_frac, near_min_px, px_diff, night_extra, night_luma,
              blur_radius):
    labels = load_labels(run_dir / "labels_snapshot.csv")
    cohorts = load_cohorts(run_dir / "cohorts.csv")
    active = load_jsonl(run_dir / "predictions_active.jsonl")
    candidate = load_jsonl(run_dir / "predictions_candidate.jsonl")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = []
    sql = (
        "SELECT f.id, f.camera, f.captured_at, f.best_score, f.alerted, "
        "f.jpg_path, MAX(d.score) AS raw "
        "FROM frames f JOIN detections d ON d.frame_id = f.id "
        "GROUP BY f.id ORDER BY f.camera, f.captured_at"
    )
    for frame in conn.execute(sql):
        key = frame["jpg_path"].split("/media/firewatch/", 1)[-1]
        if key not in labels or cohorts.get(key) != "core":
            continue
        detections = []
        for det in conn.execute(
                "SELECT label, score, x1, y1, x2, y2 FROM detections "
                "WHERE frame_id = ? ORDER BY score DESC", (frame["id"],)):
            detections.append({
                "label": det["label"],
                "score": num(det["score"]),
                "box": tuple(num(det[name])
                             for name in ("x1", "y1", "x2", "y2")),
            })
        rows.append({
            "key": key,
            "frame_id": frame["id"],
            "camera": frame["camera"],
            "captured_at": frame["captured_at"],
            "label": labels[key],
            "alerted": bool(frame["alerted"]),
            "active_db_raw": num(frame["raw"]),
            "active_raw": active[key]["max_fire"],
            "active_eff": num(frame["best_score"]),
            "active_near": num(frame["best_score"]) > num(frame["raw"]) + EPS,
            "db": detections,
            "active_near_proxy": None,
            "mode": None,
            "candidate_raw": 0.0,
            "candidate_eff": 0.0,
            "candidate_near": False,
            "candidate_boxes": [],
        })
    conn.close()

    by_camera = {}
    for row in rows:
        by_camera.setdefault(row["camera"], []).append(row)

    proxy_total = proxy_agree = 0
    proxy_tp = proxy_fp = proxy_fn = 0
    for camera_rows in by_camera.values():
        previous = None
        for row in camera_rows:
            mask = None
            if previous is not None:
                gap = row["captured_at"] - previous["captured_at"]
                a_path = evidence_dir / previous["key"]
                b_path = evidence_dir / row["key"]
                if 0 < gap <= gap_max and a_path.exists() and b_path.exists():
                    a = np.asarray(Image.open(a_path).convert("L"),
                                   dtype=np.uint8)
                    b = np.asarray(Image.open(b_path).convert("L"),
                                   dtype=np.uint8)
                    if a.shape == b.shape:
                        threshold = _adaptive_px(
                            px_diff, night_extra, night_luma, float(b.mean()))
                        mask = _changed_mask(a, b, threshold, blur_radius)

            boxes = candidate[row["key"]]["boxes"]
            if mask is not None:
                near_flags = []
                for box in boxes:
                    frac, count = _region_stats(mask, tuple(box["xy"]), margin)
                    near_flags.append(
                        frac >= near_frac
                        or (near_min_px > 0 and count >= near_min_px))
                row["mode"] = "proxy"
                if row["db"]:
                    frac, count = _region_stats(
                        mask, row["db"][0]["box"], margin)
                    proxy_near = (
                        frac >= near_frac
                        or (near_min_px > 0 and count >= near_min_px))
                    row["active_near_proxy"] = proxy_near
                    proxy_total += 1
                    if proxy_near == row["active_near"]:
                        proxy_agree += 1
                    if proxy_near and row["active_near"]:
                        proxy_tp += 1
                    elif proxy_near:
                        proxy_fp += 1
                    elif row["active_near"]:
                        proxy_fn += 1
            else:
                bonus_box = None
                if row["active_near"]:
                    candidates = []
                    for det in row["db"]:
                        effective = min(1.0, det["score"] + bonus)
                        if (row["active_eff"] > det["score"] + EPS
                                and abs(effective - row["active_eff"]) <= EPS):
                            candidates.append(det)
                    if candidates:
                        bonus_box = max(candidates,
                                        key=lambda det: det["score"])["box"]
                near_flags = [bool(bonus_box)
                              and iou(tuple(box["xy"]), bonus_box) >= 0.3
                              for box in boxes]
                row["mode"] = "fallback"

            effective = [min(1.0, box["s"] + (bonus if near else 0.0))
                         for box, near in zip(boxes, near_flags)]
            row["candidate_raw"] = max(
                [candidate[row["key"]]["max_fire"]]
                + [box["s"] for box in boxes])
            row["candidate_eff"] = max(
                [candidate[row["key"]]["max_fire"]] + effective)
            row["candidate_near"] = any(
                near and value == row["candidate_eff"]
                for near, value in zip(near_flags, effective))
            row["candidate_boxes"] = [
                [round(box["s"], 4), *[round(v, 1) for v in box["xy"]],
                 bool(near)]
                for box, near in zip(boxes, near_flags)
                if box["s"] >= 0.25
            ]
            previous = row

    validation = {
        "frames": proxy_total,
        "agreement": proxy_agree / proxy_total if proxy_total else 0.0,
        "precision": proxy_tp / (proxy_tp + proxy_fp)
        if proxy_tp + proxy_fp else 0.0,
        "recall": proxy_tp / (proxy_tp + proxy_fn)
        if proxy_tp + proxy_fn else 0.0,
        "tp": proxy_tp, "fp": proxy_fp, "fn": proxy_fn,
    }
    return rows, validation


def categories_for(row):
    cats = []
    basic = (row["active_near"] and row["active_raw"] < 0.5)
    if basic and row["label"] == "false" and row["candidate_raw"] < 0.35:
        cats.append("bonus_false_fixed")
    if basic and row["label"] == "true" and row["candidate_raw"] < 0.35:
        cats.append("bonus_true_missed")
    if basic and row["label"] == "false" and row["candidate_eff"] >= 0.5:
        cats.append("bonus_false_survives")
    if basic and row["label"] == "true" and row["candidate_eff"] >= 0.5:
        cats.append("bonus_true_rescued")
    if row["candidate_raw"] < 0.5 <= row["candidate_eff"]:
        cats.append("candidate_motion_gain")
    return cats


def draw_box(draw, box, color, width):
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)


def contact_sheet(rows, category, path, evidence_dir, limit):
    selected = [row for row in rows if category in row["cats"]]
    if len(selected) > limit:
        step = len(selected) / limit
        selected = [selected[int(index * step)] for index in range(limit)]
    if not selected:
        return
    width, height = 360, 203
    columns = 6
    lines = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (width * columns, height * lines), (20, 20, 20))
    draw_sheet = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(selected):
        image = Image.open(evidence_dir / row["key"]).convert("RGB")
        image = image.resize((width, int(width * 360 / 640)))
        draw = ImageDraw.Draw(image)
        for det in row["db"]:
            draw_box(draw, [value * width / 640 for value in det["box"]],
                     (255, 40, 40), 2)
        for box in row["candidate_boxes"]:
            if box[0] >= 0.35:
                draw_box(draw, [value * width / 640 for value in box[1:5]],
                         (40, 255, 40), 2)
        left = (index % columns) * width
        top = (index // columns) * height
        sheet.paste(image, (left, top))
        draw_sheet.rectangle((left, top + height - 34,
                              left + width, top + height), fill=(0, 0, 0))
        draw_sheet.text(
            (left + 4, top + height - 32),
            row["key"].split("/")[-1][:38], fill=(255, 255, 255), font=font)
        draw_sheet.text(
            (left + 4, top + height - 20),
            f"A r{row['active_raw']:.2f}/e{row['active_eff']:.2f}  "
            f"C r{row['candidate_raw']:.2f}/e{row['candidate_eff']:.2f}",
            fill=(210, 210, 210), font=font)
    sheet.save(path, quality=88)


def write_review(rows, path, evidence_dir, out_dir):
    relative = Path(os.path.relpath(evidence_dir, out_dir)).as_posix()
    review_rows = [row for row in rows if row["cats"]]
    counts = {name: sum(name in row["cats"] for row in review_rows)
              for name in CATEGORIES}
    data = [{
        "key": row["key"],
        "src": f"{relative}/{row['key']}",
        "label": row["label"],
        "camera": row["camera"],
        "cats": row["cats"],
        "active_raw": row["active_raw"],
        "active_eff": row["active_eff"],
        "active_near": row["active_near"],
        "candidate_raw": row["candidate_raw"],
        "candidate_eff": row["candidate_eff"],
        "candidate_near": row["candidate_near"],
        "alerted": row["alerted"],
        "active_boxes": [list(det["box"]) for det in row["db"]],
        "candidate_boxes": row["candidate_boxes"],
    } for row in review_rows]
    buttons = "".join(
        f'<button data-f="{name}">{name} ({counts[name]})</button>'
        for name in CATEGORIES
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Firewatch motion review</title>
<style>
body{{background:#111;color:#ddd;font:14px system-ui,sans-serif;margin:16px}}
.bar{{position:sticky;top:0;background:#111;padding:8px 0;z-index:5}}
button{{margin:2px;padding:6px 9px;background:#222;color:#ddd;border:1px solid #555}}
button.on{{background:#365;border-color:#7c9}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}}
.card{{background:#1b1b1b;border:1px solid #333;padding:6px}}
.wrap{{position:relative;width:100%;padding-top:56.25%;background:#000}}
.wrap img{{position:absolute;inset:0;width:100%;height:100%;object-fit:fill}}
.box{{position:absolute;border:2px solid #f33;box-sizing:border-box}}
.box.c{{border-color:#3f3}}
.meta{{font:12px monospace;line-height:1.45;margin-top:5px}}
.true{{color:#7f7}} .false{{color:#f88}} .near{{color:#ff8}}
</style></head><body>
<h2>Firewatch motion replay</h2>
<div class="bar"><button data-f="all" class="on">all motion-relevant ({len(review_rows)})</button>{buttons}</div>
<div class="grid" id="grid"></div>
<script>
const rows={json.dumps(data, separators=(",", ":"))};
const grid=document.getElementById("grid");
function addBox(wrap,box,cls){{
 const d=document.createElement("div"); d.className="box "+cls;
 d.style.left=(100*box[0]/640)+"%"; d.style.top=(100*box[1]/360)+"%";
 d.style.width=(100*(box[2]-box[0])/640)+"%";
 d.style.height=(100*(box[3]-box[1])/360)+"%";
 wrap.appendChild(d);
}}
function render(filter){{
 grid.innerHTML="";
 for(const r of rows){{
  if(filter!=="all" && !r.cats.includes(filter)) continue;
  const card=document.createElement("div"); card.className="card";
  const wrap=document.createElement("div"); wrap.className="wrap";
  const img=document.createElement("img"); img.loading="lazy"; img.src=r.src;
  wrap.appendChild(img);
  for(const b of r.active_boxes) addBox(wrap,b,"");
  for(const b of r.candidate_boxes) addBox(wrap,b.slice(1,5),"c");
  card.appendChild(wrap);
  const m=document.createElement("div"); m.className="meta";
  m.innerHTML=`<span class="${{r.label}}">${{r.label}}</span> ${{r.key}}<br>`+
   `A raw/eff ${{r.active_raw.toFixed(2)}}/${{r.active_eff.toFixed(2)}} `+
   `<span class="near">motion=${{r.active_near}}</span><br>`+
   `C raw/eff ${{r.candidate_raw.toFixed(2)}}/${{r.candidate_eff.toFixed(2)}} `+
   `<span class="near">motion=${{r.candidate_near}}</span><br>`+
   `${{r.cats.join(" ")}}`;
  card.appendChild(m); grid.appendChild(card);
 }}
}}
document.querySelectorAll("button[data-f]").forEach(btn=>btn.onclick=()=>{{
 document.querySelectorAll("button[data-f]").forEach(b=>b.classList.remove("on"));
 btn.classList.add("on"); render(btn.dataset.f);
}});
render("all");
</script></body></html>"""
    path.write_text(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path,
                        default=REPO_ROOT / "firewatch-evidence")
    parser.add_argument("--db", type=Path,
                        default=REPO_ROOT / "firewatch-evidence" / "firewatch.db")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gap-max", type=float, default=30.0)
    parser.add_argument("--bonus", type=float, default=0.15)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--near-frac", type=float, default=0.02)
    parser.add_argument("--near-min-px", type=int, default=15)
    parser.add_argument("--px-diff", type=float, default=18.0)
    parser.add_argument("--night-extra", type=float, default=8.0)
    parser.add_argument("--night-luma", type=float, default=25.0)
    parser.add_argument("--blur-radius", type=float, default=1.0)
    parser.add_argument("--contact-limit", type=int, default=24)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    evidence_dir = args.evidence_dir.resolve()
    out_dir = (args.out or (run_dir / "motion_analysis")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, validation = load_rows(
        run_dir, evidence_dir, args.db, args.gap_max, args.bonus, args.margin,
        args.near_frac, args.near_min_px, args.px_diff, args.night_extra,
        args.night_luma, args.blur_radius)
    for row in rows:
        row["cats"] = categories_for(row)

    scorers = {
        "active_raw": lambda row: row["active_raw"],
        "active_eff": lambda row: row["active_eff"],
        "candidate_raw": lambda row: row["candidate_raw"],
        "candidate_eff": lambda row: row["candidate_eff"],
    }
    thresholds = (0.25, 0.35, 0.43, 0.5, 0.61, 0.65, 0.75)
    table = {
        name: {str(threshold): metrics(rows, scorer, threshold)
               for threshold in thresholds}
        for name, scorer in scorers.items()
    }
    ranking = {}
    for name, scorer in scorers.items():
        f1, threshold = best_f1(rows, scorer)
        ranking[name] = {
            "auc": auc(rows, scorer),
            "ap": average_precision(rows, scorer),
            "best_f1": f1,
            "best_f1_threshold": threshold,
        }

    bonus_negative = [row for row in rows if row["label"] == "false"
                      and row["active_near"] and row["active_raw"] < 0.5]
    bonus_positive = [row for row in rows if row["label"] == "true"
                      and row["active_near"] and row["active_raw"] < 0.5]
    cohort = {
        "bonus_negative": len(bonus_negative),
        "bonus_positive": len(bonus_positive),
        "negatives": sum(row["label"] == "false" for row in rows),
        "positives": sum(row["label"] == "true" for row in rows),
        "bonus_negative_candidate_raw": {
            str(t): sum(row["candidate_raw"] >= t for row in bonus_negative)
            for t in (0.25, 0.35, 0.5)
        },
        "bonus_negative_candidate_eff": {
            str(t): sum(row["candidate_eff"] >= t for row in bonus_negative)
            for t in (0.25, 0.35, 0.5)
        },
        "bonus_positive_candidate_raw": {
            str(t): sum(row["candidate_raw"] >= t for row in bonus_positive)
            for t in (0.25, 0.35, 0.5)
        },
        "bonus_positive_candidate_eff": {
            str(t): sum(row["candidate_eff"] >= t for row in bonus_positive)
            for t in (0.25, 0.35, 0.5)
        },
    }

    contacts = {}
    for category in CATEGORIES:
        contact_path = out_dir / f"contact_{category}.jpg"
        contact_sheet(rows, category, contact_path, evidence_dir,
                      args.contact_limit)
        if contact_path.exists():
            contacts[category] = contact_path.name

    write_review(rows, out_dir / "review_motion.html", evidence_dir, out_dir)

    with open(out_dir / "per_image_motion.csv", "w", newline="") as handle:
        fields = [
            "key", "camera", "captured_at", "label", "alerted",
            "active_raw", "active_eff", "active_near", "active_near_proxy",
            "near_mode", "candidate_raw", "candidate_eff", "candidate_near",
            "categories",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "key": row["key"],
                "camera": row["camera"],
                "captured_at": f"{row['captured_at']:.3f}",
                "label": row["label"],
                "alerted": int(row["alerted"]),
                "active_raw": f"{row['active_raw']:.4f}",
                "active_eff": f"{row['active_eff']:.4f}",
                "active_near": int(row["active_near"]),
                "active_near_proxy": "" if row["active_near_proxy"] is None
                else int(row["active_near_proxy"]),
                "near_mode": row["mode"],
                "candidate_raw": f"{row['candidate_raw']:.4f}",
                "candidate_eff": f"{row['candidate_eff']:.4f}",
                "candidate_near": int(row["candidate_near"]),
                "categories": " ".join(row["cats"]),
            })

    summary = {
        "rows": len(rows),
        "validation": validation,
        "thresholds": thresholds,
        "metrics": table,
        "ranking": ranking,
        "cohorts": cohort,
        "contacts": contacts,
    }
    with open(out_dir / "motion_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    active_best = ranking["active_eff"]
    candidate_raw_best = ranking["candidate_raw"]
    candidate_eff_best = ranking["candidate_eff"]
    a500 = table["active_eff"]["0.5"]
    a610 = table["active_eff"]["0.61"]
    c350 = table["candidate_raw"]["0.35"]
    c500 = table["candidate_raw"]["0.5"]
    ce350 = table["candidate_eff"]["0.35"]
    ce430 = table["candidate_eff"]["0.43"]
    ce500 = table["candidate_eff"]["0.5"]

    lines = [
        "# Firewatch motion replay",
        "",
        "## Scope",
        "",
        f"{len(rows)} core frames: {cohort['positives']} true / "
        f"{cohort['negatives']} false. Mixed frames excluded.",
        "The active score is the stored Firewatch effective score (raw + motion "
        "bonus when the winning box was near motion). The candidate is replayed "
        "with the same spatial motion rule.",
        "",
        "This is a replay on the active model's own evidence, so absolute false "
        "positive rates are not representative of all camera frames. The set is "
        "deliberately enriched with frames the active model already selected.",
        "",
        "## Motion replay validation",
        "",
        f"A prior stored frame within {args.gap_max:g}s was available for "
        f"{validation['frames']} frames. Comparing the image-derived near flag "
        "with the production motion flag: "
        f"agreement {pct(validation['agreement'])}, "
        f"precision {pct(validation['precision'])}, "
        f"recall {pct(validation['recall'])}.",
        "",
        "## Active model: raw vs Firewatch effective",
        "",
        "| score | conf | TP | FP | recall | FP rate | precision | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("active_raw", "raw"), ("active_eff", "effective")):
        for threshold in (0.5, 0.61, 0.65):
            row = table[name][str(threshold)]
            lines.append(
                f"| {label} | {threshold:.2f} | {row['tp']} | {row['fp']} | "
                f"{pct(row['recall'])} | {pct(row['fpr'])} | "
                f"{pct(row['precision'])} | {pct(row['f1'])} |")
    lines += [
        "",
        f"Active raw AUC/AP: {ranking['active_raw']['auc']:.3f} / "
        f"{ranking['active_raw']['ap']:.3f}; best raw F1 "
        f"{pct(ranking['active_raw']['best_f1'])} @ "
        f"{ranking['active_raw']['best_f1_threshold']:.2f}.",
        f"Active effective AUC/AP: {active_best['auc']:.3f} / "
        f"{active_best['ap']:.3f}; best effective F1 "
        f"{pct(active_best['best_f1'])} @ "
        f"{active_best['best_f1_threshold']:.2f}.",
        "",
        "## Candidate: raw vs same motion rule",
        "",
        "| score | conf | TP | FP | recall | FP rate | precision | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("candidate_raw", "raw"),
                        ("candidate_eff", "effective")):
        for threshold in (0.35, 0.43, 0.5):
            row = table[name][str(threshold)]
            lines.append(
                f"| {label} | {threshold:.2f} | {row['tp']} | {row['fp']} | "
                f"{pct(row['recall'])} | {pct(row['fpr'])} | "
                f"{pct(row['precision'])} | {pct(row['f1'])} |")
    lines += [
        "",
        f"Candidate raw AUC/AP: {ranking['candidate_raw']['auc']:.3f} / "
        f"{ranking['candidate_raw']['ap']:.3f}; best raw F1 "
        f"{pct(candidate_raw_best['best_f1'])} @ "
        f"{candidate_raw_best['best_f1_threshold']:.2f}.",
        f"Candidate effective AUC/AP: {candidate_eff_best['auc']:.3f} / "
        f"{candidate_eff_best['ap']:.3f}; best effective F1 "
        f"{pct(candidate_eff_best['best_f1'])} @ "
        f"{candidate_eff_best['best_f1_threshold']:.2f}.",
        "",
        "## Motion-bonus cohort",
        "",
        f"Frames stored only because motion lifted a raw score below 0.50: "
        f"{cohort['bonus_negative']} false and {cohort['bonus_positive']} true.",
        "",
        "Candidate on those motion-only frames:",
        "",
        "| cohort | raw @0.35 | raw @0.50 | effective @0.35 | effective @0.50 |",
        "|---|---:|---:|---:|---:|",
        f"| motion-only false ({cohort['bonus_negative']}) | "
        f"{cohort['bonus_negative_candidate_raw']['0.35']} | "
        f"{cohort['bonus_negative_candidate_raw']['0.5']} | "
        f"{cohort['bonus_negative_candidate_eff']['0.35']} | "
        f"{cohort['bonus_negative_candidate_eff']['0.5']} |",
        f"| motion-only true ({cohort['bonus_positive']}) | "
        f"{cohort['bonus_positive_candidate_raw']['0.35']} | "
        f"{cohort['bonus_positive_candidate_raw']['0.5']} | "
        f"{cohort['bonus_positive_candidate_eff']['0.35']} | "
        f"{cohort['bonus_positive_candidate_eff']['0.5']} |",
        "",
        "## Visual review",
        "",
        f"Open `review_motion.html` for the {sum(bool(row['cats']) for row in rows)} motion-relevant frames, or the contact sheets: "
        + ", ".join(f"`{name}`" for name in contacts.values()) + ".",
        "",
        "The motion-only false positives are mainly ordinary moving objects: "
        "people, animals, vehicles, bags/tarps, sunlit patches, reflections and "
        "shadows. The candidate rejects most of them without motion. With the "
        "same motion bonus, some moving objects survive at 0.50, so the 3-of-4 "
        "follow-up confirmation still matters.",
        "",
        "## Interpretation",
        "",
        f"- Ignoring motion understates the active model. Its AUC rises from "
        f"{ranking['active_raw']['auc']:.3f} raw to {active_best['auc']:.3f} "
        f"effective, and best F1 from {pct(ranking['active_raw']['best_f1'])} "
        f"to {pct(active_best['best_f1'])}.",
        f"- But motion does not close the gap. Active effective best F1 is "
        f"{pct(active_best['best_f1'])}; candidate raw best is "
        f"{pct(candidate_raw_best['best_f1'])}, and candidate with the same "
        f"motion rule reaches {pct(candidate_eff_best['best_f1'])}.",
        f"- At the production bar 0.50, active effective is saturated by "
        f"evidence selection ({pct(a500['recall'])} recall / "
        f"{pct(a500['fpr'])} FP rate). At 0.61 it is "
        f"{pct(a610['recall'])} / {pct(a610['fpr'])} with F1 "
        f"{pct(a610['f1'])}.",
        f"- Candidate raw at 0.35: {pct(c350['recall'])} / "
        f"{pct(c350['fpr'])} with F1 {pct(c350['f1'])}. Candidate with motion "
        f"at 0.50: {pct(ce500['recall'])} / {pct(ce500['fpr'])} with F1 "
        f"{pct(ce500['f1'])}.",
        f"- Motion improves the candidate: at 0.50 it raises recall from "
        f"{pct(c500['recall'])} to {pct(ce500['recall'])} and F1 from "
        f"{pct(c500['f1'])} to {pct(ce500['f1'])}, while FP rate rises from "
        f"{pct(c500['fpr'])} to {pct(ce500['fpr'])}.",
        f"- The candidate's best replay point is effective F1 "
        f"{pct(ce430['f1'])} at 0.43: {pct(ce430['recall'])} recall / "
        f"{pct(ce430['fpr'])} FP rate. The fixed Firewatch bar is 0.50, where "
        f"F1 is {pct(ce500['f1'])}.",
        f"- Motion bonus explains {cohort['bonus_negative']} of "
        f"{cohort['negatives']} false frames "
        f"({pct(cohort['bonus_negative'] / cohort['negatives'])}) and "
        f"{cohort['bonus_positive']} of {cohort['positives']} true frames "
        f"({pct(cohort['bonus_positive'] / cohort['positives'])}). It is not "
        f"the main false-positive source; most false frames already score "
        f">=0.50 raw.",
        "",
        "## Caveats",
        "",
        "- The evidence store contains only frames the active model selected, "
        "not the full camera timeline. Candidate discovery of fires the active "
        "model never stored cannot be measured here.",
        "- Firewatch's 1.5s A/B pair and skipped frames were not archived. "
        "Motion was replayed from the nearest prior stored frame within "
        f"{args.gap_max:g}s and validated against the recorded production "
        "motion flag.",
        "- The 3-of-4 follow-up confirmation was not replayed exactly because "
        "non-stored follow-up frames are absent. Per-frame scores are the "
        "comparable signal; live confirmation remains the final test.",
        "- Many motion-only true frames are visually ambiguous at thumbnail "
        "scale (people, bags, tarps, small bright spots). If those labels are "
        "uncertain, the candidate's apparent misses are overstated.",
    ]
    (out_dir / "motion_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
