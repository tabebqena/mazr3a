#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
STORE_PREFIX = "/media/firewatch/"


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def fmt_threshold(value):
    return "%.2f" % value


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relpath(path, start):
    return Path(os.path.relpath(path, start)).as_posix()


def parse_ts(name):
    m = re.match(r"^(\d{8})_(\d{6})", name)
    if not m:
        return ""
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return ""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def camera_of(key):
    return key.split("/", 1)[0] if "/" in key else "?"


def load_manifest(evidence):
    rows = []
    with open(evidence / "manifest.csv", newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row.get("kind") != "raw":
                continue
            rel = (row.get("file") or "").replace("\\", "/")
            name = rel.rsplit("/", 1)[-1]
            if not rel or name.endswith("_annotated.jpg") or name.endswith("_annotated.jpeg"):
                continue
            if Path(name).suffix.lower() not in IMG_EXT:
                continue
            path = evidence / rel
            if not path.is_file():
                continue
            rows.append(
                {
                    "key": rel,
                    "path": str(path),
                    "camera": row.get("camera") or camera_of(rel),
                    "ts": parse_ts(name),
                    "db_boxes": 0,
                }
            )
    return rows


def load_labels(path):
    labels = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("key") or "").strip()
            verdict = (row.get("verdict") or "").strip().lower()
            if key and verdict in ("true", "false"):
                labels[key] = verdict
    return labels


def load_frame_data(evidence, db_path):
    counts = {}
    scores = {}
    if not db_path.is_file():
        return counts, scores
    con = sqlite3.connect(
        "file:{}?mode=ro".format(quote(str(db_path), safe="/")), uri=True, timeout=10
    )
    try:
        frame_to_rel = {}
        for row in con.execute("SELECT id, camera, jpg_path FROM frames"):
            frame_id, camera, jpg_path = row
            path = str(jpg_path or "").replace("\\", "/")
            if STORE_PREFIX in path:
                rel = path.split(STORE_PREFIX, 1)[1]
            else:
                rel = path.rsplit("/", 1)[-1]
                if "/" not in rel:
                    rel = "{}/{}".format(camera, rel)
            frame_to_rel[frame_id] = rel
            counts[rel] = 0
            scores[rel] = 0.0
        for frame_id, count, max_fire in con.execute(
            "SELECT frame_id, COUNT(*), MAX(CASE WHEN lower(label) = 'fire' THEN score END) "
            "FROM detections GROUP BY frame_id"
        ):
            rel = frame_to_rel.get(frame_id)
            if rel is not None:
                counts[rel] = int(count)
                scores[rel] = float(max_fire or 0.0)
    finally:
        con.close()
    return counts, scores


def build_items(manifest, labels, counts, scores):
    items = []
    for row in manifest:
        item = dict(row)
        item["db_boxes"] = counts.get(item["key"], 0)
        item["stored_max_fire"] = scores.get(item["key"], 0.0)
        verdict = labels.get(item["key"])
        if verdict:
            item["label"] = verdict
            item["cohort"] = "core"
        elif item["db_boxes"] > 0:
            item["label"] = ""
            item["cohort"] = "mixed"
        else:
            item["label"] = ""
            item["cohort"] = "excluded"
        items.append(item)
    items.sort(key=lambda r: r["key"])
    return items


def snapshot_inputs(evidence, labels_path, out_dir):
    shutil.copyfile(labels_path, out_dir / "labels_snapshot.csv")
    labels_sha = sha256_file(labels_path)
    manifest_path = evidence / "manifest.csv"
    manifest_sha = sha256_file(manifest_path) if manifest_path.is_file() else ""
    return {
        "labels_path": str(labels_path),
        "labels_sha256": labels_sha,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "snapshot_utc": utc_stamp(),
    }


def load_prediction_cache(path):
    rows = {}
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("key"):
                rows[row["key"]] = row
    return rows


def labelmap_for(tag, model_dir, out_dir):
    path = Path(model_dir) / "labelmap.txt"
    if path.is_file():
        return path
    metadata = Path(model_dir) / "metadata.yaml"
    if not metadata.is_file():
        sys.exit("no labelmap.txt or metadata.yaml in %s" % model_dir)
    import yaml

    data = yaml.safe_load(metadata.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, dict):
        labels = [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    elif isinstance(names, list):
        labels = [str(name) for name in names]
    else:
        labels = []
    if not labels:
        sys.exit("could not read class names from %s" % metadata)
    target = out_dir / ("%s_labelmap.txt" % tag)
    target.write_text("\n".join(labels) + "\n", encoding="utf-8")
    return target


def export_openvino(pt_path, out_dir, imgsz, reuse):
    from ultralytics import YOLO

    export_root = out_dir / "candidate_export"
    export_root.mkdir(parents=True, exist_ok=True)
    copied = export_root / "best.pt"
    model_dir = export_root / "best_openvino_model"
    if reuse and (model_dir / "best.xml").is_file():
        return model_dir, copied
    shutil.copyfile(pt_path, copied)
    exported = Path(YOLO(str(copied)).export(format="openvino", imgsz=imgsz))
    return exported, copied


class IRModel:
    def __init__(self, model_dir, labelmap):
        import numpy as np
        import openvino as ov
        from PIL import Image

        self.np = np
        self.Image = Image
        xml_path = Path(model_dir) / "best.xml"
        if not xml_path.is_file():
            sys.exit("model not found: %s" % xml_path)
        core = ov.Core()
        model = core.read_model(str(xml_path))
        self._compiled = core.compile_model(model, "CPU")
        self._input = self._compiled.input(0)
        self._output = self._compiled.output(0)
        shape = list(self._input.shape)
        if len(shape) != 4:
            sys.exit("unexpected input shape %s" % shape)
        if shape[1] in (1, 3):
            _, self.ch, self.height, self.width = shape
        elif shape[3] in (1, 3):
            _, self.height, self.width, self.ch = shape
        else:
            sys.exit("unexpected input shape %s" % shape)
        self.labels = []
        labelmap_path = Path(labelmap)
        if labelmap_path.is_file():
            self.labels = [
                line.strip()
                for line in labelmap_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if not self.labels:
            sys.exit("empty labelmap: %s" % labelmap_path)
        self.nc = len(self.labels)

    @staticmethod
    def _letterbox(img, size):
        from PIL import Image
        import numpy as np

        iw, ih = img.size
        scale = min(size / iw, size / ih)
        nw, nh = int(round(iw * scale)), int(round(ih * scale))
        resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))[None], scale, (size - nw) // 2, (size - nh) // 2

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        x1, y1 = max(ax1, bx1), max(ay1, by1)
        x2, y2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / (aa + ba - inter + 1e-9)

    def detect(self, image, score_thresh=0.5, iou_thresh=0.45, allowed=None):
        import numpy as np

        blob, scale, pad_x, pad_y = self._letterbox(image, self.width)
        out = self._compiled([blob])[self._output]
        data = np.asarray(out)
        if data.ndim == 3:
            data = data[0]
        if data.ndim == 2 and data.shape[0] == 4 + self.nc and data.shape[1] > data.shape[0]:
            data = data.T
        cols = data.shape[1]
        raw_cols = 4 + self.nc
        if cols == raw_cols:
            mode = "raw"
        elif cols == 6:
            mode = "e2e"
        else:
            sys.exit("unexpected output columns %d (expected %d or 6)" % (cols, raw_cols))
        dets = []
        if mode == "raw":
            boxes = data[:, :4].astype(np.float32)
            scores = data[:, 4 : 4 + self.nc].astype(np.float32)
            if np.nanmax(scores) > 1.0:
                scores = 1.0 / (1.0 + np.exp(-scores))
            for ci in range(self.nc):
                if allowed and self.labels[ci].lower() not in allowed:
                    continue
                idx = np.where(scores[:, ci] >= score_thresh)[0]
                for i in idx:
                    cx, cy, bw, bh = boxes[i]
                    x1 = (cx - bw / 2 - pad_x) / scale
                    y1 = (cy - bh / 2 - pad_y) / scale
                    x2 = (cx + bw / 2 - pad_x) / scale
                    y2 = (cy + bh / 2 - pad_y) / scale
                    dets.append(
                        {
                            "label": self.labels[ci],
                            "score": float(scores[i, ci]),
                            "box": (max(0.0, x1), max(0.0, y1), x2, y2),
                        }
                    )
        else:
            xyxy = data[:, :4].astype(np.float32)
            confs = data[:, 4].astype(np.float32)
            cids = np.clip(data[:, 5].astype(np.int64), 0, self.nc - 1)
            for i in np.where(confs >= score_thresh)[0]:
                label = self.labels[int(cids[i])]
                if allowed and label.lower() not in allowed:
                    continue
                x1, y1, x2, y2 = xyxy[i]
                dets.append(
                    {
                        "label": label,
                        "score": float(confs[i]),
                        "box": (
                            max(0.0, (x1 - pad_x) / scale),
                            max(0.0, (y1 - pad_y) / scale),
                            (x2 - pad_x) / scale,
                            (y2 - pad_y) / scale,
                        ),
                    }
                )
        dets.sort(key=lambda det: det["score"], reverse=True)
        keep = []
        for det in dets:
            if all(self._iou(det["box"], other["box"]) <= iou_thresh for other in keep):
                keep.append(det)
        return keep


def run_model(tag, model_dir, labelmap, items, args, out_dir, thresholds, source_pt):
    from PIL import Image

    cache_path = out_dir / ("predictions_%s.jsonl" % tag)
    cached = load_prediction_cache(cache_path) if args.resume else {}
    todo = [item for item in items if item["key"] not in cached]
    model = IRModel(str(model_dir), str(labelmap))
    names = {index: name for index, name in enumerate(model.labels)}
    fire_idx = next((i for i, n in names.items() if n.strip().lower() == "fire"), None)
    smoke_idx = next((i for i, n in names.items() if n.strip().lower() == "smoke"), None)
    if fire_idx is None:
        sys.exit("model %s has no fire class: %s" % (tag, names))
    config = {
        "tag": tag,
        "model_dir": str(model_dir),
        "source_pt": str(source_pt) if source_pt else "",
        "source_pt_md5": md5_file(source_pt) if source_pt and Path(source_pt).is_file() else "",
        "xml_md5": md5_file(Path(model_dir) / "best.xml"),
        "bin_md5": md5_file(Path(model_dir) / "best.bin"),
        "bytes": os.path.getsize(Path(model_dir) / "best.xml")
        + os.path.getsize(Path(model_dir) / "best.bin"),
        "names": names,
        "fire_index": fire_idx,
        "smoke_index": smoke_idx,
        "input_width": model.width,
        "input_height": model.height,
        "output_shape": list(model._output.shape),
        "cached_images": len(cached),
        "pending_images": len(todo),
    }
    if not todo:
        return cached, config, 0.0
    min_threshold = min(thresholds)
    started = time.time()
    processed = 0
    mode = "a" if args.resume else "w"
    with open(cache_path, mode, encoding="utf-8") as fh:
        for item in todo:
            with Image.open(item["path"]) as image:
                dets = model.detect(
                    image.convert("RGB"),
                    score_thresh=args.floor,
                    iou_thresh=args.iou,
                    allowed=None,
                )
            max_fire = 0.0
            max_smoke = 0.0
            boxes = []
            for det in dets:
                score = float(det["score"])
                label = str(det["label"]).strip().lower()
                if label == "fire":
                    max_fire = max(max_fire, score)
                    if score >= min_threshold:
                        boxes.append(
                            {
                                "s": round(score, 4),
                                "xy": [round(float(v), 2) for v in det["box"]],
                            }
                        )
                elif label == "smoke":
                    max_smoke = max(max_smoke, score)
            n_fire = {}
            hit = {}
            for threshold in thresholds:
                key = fmt_threshold(threshold)
                n_fire[key] = sum(1 for box in boxes if box["s"] >= threshold)
                hit[key] = max_fire >= threshold
            row = {
                "key": item["key"],
                "max_fire": round(max_fire, 4),
                "max_smoke": round(max_smoke, 4),
                "boxes": boxes,
                "n_fire": n_fire,
                "hit": hit,
            }
            cached[row["key"]] = row
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            processed += 1
            if processed % 100 == 0:
                fh.flush()
                elapsed = time.time() - started
                rate = processed / elapsed if elapsed else 0.0
                print(
                    "  %-9s %d/%d  %.1f img/s  %.0fs"
                    % (tag, processed, len(todo), rate, elapsed),
                    flush=True,
                )
        fh.flush()
    return cached, config, time.time() - started


def distribution(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {"n": 0}
    def pct(p):
        if len(values) == 1:
            return values[0]
        pos = (len(values) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        frac = pos - lo
        return values[lo] * (1 - frac) + values[hi] * frac
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(values[0], 4),
        "p10": round(pct(0.10), 4),
        "p90": round(pct(0.90), 4),
        "max": round(values[-1], 4),
    }


def roc_auc(rows, tag):
    values = sorted((row[tag]["max_fire"], row["label"] == "true") for row in rows)
    n_pos = sum(1 for _, positive in values if positive)
    n_neg = len(values) - n_pos
    if not n_pos or not n_neg:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(values):
        end = index + 1
        score = values[index][0]
        while end < len(values) and values[end][0] == score:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        rank_sum += avg_rank * sum(1 for _, positive in values[index:end] if positive)
        index = end
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(rows, tag):
    values = sorted(
        ((row[tag]["max_fire"], row["label"] == "true") for row in rows),
        key=lambda item: item[0],
        reverse=True,
    )
    n_pos = sum(1 for _, positive in values if positive)
    if not n_pos:
        return None
    tp = 0
    fp = 0
    previous_recall = 0.0
    total = 0.0
    index = 0
    while index < len(values):
        score = values[index][0]
        end = index
        while end < len(values) and values[end][0] == score:
            if values[end][1]:
                tp += 1
            else:
                fp += 1
            end += 1
        recall = tp / n_pos
        precision = tp / (tp + fp) if tp + fp else 0.0
        total += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return total


def metrics_for_model(rows, tag, threshold):
    pos = [row for row in rows if row["label"] == "true"]
    neg = [row for row in rows if row["label"] == "false"]
    key = fmt_threshold(threshold)
    hit = lambda row: row[tag]["hit"][key]
    tp = sum(1 for row in pos if hit(row))
    fn = len(pos) - tp
    fp = sum(1 for row in neg if hit(row))
    tn = len(neg) - fp
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    precision = tp / (tp + fp) if tp + fp else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if precision is not None and recall is not None and 4 * precision + recall
        else None
    )
    accuracy = (tp + tn) / len(rows) if rows else None
    balanced = (
        (recall + specificity) / 2.0
        if recall is not None and specificity is not None
        else None
    )
    per_camera = {}
    for row in neg:
        camera = row["camera"]
        state = per_camera.setdefault(camera, {"images": 0, "fp": 0})
        state["images"] += 1
        if hit(row):
            state["fp"] += 1
    return {
        "threshold": threshold,
        "positives": len(pos),
        "negatives": len(neg),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": recall,
        "fpr": fpr,
        "precision": precision,
        "specificity": specificity,
        "f1": f1,
        "f2": f2,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "per_camera_fp": per_camera,
    }


def flip_group(label, active_hit, candidate_hit):
    if label == "true":
        if active_hit and candidate_hit:
            return "true_both_hit"
        if active_hit and not candidate_hit:
            return "true_lost_by_candidate"
        if candidate_hit and not active_hit:
            return "true_fixed_by_candidate"
        return "true_both_miss"
    if active_hit and candidate_hit:
        return "false_both_fp"
    if active_hit and not candidate_hit:
        return "false_fixed_by_candidate"
    if candidate_hit and not active_hit:
        return "false_new_fp_by_candidate"
    return "false_both_clean"


def build_joined(core, predictions):
    joined = []
    for item in core:
        active = predictions["active"].get(item["key"])
        candidate = predictions["candidate"].get(item["key"])
        if active is None or candidate is None:
            if item.get("partial"):
                continue
            sys.exit("missing prediction for %s" % item["key"])
        row = dict(item)
        row["active"] = active
        row["candidate"] = candidate
        joined.append(row)
    return joined


def threshold_sweep(rows, tag):
    table = {}
    for step in range(5, 96, 5):
        threshold = step / 100.0
        key = fmt_threshold(threshold)
        pos = [row for row in rows if row["label"] == "true"]
        neg = [row for row in rows if row["label"] == "false"]
        tp = sum(1 for row in pos if row[tag]["max_fire"] >= threshold)
        fp = sum(1 for row in neg if row[tag]["max_fire"] >= threshold)
        recall = tp / len(pos) if pos else None
        fpr = fp / len(neg) if neg else None
        precision = tp / (tp + fp) if tp + fp else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        table[key] = {
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "recall": recall,
            "fpr": fpr,
            "precision": precision,
            "f1": f1,
        }
    best_f1 = max(
        (entry for entry in table.values() if entry["f1"] is not None),
        key=lambda entry: (entry["f1"], entry["threshold"]),
        default=None,
    )
    low_fp = [
        entry
        for entry in table.values()
        if entry["fpr"] is not None and entry["fpr"] <= 0.05 and entry["recall"] is not None
    ]
    best_low_fp = max(low_fp, key=lambda entry: entry["recall"], default=None)
    return {"table": table, "best_f1": best_f1, "best_recall_at_5pct_fpr": best_low_fp}


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_per_image(out_dir, rows, thresholds):
    fields = [
        "key",
        "camera",
        "ts",
        "cohort",
        "label",
        "db_boxes",
        "active_max_fire",
        "candidate_max_fire",
    ]
    for threshold in thresholds:
        key = fmt_threshold(threshold)
        fields.extend(
            [
                "active_n_fire_%s" % key,
                "candidate_n_fire_%s" % key,
                "active_hit_%s" % key,
                "candidate_hit_%s" % key,
                "group_%s" % key,
            ]
        )
    records = []
    for row in rows:
        record = {
            "key": row["key"],
            "camera": row["camera"],
            "ts": row["ts"],
            "cohort": row["cohort"],
            "label": row["label"],
            "db_boxes": row["db_boxes"],
            "active_max_fire": row["active"]["max_fire"],
            "candidate_max_fire": row["candidate"]["max_fire"],
        }
        for threshold in thresholds:
            key = fmt_threshold(threshold)
            active_hit = row["active"]["hit"][key]
            candidate_hit = row["candidate"]["hit"][key]
            record["active_n_fire_%s" % key] = row["active"]["n_fire"][key]
            record["candidate_n_fire_%s" % key] = row["candidate"]["n_fire"][key]
            record["active_hit_%s" % key] = int(active_hit)
            record["candidate_hit_%s" % key] = int(candidate_hit)
            record["group_%s" % key] = flip_group(row["label"], active_hit, candidate_hit)
        records.append(record)
    write_csv(out_dir / "per_image_comparison.csv", records, fields)
    return records


def save_flips(out_dir, records, thresholds):
    counts = {}
    for threshold in thresholds:
        key = fmt_threshold(threshold)
        group_key = "group_%s" % key
        fields = [
            "key",
            "camera",
            "ts",
            "label",
            "active_max_fire",
            "candidate_max_fire",
            "active_hit",
            "candidate_hit",
            "group",
        ]
        rows = []
        counter = Counter()
        for record in records:
            group = record[group_key]
            counter[group] += 1
            if group in (
                "true_lost_by_candidate",
                "true_fixed_by_candidate",
                "false_new_fp_by_candidate",
                "false_fixed_by_candidate",
            ):
                rows.append(
                    {
                        "key": record["key"],
                        "camera": record["camera"],
                        "ts": record["ts"],
                        "label": record["label"],
                        "active_max_fire": record["active_max_fire"],
                        "candidate_max_fire": record["candidate_max_fire"],
                        "active_hit": record["active_hit_%s" % key],
                        "candidate_hit": record["candidate_hit_%s" % key],
                        "group": group,
                    }
                )
        write_csv(out_dir / ("flips_%s.csv" % key), rows, fields)
        counts[key] = dict(counter)
    return counts


def build_review(out_dir, mixed, predictions, thresholds):
    from PIL import Image

    rows = []
    for item in mixed:
        active = predictions["active"].get(item["key"])
        candidate = predictions["candidate"].get(item["key"])
        if active is None or candidate is None:
            sys.exit("missing prediction for mixed image %s" % item["key"])
        with Image.open(item["path"]) as image:
            width, height = image.size
        rows.append(
            {
                "key": item["key"],
                "src": relpath(item["path"], out_dir),
                "camera": item["camera"],
                "ts": item["ts"],
                "db_boxes": item["db_boxes"],
                "width": width,
                "height": height,
                "active": {
                    "max": active["max_fire"],
                    "boxes": active["boxes"],
                    "n_fire": active["n_fire"],
                },
                "candidate": {
                    "max": candidate["max_fire"],
                    "boxes": candidate["boxes"],
                    "n_fire": candidate["n_fire"],
                },
            }
        )
    data = {"thresholds": thresholds, "rows": rows}
    with open(out_dir / "review_mixed_data.json", "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    options = "".join(
        '<option value="%s">%s</option>' % (fmt_threshold(t), fmt_threshold(t))
        for t in thresholds
    )
    page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mixed evidence review</title>
<style>
:root { color-scheme: dark; --bg:#101418; --panel:#171d24; --line:#2b3540; --text:#e6edf3; --muted:#95a3b3; --active:#3b82f6; --candidate:#f97316; --ok:#22c55e; --bad:#ef4444; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; }
header { padding:18px 22px 8px; }
h1 { margin:0 0 6px; font-size:20px; }
p { margin:4px 0; color:var(--muted); }
.controls { position:sticky; top:0; z-index:20; display:flex; flex-wrap:wrap; gap:12px; align-items:end; padding:12px 22px; background:rgba(16,20,24,.96); border-bottom:1px solid var(--line); }
label { display:grid; gap:4px; color:var(--muted); font-size:12px; }
select,input,button,textarea { background:#0d1116; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:6px 8px; font:inherit; }
button { cursor:pointer; }
button:hover { border-color:#4b5d70; }
#stats { padding:8px 22px 0; color:var(--muted); }
main { display:grid; gap:18px; padding:18px 22px 40px; max-width:1180px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
.card-head { display:flex; flex-wrap:wrap; gap:10px; align-items:baseline; justify-content:space-between; padding:10px 14px; border-bottom:1px solid var(--line); }
.key { font-size:12px; color:var(--muted); overflow-wrap:anywhere; }
.pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; border:1px solid var(--line); margin-left:6px; }
.active { color:#bfdbfe; border-color:#1d4ed8; }
.candidate { color:#fed7aa; border-color:#c2410c; }
.media { position:relative; line-height:0; background:#05070a; }
.media img { display:block; width:100%; height:auto; }
.overlay { position:absolute; inset:0; pointer-events:none; }
.box { position:absolute; border:2px solid var(--active); }
.box.candidate { border-color:var(--candidate); }
.box span { position:absolute; left:0; top:-18px; background:rgba(0,0,0,.78); color:#fff; font:11px/1.4 monospace; padding:1px 4px; white-space:nowrap; }
.card-body { display:grid; gap:10px; padding:12px 14px; }
.metrics { display:flex; flex-wrap:wrap; gap:14px; color:var(--muted); font-size:12px; }
.review { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; align-items:end; }
.review .wide { grid-column:1/-1; }
.notes { width:100%; }
.empty { padding:30px; color:var(--muted); }
</style>
</head>
<body>
<header>
<h1>Mixed evidence review</h1>
<p>These images are excluded from the headline comparison. They contain at least one box but were left untouched because the active model reports a mix of plausible fire and false boxes.</p>
<p>Review the new model's boxes at each confidence. Saving and exporting decisions is optional.</p>
</header>
<div class="controls">
<label>Threshold<select id="threshold">__OPTIONS__</select></label>
<label>Overlay<select id="overlay"><option value="both">Both models</option><option value="active">Active only</option><option value="candidate">Candidate only</option></select></label>
<label>Filter<select id="filter"><option value="all">All mixed</option><option value="disagree">Models disagree</option><option value="active_only">Active only</option><option value="candidate_only">Candidate only</option><option value="both">Both hit</option><option value="neither">Neither</option></select></label>
<label>Sort<select id="sort"><option value="delta_desc">Largest candidate-active gap</option><option value="delta_asc">Largest active-candidate gap</option><option value="active_desc">Active score</option><option value="candidate_desc">Candidate score</option><option value="key">Filename</option></select></label>
<button id="export">Download decisions CSV</button>
<button id="reset">Reset decisions</button>
</div>
<div id="stats"></div>
<main id="cards"></main>
<script>
const DATA = __DATA__;
const STORAGE_KEY = "firewatch-mixed-review-v1";
const decisions = loadDecisions();
const $ = (id) => document.getElementById(id);
function loadDecisions() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  } catch (error) {
    return {};
  }
}
function saveDecisions() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(decisions));
  } catch (error) {}
}
function decisionFor(key) {
  if (!decisions[key]) {
    decisions[key] = {fire_present: "", old_false_boxes: "", new_false_boxes: "", notes: ""};
  }
  return decisions[key];
}
function threshold() {
  return Number($("threshold").value);
}
function hit(row, tag, t) {
  return row[tag].max >= t;
}
function boxesFor(row, tag, t) {
  return row[tag].boxes.filter((box) => box.s >= t);
}
function status(row, t) {
  const a = hit(row, "active", t);
  const c = hit(row, "candidate", t);
  if (a && c) return "both";
  if (a) return "active_only";
  if (c) return "candidate_only";
  return "neither";
}
function boxHtml(box, row, tag) {
  const x1 = box.xy[0] / row.width * 100;
  const y1 = box.xy[1] / row.height * 100;
  const x2 = box.xy[2] / row.width * 100;
  const y2 = box.xy[3] / row.height * 100;
  const prefix = tag === "active" ? "A" : "N";
  const cls = tag === "candidate" ? "candidate" : "";
  return '<div class="box ' + cls + '" style="left:' + x1 + '%;top:' + y1 + '%;width:' + (x2 - x1) + '%;height:' + (y2 - y1) + '%"><span>' + prefix + ' ' + box.s.toFixed(2) + '</span></div>';
}
function escapeCsv(value) {
  const text = String(value == null ? "" : value);
  return /[",\\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
}
function renderStats(t) {
  let active = 0;
  let candidate = 0;
  let both = 0;
  let neither = 0;
  for (const row of DATA.rows) {
    const state = status(row, t);
    if (hit(row, "active", t)) active += 1;
    if (hit(row, "candidate", t)) candidate += 1;
    if (state === "both") both += 1;
    if (state === "neither") neither += 1;
  }
  $("stats").textContent = DATA.rows.length + " mixed images | active " + active + " hit | candidate " + candidate + " hit | both " + both + " | neither " + neither;
}
function filteredRows(t) {
  const mode = $("filter").value;
  const rows = DATA.rows.filter((row) => {
    if (mode === "all") return true;
    const state = status(row, t);
    if (mode === "disagree") return state === "active_only" || state === "candidate_only";
    return state === mode;
  });
  const sort = $("sort").value;
  rows.sort((a, b) => {
    const da = a.candidate.max - a.active.max;
    const db = b.candidate.max - b.active.max;
    if (sort === "delta_desc") return db - da;
    if (sort === "delta_asc") return da - db;
    if (sort === "active_desc") return b.active.max - a.active.max;
    if (sort === "candidate_desc") return b.candidate.max - a.candidate.max;
    return a.key.localeCompare(b.key);
  });
  return rows;
}
function render() {
  const t = threshold();
  const overlayMode = $("overlay").value;
  renderStats(t);
  const cards = [];
  for (const row of filteredRows(t)) {
    const decision = decisionFor(row.key);
    const activeBoxes = overlayMode === "candidate" ? [] : boxesFor(row, "active", t);
    const candidateBoxes = overlayMode === "active" ? [] : boxesFor(row, "candidate", t);
    const overlay = activeBoxes.map((box) => boxHtml(box, row, "active")).join("") + candidateBoxes.map((box) => boxHtml(box, row, "candidate")).join("");
    cards.push(
      '<section class="card" data-key="' + row.key + '">' +
      '<div class="card-head"><div class="key">' + row.key + '</div><div>' +
      '<span class="pill active">active ' + row.active.max.toFixed(3) + '</span>' +
      '<span class="pill candidate">candidate ' + row.candidate.max.toFixed(3) + '</span>' +
      '<span class="pill">DB boxes ' + row.db_boxes + '</span>' +
      '</div></div>' +
      '<div class="media"><img loading="lazy" src="' + row.src + '" width="' + row.width + '" height="' + row.height + '"><div class="overlay">' + overlay + '</div></div>' +
      '<div class="card-body"><div class="metrics">' +
      '<span>active fire boxes >= ' + t.toFixed(2) + ': ' + boxesFor(row, "active", t).length + '</span>' +
      '<span>candidate fire boxes >= ' + t.toFixed(2) + ': ' + boxesFor(row, "candidate", t).length + '</span>' +
      '<span>' + row.ts + ' UTC</span>' +
      '</div><div class="review">' +
      '<label>Fire present<select data-field="fire_present"><option value="">unset</option><option value="yes"' + (decision.fire_present === "yes" ? " selected" : "") + '>yes</option><option value="no"' + (decision.fire_present === "no" ? " selected" : "") + '>no</option><option value="unsure"' + (decision.fire_present === "unsure" ? " selected" : "") + '>unsure</option></select></label>' +
      '<label>Active false boxes<input data-field="old_false_boxes" type="number" min="0" step="1" value="' + decision.old_false_boxes + '"></label>' +
      '<label>New false boxes<input data-field="new_false_boxes" type="number" min="0" step="1" value="' + decision.new_false_boxes + '"></label>' +
      '<label class="wide">Notes<input class="notes" data-field="notes" type="text" value="' + String(decision.notes).replace(/"/g, "&quot;") + '"></label>' +
      '</div></div></section>'
    );
  }
  $("cards").innerHTML = cards.length ? cards.join("") : '<div class="empty">No images match this filter.</div>';
}
$("cards").addEventListener("input", (event) => {
  const field = event.target.dataset.field;
  if (!field) return;
  const card = event.target.closest(".card");
  if (!card) return;
  decisionFor(card.dataset.key)[field] = event.target.value;
  saveDecisions();
});
$("cards").addEventListener("change", (event) => {
  const field = event.target.dataset.field;
  if (!field) return;
  const card = event.target.closest(".card");
  if (!card) return;
  decisionFor(card.dataset.key)[field] = event.target.value;
  saveDecisions();
});
for (const id of ["threshold", "overlay", "filter", "sort"]) {
  $(id).addEventListener("change", render);
}
$("reset").addEventListener("click", () => {
  if (!confirm("Reset all review decisions?")) return;
  for (const key of Object.keys(decisions)) delete decisions[key];
  saveDecisions();
  render();
});
$("export").addEventListener("click", () => {
  const lines = ["key,camera,fire_present,old_false_boxes,new_false_boxes,notes"];
  for (const row of DATA.rows) {
    const decision = decisionFor(row.key);
    lines.push([row.key, row.camera, decision.fire_present, decision.old_false_boxes, decision.new_false_boxes, decision.notes].map(escapeCsv).join(","));
  }
  const blob = new Blob([lines.join("\\n") + "\\n"], {type: "text/csv;charset=utf-8"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "review_mixed_decisions.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
});
render();
</script>
</body>
</html>
"""
    page = page.replace("__OPTIONS__", options).replace("__DATA__", payload)
    with open(out_dir / "review_mixed.html", "w", encoding="utf-8") as fh:
        fh.write(page)
    with open(out_dir / "review_mixed_decisions.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "key",
                "camera",
                "fire_present",
                "old_false_boxes",
                "new_false_boxes",
                "notes",
            ]
        )
    stats = {}
    for threshold in thresholds:
        key = fmt_threshold(threshold)
        stats[key] = {
            "active": sum(1 for row in rows if row["active"]["max"] >= threshold),
            "candidate": sum(
                1 for row in rows if row["candidate"]["max"] >= threshold
            ),
            "both": sum(
                1
                for row in rows
                if row["active"]["max"] >= threshold
                and row["candidate"]["max"] >= threshold
            ),
        }
    return {
        "rows": len(rows),
        "html": str(out_dir / "review_mixed.html"),
        "stats": stats,
    }


def pct(value):
    return "n/a" if value is None else "%.1f%%" % (100.0 * value)


def build_summary(config, counts, metrics, aucs, sweeps, flip_counts, sanity, review):
    lines = []
    lines.append("# Active vs candidate fire model on firewatch evidence")
    lines.append("")
    lines.append("**Generated:** %s UTC" % config["generated"])
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("| model | model dir | classes | input | output | IR md5 (xml / bin) | source pt md5 |")
    lines.append("|---|---|---|---|---|---|---|")
    for tag in ("active", "candidate"):
        model = config["models"][tag]
        lines.append(
            "| %s | `%s` | %s | %dx%d | %s | `%s` / `%s` | `%s` |"
            % (
                tag,
                model["model_dir"],
                ", ".join("%d=%s" % (k, v) for k, v in sorted(model["names"].items())),
                model["input_width"],
                model["input_height"],
                model["output_shape"],
                model["xml_md5"],
                model["bin_md5"],
                model["source_pt_md5"],
            )
        )
    lines.append("")
    lines.append("Inference uses the deployed OpenVINO IR path and firewatch preprocessing: letterbox to %dx%d, NMS IoU %.2f, inference floor %.2f."
                 % (
                     config["models"]["active"]["input_width"],
                     config["models"]["active"]["input_height"],
                     config["args"]["iou"],
                     config["args"]["floor"],
                 ))
    lines.append("")
    lines.append("The remote firewatch service loads `best.xml`, not `best.pt`. The active row above is the IR actually deployed. The tracked active `best.pt` does not reproduce the IR's scores, so the IR is the authoritative active model for this comparison.")
    lines.append("")
    lines.append("Production tracks fire only (`TRACK_SMOKE=false`); smoke/other boxes are ignored.")
    lines.append("")
    lines.append("## Cohort")
    lines.append("")
    lines.append("| cohort | images |")
    lines.append("|---|---:|")
    for name in ("core", "mixed", "excluded"):
        lines.append("| %s | %d |" % (name, counts[name]))
    lines.append("")
    lines.append(
        "Core labels: %d true / %d false. Mixed images are excluded from the headline and reviewed separately."
        % (counts["true"], counts["false"])
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        "| model | conf | TP | FP | FN | TN | recall | FP rate | precision | F1 | F2 | balanced acc |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for threshold in config["thresholds"]:
        key = fmt_threshold(threshold)
        for tag in ("active", "candidate"):
            m = metrics[tag][key]
            lines.append(
                "| %s | %s | %d | %d | %d | %d | %s | %s | %s | %s | %s | %s |"
                % (
                    tag,
                    key,
                    m["tp"],
                    m["fp"],
                    m["fn"],
                    m["tn"],
                    pct(m["recall"]),
                    pct(m["fpr"]),
                    pct(m["precision"]),
                    pct(m["f1"]),
                    pct(m["f2"]),
                    pct(m["balanced_accuracy"]),
                )
            )
    lines.append("")
    lines.append("## Candidate minus active")
    lines.append("")
    lines.append("| conf | recall pp | FP-rate pp | FP count | precision pp | F1 pp |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for threshold in config["thresholds"]:
        key = fmt_threshold(threshold)
        a = metrics["active"][key]
        c = metrics["candidate"][key]
        lines.append(
            "| %s | %+.1f | %+.1f | %+d | %+.1f | %+.1f |"
            % (
                key,
                100 * (c["recall"] - a["recall"]),
                100 * (c["fpr"] - a["fpr"]),
                c["fp"] - a["fp"],
                100 * (c["precision"] - a["precision"]),
                100 * (c["f1"] - a["f1"]),
            )
        )
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append("| model | ROC-AUC | average precision | best F1 | best F1 at conf | best recall at <=5% FP |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for tag in ("active", "candidate"):
        best = sweeps[tag]["best_f1"]
        low = sweeps[tag]["best_recall_at_5pct_fpr"]
        lines.append(
            "| %s | %.4f | %.4f | %s | %s | %s |"
            % (
                tag,
                aucs[tag]["roc_auc"],
                aucs[tag]["average_precision"],
                pct(best["f1"]) if best else "n/a",
                ("%.2f" % best["threshold"]) if best else "n/a",
                ("%s @ %.2f" % (pct(low["recall"]), low["threshold"])) if low else "n/a",
            )
        )
    lines.append("")
    lines.append("## Max fire confidence")
    lines.append("")
    lines.append("| model | set | n | mean | median | p10 | p90 | min | max |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for tag in ("active", "candidate"):
        for label in ("true", "false"):
            d = metrics[tag]["dist_" + label]
            lines.append(
                "| %s | %s | %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |"
                % (
                    tag,
                    label,
                    d["n"],
                    d["mean"],
                    d["median"],
                    d["p10"],
                    d["p90"],
                    d["min"],
                    d["max"],
                )
            )
    lines.append("")
    lines.append("## Candidate vs active flips")
    lines.append("")
    lines.append("| conf | fires lost | fires fixed | false boxes fixed | new false boxes |")
    lines.append("|---:|---:|---:|---:|---:|")
    for threshold in config["thresholds"]:
        key = fmt_threshold(threshold)
        f = flip_counts[key]
        lines.append(
            "| %s | %d | %d | %d | %d |"
            % (
                key,
                f.get("true_lost_by_candidate", 0),
                f.get("true_fixed_by_candidate", 0),
                f.get("false_fixed_by_candidate", 0),
                f.get("false_new_fp_by_candidate", 0),
            )
        )
    lines.append("")
    lines.append("## False positives per camera")
    lines.append("")
    for threshold in config["thresholds"]:
        key = fmt_threshold(threshold)
        lines.append("### conf %s" % key)
        lines.append("")
        lines.append("| camera | negative images | active FP | candidate FP |")
        lines.append("|---|---:|---:|---:|")
        cameras = sorted(
            set(metrics["active"][key]["per_camera_fp"])
            | set(metrics["candidate"][key]["per_camera_fp"])
        )
        for camera in cameras:
            a = metrics["active"][key]["per_camera_fp"].get(
                camera, {"images": 0, "fp": 0}
            )
            c = metrics["candidate"][key]["per_camera_fp"].get(
                camera, {"images": 0, "fp": 0}
            )
            lines.append(
                "| %s | %d | %d | %d |"
                % (camera, a["images"], a["fp"], c["fp"])
            )
        lines.append("")
    lines.append("## Harness sanity")
    lines.append("")
    lines.append(
        "Active IR re-measured max fire vs the stored raw fire score: n=%d, mean absolute difference=%.4f, max=%.4f, within 0.05=%d/%d."
        % (
            sanity["n"],
            sanity["mean_abs_diff"],
            sanity["max_abs_diff"],
            sanity["within_0_05"],
            sanity["n"],
        )
    )
    lines.append("")
    lines.append("## Mixed review")
    lines.append("")
    lines.append(
        "%d boxed untouched images are in `review_mixed.html`; they are excluded from headline metrics."
        % review["rows"]
    )
    lines.append("")
    if review.get("stats"):
        lines.append("| conf | active hit | candidate hit | both hit |")
        lines.append("|---:|---:|---:|---:|")
        for threshold in config["thresholds"]:
            key = fmt_threshold(threshold)
            state = review["stats"][key]
            lines.append(
                "| %s | %d | %d | %d |"
                % (key, state["active"], state["candidate"], state["both"])
            )
        lines.append("")
    lines.append("## Caveat")
    lines.append("")
    lines.append(
        "The negatives are frames the active model already stored, so this is an adversarial evidence-set comparison, not a full production false-alarm-rate measurement. All true labels are cam01."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for name in (
        "labels_snapshot.csv",
        "cohorts.csv",
        "per_image_comparison.csv",
        "predictions_active.jsonl",
        "predictions_candidate.jsonl",
        "summary.json",
        "review_mixed.html",
        "review_mixed_data.json",
        "candidate_export/best_openvino_model/",
    ):
        lines.append("- `%s`" % name)
    for threshold in config["thresholds"]:
        lines.append("- `flips_%s.csv`" % fmt_threshold(threshold))
    lines.append("")
    return "\n".join(lines)


def sanity_against_store(rows):
    diffs = []
    for row in rows:
        if row.get("db_boxes", 0) > 0 and row.get("stored_max_fire", 0.0) > 0:
            diffs.append(abs(row["stored_max_fire"] - row["active"]["max_fire"]))
    if not diffs:
        return {
            "n": 0,
            "mean_abs_diff": 0.0,
            "max_abs_diff": 0.0,
            "within_0_05": 0,
        }
    return {
        "n": len(diffs),
        "mean_abs_diff": statistics.fmean(diffs),
        "max_abs_diff": max(diffs),
        "within_0_05": sum(1 for diff in diffs if diff <= 0.05),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default="./firewatch-evidence")
    parser.add_argument("--labels", default="")
    parser.add_argument("--active", default="models/fire")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.25, 0.35, 0.5])
    parser.add_argument("--floor", type=float, default=0.02)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--out", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    evidence = Path(args.evidence).resolve()
    labels_path = (
        Path(args.labels).resolve() if args.labels else evidence / "classifications.csv"
    )
    thresholds = sorted(set(round(value, 2) for value in args.thresholds))
    if not evidence.is_dir():
        sys.exit("evidence dir not found: %s" % evidence)
    if not labels_path.is_file():
        sys.exit("labels file not found: %s" % labels_path)
    if args.floor > min(thresholds):
        sys.exit("--floor must be <= the smallest threshold")

    out_dir = (
        Path(args.out).resolve()
        if args.out
        else evidence / "model-comparison" / ("run-%s" % utc_stamp())
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(evidence)
    labels = load_labels(labels_path)
    counts, stored_scores = load_frame_data(evidence, evidence / "firewatch.db")
    items = build_items(manifest, labels, counts, stored_scores)
    core = [item for item in items if item["cohort"] == "core"]
    mixed = [item for item in items if item["cohort"] == "mixed"]
    excluded = [item for item in items if item["cohort"] == "excluded"]
    if args.limit > 0:
        selected = (core + mixed)[: args.limit]
        for item in selected:
            item["partial"] = True
        core = [item for item in selected if item["cohort"] == "core"]
        mixed = [item for item in selected if item["cohort"] == "mixed"]
    run_items = core + mixed

    snapshot = snapshot_inputs(evidence, labels_path, out_dir)
    cohort_rows = [
        {
            "key": item["key"],
            "camera": item["camera"],
            "ts": item["ts"],
            "db_boxes": item["db_boxes"],
            "label": item["label"],
            "cohort": item["cohort"],
        }
        for item in items
    ]
    write_csv(
        out_dir / "cohorts.csv",
        cohort_rows,
        ["key", "camera", "ts", "db_boxes", "label", "cohort"],
    )

    print(
        "cohort: core=%d true=%d false=%d mixed=%d excluded=%d"
        % (
            len(core),
            len([item for item in core if item["label"] == "true"]),
            len([item for item in core if item["label"] == "false"]),
            len(mixed),
            len(excluded),
        ),
        flush=True,
    )
    if args.limit <= 0 and len(core) == 0:
        sys.exit("no labelled core images found")

    active_dir = Path(args.active).resolve()
    if not (active_dir / "best.xml").is_file():
        sys.exit(
            "--active must be the deployed OpenVINO model directory (best.xml), got %s"
            % active_dir
        )
    active_pt = active_dir / "best.pt" if (active_dir / "best.pt").is_file() else None

    candidate_path = Path(args.candidate).resolve()
    if candidate_path.is_dir():
        candidate_dir = candidate_path
        candidate_pt = (
            candidate_dir / "best.pt" if (candidate_dir / "best.pt").is_file() else None
        )
    elif candidate_path.is_file() and candidate_path.suffix.lower() == ".pt":
        print("exporting candidate to OpenVINO: %s" % candidate_path, flush=True)
        candidate_dir, candidate_pt = export_openvino(
            candidate_path, out_dir, args.imgsz, args.resume
        )
    else:
        sys.exit("candidate must be a .pt file or an OpenVINO model directory")

    model_dirs = {"active": active_dir, "candidate": Path(candidate_dir)}
    source_pts = {"active": active_pt, "candidate": candidate_pt}
    labelmaps = {
        tag: labelmap_for(tag, model_dirs[tag], out_dir) for tag in ("active", "candidate")
    }
    for tag in ("active", "candidate"):
        if not (model_dirs[tag] / "best.xml").is_file():
            sys.exit("%s model missing best.xml: %s" % (tag, model_dirs[tag]))

    predictions = {}
    models = {}
    elapsed = {}
    for tag in ("active", "candidate"):
        print("model %s: %s" % (tag, model_dirs[tag]), flush=True)
        predictions[tag], models[tag], elapsed[tag] = run_model(
            tag,
            model_dirs[tag],
            labelmaps[tag],
            run_items,
            args,
            out_dir,
            thresholds,
            source_pts[tag],
        )

    if args.limit > 0:
        for item in core:
            if item["key"] in predictions["active"]:
                item.pop("partial", None)

    joined = build_joined(core, predictions)
    metrics = {}
    aucs = {}
    sweeps = {}
    flip_counts = {}
    for tag in ("active", "candidate"):
        metrics[tag] = {}
        for threshold in thresholds:
            metrics[tag][fmt_threshold(threshold)] = metrics_for_model(
                joined, tag, threshold
            )
        true_scores = [
            row[tag]["max_fire"] for row in joined if row["label"] == "true"
        ]
        false_scores = [
            row[tag]["max_fire"] for row in joined if row["label"] == "false"
        ]
        metrics[tag]["dist_true"] = distribution(true_scores)
        metrics[tag]["dist_false"] = distribution(false_scores)
        aucs[tag] = {
            "roc_auc": roc_auc(joined, tag),
            "average_precision": average_precision(joined, tag),
        }
        sweeps[tag] = threshold_sweep(joined, tag)

    records = save_per_image(out_dir, joined, thresholds)
    flip_counts = save_flips(out_dir, records, thresholds)
    sanity = sanity_against_store(joined)
    review = build_review(out_dir, mixed, predictions, thresholds)

    config = {
        "generated": utc_stamp(),
        "args": vars(args),
        "thresholds": thresholds,
        "models": models,
        "elapsed_seconds": elapsed,
        "snapshot": snapshot,
        "partial": bool(args.limit),
    }
    counts_summary = {
        "core": len(joined),
        "mixed": len(mixed),
        "excluded": len(excluded),
        "true": sum(1 for row in joined if row["label"] == "true"),
        "false": sum(1 for row in joined if row["label"] == "false"),
        "total": len(items),
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    summary_json = {
        "generated": config["generated"],
        "partial": config["partial"],
        "counts": counts_summary,
        "thresholds": thresholds,
        "models": models,
        "metrics": metrics,
        "auc": aucs,
        "sweeps": sweeps,
        "flips": flip_counts,
        "sanity": sanity,
        "review": review,
        "snapshot": snapshot,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2)
    summary = build_summary(
        config, counts_summary, metrics, aucs, sweeps, flip_counts, sanity, review
    )
    with open(out_dir / "summary.md", "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print()
    print(summary)
    print()
    print("wrote:", out_dir)


if __name__ == "__main__":
    main()
