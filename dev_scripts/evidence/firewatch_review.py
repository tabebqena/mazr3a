#!/usr/bin/env python3
import argparse
import csv
import io
import json
import mimetypes
import os
import re
import sqlite3
import struct
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

STORE_PREFIX = "/media/firewatch/"
NAME_RE = re.compile(
    r"^(\d{8})_(\d{6})_(\d{3})_cam[0-9]+_conf[0-9.]+(_annotated)?\.jpe?g$", re.I
)
LABEL_FIELDS = ["key", "frame_id", "camera", "file", "verdict", "updated_at"]


def num(value):
    if isinstance(value, (bytes, bytearray)):
        if len(value) >= 4:
            return struct.unpack("<f", bytes(value[:4]))[0]
        return 0.0
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def frame_rel(path, camera):
    p = str(path or "").replace("\\", "/")
    if STORE_PREFIX in p:
        p = p.split(STORE_PREFIX, 1)[1]
    else:
        p = p.rsplit("/", 1)[-1]
    if "/" not in p:
        p = "{}/{}".format(camera, p)
    return p


def filename_ts(name):
    m = re.match(r"^(\d{8})_(\d{6})_(\d{3})", name)
    if not m:
        return 0.0
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return dt.replace(tzinfo=timezone.utc).timestamp() + int(m.group(3)) / 1000.0
    except ValueError:
        return 0.0


def load_frames(db_path):
    frames = {}
    detections = {}
    if not db_path.is_file():
        return frames
    try:
        con = sqlite3.connect(
            "file:{}?mode=ro".format(quote(str(db_path), safe="/")),
            uri=True,
            timeout=10,
        )
        con.row_factory = sqlite3.Row
        try:
            for row in con.execute(
                "SELECT id, camera, captured_at, ts_utc, jpg_path, best_score, "
                "score_threshold, alerted, annotated_path FROM frames ORDER BY id"
            ):
                rel = frame_rel(row["jpg_path"], row["camera"])
                frames[rel] = {
                    "frame_id": row["id"],
                    "camera": row["camera"],
                    "captured_at": num(row["captured_at"]),
                    "ts_utc": row["ts_utc"] or "",
                    "best_score": num(row["best_score"]),
                    "score_threshold": num(row["score_threshold"]),
                    "alerted": bool(row["alerted"]),
                    "annotated_rel": (
                        frame_rel(row["annotated_path"], row["camera"])
                        if row["annotated_path"]
                        else ""
                    ),
                    "detections": [],
                }
            for row in con.execute(
                "SELECT frame_id, label, score, x1, y1, x2, y2 "
                "FROM detections ORDER BY id"
            ):
                detections.setdefault(row["frame_id"], []).append(
                    {
                        "label": row["label"],
                        "score": num(row["score"]),
                        "x1": num(row["x1"]),
                        "y1": num(row["y1"]),
                        "x2": num(row["x2"]),
                        "y2": num(row["y2"]),
                    }
                )
        finally:
            con.close()
    except sqlite3.Error as exc:
        print("warning: could not read {}: {}".format(db_path, exc))
        return {}
    for frame in frames.values():
        frame["detections"] = detections.get(frame["frame_id"], [])
    return frames


def load_manifest(path):
    rows = []
    if not path.is_file():
        return rows
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def build_items(evidence, frames, manifest):
    raw = {}
    annotated = {}
    for row in manifest:
        rel = (row.get("file") or "").replace("\\", "/")
        if not rel:
            continue
        name = rel.rsplit("/", 1)[-1]
        if row.get("kind") == "annotated" or name.endswith("_annotated.jpg"):
            annotated[rel] = row
        elif NAME_RE.match(name):
            raw[rel] = row
    if not raw:
        for p in sorted(evidence.rglob("*")):
            if not p.is_file():
                continue
            name = p.name
            if not NAME_RE.match(name):
                continue
            rel = p.relative_to(evidence).as_posix()
            if name.endswith("_annotated.jpg"):
                annotated[rel] = {"file": rel}
            else:
                raw[rel] = {"file": rel}

    items = []
    for rel, row in raw.items():
        path = evidence / rel
        if not path.is_file():
            continue
        frame = frames.get(rel)
        camera = (frame or {}).get("camera") or row.get("camera") or rel.split("/")[0]
        captured_at = (frame or {}).get("captured_at") or filename_ts(path.name)
        ts_utc = (frame or {}).get("ts_utc") or ""
        if not ts_utc and captured_at:
            ts_utc = datetime.fromtimestamp(
                captured_at, timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S")
        base = rel.rsplit(".", 1)[0]
        annotated_rel = ""
        if frame and frame.get("annotated_rel"):
            candidate = frame["annotated_rel"]
            if (evidence / candidate).is_file():
                annotated_rel = candidate
        if not annotated_rel:
            candidate = base + "_annotated.jpg"
            if candidate in annotated or (evidence / candidate).is_file():
                annotated_rel = candidate
        dets = (frame or {}).get("detections") or []
        items.append(
            {
                "key": rel,
                "file": rel,
                "annotated": annotated_rel,
                "camera": camera,
                "ts_utc": ts_utc,
                "captured_at": captured_at,
                "best_score": (frame or {}).get("best_score", 0.0),
                "score_threshold": (frame or {}).get("score_threshold", 0.0),
                "alerted": bool((frame or {}).get("alerted")),
                "frame_id": (frame or {}).get("frame_id"),
                "orphan": frame is None,
                "labels": sorted({d["label"] for d in dets}),
                "detections": dets,
                "verdict": "",
            }
        )
    items.sort(key=lambda i: (i["captured_at"], i["key"]), reverse=True)
    return items


def load_labels(path):
    labels = {}
    if not path.is_file():
        return labels
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("key") or "").strip()
            verdict = (row.get("verdict") or "").strip().lower()
            if key and verdict in ("true", "false"):
                labels[key] = {
                    "key": key,
                    "frame_id": row.get("frame_id") or "",
                    "camera": row.get("camera") or "",
                    "file": row.get("file") or key,
                    "verdict": verdict,
                    "updated_at": row.get("updated_at") or "",
                }
    return labels


def write_labels(path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for record in sorted(labels.values(), key=lambda r: r["key"]):
            writer.writerow(record)
    os.replace(tmp, path)


class State:
    def __init__(self, evidence, labels_path, items):
        self.evidence = evidence
        self.labels_path = labels_path
        self.items = items
        self.by_key = {item["key"]: item for item in items}
        self.lock = threading.Lock()
        self.labels = load_labels(labels_path)
        for key, record in self.labels.items():
            if key in self.by_key:
                self.by_key[key]["verdict"] = record["verdict"]

    def classify(self, key, verdict):
        item = self.by_key.get(key)
        if item is None:
            return None
        with self.lock:
            item["verdict"] = verdict
            if verdict:
                self.labels[key] = {
                    "key": key,
                    "frame_id": item.get("frame_id") or "",
                    "camera": item.get("camera") or "",
                    "file": item.get("file") or key,
                    "verdict": verdict,
                    "updated_at": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                }
            else:
                self.labels.pop(key, None)
            write_labels(self.labels_path, self.labels)
        return item

    def export_csv(self):
        with self.lock:
            records = sorted(self.labels.values(), key=lambda r: r["key"])
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(records)
        return buf.getvalue().encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "FirewatchReview/1.0"

    def log_message(self, fmt, *args):
        pass

    def _json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _bytes(self, data, content_type, cache=True, download=None):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header(
            "Cache-Control", "public, max-age=3600" if cache else "no-store"
        )
        if download:
            self.send_header(
                "Content-Disposition", 'attachment; filename="{}"'.format(download)
            )
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path, content_type=None, cache=True, download=None):
        if path is None or not path.is_file():
            self._json({"error": "not found"}, 404)
            return
        self._bytes(
            path.read_bytes(),
            content_type
            or mimetypes.guess_type(path.name)[0]
            or "application/octet-stream",
            cache=cache,
            download=download,
        )

    def _safe_path(self, root, rel):
        try:
            target = (root / unquote(rel)).resolve()
            target.relative_to(root.resolve())
        except (ValueError, OSError):
            return None
        return target if target.is_file() else None

    def do_GET(self):
        path = urlparse(self.path).path
        state = self.server.state
        if path in ("/", "/index.html"):
            self._file(STATIC / "index.html", "text/html; charset=utf-8", cache=False)
        elif path == "/api/items":
            self._json({"items": state.items})
        elif path == "/api/export":
            self._bytes(
                state.export_csv(),
                "text/csv; charset=utf-8",
                cache=False,
                download="firewatch-classifications.csv",
            )
        elif path.startswith("/static/"):
            target = self._safe_path(STATIC, path[len("/static/") :])
            self._file(target, cache=True)
        elif path.startswith("/evidence/"):
            target = self._safe_path(state.evidence, path[len("/evidence/") :])
            self._file(target, cache=True)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/classify":
            self._json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json({"error": "invalid json"}, 400)
            return
        key = str(payload.get("key") or "")
        verdict = str(payload.get("verdict") or "").strip().lower()
        if verdict not in ("true", "false", ""):
            self._json({"error": "verdict must be true, false, or clear"}, 400)
            return
        item = self.server.state.classify(key, verdict)
        if item is None:
            self._json({"error": "unknown key"}, 404)
            return
        self._json({"ok": True, "key": key, "verdict": item["verdict"]})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local gallery to review firewatch evidence and mark true/false detections."
    )
    parser.add_argument(
        "--evidence-dir",
        default="./firewatch-evidence",
        help="evidence export directory (default: ./firewatch-evidence)",
    )
    parser.add_argument(
        "--db",
        default="",
        help="firewatch.db snapshot (default: <evidence-dir>/firewatch.db)",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="classification CSV (default: <evidence-dir>/classifications.csv)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8099, help="bind port")
    return parser.parse_args()


STATIC = Path(__file__).resolve().parent / "static"


def main():
    args = parse_args()
    evidence = Path(args.evidence_dir).expanduser().resolve()
    if not evidence.is_dir():
        raise SystemExit("evidence directory not found: {}".format(evidence))
    db_path = Path(args.db).expanduser().resolve() if args.db else evidence / "firewatch.db"
    labels_path = (
        Path(args.labels).expanduser().resolve()
        if args.labels
        else evidence / "classifications.csv"
    )
    frames = load_frames(db_path)
    manifest = load_manifest(evidence / "manifest.csv")
    items = build_items(evidence, frames, manifest)
    state = State(evidence, labels_path, items)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.state = state
    print("Firewatch review: http://{}:{}/".format(args.host, args.port))
    print("Evidence: {}".format(evidence))
    print("Labels:   {}".format(labels_path))
    print("Items:    {}".format(len(items)))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
