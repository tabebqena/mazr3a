#!/usr/bin/env python3
"""Rebuild a listener capture's `events.jsonl` index offline.

`isapi_alert_listener.py` appends one JSON object per KEPT multipart part to
`<outdir>/events.jsonl`. If that index is lost or truncated but the capture
itself survives, this tool regenerates it **offline** with the *same* field
extraction the listener uses. It never touches the raw capture and needs no
network.

Sources (auto-detected, best first):

    parts/          each kept part is stored raw (headers + body) as
                    `<run>-NNNNNN-<type>.<ext>`. `NNNNNN` IS the listener's
                    `part` counter, so discarded parts leave numbered gaps
                    (preserved faithfully), and the file mtime is the host
                    receive time (the listener's `ts_host`).  <- exact
    _stream.bin     the same parts concatenated. Fallback only: it is
                    block-buffered, so an abruptly-killed run can be missing
                    its tail; per-part mtimes and part-number gaps cannot be
                    recovered either, so parts are re-numbered sequentially
                    and ts_host is taken from the NVR `<dateTime>` (else "-").

The generated record has the same keys, in the same order, as the listener:
`part, ts_host, content_type, bytes, <xml fields...>, file`.

Usage:
    python3 isapi_rebuild_index.py --outdir /home/dr/isapi_events
        [--source auto|parts|stream]
        [--output FILE]      # default: <outdir>/events.jsonl
        [--backup]           # move a non-empty index to <output>.bak-<ts>
        [--force]            # overwrite a non-empty index in place
        [--dry-run]          # parse + summarise, write nothing
        [--print]            # also echo each record to stdout

Read-only w.r.t. the capture: only the index (or its backup) is written.
No third-party dependencies.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

# Mirror isapi_alert_listener.py (keep in sync).
XML_FIELDS = ("eventType", "channelID", "channelName", "dateTime", "eventState",
              "eventDescription", "activePostCount", "ipAddress", "macAddress",
              "serialNo", "pictureURL", "faceScore", "FaceRect")

PART_RE = re.compile(
    r"^(?P<run>\d{8}-\d{6})-(?P<num>\d+)-(?P<base>.+)\.(?P<ext>[A-Za-z0-9]+)$")


def _xml_field(text: str, name: str) -> str | None:
    m = re.search(rf"<(?:\w+:)?{name}>([^<]*)</(?:\w+:)?{name}>", text, re.I)
    return m.group(1).strip() if m else None


def _ext_for(content_type: str, body: bytes) -> str:
    ct = (content_type or "").lower()
    if "jpeg" in ct or "jpg" in ct or body[:3] == b"\xff\xd8\xff":
        return "jpg"
    if "png" in ct or body[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if "xml" in ct or body.lstrip()[:1] == b"<":
        return "xml"
    if "text" in ct:
        return "txt"
    return "bin"


def _content_type(head: bytes) -> str:
    headers = head.decode("latin-1", "replace")
    for line in headers.splitlines():
        if line.lower().startswith("content-type:"):
            return line.split(":", 1)[1].strip()
    return ""


def _xml_fields(body: bytes) -> dict:
    text = body.decode("utf-8", "replace")
    fields = {f: _xml_field(text, f) for f in XML_FIELDS}
    return {k: v for k, v in fields.items() if v}


def _ts(mtime: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(mtime))


def records_from_parts(outdir: str) -> list[dict]:
    """Exact rebuild from <outdir>/parts/* (part numbers + mtimes preserved)."""
    parts_dir = os.path.join(outdir, "parts")
    entries: list[tuple[int, str, str]] = []
    for name in os.listdir(parts_dir):
        m = PART_RE.match(name)
        if m:
            entries.append((int(m.group("num")), name, os.path.join(parts_dir, name)))
    entries.sort()
    rows: list[dict] = []
    for num, _name, path in entries:
        with open(path, "rb") as fh:
            raw = fh.read()
        head, _, body = raw.partition(b"\r\n\r\n")
        ctype = _content_type(head)
        rec: dict = {
            "part": num,
            "ts_host": _ts(os.path.getmtime(path)),
            "content_type": ctype,
            "bytes": len(body),
        }
        if _ext_for(ctype, body) == "xml":
            rec.update(_xml_fields(body))
        rec["file"] = os.path.relpath(path, outdir)
        rows.append(rec)
    return rows


def _boundary(data: bytes) -> bytes:
    m = re.match(rb"\r?\n?--([^\r\n]+)", data)
    return m.group(1) if m else b"boundary"


def records_from_stream(outdir: str) -> list[dict]:
    """Fallback rebuild from <outdir>/_stream.bin (re-numbered, best-effort)."""
    with open(os.path.join(outdir, "_stream.bin"), "rb") as fh:
        data = fh.read()
    delim = b"--" + _boundary(data)
    rows: list[dict] = []
    i = data.find(delim)
    while i != -1:
        j = data.find(delim, i + len(delim))
        if j == -1:
            break
        part = data[i + len(delim):j]
        if part.startswith(b"\r\n"):
            part = part[2:]
        head, _, body = part.partition(b"\r\n\r\n")
        ctype = _content_type(head)
        num = len(rows) + 1
        nvr_dt = None
        if _ext_for(ctype, body) == "xml":
            nvr_dt = _xml_field(body.decode("utf-8", "replace"), "dateTime")
        rec: dict = {
            "part": num,
            "ts_host": nvr_dt or "-",
            "content_type": ctype,
            "bytes": len(body),
        }
        if _ext_for(ctype, body) == "xml":
            rec.update(_xml_fields(body))
        rec["file"] = None
        rows.append(rec)
        i = j
    return rows


def collect(outdir: str, source: str) -> tuple[list[dict], str]:
    parts_dir = os.path.join(outdir, "parts")
    stream = os.path.join(outdir, "_stream.bin")
    have_parts = os.path.isdir(parts_dir) and any(os.scandir(parts_dir))
    have_stream = os.path.isfile(stream)
    if source in ("auto", "parts") and have_parts:
        return records_from_parts(outdir), "parts"
    if source == "parts":
        raise SystemExit(f"error: no parts/ under {outdir}")
    if source in ("auto", "stream") and have_stream:
        return records_from_stream(outdir), "stream"
    raise SystemExit(f"error: no parts/ or _stream.bin under {outdir}")


def _summary(rows: list[dict]) -> None:
    by_type: dict[str, int] = {}
    by_run: dict[str, int] = {}
    for r in rows:
        et = r.get("eventType", f"<{r.get('content_type') or '?'}>")
        by_type[et] = by_type.get(et, 0) + 1
        f = r.get("file") or ""
        m = PART_RE.match(os.path.basename(f)) if f else None
        run = m.group("run") if m else "?"
        by_run[run] = by_run.get(run, 0) + 1
    print(f"# records={len(rows)}")
    for run, n in sorted(by_run.items()):
        print(f"    run {run}: {n}")
    for et, n in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"    {n:>6}  {et}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="/tmp/isapi_events",
                    help="capture dir (default: %(default)s)")
    ap.add_argument("--source", choices=("auto", "parts", "stream"), default="auto")
    ap.add_argument("--output", default=None,
                    help="index path (default: <outdir>/events.jsonl)")
    ap.add_argument("--backup", action="store_true",
                    help="move a non-empty index to <output>.bak-<ts> first")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a non-empty index in place")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse + summarise, write nothing")
    ap.add_argument("--print", dest="echo", action="store_true",
                    help="also echo each record to stdout")
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    if not os.path.isdir(outdir):
        print(f"error: not a directory: {outdir}", file=sys.stderr)
        return 2
    output = os.path.abspath(args.output or os.path.join(outdir, "events.jsonl"))

    rows, used = collect(outdir, args.source)
    print(f"# source={used}  outdir={outdir}")
    _summary(rows)

    if args.dry_run:
        print("# dry-run: nothing written")
        return 0

    if os.path.exists(output) and os.path.getsize(output) > 0:
        if not (args.backup or args.force):
            print(f"error: {output} is non-empty; use --backup or --force",
                  file=sys.stderr)
            return 2
        if args.backup:
            bak = f"{output}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
            os.replace(output, bak)
            print(f"# backed up existing index -> {os.path.basename(bak)}")

    # Write in place (truncate) so the existing file's owner/group/mode survive.
    with open(output, "w", encoding="utf-8") as fh:
        for rec in rows:
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            fh.write(line)
            if args.echo:
                print(line, end="")
    print(f"# wrote {len(rows)} records -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
