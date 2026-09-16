#!/usr/bin/env python3

"""Read-only Hikvision ISAPI — list the notifications currently ACTIVE on the NVR.

Strictly read-only: it opens one streaming **GET** to
`/ISAPI/Event/notification/alertStream` (HTTPS + HTTP Digest, self-signed cert)
and never POSTs/PUTs/DELETEs, so no device configuration can change.

Hikvision does not expose a "tell me the alarms that are active right now"
snapshot endpoint. Events arrive on `alertStream` as state transitions: an
`<EventNotificationAlert>` whose `<eventState>` is `active` means the
notification turned on; `inactive` means it turned off. This script keeps a
running set of notifications whose last observed state is `active`, keyed by
(eventType, channel, eventDescription), and reports:

    * a line per transition ([ACTIVE] / [INACTIVE])
    * the full current active list after every change
    * a final active list + totals when stopped (Ctrl-C) or after --seconds

Caveats
    * Only transitions actually OBSERVED count. An alarm that was already active
      before this script connected will not appear until its next state change,
      so leave it running to converge.
    * This NVR limits concurrent alertStream clients (typically one). Do not run
      this while the long-running recorder (isapi_alert_listener.py) already
      holds the stream, or the NVR may drop one of the two connections.

Usage:
    python3 isapi_active_notifications.py [--host 192.168.1.4] [--user admin]
        [--pass P] [--seconds 0] [--reconnect] [--discard-types videoloss]
        [--json]

Transport: HTTPS + HTTP Digest. Requires: requests
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time

import requests
import urllib3
from requests.auth import HTTPDigestAuth

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_STREAM = "/ISAPI/Event/notification/alertStream"
# fields lifted out of an XML EventNotificationAlert (best-effort)
XML_FIELDS = ("eventType", "channelID", "channelName", "dateTime",
              "eventState", "eventDescription", "activePostCount")

_stop = False


def _on_sigint(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True
    print("\n# interrupt received, finishing current read…", file=sys.stderr)


def _xml_field(text: str, name: str) -> str | None:
    m = re.search(rf"<(?:\w+:)?{name}>([^<]*)</(?:\w+:)?{name}>", text, re.I)
    return m.group(1).strip() if m else None


def _boundary(content_type: str) -> bytes | None:
    m = re.search(r'boundary="?([^";]+)"?', content_type or "")
    return m.group(1).encode() if m else None


def parse_event(body: bytes) -> dict | None:
    """Best-effort EventNotificationAlert -> {field: value}; None if not an event."""
    text = body.decode("utf-8", "replace")
    if "<eventType>" not in text and "EventNotificationAlert" not in text:
        return None
    fields = {f: _xml_field(text, f) for f in XML_FIELDS}
    return {k: v for k, v in fields.items() if v}


def _key(rec: dict) -> tuple:
    et = rec.get("eventType", "?")
    ch = rec.get("channelID") or rec.get("channelName") or "-"
    desc = rec.get("eventDescription", "")
    return (et, ch, desc)


class ActiveTracker:
    """Tracks the set of notifications whose <eventState> is currently active."""

    def __init__(self, discard_types: set[str] | None = None):
        self.discard = {t.strip().lower() for t in (discard_types or set())}
        self.active: dict[tuple, dict] = {}
        self.seen: dict[str, int] = {}          # kept eventType -> count
        self.discarded: dict[str, int] = {}     # dropped eventType -> count

    def handle(self, rec: dict) -> str | None:
        """Apply one event; returns 'active', 'inactive' or None (ignored/discarded)."""
        et = rec.get("eventType", "?")
        if et.lower() in self.discard:
            self.discarded[et] = self.discarded.get(et, 0) + 1
            return None

        state = (rec.get("eventState") or "").lower()
        self.seen[et] = self.seen.get(et, 0) + 1

        if state == "active":
            self.active[_key(rec)] = {
                "eventType": et,
                "channel": rec.get("channelID") or rec.get("channelName") or "-",
                "description": rec.get("eventDescription", ""),
                "activePostCount": rec.get("activePostCount", ""),
                "since_host": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "nvr": rec.get("dateTime", ""),
            }
            return "active"
        if state == "inactive":
            self.active.pop(_key(rec), None)
            return "inactive"
        return None

    def active_list(self) -> list[dict]:
        return [r for _, r in sorted(
            self.active.items(), key=lambda kv: (kv[1]["eventType"], kv[1]["channel"]))]

    def print_active(self) -> None:
        rows = self.active_list()
        print(f"\n# currently ACTIVE notifications: {len(rows)}")
        if not rows:
            print("  (none)")
            return
        for r in rows:
            print(f"  {r['eventType']:<20} ch={r['channel']:<4} "
                  f"{r['description']:<28} posts={r['activePostCount']:<3} "
                  f"since={r['since_host']} nvr={r['nvr']}")


def _iter_parts(resp, boundary: bytes):
    """Yield each multipart part body (without boundary lines)."""
    buf = bytearray()
    delim = b"--" + boundary
    for chunk in resp.iter_content(chunk_size=4096):
        if _stop:
            return
        if not chunk:
            continue
        buf.extend(chunk)
        while True:
            i = buf.find(delim)
            if i == -1:
                break
            j = buf.find(delim, i + len(delim))
            if j == -1:
                break
            part = bytes(buf[i + len(delim):j])
            del buf[:j]
            if part.startswith(b"\r\n"):
                part = part[2:]
            if not part or part.startswith(b"--"):
                continue
            yield part


def run_once(session, url: str, tracker: ActiveTracker) -> bool:
    """One alertStream connection. Returns True if it ended cleanly."""
    with session.get(url, stream=True, verify=False, timeout=(15, 120)) as resp:
        print(f"# connected http={resp.status_code} "
              f"type={resp.headers.get('Content-Type', '')}")
        if resp.status_code != 200:
            print(resp.text[:400])
            return False
        boundary = _boundary(resp.headers.get("Content-Type", "")) or b"boundary"
        print(f"# boundary={boundary!r}")
        for part in _iter_parts(resp, boundary):
            _head, _, body = part.partition(b"\r\n\r\n")
            rec = parse_event(body)
            if rec is None:
                continue
            kind = tracker.handle(rec)
            et = rec.get("eventType", "?")
            ch = rec.get("channelID") or rec.get("channelName") or "-"
            desc = rec.get("eventDescription", "")
            if kind == "active":
                print(f"[ACTIVE]   {et:<20} ch={ch:<4} {desc:<28} "
                      f"nvr={rec.get('dateTime', '')}")
                tracker.print_active()
            elif kind == "inactive":
                print(f"[INACTIVE] {et:<20} ch={ch:<4} {desc:<28} "
                      f"nvr={rec.get('dateTime', '')}")
                tracker.print_active()
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("ISAPI_HOST", "192.168.1.4"))
    ap.add_argument("--user", default=os.environ.get("ISAPI_USER", "admin"))
    ap.add_argument("--pass", dest="pwd", default=os.environ.get("ISAPI_PASS", ""))
    ap.add_argument("--stream", default=DEFAULT_STREAM,
                    help="ISAPI stream path (default: alertStream)")
    ap.add_argument("--seconds", type=int, default=0,
                    help="run window; 0 = run until Ctrl-C (default: %(default)s)")
    ap.add_argument("--discard-types", default="",
                    help="csv of XML eventTypes to ignore, case-insensitive "
                         "(default: keep everything)")
    ap.add_argument("--reconnect", action="store_true",
                    help="reopen the stream if it drops")
    ap.add_argument("--json", action="store_true",
                    help="also print the final active list as JSON")
    args = ap.parse_args()

    if not args.pwd:
        print("error: no password (--pass or ISAPI_PASS)", file=sys.stderr)
        return 2

    signal.signal(signal.SIGINT, _on_sigint)
    base = f"https://{args.host}"
    url = base + args.stream
    session = requests.Session()
    session.auth = HTTPDigestAuth(args.user, args.pwd)
    session.headers["Accept"] = "multipart/mixed"

    discard = {t.strip() for t in args.discard_types.split(",") if t.strip()}
    tracker = ActiveTracker(discard)

    print(f"# read-only ISAPI active-notification tracker: {url}")
    print(f"# discard-types={','.join(sorted(discard)) or '(none)'}  "
          f"window={'∞' if args.seconds <= 0 else str(args.seconds) + 's'}  "
          f"reconnect={args.reconnect}")

    deadline = None if args.seconds <= 0 else time.time() + args.seconds
    while True:
        if deadline and time.time() >= deadline:
            print("# time window elapsed")
            break
        try:
            clean = run_once(session, url, tracker)
        except requests.RequestException as exc:
            print(f"# stream error: {type(exc).__name__}: {exc}")
            clean = False
        if _stop:
            break
        if not args.reconnect:
            break
        if deadline and time.time() >= deadline:
            break
        if not clean:
            time.sleep(3)

    tracker.print_active()

    print("\n# SUMMARY seen:")
    for et, n in sorted(tracker.seen.items(), key=lambda x: -x[1]):
        print(f"    {n:>5}  {et}")
    for et, n in sorted(tracker.discarded.items(), key=lambda x: -x[1]):
        print(f"    {n:>5}  {et} (discarded)")

    if args.json:
        print("\n# JSON:")
        print(json.dumps(tracker.active_list(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
