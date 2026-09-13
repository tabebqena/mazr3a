#!/usr/bin/env python3
"""General TCP+UDP port listener — records EVERYTHING received (default :514).

Protocol-agnostic: it does not care what the payload is. Whatever bytes arrive
on the bound port are written to disk **exactly as received**, with source
metadata. An optional (best-effort) syslog decode enriches the JSON index but
never replaces the raw capture.

Typical use: the Hikvision NVR's Log Server (`isSupportLogServer=true`) pushes
its logs to `<host>:514`; this captures them verbatim.

    # port 514 is privileged (<1024) — see --help / the install helper
    sudo python3 port_listener.py --port 514 --outdir /var/log/nvr
    # or run via a systemd unit that grants CAP_NET_BIND_SERVICE (no root user)

Storage under --outdir:
    capture.raw    framed records, byte-exact:
                   [u32 BE len][u8 proto][u16 BE src_port][u32 BE src_ip]
                   [u64 BE epoch_ms][payload]
                   proto: 17 = UDP, 6 = TCP
    records.jsonl  one JSON object per received chunk:
                   seq, ts, proto, src, bytes, b64 (payload, base64),
                   text (if printable) + syslog fields when detected
    capture.log    human-readable "<ts> <proto> <src> <len>B <preview>"

Usage:
    python3 port_listener.py [--bind 0.0.0.0] [--port 514]
        [--proto udp|tcp|both] [--seconds 0] [--max-records 0]
        [--outdir /var/log/nvr] [--fallback-port N] [--no-syslog]
        [--hexdump] [--quiet]

Requires: Python 3.8+ (stdlib only).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import selectors
import signal
import socket
import struct
import sys
import time
from typing import Any, cast

PROTO_NUM = {"tcp": 6, "udp": 17}

# --- optional syslog decode (best-effort; raw is always stored) -------------
PRI_RE = re.compile(rb"^<(\d{1,3})>")
RFC5424_RE = re.compile(
    rb"^<(\d{1,3})>(\d) (\S+) (\S+) (\S+) (\S+) (\S+) (.*)$", re.S)
RFC3164_TS_RE = re.compile(
    rb"^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(.*)$", re.S)
ISO_TS_RE = re.compile(rb"^(\d{4}-\d{2}-\d{2}T\S+)\s+(\S+)\s+(.*)$", re.S)
TAG_RE = re.compile(rb"^\[?([A-Za-z0-9_.\-/]+)\]?(?:\[\d+\])?:\s?(.*)$", re.S)
FACILITIES = {0: "kern", 1: "user", 2: "mail", 3: "daemon", 4: "auth",
              5: "syslog", 6: "lpr", 7: "news", 8: "uucp", 9: "cron",
              10: "authpriv", 11: "ftp", 12: "ntp", 13: "security",
              14: "console", 15: "solaris-cron", 16: "local0", 17: "local1",
              18: "local2", 19: "local3", 20: "local4", 21: "local5",
              22: "local6", 23: "local7"}
SEVERITIES = {0: "emerg", 1: "alert", 2: "crit", 3: "err", 4: "warning",
              5: "notice", 6: "info", 7: "debug"}

_stop = False


def _on_signal(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True


def decode_syslog(data: bytes) -> dict:
    """Best-effort; returns {} if it doesn't look like syslog."""
    line = data.rstrip(b"\r\n\x00")
    m = RFC5424_RE.match(line)
    if m:
        pri = int(m.group(1))
        return {"pri": pri, "severity": SEVERITIES.get(pri & 7),
                "facility": FACILITIES.get(pri >> 3),
                "syslog_version": int(m.group(2)),
                "syslog_ts": m.group(3).decode("latin-1"),
                "hostname": m.group(4).decode("latin-1"),
                "app": m.group(5).decode("latin-1"),
                "procid": m.group(6).decode("latin-1"),
                "msgid": m.group(7).decode("latin-1"),
                "msg": m.group(8).decode("latin-1", "replace")}
    pm = PRI_RE.match(line)
    if not pm:
        return {}
    pri = int(pm.group(1))
    out: dict[str, Any] = {"pri": pri, "severity": SEVERITIES.get(pri & 7),
                           "facility": FACILITIES.get(pri >> 3)}
    rest = line[pm.end():]
    for rx in (RFC3164_TS_RE, ISO_TS_RE):
        tm = rx.match(rest.strip())
        if tm:
            out["syslog_ts"] = tm.group(1).decode("latin-1")
            out["hostname"] = tm.group(2).decode("latin-1")
            rest = tm.group(3)
            break
    tm = TAG_RE.match(rest)
    if tm:
        out["tag"] = tm.group(1).decode("latin-1")
        out["msg"] = tm.group(2).decode("latin-1", "replace")
    else:
        out["msg"] = rest.decode("latin-1", "replace")
    return out


class Store:
    def __init__(self, outdir: str, quiet: bool, decode: bool, hexdump: bool):
        os.makedirs(outdir, exist_ok=True)
        self.quiet = quiet
        self.decode = decode
        self.hexdump = hexdump
        self.raw_fh = open(os.path.join(outdir, "capture.raw"), "ab", buffering=0)
        self.json_fh = open(os.path.join(outdir, "records.jsonl"), "a",
                            encoding="utf-8", buffering=1)
        self.log_fh = open(os.path.join(outdir, "capture.log"), "a",
                           encoding="utf-8", buffering=1)
        self.count = 0
        self.bytes_total = 0

    def close(self):
        for fh in (self.raw_fh, self.json_fh, self.log_fh):
            try:
                fh.close()
            except OSError:
                pass

    def add(self, proto: str, ip: str, port: int, payload: bytes) -> None:
        self.count += 1
        self.bytes_total += len(payload)
        now = time.time()
        # framed raw record (byte-exact)
        hdr = struct.pack(">IBHIQ", len(payload), PROTO_NUM[proto], port,
                          struct.unpack(">I", socket.inet_aton(ip))[0],
                          int(now * 1000))
        self.raw_fh.write(hdr + payload)

        printable = all(9 <= b < 127 or b in (10, 13) for b in payload[:64])
        rec: dict[str, Any] = {
            "seq": self.count,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
            "proto": proto,
            "src": f"{ip}:{port}",
            "bytes": len(payload),
            "b64": base64.b64encode(payload).decode("ascii"),
        }
        if printable:
            rec["text"] = payload.decode("latin-1")
        if self.decode:
            rec.update(decode_syslog(payload))
        self.json_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        preview = payload[:120].decode("latin-1", "replace").replace("\n", " ")
        self.log_fh.write(f"{rec['ts']} {proto} {ip}:{port} {len(payload)}B {preview}\n")

        if not self.quiet:
            sev = rec.get("severity", "-")
            print(f"[{self.count:07d}] {proto}/{ip}:{port} {len(payload):>6}B "
                  f"sev={sev:<7} {preview!r}")
            if self.hexdump:
                for off in range(0, min(len(payload), 128), 16):
                    chunk = payload[off:off + 16]
                    hx = " ".join(f"{b:02x}" for b in chunk)
                    asc = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
                    print(f"            {off:04x}  {hx:<47}  {asc}")


def make_sockets(bind: str, port: int, proto: str, reuse_port: bool):
    out = []

    def reuse(s: socket.socket) -> None:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if reuse_port and hasattr(socket, "SO_REUSEPORT"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

    if proto in ("udp", "both"):
        u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        reuse(u)
        u.bind((bind, port))
        u.setblocking(False)
        out.append(("udp", u))
    if proto in ("tcp", "both"):
        t = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        reuse(t)
        t.bind((bind, port))
        t.listen(64)
        t.setblocking(False)
        out.append(("tcp", t))
    return out


def bind_or_explain(bind: str, port: int, proto: str, fallback: int | None,
                    reuse_port: bool):
    try:
        return make_sockets(bind, port, proto, reuse_port), port
    except PermissionError:
        print(f"# cannot bind {bind}:{port}: Permission denied.\n"
              f"#   port {port} is privileged (<1024). Fix by one of:\n"
              f"#     a) run as root:            sudo python3 {sys.argv[0]} --port {port}\n"
              f"#     b) grant the capability:   sudo setcap cap_net_bind_service=+ep "
              f"$(readlink -f $(command -v python3))\n"
              f"#     c) install the systemd unit (see dev_scripts/install_log_listener.sh)\n"
              f"#     d) test on an unprivileged port: --port 5514",
              file=sys.stderr)
    except OSError as exc:
        print(f"# cannot bind {bind}:{port}: {exc}", file=sys.stderr)
    if fallback:
        print(f"# falling back to port {fallback}", file=sys.stderr)
        return make_sockets(bind, fallback, proto, reuse_port), fallback
    raise SystemExit(3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default=os.environ.get("PORT_LISTEN_BIND", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT_LISTEN_PORT", "514")))
    ap.add_argument("--proto", choices=("udp", "tcp", "both"), default="both")
    ap.add_argument("--seconds", type=int, default=0, help="0 = forever")
    ap.add_argument("--max-records", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--outdir", default=os.environ.get("PORT_LISTEN_OUTDIR", "/var/log/nvr"))
    ap.add_argument("--fallback-port", type=int, default=None)
    ap.add_argument("--no-syslog", action="store_true", help="skip syslog decoding")
    ap.add_argument("--reuse-port", action="store_true", help="set SO_REUSEPORT")
    ap.add_argument("--hexdump", action="store_true", help="hex-dump payloads")
    ap.add_argument("--quiet", action="store_true", help="don't print records")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    socks, port = bind_or_explain(args.bind, args.port, args.proto,
                                  args.fallback_port, args.reuse_port)
    sel = selectors.DefaultSelector()
    for proto, s in socks:
        sel.register(s, selectors.EVENT_READ, proto)

    store = Store(args.outdir, args.quiet, not args.no_syslog, args.hexdump)
    deadline = None if args.seconds <= 0 else time.time() + args.seconds
    print(f"# general port listener on {args.bind}:{port} proto={args.proto} "
          f"outdir={args.outdir} window={'inf' if deadline is None else str(args.seconds)+'s'}")

    conns: dict[socket.socket, tuple[str, int]] = {}
    try:
        while not _stop:
            if deadline and time.time() > deadline:
                print("# time window elapsed")
                break
            if args.max_records and store.count >= args.max_records:
                print("# reached --max-records")
                break
            for key, _ in sel.select(timeout=1.0):
                kind = cast(str, key.data)
                sock = cast(socket.socket, key.fileobj)
                if kind == "udp":
                    data, addr = sock.recvfrom(65535)
                    if data:
                        store.add("udp", addr[0], addr[1], data)
                elif kind == "tcp":
                    try:
                        conn, addr = sock.accept()
                    except OSError as exc:
                        print(f"# accept skipped: {exc}")
                        continue
                    conn.setblocking(False)
                    conns[conn] = (addr[0], addr[1])
                    sel.register(conn, selectors.EVENT_READ, "conn")
                    print(f"# tcp connect {addr[0]}:{addr[1]}")
                elif kind == "conn":
                    try:
                        chunk = sock.recv(65535)
                    except OSError:
                        chunk = b""
                    if not chunk:
                        ip, pt = conns.pop(sock, ("?", 0))
                        sel.unregister(sock)
                        sock.close()
                        print(f"# tcp close {ip}:{pt}")
                        continue
                    ip, pt = conns.get(sock, ("?", 0))
                    store.add("tcp", ip, pt, chunk)
    finally:
        for c in list(conns):
            try:
                sel.unregister(c)
                c.close()
            except OSError:
                pass
        for _, s in socks:
            s.close()
        store.close()

    print(f"\n# SUMMARY records={store.count} bytes={store.bytes_total} -> "
          f"{args.outdir}/capture.raw, records.jsonl, capture.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
