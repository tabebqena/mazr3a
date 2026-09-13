#!/usr/bin/env python3
"""Syslog listener for NVR logs (default UDP/TCP port 514).

Stands up a **receiving** socket (UDP and/or TCP) so a device configured to
"send logs to <host>:514" (e.g. the Hikvision NVR's Log Server, which reports
`isSupportLogServer=true`) can push its syslog there. It is a passive receiver:
it opens listening sockets only — it never contacts or configures any device.

Everything is stored **raw** plus a parsed index, under --outdir:

    syslog.raw        exact bytes, framed as [u32 BIG-ENDIAN length][payload]
    syslog.jsonl      one JSON object per message: ts, proto, src, bytes, pri,
                      facility, severity, hostname, tag, msg, raw (latin-1, so
                      round-trips byte-exact)
    syslog.log        human-readable: "<ts> <src> <msg>"

Usage:
    # port 514 is privileged (<1024): run as root, or grant the capability
    sudo python3 syslog_listener.py --port 514
    # or, without root:
    sudo setcap cap_net_bind_service=+ep "$(command -v python3)"
    python3 syslog_listener.py --port 514

    # quick local test on an unprivileged port
    python3 syslog_listener.py --port 5514 --seconds 30
    # ... in another shell:
    logger -n 127.0.0.1 -P 5514 "hello from host"

    # allow a fallback port if 514 cannot be bound (handy for testing)
    python3 syslog_listener.py --port 514 --fallback-port 5514

Options: --bind, --port, --proto udp|tcp|both, --seconds (0 = forever),
--max-messages, --outdir, --fallback-port, --quiet.
Requires: Python 3.8+ (stdlib only).
"""
from __future__ import annotations

import argparse
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

# syslog parsing (best-effort): RFC5424 first, then RFC3164, then raw.
PRI_RE = re.compile(r"^<(?P<pri>\d{1,3})>")
RFC5424_RE = re.compile(r"^<(?P<pri>\d{1,3})>(?P<ver>\d)\s+(?P<ts>\S+)\s+"
                        r"(?P<host>\S+)\s+(?P<app>\S+)\s+(?P<proc>\S+)\s+"
                        r"(?P<mid>\S+)\s+(?P<msg>.*)$", re.S)
# RFC3164 timestamp "Mmm dd HH:MM:SS" (day may be space-padded) then hostname
RFC3164_TS_RE = re.compile(
    r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+(?P<rest>.*)$", re.S)
# ISO8601 timestamp then hostname
ISO_TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\S+)\s+(?P<host>\S+)\s+"
                       r"(?P<rest>.*)$", re.S)
TAG_RE = re.compile(r"^\[?(?P<tag>[A-Za-z0-9_.\-/]+)\]?(?:\[\d+\])?:\s?(?P<msg>.*)$",
                    re.S)

# syslog facility/severity tables
FACILITIES = {
    0: "kern", 1: "user", 2: "mail", 3: "daemon", 4: "auth", 5: "syslog",
    6: "lpr", 7: "news", 8: "uucp", 9: "cron", 10: "authpriv", 11: "ftp",
    16: "local0", 17: "local1", 18: "local2", 19: "local3", 20: "local4",
    21: "local5", 22: "local6", 23: "local7",
}
SEVERITIES = {0: "emerg", 1: "alert", 2: "crit", 3: "err", 4: "warning",
              5: "notice", 6: "info", 7: "debug"}

_stop = False


def _on_signal(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True


def parse_syslog(data: bytes) -> dict:
    """Best-effort parse of a syslog datagram/line into fields."""
    text = data.decode("utf-8", "replace").rstrip("\r\n\x00")
    out: dict[str, Any] = {"msg": text}

    m = RFC5424_RE.match(text)
    if m:
        pri = int(m.group("pri"))
        out.update(version=int(m.group("ver")), timestamp=m.group("ts"),
                   hostname=m.group("host"), tag=m.group("app"),
                   procid=m.group("proc"), msgid=m.group("mid"),
                   msg=m.group("msg"))
    else:
        pri_m = PRI_RE.match(text)
        pri = int(pri_m.group("pri")) if pri_m else None
        rest = text[pri_m.end():] if pri_m else text
        ts = host = None
        for rx in (RFC3164_TS_RE, ISO_TS_RE):
            tm = rx.match(rest.strip())
            if tm:
                ts, host, rest = tm.group("ts"), tm.group("host"), tm.group("rest")
                break
        tm = TAG_RE.match(rest)
        if tm:
            out["tag"] = tm.group("tag")
            out["msg"] = tm.group("msg")
        else:
            out["msg"] = rest
        if ts:
            out["timestamp"] = ts
        if host:
            out["hostname"] = host

    if pri is not None:
        out["pri"] = pri
        out["facility"] = FACILITIES.get(pri >> 3, str(pri >> 3))
        out["severity"] = SEVERITIES.get(pri & 7, str(pri & 7))
    return out


class Store:
    def __init__(self, outdir: str, quiet: bool):
        os.makedirs(outdir, exist_ok=True)
        self.quiet = quiet
        self.raw_fh = open(os.path.join(outdir, "syslog.raw"), "ab", buffering=0)
        self.json_fh = open(os.path.join(outdir, "syslog.jsonl"), "a",
                            encoding="utf-8", buffering=1)
        self.log_fh = open(os.path.join(outdir, "syslog.log"), "a",
                           encoding="utf-8", buffering=1)
        self.count = 0

    def close(self):
        for fh in (self.raw_fh, self.json_fh, self.log_fh):
            try:
                fh.close()
            except OSError:
                pass

    def add(self, proto: str, src: str, data: bytes) -> None:
        self.count += 1
        # exact bytes, length-framed
        self.raw_fh.write(struct.pack(">I", len(data)) + data)
        rec = {
            "seq": self.count,
            "ts_host": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "proto": proto,
            "src": src,
            "bytes": len(data),
            # latin-1 keeps a 1:1 byte<->codepoint map, so re-encoding is exact
            "raw": data.decode("latin-1"),
        }
        rec.update(parse_syslog(data))
        self.json_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.log_fh.write(f"{rec['ts_host']} {proto}/{src} {rec.get('msg','')}\n")
        if not self.quiet:
            sev = rec.get("severity", "-")
            tag = rec.get("tag", "-")
            host = rec.get("hostname", "-")
            msg = rec.get("msg", "").replace("\n", " ")[:160]
            print(f"[{self.count:06d}] {proto}/{src} sev={sev:<7} "
                  f"{host}/{tag}: {msg}")


def make_sockets(bind: str, port: int, proto: str):
    socks = []
    if proto in ("udp", "both"):
        u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        u.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        u.bind((bind, port))
        u.setblocking(False)
        socks.append(("udp", u))
    if proto in ("tcp", "both"):
        t = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        t.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        t.bind((bind, port))
        t.listen(16)
        t.setblocking(False)
        socks.append(("tcp", t))
    return socks


def bind_with_fallback(bind: str, port: int, proto: str, fallback: int | None):
    """Bind (host, port); on EACCES/EADDRINUSE, try --fallback-port."""
    try:
        return make_sockets(bind, port, proto), port
    except PermissionError:
        print(f"# cannot bind {bind}:{port} (privileged port <1024).\n"
              f"#   run with sudo, or: sudo setcap cap_net_bind_service=+ep "
              f"$(command -v python3)", file=sys.stderr)
    except OSError as exc:
        print(f"# cannot bind {bind}:{port}: {exc}", file=sys.stderr)
    if fallback:
        print(f"# falling back to port {fallback}", file=sys.stderr)
        return make_sockets(bind, fallback, proto), fallback
    raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default=os.environ.get("SYSLOG_BIND", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("SYSLOG_PORT", "514")))
    ap.add_argument("--proto", choices=("udp", "tcp", "both"), default="udp")
    ap.add_argument("--seconds", type=int, default=0,
                    help="run window; 0 = forever (Ctrl-C to stop)")
    ap.add_argument("--max-messages", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--outdir", default="/tmp/syslog")
    ap.add_argument("--fallback-port", type=int, default=None,
                    help="use this port if --port cannot be bound")
    ap.add_argument("--quiet", action="store_true", help="don't print messages")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    socks, port = bind_with_fallback(args.bind, args.port, args.proto,
                                     args.fallback_port)
    sel = selectors.DefaultSelector()
    for proto, s in socks:
        sel.register(s, selectors.EVENT_READ, proto)

    store = Store(args.outdir, args.quiet)
    deadline = None if args.seconds <= 0 else time.time() + args.seconds
    print(f"# syslog listener on {args.bind}:{port} proto={args.proto} "
          f"outdir={args.outdir} window={'∞' if deadline is None else str(args.seconds)+'s'}")

    tcp_conns: dict[socket.socket, bytearray] = {}
    try:
        while not _stop:
            if deadline and time.time() > deadline:
                print("# time window elapsed")
                break
            if args.max_messages and store.count >= args.max_messages:
                print("# reached --max-messages")
                break
            for key, _ in sel.select(timeout=1.0):
                proto = key.data
                sock = cast(socket.socket, key.fileobj)
                if proto == "udp":
                    data, addr = sock.recvfrom(65535)
                    if data:
                        store.add("udp", f"{addr[0]}:{addr[1]}", data)
                elif proto == "tcp":  # tcp listening socket
                    try:
                        conn, addr = sock.accept()
                    except OSError as exc:  # spurious/again -> just retry
                        print(f"# accept skipped: {exc}")
                        continue
                    conn.setblocking(False)
                    tcp_conns[conn] = bytearray()
                    sel.register(conn, selectors.EVENT_READ, "tcp-conn")
                    print(f"# tcp connection from {addr[0]}:{addr[1]}")
                elif proto == "tcp-conn":
                    conn = sock
                    try:
                        chunk = conn.recv(65535)
                    except OSError:
                        chunk = b""
                    if not chunk:  # closed
                        sel.unregister(conn)
                        conn.close()
                        tcp_conns.pop(conn, None)
                        continue
                    buf = tcp_conns.setdefault(conn, bytearray())
                    buf.extend(chunk)
                    while b"\n" in buf:
                        line, _, rest = bytes(buf).partition(b"\n")
                        buf[:] = rest
                        if line:
                            a = conn.getpeername()
                            store.add("tcp", f"{a[0]}:{a[1]}", line + b"\n")
    finally:
        for conn in list(tcp_conns):
            try:
                conn.close()
            except OSError:
                pass
        for _, s in socks:
            s.close()
        store.close()

    print(f"\n# SUMMARY messages={store.count} -> {args.outdir}/syslog.raw, "
          f"syslog.jsonl, syslog.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
