#!/usr/bin/env python3
"""cam_event_listener.py — overnight cam01 (UNV IPC) event listener.

WHY
    cam01 supports the UNV HTTP *alarm subscription* API (subscribe -> the
    device pushes alarm data to a receiver IP:port -> keepalive -> delete).  We
    want to observe what the camera actually pushes, overnight, without any
    dependency and without endangering the camera or the host.

WHAT IT DOES
    1. Creates ONE subscription on the camera:
           POST /LAPI/V1.0/System/Event/Subscription
           {"AddressType":0,"IPAddress":<advertise-ip>,"Port":<listen-port>,
            "Duration":<duration>}
       persists the returned ID, and refreshes it with
           PUT  /LAPI/V1.0/System/Event/Subscription/<ID> {"Duration":<duration>}
       every duration/2 seconds.  On exit it DELETEs the subscription.
    2. Serves the pushes on 0.0.0.0:<listen-port>.  Every connection is logged
       (first bytes) and its payload is streamed INCREMENTALLY to disk — so a
       one-shot HTTP POST and a long-lived streaming channel are both handled
       with bounded memory.  Each push also gets one NDJSON line; base64
       pictures are decoded to image files.
    3. Optionally ALSO opens the camera WebSocket event-status subscription
       (ws://cam/.../WsSubscription/WsSubscribers, WsCreate Type 8) and appends
       its {"url":...,"data":...} frames to ws_events.ndjson.

SAFETY — CAMERA
    * Exactly ONE subscription at a time.  The ID is persisted so a restart
      DELETEs a stale one first (never a dangling subscription).
    * Duration is clamped to [30, 3600] s; keepalive at duration/2.
    * Subscribe/keepalive use bounded exponential backoff (cap 300 s); three
      consecutive keepalive failures trigger a clean resubscribe.  No tight loop.
    * Every camera call has a socket timeout.  No configuration is written.

SAFETY — SERVER
    * Bounded concurrency (--max-conn, default 16 — the camera may hold several
      channels open), a per-connection idle timeout (--stream-idle-s, default 20),
      and hard caps on each connection's stored bytes (--max-body-mb, default 32).
      Nothing is buffered unbounded: payloads are written to disk as they arrive.
    * Output is size-capped (--max-total-mb, default 512) with oldest-first
      pruning; the log rotates (5 MB x 3).  Stdlib only; writes ONLY under --out.
    * Every worker is a daemon thread; the process exits on SIGINT/SIGTERM.

RUN  (ON THE HOST, from the deploy dir, e.g. /home/dr/frigate)
    set -a; . ./.env; set +a
    mkdir -p media/cam_events
    nohup setsid python3 scripts/cam_event_listener.py \
        --cam-ip 192.168.1.200 --advertise-ip 192.168.1.5 \
        --listen-port 50235 --out media/cam_events \
        >> media/cam_events/nohup.log 2>&1 &
    # NOTE: the script writes its OWN pid file (media/cam_events/listener.pid).
    # Do NOT `echo $!` -- `setsid` forks, so $! is setsid's pid, not python's.
    # A single-instance lock (listener.lock) refuses a second copy, and a bind
    # failure is FATAL (so a port collision is visible, not silent).

STOP
    kill "$(cat media/cam_events/listener.pid)"   # SIGTERM -> DELETE + clean exit

WATCH
    tail -f media/cam_events/listener.log
    cat media/cam_events/events.ndjson
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import logging
import os
import re
import signal
import socket
import struct
import sys
import threading
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

LOG = logging.getLogger("cam-events")

# --- camera LAPI -----------------------------------------------------------
SUB_PATH = "/LAPI/V1.0/System/Event/Subscription"
WS_PATH = "/LAPI/V1.0/Channel/0/Event/WsSubscription/WsSubscribers"

# --- limits ----------------------------------------------------------------
MAX_HEADER_BYTES = 16 * 1024
MAX_CAM_RESPONSE = 4 * 1024 * 1024
READ_TIMEOUT_S = 15.0
PARSE_KEEP_BYTES = 4 * 1024 * 1024     # body prefix kept in RAM for JSON parsing
MIN_DURATION, MAX_DURATION = 30, 3600

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_B64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")
_HTTP_LINE_RE = re.compile(rb"^[A-Z]{3,7} \S+ HTTP/1\.[01]\r\n")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ts_compact() -> str:
    return utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def brief(data: bytes, n: int = 200) -> str:
    """A compact, log-safe view of the first bytes of a connection."""
    line = data.split(b"\r\n", 1)[0][:80]
    return (f"first-line={line!r} hex={data[:16].hex()}" if line
            else f"hex={data[:16].hex()} (n={len(data)})")


def _parse_digest(text: str) -> dict:
    """Extract Digest challenge attributes from a raw HTTP response."""
    m = re.search(r"WWW-Authenticate:\s*Digest\s+(.*)", text, re.I)
    if not m:
        return {}
    attrs: dict[str, str] = {}
    for k, a, b in re.findall(r'(\w+)\s*=\s*(?:"([^"]*)"|([^,\s]+))', m.group(1)):
        attrs[k.lower()] = a if a != "" else b
    return attrs


def detect_local_ip(target: str) -> str | None:
    """Source IP the kernel would use to reach `target` (no packets sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((target, 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def load_env(path: str) -> dict:
    env: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return env


# ---------------------------------------------------------------------------
# Camera HTTP (Digest, MD5/qop=auth) — minimal, single-shot, timed out.
# ---------------------------------------------------------------------------
class CameraHTTP:
    def __init__(self, host: str, port: int, user: str, password: str,
                 timeout: float = 8.0):
        self.host, self.port = host, port
        self.user, self.password = user, password
        self.timeout = timeout

    def _send(self, req: bytes, timeout: float) -> str:
        s = socket.create_connection((self.host, self.port), timeout=timeout)
        s.settimeout(timeout)
        try:
            s.sendall(req)
            buf = bytearray()
            while len(buf) < MAX_CAM_RESPONSE:
                try:
                    d = s.recv(65536)
                except socket.timeout:
                    break
                if not d:
                    break
                buf += d
            return bytes(buf).decode("latin1", "replace")
        finally:
            s.close()

    def request(self, method: str, path: str, body=None,
                timeout: float | None = None) -> tuple[str, str]:
        timeout = timeout or self.timeout
        body_b = b"" if body is None else (
            body.encode() if isinstance(body, str) else bytes(body))
        hdr = ""
        if body is not None:
            hdr = (f"Content-Type: application/json\r\n"
                   f"Content-Length: {len(body_b)}\r\n")
        r1 = self._send(
            (f"{method} {path} HTTP/1.1\r\nHost: {self.host}\r\n"
             f"Connection: close\r\n\r\n").encode(), timeout)
        first = r1.split("\r\n", 1)[0]
        if " 401" not in first:
            return first, r1

        ch = _parse_digest(r1)
        if not ch.get("realm") or not ch.get("nonce"):
            return first, r1
        qop = ch.get("qop") or "auth"
        nc, cnonce = "00000001", os.urandom(8).hex()
        ha1 = md5(f"{self.user}:{ch['realm']}:{self.password}")
        ha2 = md5(f"{method}:{path}")
        if qop:
            resp = md5(f"{ha1}:{ch['nonce']}:{nc}:{cnonce}:{qop}:{ha2}")
        else:
            resp = md5(f"{ha1}:{ch['nonce']}:{ha2}")
        auth = (f'Digest username="{self.user}", realm="{ch["realm"]}", '
                f'nonce="{ch["nonce"]}", algorithm="{ch.get("algorithm", "MD5")}", '
                f'uri="{path}", response="{resp}"')
        if qop:
            auth += f', qop="{qop}", nc="{nc}", cnonce="{cnonce}"'
        if ch.get("opaque"):
            auth += f', opaque="{ch["opaque"]}"'
        req = (f"{method} {path} HTTP/1.1\r\nHost: {self.host}\r\n"
               f"Authorization: {auth}\r\n{hdr}Connection: close\r\n\r\n").encode()
        if body is not None:
            req += body_b
        r2 = self._send(req, timeout)
        return r2.split("\r\n", 1)[0], r2


def lapi_json(text: str) -> dict:
    """Return the parsed LAPI body (after the HTTP headers), else {}."""
    _, _, body = text.partition("\r\n\r\n")
    try:
        return json.loads(body)
    except (ValueError, TypeError):
        return {}


def lapi_ok(text: str) -> bool:
    return lapi_json(text).get("Response", {}).get("ResponseCode") == 0


# ---------------------------------------------------------------------------
# Output store — verbatim raw + NDJSON + decoded pictures, size-capped.
# ---------------------------------------------------------------------------
class EventStore:
    PROTECTED_SUFFIXES = (".log", ".pid", "subscription.json")

    def __init__(self, root: str, max_body_mb: int, max_total_mb: int):
        self.root = os.path.abspath(root)
        self.max_body = max_body_mb * 1024 * 1024
        self.max_total = max_total_mb * 1024 * 1024
        self.raw_dir = os.path.join(self.root, "raw")
        self.img_dir = os.path.join(self.root, "images")
        for d in (self.root, self.raw_dir, self.img_dir):
            os.makedirs(d, exist_ok=True)
        self.events_path = os.path.join(self.root, "events.ndjson")
        self.ws_path = os.path.join(self.root, "ws_events.ndjson")
        self._lock = threading.Lock()
        self._last_prune = 0.0
        self.events = 0
        self.ws_frames = 0

    # -- paths --------------------------------------------------------------
    def raw_path(self, tag: str) -> str:
        day = utcnow().strftime("%Y%m%d")
        d = os.path.join(self.raw_dir, day)
        os.makedirs(d, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9]+", "_", tag.strip("/"))[:40] or "push"
        return os.path.join(d, f"{ts_compact()}_{safe}.bin")

    def rel(self, path: str) -> str:
        return os.path.relpath(path, self.root)

    # -- pruning ------------------------------------------------------------
    def _prune_locked(self) -> int:
        files, total = [], 0
        for base, _dirs, names in os.walk(self.root):
            for n in names:
                p = os.path.join(base, n)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                files.append((st.st_mtime, p, st.st_size))
                total += st.st_size
        if total <= self.max_total:
            return 0
        files.sort()
        removed = 0
        for _m, p, sz in files:
            if total <= self.max_total:
                break
            if p.endswith(self.PROTECTED_SUFFIXES):
                continue
            try:
                os.remove(p)
                total -= sz
                removed += 1
            except OSError:
                pass
        return removed

    def prune(self, force: bool = False) -> None:
        with self._lock:
            now = time.time()
            if not force and now - self._last_prune < 60:
                return
            self._last_prune = now
            n = self._prune_locked()
            if n:
                LOG.warning("prune: removed %d old files (cap %d MB)",
                            n, self.max_total // (1024 * 1024))

    # -- writers ------------------------------------------------------------
    def _append(self, path: str, obj: dict) -> None:
        with self._lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def append_event(self, obj: dict) -> None:
        self._append(self.events_path, obj)
        self.events += 1

    def append_ws(self, obj: dict) -> None:
        self._append(self.ws_path, obj)
        self.ws_frames += 1

    def save_image(self, data: bytes, fmt: str) -> str:
        day = utcnow().strftime("%Y%m%d")
        d = os.path.join(self.img_dir, day)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{ts_compact()}.{fmt}")
        with open(p, "wb") as fh:
            fh.write(data)
        return os.path.relpath(p, self.root)


def find_images(obj, min_len: int = 1500, max_count: int = 4,
                max_scan: int = 32 * 1024 * 1024) -> list[tuple[str, bytes]]:
    """Recursively find base64-encoded JPEG/PNG blobs in parsed JSON."""
    found: list[tuple[str, bytes]] = []

    def walk(x) -> None:
        if len(found) >= max_count:
            return
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and min_len <= len(x) <= max_scan \
                and _B64_RE.match(x[:120]):
            s = re.sub(r"\s+", "", x)
            s = s[: len(s) - (len(s) % 4)]
            try:
                data = base64.b64decode(s, validate=True)
            except Exception:
                return
            if data.startswith(JPEG_MAGIC):
                found.append(("jpg", data))
            elif data.startswith(PNG_MAGIC):
                found.append(("png", data))

    walk(obj)
    return found


def summarize(obj, limit: int = 16) -> dict:
    """A small, log/NDJSON-friendly view of a pushed JSON object."""
    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if len(out) >= limit:
                break
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v if not isinstance(v, str) else v[:200]
            elif isinstance(v, list):
                out[k] = f"<list len={len(v)}>"
            elif isinstance(v, dict):
                out[k] = f"<obj keys={len(v)}>"
    return out


# ---------------------------------------------------------------------------
# Receiver — logs every connection, streams each payload to disk (bounded RAM).
# ---------------------------------------------------------------------------
class Receiver(threading.Thread):
    def __init__(self, store: EventStore, bind_ip: str, port: int,
                 max_body: int, max_conn: int, stream_idle: float,
                 stop: threading.Event):
        super().__init__(name="receiver", daemon=True)
        self.store, self.bind_ip, self.port = store, bind_ip, port
        self.max_body, self.stop = max_body, stop
        self.stream_idle = stream_idle
        self._sem = threading.BoundedSemaphore(max_conn)
        self._sock: socket.socket | None = None
        self.pushes = 0

    def bind(self) -> None:
        """Bind NOW (in the main thread) so a bind failure is fatal + visible."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.bind_ip, self.port))
        s.listen(32)
        s.settimeout(1.0)
        self._sock = s

    def run(self) -> None:
        s = self._sock
        if s is None:
            LOG.error("receiver not bound - thread exiting")
            return
        LOG.info("receiver listening on %s:%d", self.bind_ip or "0.0.0.0", self.port)
        while not self.stop.is_set():
            try:
                conn, addr = s.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            if not self._sem.acquire(blocking=False):
                LOG.warning("refused connection from %s (max concurrency)",
                            addr[0])
                try:
                    conn.close()
                except OSError:
                    pass
                continue
            threading.Thread(target=self._handle, args=(conn, addr),
                             name="rx", daemon=True).start()
        try:
            s.close()
        except OSError:
            pass
        LOG.info("receiver stopped (%d pushes)", self.pushes)

    # -- one connection -----------------------------------------------------
    def _handle(self, conn: socket.socket, addr) -> None:
        peer = f"{addr[0]}:{addr[1]}"
        t0 = time.monotonic()
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        try:
            conn.settimeout(self.stream_idle)
            first = conn.recv(65536)
            if not first:
                LOG.info("conn %s closed with no data", peer)
                return
            LOG.info("conn %s opened: %s", peer, brief(first))
            if _HTTP_LINE_RE.match(first):
                self._handle_http(conn, peer, first)
            else:
                self._handle_stream(conn, peer, first)
        except Exception as exc:  # noqa: BLE001 - a push must never kill the loop
            LOG.warning("receiver error from %s: %s", addr[0], exc)
        finally:
            try:
                conn.close()
            except OSError:
                pass
            self._sem.release()
            LOG.info("conn %s closed (%.1fs)", peer, time.monotonic() - t0)

    # -- HTTP request -------------------------------------------------------
    def _handle_http(self, conn, peer: str, buf: bytes) -> None:
        end = buf.find(b"\r\n\r\n")
        while end < 0 and len(buf) < MAX_HEADER_BYTES:
            more = conn.recv(4096)
            if not more:
                break
            buf += more
            end = buf.find(b"\r\n\r\n")
        if end < 0:
            LOG.warning("conn %s: HTTP head incomplete (%d bytes)", peer, len(buf))
            return
        head, rest = buf[:end], buf[end + 4:]
        lines = head.split(b"\r\n")
        parts = lines[0].decode("latin1").split(" ")
        method, path = parts[0], (parts[1] if len(parts) > 1 else "/")
        headers: dict[str, str] = {}
        for ln in lines[1:]:
            if b":" in ln:
                k, v = ln.split(b":", 1)
                headers[k.decode("latin1").strip().lower()] = \
                    v.decode("latin1").strip().lower()

        if headers.get("expect", "").startswith("100-"):
            try:
                conn.sendall(b"HTTP/1.1 100 Continue\r\n\r\n")
            except OSError:
                return

        try:
            if "content-length" in headers:
                need = int(headers["content-length"] or "0")
                path_raw, size, keep, sha = self._stream_exact(conn, rest, need,
                                                               peer, path)
                complete = size >= need
            elif "chunked" in headers.get("transfer-encoding", ""):
                path_raw, size, keep, sha = self._stream_chunked(conn, rest, peer)
                complete = False
            else:
                path_raw, size, keep, sha = self._stream_raw(conn, rest, peer)
                complete = False
        except Exception as exc:  # noqa: BLE001
            LOG.warning("bad push from %s (%s %s): %s", peer, method, path, exc)
            self._respond(conn, path, 400)
            return

        self._emit(peer, method, path, path_raw, size, keep, sha,
                   complete=complete)
        self._respond(conn, path, 200)

    # -- raw / streaming connection ----------------------------------------
    def _handle_stream(self, conn, peer: str, first: bytes) -> None:
        path_raw, size, keep, sha = self._stream_raw(conn, first, peer)
        self._emit(peer, "RAW", "", path_raw, size, keep, sha, complete=False)

    # -- stream helpers (bounded RAM: write through to disk) ---------------
    def _stream_exact(self, conn, initial: bytes, need: int, peer: str,
                      tag: str):
        p = self.store.raw_path(tag)
        keep = bytearray()
        h = hashlib.sha256()
        written = 0
        with open(p, "wb") as fh:
            if initial:
                take = initial[:max(need, 0)]
                fh.write(take)
                h.update(take)
                written += len(take)
                keep += take[:PARSE_KEEP_BYTES]
            while written < need and written < self.max_body:
                try:
                    more = conn.recv(min(65536, need - written))
                except socket.timeout:
                    LOG.warning("conn %s: body idle at %d/%d bytes",
                                peer, written, need)
                    break
                if not more:
                    break
                fh.write(more)
                h.update(more)
                written += len(more)
                if len(keep) < PARSE_KEEP_BYTES:
                    keep += more[:PARSE_KEEP_BYTES - len(keep)]
        return p, written, bytes(keep), h.hexdigest()

    def _stream_chunked(self, conn, initial: bytes, peer: str):
        p = self.store.raw_path("chunked")
        keep = bytearray()
        h = hashlib.sha256()
        written = 0
        buf = bytearray(initial)

        def readline() -> bytes:
            nonlocal buf
            while b"\r\n" not in buf:
                d = conn.recv(4096)
                if not d:
                    raise IOError("eof in chunk header")
                buf += d
            line, _, rest = buf.partition(b"\r\n")
            buf.clear()
            buf += rest
            return bytes(line)

        with open(p, "wb") as fh:
            while written < self.max_body:
                try:
                    line = readline()
                except socket.timeout:
                    LOG.warning("conn %s: chunk header idle at %d bytes",
                                peer, written)
                    break
                size = int(line.split(b";")[0] or b"0", 16)
                if size == 0:
                    try:
                        readline()
                    except (socket.timeout, IOError, OSError):
                        pass
                    break
                while len(buf) < size + 2:
                    try:
                        d = conn.recv(min(65536, size + 2 - len(buf)))
                    except socket.timeout:
                        LOG.warning("conn %s: chunk body idle at %d bytes",
                                    peer, written)
                        break
                    if not d:
                        break
                    buf += d
                if len(buf) < size:
                    break
                chunk = bytes(buf[:size])
                buf = buf[size + 2:]
                fh.write(chunk)
                h.update(chunk)
                written += len(chunk)
                if len(keep) < PARSE_KEEP_BYTES:
                    keep += chunk[:PARSE_KEEP_BYTES - len(keep)]
        return p, written, bytes(keep), h.hexdigest()

    def _stream_raw(self, conn, initial: bytes, peer: str):
        p = self.store.raw_path("stream")
        keep = bytearray()
        h = hashlib.sha256()
        written = 0
        with open(p, "wb") as fh:
            for chunk in (initial,):
                if chunk:
                    fh.write(chunk)
                    h.update(chunk)
                    written += len(chunk)
                    keep += chunk[:PARSE_KEEP_BYTES]
            while written < self.max_body:
                try:
                    more = conn.recv(65536)
                except socket.timeout:
                    LOG.info("conn %s: stream idle at %d bytes", peer, written)
                    break
                if not more:
                    break
                fh.write(more)
                h.update(more)
                written += len(more)
                if len(keep) < PARSE_KEEP_BYTES:
                    keep += more[:PARSE_KEEP_BYTES - len(keep)]
        return p, written, bytes(keep), h.hexdigest()

    # -- record + respond ---------------------------------------------------
    def _emit(self, peer: str, method: str, path: str, raw_path: str,
              size: int, keep: bytes, sha: str, complete: bool) -> None:
        self.pushes += 1
        self.store.prune()
        record: dict = {
            "ts": utcnow().isoformat(),
            "client": peer,
            "method": method,
            "path": path,
            "bytes": size,
            "sha256": sha[:16] if size else None,
            "complete": complete,
            "raw": self.store.rel(raw_path),
        }
        obj = None
        if keep:
            try:
                obj = json.loads(keep)
            except (ValueError, TypeError):
                obj = None
        if isinstance(obj, (dict, list)):
            record["summary"] = summarize(obj)
            try:
                imgs = find_images(obj)
            except Exception:  # noqa: BLE001
                imgs = []
            if imgs:
                record["images"] = []
                for fmt, data in imgs:
                    try:
                        record["images"].append(self.store.save_image(data, fmt))
                    except OSError as exc:
                        LOG.warning("save image failed: %s", exc)
        self.store.append_event(record)
        LOG.info("push #%d %s %s (%d bytes, complete=%s) from %s",
                 self.pushes, method, path or "(raw)", size, complete,
                 peer.split(":")[0])

    @staticmethod
    def _respond(conn, path: str, code: int = 200) -> None:
        payload = json.dumps({
            "Response": {
                "ResponseURL": path or SUB_PATH, "CreatedID": -1,
                "ResponseCode": 0 if code == 200 else 1, "SubResponseCode": 0,
                "ResponseString": "Succeed" if code == 200 else "Fail",
                "StatusCode": 0 if code == 200 else 8,
                "StatusString": "Succeed" if code == 200 else "Fail",
                "Data": None,
            }
        }).encode()
        reason = "OK" if code == 200 else "Bad Request"
        try:
            conn.sendall(
                f"HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
                .encode() + payload)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Subscription manager — one live subscription, persisted, backoff, delete.
# ---------------------------------------------------------------------------
class SubscriptionManager(threading.Thread):
    def __init__(self, cam: CameraHTTP, advertise_ip: str, listen_port: int,
                 duration: int, state_path: str, stop: threading.Event):
        super().__init__(name="subscription", daemon=True)
        self.cam = cam
        self.advertise_ip, self.listen_port = advertise_ip, listen_port
        self.duration = max(MIN_DURATION, min(MAX_DURATION, duration))
        self.state_path, self.stop = state_path, stop
        self.sub_id: int | None = None
        self.fail = 0

    # -- state --------------------------------------------------------------
    def _load_state(self) -> None:
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                obj = json.load(fh)
            if isinstance(obj.get("id"), int):
                self.sub_id = obj["id"]
        except (OSError, ValueError, TypeError):
            self.sub_id = None

    def _save_state(self) -> None:
        try:
            if self.sub_id is None:
                try:
                    os.remove(self.state_path)
                except OSError:
                    pass
                return
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump({"id": self.sub_id, "saved": utcnow().isoformat()}, fh)
        except OSError as exc:
            LOG.warning("state write failed: %s", exc)

    # -- lifecycle ----------------------------------------------------------
    def run(self) -> None:
        self._load_state()
        if self.sub_id is not None:
            LOG.info("found stale subscription %s from a previous run - deleting",
                     self.sub_id)
            self.delete(log=True)
        backoff = 5
        while not self.stop.is_set():
            if not self.ensure():
                LOG.warning("subscribe failed - retry in %ds", backoff)
                if self.stop.wait(backoff):
                    break
                backoff = min(backoff * 2, 300)
                continue
            backoff = 5
            self.fail = 0
            while not self.stop.is_set():
                if self.stop.wait(self.duration / 2.0):
                    break
                if self.keepalive():
                    if self.fail:
                        LOG.info("keepalive recovered")
                    self.fail = 0
                else:
                    self.fail += 1
                    LOG.warning("keepalive failed (%d/3)", self.fail)
                    if self.fail >= 3:
                        LOG.warning("3 keepalive failures - resubscribing")
                        self.delete(log=False)
                        break
        self.delete(log=True)

    def ensure(self) -> bool:
        if self.sub_id is not None:
            return True
        body = json.dumps({
            "AddressType": 0, "IPAddress": self.advertise_ip,
            "Port": self.listen_port, "Duration": self.duration,
        })
        try:
            code, text = self.cam.request("POST", SUB_PATH, body)
        except OSError as exc:
            LOG.warning("subscribe transport error: %s", exc)
            return False
        data = lapi_json(text).get("Response", {}).get("Data", {})
        sid = data.get("ID") if isinstance(data, dict) else None
        if sid is None:
            LOG.warning("subscribe rejected: %s", code)
            return False
        self.sub_id = int(sid)
        self._save_state()
        LOG.info("subscribed id=%s -> %s:%d for %ds (SupportType=%s)",
                 self.sub_id, self.advertise_ip, self.listen_port,
                 self.duration, data.get("SupportType"))
        return True

    def keepalive(self) -> bool:
        if self.sub_id is None:
            return False
        try:
            code, text = self.cam.request(
                "PUT", f"{SUB_PATH}/{self.sub_id}",
                json.dumps({"Duration": self.duration}))
        except OSError as exc:
            LOG.warning("keepalive transport error: %s", exc)
            return False
        resp = lapi_json(text).get("Response", {})
        rc = resp.get("ResponseCode")
        if " 200" in code and (rc is None or rc == 0):
            LOG.info("keepalive ok (id=%s)", self.sub_id)
            return True
        LOG.warning("keepalive rejected: %s rc=%s body=%s", code, rc,
                    text.split("\r\n\r\n", 1)[-1][:160].replace("\n", " "))
        return False

    def delete(self, log: bool) -> None:
        if self.sub_id is None:
            return
        try:
            code, text = self.cam.request("DELETE", f"{SUB_PATH}/{self.sub_id}")
            if log:
                LOG.info("unsubscribed id=%s (%s)", self.sub_id, code)
        except OSError as exc:
            LOG.warning("unsubscribe transport error: %s", exc)
        self.sub_id = None
        self._save_state()


# ---------------------------------------------------------------------------
# WebSocket status subscription (Type 8) — optional mirror, auto-reconnect.
# ---------------------------------------------------------------------------
class WSStatusClient(threading.Thread):
    def __init__(self, cam_host: str, cam_port: int, user: str, password: str,
                 ws_type: int, store: EventStore, stop: threading.Event):
        super().__init__(name="ws", daemon=True)
        self.host, self.port = cam_host, cam_port
        self.user, self.password = user, password
        self.ws_type = ws_type
        self.store, self.stop = store, stop

    def run(self) -> None:
        backoff = 5
        while not self.stop.is_set():
            try:
                self._session()
                backoff = 5
            except Exception as exc:  # noqa: BLE001
                LOG.warning("ws session ended: %s", exc)
            if self.stop.wait(backoff):
                break
            backoff = min(backoff * 2, 120)

    # -- ws helpers ---------------------------------------------------------
    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            d = sock.recv(n - len(buf))
            if not d:
                raise IOError("ws eof")
            buf += d
        return bytes(buf)

    def _send_text(self, sock: socket.socket, text: str) -> None:
        payload = text.encode()
        ln = len(payload)
        if ln < 126:
            hdr = bytes([0x81, 0x80 | ln])
        elif ln < 65536:
            hdr = bytes([0x81, 0x80 | 126]) + struct.pack(">H", ln)
        else:
            hdr = bytes([0x81, 0x80 | 127]) + struct.pack(">Q", ln)
        mask = os.urandom(4)
        masked = bytes(payload[i] ^ mask[i % 4] for i in range(ln))
        sock.sendall(hdr + mask + masked)

    def _recv_msg(self, sock: socket.socket) -> tuple[int, bytes]:
        frags: list[bytes] = []
        frag_op = None
        while True:
            b = self._recv_exact(sock, 2)
            fin, op, ln = b[0] & 0x80, b[0] & 0x0F, b[1] & 0x7F
            if ln == 126:
                ln = struct.unpack(">H", self._recv_exact(sock, 2))[0]
            elif ln == 127:
                ln = struct.unpack(">Q", self._recv_exact(sock, 8))[0]
            if b[1] & 0x80:
                mask = self._recv_exact(sock, 4)
                raw = self._recv_exact(sock, ln)
                data = bytes(raw[i] ^ mask[i % 4] for i in range(ln))
            else:
                data = self._recv_exact(sock, ln)
            if op == 0x9:
                continue
            if op == 0xA:
                continue
            if op == 0x8:
                return 0x8, data
            if op in (0x1, 0x2):
                frag_op, frags = op, [data]
            elif op == 0x0:
                frags.append(data)
            if fin:
                return frag_op or op, b"".join(frags)

    def _session(self) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=10)
        sock.settimeout(30)
        key = base64.b64encode(os.urandom(16)).decode()
        sock.sendall(
            (f"GET {WS_PATH} HTTP/1.1\r\nHost: {self.host}\r\n"
             f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
             f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
             f"Origin: http://{self.host}\r\n\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            d = sock.recv(4096)
            if not d:
                raise IOError("ws handshake eof")
            head += d
        if b" 101" not in head.split(b"\r\n", 1)[0]:
            raise IOError("ws handshake not 101")

        index = None
        try:
            op, data = self._recv_msg(sock)
            challenge = json.loads(data)
            if challenge.get("errorCode") == 401:
                attrs = {}
                for part in challenge.get("detail", "").replace(
                        "Digest", "").replace('"', "").split(","):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        attrs[k.strip()] = v.strip()
                uri = f"ws://{self.host}:{self.port}{WS_PATH}"
                auth = self._digest(attrs, "GET", uri)
                self._send_text(sock, "Authorization=" + auth)
                op, data = self._recv_msg(sock)
                if json.loads(data).get("errorCode") != 0:
                    raise IOError("ws auth failed")
            self._send_text(sock, json.dumps(
                {"WsCreate": {"Index": -1, "Type": self.ws_type,
                              "Duration": 3600}}))
            LOG.info("ws subscribed (Type=%d)", self.ws_type)

            keepalive_at = time.monotonic() + 3500
            while not self.stop.is_set():
                if time.monotonic() >= keepalive_at and index is not None:
                    self._send_text(sock, json.dumps(
                        {"WsModify": {"Index": index, "Type": self.ws_type,
                                      "Duration": 3600}}))
                    keepalive_at = time.monotonic() + 3500
                try:
                    op, data = self._recv_msg(sock)
                except socket.timeout:
                    continue
                if op == 0x8:
                    break
                if op != 0x1:
                    LOG.info("ws binary frame len=%d", len(data))
                    continue
                try:
                    obj = json.loads(data)
                except ValueError:
                    obj = None
                create = obj.get("WsCreate") if isinstance(obj, dict) else None
                if isinstance(create, dict) and create.get("Index") is not None \
                        and create.get("Index") != -1:
                    index = create["Index"]
                    LOG.info("ws subscription index=%s", index)
                    continue
                self.store.append_ws({
                    "ts": utcnow().isoformat(), "data": obj
                    if obj is not None else data.decode("utf-8", "replace")[:2000],
                })
                LOG.info("ws push: %s",
                         (json.dumps(obj)[:200] if obj is not None else "text"))
        finally:
            if index is not None:
                try:
                    self._send_text(sock, json.dumps(
                        {"WsDelete": {"Index": index}}))
                except OSError:
                    pass
            try:
                sock.close()
            except OSError:
                pass

    def _digest(self, attrs: dict, method: str, uri: str) -> str:
        realm = attrs.get("realm", "")
        nonce = attrs.get("nonce", "")
        qop = attrs.get("qop", "auth") or "auth"
        nc, cnonce = "00000001", os.urandom(8).hex()
        ha1 = md5(f"{self.user}:{realm}:{self.password}")
        ha2 = md5(f"{method}:{uri}")
        resp = md5(f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}")
        out = (f'Digest username="{self.user}", realm="{realm}", '
               f'nonce="{nonce}", algorithm="MD5", uri="{uri}", '
               f'response="{resp}", qop="{qop}", nc="{nc}", cnonce="{cnonce}"')
        if attrs.get("opaque"):
            out += f', opaque="{attrs["opaque"]}"'
        return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Overnight cam01 (UNV) alarm/event listener (subscribe + "
                    "keepalive + delete, receive + store pushes).")
    p.add_argument("--cam-ip", default="192.168.1.200")
    p.add_argument("--cam-port", type=int, default=80)
    p.add_argument("--user", default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--env-file", default=None,
                   help="default: <repo>/.env (FRIGATE_RTSP_USER/PASS)")
    p.add_argument("--listen-ip", default="0.0.0.0")
    p.add_argument("--listen-port", type=int, default=50235)
    p.add_argument("--advertise-ip", default=None,
                   help="IP the camera pushes to (default: auto-detect via cam-ip)")
    p.add_argument("--duration", type=int, default=3600,
                   help=f"subscription seconds, clamped [{MIN_DURATION},{MAX_DURATION}]")
    p.add_argument("--out", default=None,
                   help="output dir (default: <repo>/media/cam_events)")
    p.add_argument("--pidfile", default=None,
                   help="default: <out>/listener.pid (written by the script)")
    p.add_argument("--max-conn", type=int, default=16,
                   help="max concurrent camera connections (default 16)")
    p.add_argument("--stream-idle-s", type=float, default=20.0,
                   help="close a connection after this many idle seconds")
    p.add_argument("--max-body-mb", type=int, default=32)
    p.add_argument("--max-total-mb", type=int, default=512)
    p.add_argument("--no-ws", action="store_true",
                   help="do not open the WebSocket status subscription")
    p.add_argument("--ws-type", type=int, default=8)
    p.add_argument("--selftest", action="store_true",
                   help="resolve + print config, touch nothing on the network")
    return p


def setup_logging(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = RotatingFileHandler(os.path.join(out_dir, "listener.log"),
                             maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    LOG.addHandler(fh)
    LOG.addHandler(sh)


def _acquire_lock(path: str):
    """Single-instance lock; returns the open file handle, or None if held."""
    try:
        fh = open(path, "w")
    except OSError:
        return None
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    return fh


def _write_pidfile(path: str) -> None:
    """Write OUR pid (not the shell's $! -- that is setsid's when setsid forks)."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()}\n")
    except OSError as exc:
        LOG.warning("could not write pid file %s: %s", path, exc)


def _release_lock(lock_path: str, pid_path: str) -> None:
    for p in (pid_path, lock_path):
        try:
            os.remove(p)
        except OSError:
            pass


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    out_dir = args.out or os.path.join(repo_root, "media", "cam_events")
    env_path = args.env_file or os.path.join(repo_root, ".env")
    env = load_env(env_path)

    user = args.user or os.environ.get("FRIGATE_RTSP_USER") \
        or env.get("FRIGATE_RTSP_USER", "admin")
    password = args.password or os.environ.get("FRIGATE_RTSP_PASS") \
        or env.get("FRIGATE_RTSP_PASS", "")
    advertise_ip = args.advertise_ip or detect_local_ip(args.cam_ip)

    if args.selftest:
        print(json.dumps({
            "cam": f"{args.cam_ip}:{args.cam_port}", "user": user,
            "password_set": bool(password),
            "advertise_ip": advertise_ip, "listen": f"{args.listen_ip}:{args.listen_port}",
            "duration": max(MIN_DURATION, min(MAX_DURATION, args.duration)),
            "out": out_dir, "env_file": env_path,
            "max_conn": args.max_conn, "stream_idle_s": args.stream_idle_s,
            "ws": not args.no_ws,
        }, indent=2))
        return 0

    if not password:
        print("ERROR: camera password not set (use --password or .env "
              "FRIGATE_RTSP_PASS)", file=sys.stderr)
        return 2
    if not advertise_ip:
        print("ERROR: could not determine --advertise-ip (camera cannot reach the "
              "host); pass it explicitly", file=sys.stderr)
        return 2

    setup_logging(out_dir)
    LOG.info("starting: cam=%s:%d advertise=%s:%d duration=%ds out=%s ws=%s "
             "max_conn=%d stream_idle=%.0fs",
             args.cam_ip, args.cam_port, advertise_ip, args.listen_port,
             args.duration, out_dir, not args.no_ws, args.max_conn,
             args.stream_idle_s)

    store = EventStore(out_dir, args.max_body_mb, args.max_total_mb)

    lock_path = os.path.join(out_dir, "listener.lock")
    lock = _acquire_lock(lock_path)
    if lock is None:
        LOG.error("another listener is already running (lock %s) - exiting",
                  lock_path)
        return 4
    pid_path = args.pidfile or os.path.join(out_dir, "listener.pid")
    _write_pidfile(pid_path)

    stop = threading.Event()
    cam = CameraHTTP(args.cam_ip, args.cam_port, user, password)

    def _signal(signum, _frame):
        LOG.info("signal %s received - shutting down", signum)
        stop.set()

    signal.signal(signal.SIGINT, _signal)
    signal.signal(signal.SIGTERM, _signal)

    state_path = os.path.join(out_dir, "subscription.json")
    receiver = Receiver(store, args.listen_ip, args.listen_port,
                        args.max_body_mb * 1024 * 1024, args.max_conn,
                        args.stream_idle_s, stop)
    try:
        receiver.bind()
    except OSError as exc:
        LOG.error("cannot bind %s:%d: %s (another listener running?)",
                  args.listen_ip, args.listen_port, exc)
        _release_lock(lock_path, pid_path)
        return 3
    manager = SubscriptionManager(cam, advertise_ip, args.listen_port,
                                  args.duration, state_path, stop)
    ws = None if args.no_ws else WSStatusClient(
        args.cam_ip, args.cam_port, user, password, args.ws_type, store, stop)

    receiver.start()
    manager.start()
    if ws:
        ws.start()

    last_hb = time.monotonic()
    try:
        while not stop.is_set():
            if stop.wait(5.0):
                break
            if time.monotonic() - last_hb >= 300:
                last_hb = time.monotonic()
                LOG.info("heartbeat: pushes=%d ws_frames=%d sub_id=%s",
                         receiver.pushes, store.ws_frames, manager.sub_id)
                store.prune()
    finally:
        stop.set()
        for t in (receiver, manager, ws):
            if t is not None:
                t.join(timeout=10)
        LOG.info("stopped cleanly (pushes=%d ws_frames=%d)",
                 receiver.pushes, store.ws_frames)
        _release_lock(lock_path, pid_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
