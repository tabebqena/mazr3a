#!/usr/bin/env python3
"""container_logs - read-only Docker logs sidecar (stdlib only).

Sits IDLE as a sibling container of the portal (docker-compose service `logs`)
and owns a mount of the host Docker socket (/var/run/docker.sock). It exposes
TWO read-only endpoints on the internal compose network (no published ports)
that the portal calls ONLY when the admin opens the Debug tab:

    GET /api/containers            -> JSON list of every container
    GET /api/logs?name=<c>&tail=N  -> last N lines of a container's stdout+stderr

It issues Docker READ calls only (list + logs), never writes to Docker and
never exposes the raw socket. Runs as root (the stock python image default)
because the host socket is root-owned; this file is mounted read-only.

Example run (see docker-compose.yml):
    python /scripts/container_logs.py        # binds 0.0.0.0:8090
"""
import http.client
import json
import os
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DOCKER_SOCKET = os.environ.get("DOCKER_SOCKET", "/var/run/docker.sock")
LISTEN = os.environ.get("LOGS_LISTEN", "0.0.0.0")
PORT = int(os.environ.get("LOGS_PORT", "8090"))
MAX_TAIL = int(os.environ.get("LOGS_MAX_TAIL", "2000"))
REQUEST_TIMEOUT = float(os.environ.get("LOGS_TIMEOUT", "20"))


def docker_req(method, path, timeout=REQUEST_TIMEOUT):
    """Issue an HTTP request to the Docker Engine API over its unix socket."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(DOCKER_SOCKET)
    conn = http.client.HTTPConnection("localhost", timeout=timeout)
    conn.sock = sock  # talk HTTP over our already-connected unix socket
    try:
        conn.request(method, path)
        resp = conn.getresponse()
        body = resp.read()
        return resp.status, body
    finally:
        conn.close()


def demux_frames(data):
    """Split Docker's multiplexed stdout/stderr stream into frame payloads.

    Non-tty containers multiplex stdout(1)/stderr(2) with an 8-byte header per
    frame: stream byte + 3 reserved bytes + 4-byte big-endian payload length,
    then that many payload bytes. Frames arrive in chronological order.
    """
    frames = []
    i, n = 0, len(data)
    while i + 8 <= n:
        stream = data[i]
        size = int.from_bytes(data[i + 4:i + 8], "big")
        if i + 8 + size > n:
            break  # truncated trailing frame - stop cleanly
        if stream in (1, 2):
            frames.append((stream, data[i + 8:i + 8 + size]))
        i += 8 + size
    if i < n:  # unexpected leftover bytes: treat as stdout
        frames.append((1, data[i:]))
    return frames


def list_containers():
    status, body = docker_req("GET", "/containers/json?all=1")
    if status != 200:
        raise RuntimeError("docker containers list failed: HTTP %s" % status)
    items = json.loads(body.decode("utf-8", "replace"))
    out = []
    for c in items:
        names = c.get("Names") or []
        name = (names[0].lstrip("/") if names else "") \
            or (c.get("Id") or "?")[:12]
        state = (c.get("State") or "").lower()
        out.append({
            "name": name,
            "id": (c.get("Id") or "")[:12],
            "image": c.get("Image") or "",
            "state": state,
            "status": c.get("Status") or "",
            "running": state == "running",
        })
    out.sort(key=lambda x: (not x["running"], x["name"].lower()))
    return out


def container_logs(name, tail):
    """Return the last `tail` lines of a container's stdout+stderr (text)."""
    q = "stdout=1&stderr=1&timestamps=1&tail=%d" % tail
    # Docker accepts the (no-slash) container name directly in the path.
    safe = name.replace("/", "%2F")
    path = "/containers/%s/logs?%s" % (safe, q)
    status, body = docker_req("GET", path)
    if status == 404:
        raise LookupError("container '%s' not found" % name)
    if status != 200:
        raise RuntimeError("docker logs '%s' failed: HTTP %s" % (name, status))
    chunks = []
    for _stream, payload in demux_frames(body):
        chunks.append(payload.decode("utf-8", "replace"))
    return "".join(chunks)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):      # stay quiet while idle
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        try:
            parsed = urlparse(self.path)

            if parsed.path == "/api/containers":
                try:
                    self._json({"containers": list_containers()})
                except Exception as exc:
                    self._json({"error": "docker list failed: %s" % exc}, 502)
                return

            if parsed.path == "/api/logs":
                qs = parse_qs(parsed.query)
                name = (qs.get("name") or [""])[0].strip().lstrip("/")
                try:
                    tail = int((qs.get("tail") or ["200"])[0])
                except ValueError:
                    tail = 200
                tail = max(1, min(int(MAX_TAIL), tail))
                if not name:
                    self._json({"error": "missing 'name' query param"}, 400)
                    return
                try:
                    self._json({"name": name, "tail": tail,
                                "logs": container_logs(name, tail)})
                except LookupError as exc:
                    self._json({"error": str(exc)}, 404)
                except Exception as exc:
                    self._json({"error": "docker logs failed: %s" % exc}, 502)
                return

            self._json({"error": "not found"}, 404)
        except Exception as exc:          # never let a handler crash the server
            try:
                self._json({"error": "internal error: %s" % exc}, 500)
            except Exception:
                pass


def main():
    try:
        httpd = ThreadingHTTPServer((LISTEN, PORT), Handler)
    except OSError as exc:
        print("container_logs: cannot bind %s:%s: %s" % (LISTEN, PORT, exc),
              file=sys.stderr)
        sys.exit(1)
    print("container_logs: idle on http://%s:%s (docker socket %s)"
          % (LISTEN, PORT, DOCKER_SOCKET), file=sys.stderr)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
