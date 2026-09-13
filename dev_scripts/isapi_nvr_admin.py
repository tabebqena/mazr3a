#!/usr/bin/env python3
"""NVR ISAPI admin helper: Log Server + per-channel Face Detection.

⚠️  THIS TOOL WRITES DEVICE CONFIGURATION (PUT). Use deliberately. `*-get`
subcommands and `--dry-run` are read-only / safe.

Target: Hikvision NVR (DS-7616NXI-K2(E)) served over HTTPS + HTTP Digest.

Key facts learned on this NVR:
  * Log Server ISAPI path is `/ISAPI/System/LogServer` (alias `/ISAPI/System/log`),
    fields: enabled, addressingFormatType, ipAddress, portNo, uploadInterval.
    Use a NON-PRIVILEGED port (e.g. 1515) so the listener needs no root.
  * Face Detection is limited to **one channel at a time** (single analysis
    unit). Enabling a 2nd channel returns HTTP 403 invalidOperation /
    "isapi_srv_set_face_detection_do failed". Enabling it anywhere turns the
    previously-enabled channel off is NOT automatic — you must free the slot.

Usage:
    # read
    isapi_nvr_admin.py ls-get
    isapi_nvr_admin.py fd-get --all
    isapi_nvr_admin.py fd-get --channel 1

    # write (device config!)
    isapi_nvr_admin.py ls-set --ip 192.168.1.5 --port 1515 --enabled true
    isapi_nvr_admin.py fd-set --channel 1 --enabled true
    isapi_nvr_admin.py fd-set --channel 1 --enabled false

    # show the XML that would be sent, without sending it
    isapi_nvr_admin.py ls-set --ip 192.168.1.5 --port 1515 --dry-run

Credentials: --user/--pass or env ISAPI_USER/ISAPI_PASS; host via --host or
ISAPI_HOST. Requires: requests.
"""
from __future__ import annotations

import argparse
import os
import sys

import requests
import urllib3
from requests.auth import HTTPDigestAuth

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

NS = "http://www.hikvision.com/ver20/XMLSchema"


def make_session(user: str, pwd: str) -> requests.Session:
    s = requests.Session()
    s.auth = HTTPDigestAuth(user, pwd)
    s.verify = False
    return s


def url(base: str, path: str) -> str:
    return base + path


def get_xml(sess, u: str) -> str:
    r = sess.get(u, timeout=20)
    return f"http={r.status_code}\n{r.text}"


def put_xml(sess, u: str, body: str, dry: bool) -> str:
    if dry:
        return f"[dry-run] PUT {u}\n{body}"
    r = sess.put(u, data=body.encode("utf-8"),
                 headers={"Content-Type": "application/xml"}, timeout=20)
    return f"http={r.status_code}\n{r.text}"


def logserver_xml(enabled: str, ip: str, port: int, interval: int, fmt: str) -> str:
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<LogServer version="1.0" xmlns="{NS}">'
            f"<enabled>{enabled}</enabled>"
            f"<addressingFormatType>{fmt}</addressingFormatType>"
            f"<ipAddress>{ip}</ipAddress>"
            f"<portNo>{port}</portNo>"
            f"<uploadInterval>{interval}</uploadInterval>"
            f"</LogServer>")


def facedetect_xml(ch: int, enabled: str, sensitivity: int, highlight: str) -> str:
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<FaceDetect version="1.0" xmlns="{NS}">'
            f"<id>{ch}</id>"
            f"<enabled>{enabled}</enabled>"
            f"<sensitivityLevel>{sensitivity}</sensitivityLevel>"
            f"<highlightsenabled>{highlight}</highlightsenabled>"
            f"</FaceDetect>")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("ls-get", "ls-set", "fd-get", "fd-set"))
    ap.add_argument("--host", default=os.environ.get("ISAPI_HOST", "192.168.1.4"))
    ap.add_argument("--user", default=os.environ.get("ISAPI_USER", "admin"))
    ap.add_argument("--pass", dest="pwd", default=os.environ.get("ISAPI_PASS", ""))
    ap.add_argument("--dry-run", action="store_true")
    # ls-set
    ap.add_argument("--enabled", default="true")
    ap.add_argument("--ip", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--interval", type=int, default=1)
    ap.add_argument("--format", dest="fmt", default="ipaddress")
    # fd-*
    ap.add_argument("--channel", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sensitivity", type=int, default=3)
    ap.add_argument("--highlight", default="false")
    args = ap.parse_args()

    if not args.pwd:
        print("error: no password (--pass or ISAPI_PASS)", file=sys.stderr)
        return 2
    base = f"https://{args.host}"
    sess = make_session(args.user, args.pwd)

    if args.cmd == "ls-get":
        print(get_xml(sess, url(base, "/ISAPI/System/LogServer")))
        return 0

    if args.cmd == "ls-set":
        if not args.ip or not args.port:
            print("error: --ip and --port are required", file=sys.stderr)
            return 2
        body = logserver_xml(args.enabled, args.ip, args.port, args.interval, args.fmt)
        print(put_xml(sess, url(base, "/ISAPI/System/LogServer"), body, args.dry_run))
        if not args.dry_run:
            print("# read-back:\n" + get_xml(sess, url(base, "/ISAPI/System/LogServer")))
        return 0

    if args.cmd == "fd-get":
        chans = list(range(1, 11)) if args.all else [args.channel or 1]
        for c in chans:
            r = sess.get(url(base, f"/ISAPI/Smart/FaceDetect/{c}"), timeout=20)
            on = "?"
            if "<enabled>" in r.text:
                on = r.text.split("<enabled>", 1)[1].split("<", 1)[0]
            print(f"ch{c:<3} http={r.status_code} enabled={on}")
        return 0

    # fd-set
    if args.channel is None:
        print("error: --channel required for fd-set", file=sys.stderr)
        return 2
    body = facedetect_xml(args.channel, args.enabled, args.sensitivity, args.highlight)
    print(put_xml(sess, url(base, f"/ISAPI/Smart/FaceDetect/{args.channel}"),
                  body, args.dry_run))
    if not args.dry_run:
        print("# read-back:\n"
              + get_xml(sess, url(base, f"/ISAPI/Smart/FaceDetect/{args.channel}")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
