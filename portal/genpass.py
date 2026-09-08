#!/usr/bin/env python3
"""Generate a `user ...` line to paste into config/portal.conf.

Usage:
    python portal/genpass.py --username admin                 # hidden prompt
    echo 'hunter2' | python portal/genpass.py --username admin
    python portal/genpass.py --username admin --password 'pw' --default-camera cam01

Prints (to stdout):
    user admin pbkdf2$600000$<salt>$<hash> [default_camera=cam01]
"""
import argparse
import getpass
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auth  # noqa: E402  (portal sibling module)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", default=None,
                    help="if omitted: prompt on a TTY or read one line from stdin")
    ap.add_argument("--default-camera", default="")
    ap.add_argument("--iterations", type=int, default=auth._DEFAULT_ITERS)
    a = ap.parse_args(argv)

    pw = a.password
    if pw is None:
        pw = getpass.getpass("Password: ") if sys.stdin.isatty() \
            else sys.stdin.readline().rstrip("\n")
    if not pw:
        print("error: empty password", file=sys.stderr)
        return 2

    line = "user {} {}".format(a.username, auth.hash_password(pw, iterations=a.iterations))
    if a.default_camera:
        line += " default_camera={}".format(a.default_camera)
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
