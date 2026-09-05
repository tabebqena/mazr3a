#!/usr/bin/env python3
"""Collect sensor readings via `sensors` and cache them to a file.

Drop-in Python replacement for collect_sensors.sh.  Runs every POLL_INTERVAL
seconds and writes KEY=VALUE lines to CACHE_FILE, which conky reads.

The live, cache-free read/parse helpers (run_sensors()/collect()) live in the
shared monitoring library monitor_lib.py; this daemon only writes the freshly
parsed values to CACHE_FILE for conky.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import monitor_lib as lib  # noqa: E402

CACHE_FILE = "/dev/shm/conky_sensors_cache"
POLL_INTERVAL = 5  # seconds, matching the shell script's `sleep 5`


def write_cache(values):
    """Write the KEY=VALUE lines to CACHE_FILE (truncating it each cycle)."""
    with open(CACHE_FILE, "w", encoding="utf-8") as cache:
        for key, value in values:
            cache.write(f"{key}={value}\n")


def main():
    while True:
        # Live read on demand via the shared library; no cache is read here.
        values = lib.collect(lib.run_sensors())
        try:
            write_cache(values)
        except OSError as exc:
            sys.stderr.write(f"failed to write {CACHE_FILE}: {exc}\n")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
