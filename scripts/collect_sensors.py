#!/usr/bin/env python3
"""Collect sensor readings via the `sensors` command and cache them to a file.

Drop-in Python replacement for collect_sensors.sh.  Runs every POLL_INTERVAL
seconds and writes KEY=VALUE lines to CACHE_FILE, which conky reads.
"""

import subprocess
import sys
import time

CACHE_FILE = "/dev/shm/conky_sensors_cache"
POLL_INTERVAL = 5  # seconds, matching the shell script's `sleep 5`


def run_sensors():
    """Run `sensors` and return its stdout as a list of lines."""
    try:
        result = subprocess.run(
            ["sensors"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        sys.stderr.write("sensors: command not found\n")
        return []
    if result.returncode != 0:
        sys.stderr.write(f"sensors failed (rc={result.returncode}): "
                         f"{result.stderr.strip()}\n")
        return []
    return result.stdout.splitlines()


def find_value(lines, label, field_index, section=None, context_lines=0):
    """Return the field at ``field_index`` (0-based) of the first line that
    contains ``label``.

    Mirrors the shell pipeline::

        grep 'label' | awk '{print $N}'           # field_index = N - 1

    When ``section`` is given, only lines within ``context_lines`` lines after
    a line containing ``section`` are considered, mirroring::

        grep -A N 'section' | grep 'label' | awk '{print $N}'
    """
    if section is not None:
        # Collect the section line plus the N lines that follow it.
        region = []
        for i, line in enumerate(lines):
            if section in line:
                region.extend(lines[i:i + context_lines + 1])
    else:
        region = lines

    for line in region:
        if label in line:
            fields = line.split()
            if len(fields) > field_index:
                return fields[field_index]
    return None


def collect(lines):
    """Parse ``sensors`` output into an ordered list of (key, value)."""
    return [
        ("CPU_PACK", find_value(lines, "Package id 0", 3)),
        ("CORE_0",   find_value(lines, "Core 0", 2)),
        ("CORE_1",   find_value(lines, "Core 1", 2)),
        ("CORE_2",   find_value(lines, "Core 2", 2)),
        ("CORE_3",   find_value(lines, "Core 3", 2)),
        ("CORE_4",   find_value(lines, "Core 4", 2)),
        ("CORE_5",   find_value(lines, "Core 5", 2)),
        # Values sourced from the alienware_wmi block.
        ("GPU_TEMP", find_value(lines, "GPU:", 1,
                                section="alienware_wmi", context_lines=5)),
        ("CPU_FAN",  find_value(lines, "CPU Fan:", 2,
                                section="alienware_wmi", context_lines=5)),
        ("GPU_FAN",  find_value(lines, "GPU Fan:", 2,
                                section="alienware_wmi", context_lines=5)),
        # NVMe SSD temperature (Composite) from the nvme block.
        ("NVME_SSD", find_value(lines, "Composite", 1,
                                section="nvme-pci", context_lines=2)),
        # RAM/SODIMM temperature from the dell_ddv block.
        ("RAM_TEMP", find_value(lines, "SODIMM:", 1,
                                section="dell_ddv", context_lines=10)),
    ]


def write_cache(values):
    """Write the KEY=VALUE lines to CACHE_FILE (truncating it each cycle)."""
    with open(CACHE_FILE, "w", encoding="utf-8") as cache:
        for key, value in values:
            cache.write(f"{key}={value}\n")


def main():
    while True:
        lines = run_sensors()
        values = collect(lines)
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
