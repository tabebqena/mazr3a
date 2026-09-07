#!/usr/bin/env python3
"""Live lm-sensors reading helpers for the host scripts.

Sensors-only module for the host monitoring stack. The Telegram/config helpers
that used to live here moved to scripts/telegram_notify.py (see
plans/machine-monitor-debounce-and-telegram-module.md) - keep this module free
of networking so it stays a pure, dependency-light lm-sensors reader.

Provides:
  - run_sensors():       run `sensors` and return its stdout lines (live).
  - find_value():        extract one whitespace-separated field from sensor lines.
  - collect():           parse `sensors` output into ordered (key, value) pairs.
  - get_cpu_temp_max():  hottest live CPU temperature in °C (read on demand).

The sensor helpers are cache-free: they read live data from lm-sensors on
every call and never read/write any cache file.

Run directly to print the current live readings on demand (nothing is read
from or written to disk):

    python3 collect_sensors.py      # prints CPU_PACK=.. / CORE_0=.. lines
"""
import os
import subprocess
import sys


def run_sensors():
    """Run `sensors` and return its stdout as a list of lines.

    Live, on-demand read - no cache file is read or written. On failure it
    reports to stderr and returns [] so callers can fall back gracefully.
    """
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
    """Parse ``sensors`` output into an ordered list of (key, value).

    Cache-free: parses whatever lines it is given (usually run_sensors()).
    """
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


def get_cpu_temp_max():
    """Hottest live CPU temperature in °C (int) or None if unavailable.

    Reads live sensors on demand via run_sensors()/collect() - no cache file
    involved. Only the CPU fields (CPU_PACK, CORE_*) are considered; the first
    °C value on each sensor line is the live reading (high/crit are not).
    """
    try:
        values = collect(run_sensors())
    except Exception:
        return None
    temps = []
    for key, value in values:
        if not (key == "CPU_PACK" or key.startswith("CORE_")):
            continue
        if value is None:
            continue
        try:
            temps.append(float(value.replace("°C", "").lstrip("+")))
        except ValueError:
            continue
    return int(round(max(temps))) if temps else None


def main():
    """On-demand: read live sensors once and print KEY=VALUE lines to stdout."""
    for key, value in collect(run_sensors()):
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
