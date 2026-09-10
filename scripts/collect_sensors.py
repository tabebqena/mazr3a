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
  - get_gpu_usage():     Intel iGPU busy % from the i915 RC6 idle counter.
  - get_gpu_freq():      Intel iGPU act/cur/max frequency in MHz.
  - get_gpu_temp():      GPU temperature in °C when the host exposes one.

The sensor helpers are cache-free: they read live data from lm-sensors (and
sysfs) on every call and never read/write any cache file. get_gpu_usage() is
the one exception to "instant" - it needs TWO readings of a cumulative counter
a short interval apart, so it samples for ~1 s by default.

Run directly to print the current live readings on demand (nothing is read
from or written to disk):

    python3 collect_sensors.py      # prints CPU_PACK=.. / CORE_0=.. lines
"""
import glob
import os
import subprocess
import sys
import time


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


# ---------------------------------------------------------------------------
# GPU (Intel iGPU) helpers - best-effort, None when unavailable
# ---------------------------------------------------------------------------
# The Frigate OpenVINO detector runs on the Intel UHD 630 iGPU, so the host
# watchdogs also report GPU load/temperature. Availability is host-dependent and
# both helpers degrade to None (callers then simply omit the metric):
#
#   * USAGE - busy % derived from the i915 RC6 (GPU idle) residency counter at
#     /sys/class/drm/card*/gt/gt*/rc6_residency_ms: RC6 is the GPU idle state, so
#     busy% = 1 - d(rc6_ms) / d(t_ms) over a short window. Readable by an
#     ordinary user. `intel_gpu_top` would give a richer per-engine breakdown but
#     it reads the i915 PMU and needs CAP_PERFMON/root - verified on this host
#     ("Failed to initialize PMU! (Permission denied)" as `ai`, with
#     perf_event_paranoid=3) - so it is deliberately NOT used here.
#   * TEMPERATURE - only if the host exposes a GPU hwmon (amdgpu/nouveau/xe/...)
#     or an lm-sensors GPU field (e.g. alienware_wmi "GPU:"). An Intel iGPU
#     shares the CPU die and normally has NO separate sensor, so this is
#     typically None here (coretemp reports CPU cores only).
_GPU_HWMON_NAMES = ("i915", "xe", "amdgpu", "nouveau", "radeon", "intel_gpu")


def _read_int(path):
    """Read a single integer from a sysfs file, or None."""
    try:
        with open(path, encoding="utf-8") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def _read_str(path):
    """Read a sysfs file as a stripped string, or None."""
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None


def _intel_gt_dir():
    """sysfs 'gt' dir of the Intel iGPU (i915/xe) exposing an RC6 counter.

    Returns None when there is no Intel GPU (or RC6 is not exposed). Walks the
    DRM cards and checks the PCI driver name, so connector nodes such as
    card0-DP-1 and any second (discrete) card are skipped safely.
    """
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        if "-" in os.path.basename(card):
            continue  # card0-DP-1 / card0-HDMI-A-1 connector symlinks
        driver = os.path.basename(
            os.path.realpath(os.path.join(card, "device", "driver"))
        )
        if driver not in ("i915", "xe"):
            continue
        for gt in sorted(glob.glob(os.path.join(card, "gt", "gt*"))):
            if os.path.isfile(os.path.join(gt, "rc6_residency_ms")):
                return gt
    return None


def get_gpu_usage(sample_s=1.0):
    """Intel iGPU busy % (0-100 int), or None when unavailable.

    Busy % = 100 * (1 - d(rc6_residency_ms) / d(t_ms)) over ``sample_s`` seconds
    (RC6 is the GPU's idle state). Never raises. Returns None when there is no
    Intel GPU, RC6 is disabled, the counter is unreadable, or the counter does
    not advance - so an absent/broken counter can never be reported as "100%".
    """
    gt = _intel_gt_dir()
    if gt is None:
        return None
    if _read_str(os.path.join(gt, "rc6_enable")) == "0":
        return None  # RC6 off -> the residency counter is not meaningful
    rc6_a = _read_int(os.path.join(gt, "rc6_residency_ms"))
    t_a = time.monotonic()
    if rc6_a is None:
        return None
    time.sleep(max(0.0, float(sample_s)))
    rc6_b = _read_int(os.path.join(gt, "rc6_residency_ms"))
    t_b = time.monotonic()
    if rc6_b is None:
        return None
    elapsed_ms = (t_b - t_a) * 1000.0
    if elapsed_ms <= 0 or rc6_b < rc6_a:
        return None
    idle = min(1.0, max(0.0, (rc6_b - rc6_a) / elapsed_ms))
    return int(round(100.0 * (1.0 - idle)))


def get_gpu_freq():
    """Intel iGPU (act, cur, max) frequency in MHz; any element may be None."""
    gt = _intel_gt_dir()
    if gt is None:
        return None, None, None
    return (
        _read_int(os.path.join(gt, "rps_act_freq_mhz")),
        _read_int(os.path.join(gt, "rps_cur_freq_mhz")),
        _read_int(os.path.join(gt, "rps_max_freq_mhz")),
    )


def get_gpu_temp():
    """GPU temperature in °C (int), or None when the host exposes no sensor.

    Prefers a GPU hwmon (amdgpu/nouveau/xe/i915), then falls back to the
    lm-sensors "GPU_TEMP" field already parsed by collect() (e.g. the
    alienware_wmi "GPU:" reading). Intel iGPUs usually have no separate sensor
    -> None on this host.
    """
    for hwmon in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name = (_read_str(os.path.join(hwmon, "name")) or "").lower()
        if name not in _GPU_HWMON_NAMES:
            continue
        for temp_input in sorted(glob.glob(os.path.join(hwmon, "temp*_input"))):
            milli = _read_int(temp_input)
            if milli is not None:
                return int(round(milli / 1000.0))
        return None
    try:
        for key, value in collect(run_sensors()):
            if key == "GPU_TEMP" and value:
                return int(round(float(value.replace("°C", "").lstrip("+"))))
    except (TypeError, ValueError):
        pass
    return None


def main():
    """On-demand: read live sensors once and print KEY=VALUE lines to stdout."""
    for key, value in collect(run_sensors()):
        print(f"{key}={value}")
    usage = get_gpu_usage()
    act, cur, mx = get_gpu_freq()
    gpu_temp = get_gpu_temp()
    print(f"GPU_USAGE={'n/a' if usage is None else str(usage) + '%'}")
    print(f"GPU_FREQ_ACT={'n/a' if act is None else str(act) + 'MHz'}")
    print(f"GPU_FREQ_CUR={'n/a' if cur is None else str(cur) + 'MHz'}")
    print(f"GPU_FREQ_MAX={'n/a' if mx is None else str(mx) + 'MHz'}")
    print(f"GPU_TEMP_SYSFS={'n/a' if gpu_temp is None else str(gpu_temp) + '°C'}")


if __name__ == "__main__":
    main()
