# Reuse collect_sensors.py parsing in monitor_lib.py

## 1. Problem

[`scripts/collect_sensors.py`](../scripts/collect_sensors.py) is a reference
for the lm-sensors parsing used by the monitoring stack. The Telegram
monitoring scripts ([`machine-monitor.py`](../scripts/machine-monitor.py),
[`machine-status.py`](../scripts/machine-status.py)) only need the parsed live
sensor values (CPU package/cores, Alienware GPU/fan, NVMe, RAM temps).

Decision (user): there is **no conky here** - the conky/cache approach was a
reference only. Remove **all** read & write to the cache file
(`/dev/shm/conky_sensors_cache`). The monitoring scripts read **live `sensors`
data on demand**; no file I/O at all.

## 2. Plan (final - cache-free, on-demand, no file I/O)

- Host the live, cache-free helpers in the shared library
  [`scripts/monitor_lib.py`](../scripts/monitor_lib.py):
  - `run_sensors()` - runs the `sensors` binary on demand and returns stdout
    lines (errors go to stderr, returns `[]` on failure).
  - `find_value()` - whitespace-field extraction mirroring `grep | awk`.
  - `collect()` - parses lines into ordered `(key, value)` pairs
    (CPU_PACK, CORE_*, GPU_TEMP, CPU_FAN, GPU_FAN, NVME_SSD, RAM_TEMP).
  - `get_cpu_temp_max()` - hottest live CPU temp (`int °C | None`) built on
    `run_sensors()`/`collect()`, preserving the contract used by
    `machine-monitor.py` and `machine-status.py`.
  - The library contains **no** `CACHE_FILE`, `/dev/shm`, or conky reference.
- [`scripts/collect_sensors.py`](../scripts/collect_sensors.py) is kept as an
  **on-demand reference tool**: no daemon loop, no cache file, no disk I/O.
  It imports `run_sensors()`/`collect()` from `monitor_lib`, reads live data
  once, and prints `KEY=VALUE` lines to stdout.
- Public APIs of both scripts unchanged for their consumers.

## 3. Verification

- `python3 -m py_compile scripts/monitor_lib.py scripts/collect_sensors.py`
- Confirm neither script references `CACHE_FILE` / `/dev/shm` / a cache file.
- Parse simulation: feed representative `sensors` output (coretemp /
  alienware_wmi / nvme-pci / dell_ddv blocks) to `collect()` and check
  `get_cpu_temp_max()` returns the expected value (live on-demand read).
- Confirm `collect_sensors.py` prints `KEY=VALUE` lines from `monitor_lib`.
- Move to the remote host (ssh.mazr3a.garden, user `ai`) and re-run the two
  cron consumers (`machine-monitor.py --dry-run`, `machine-status.py
  --dry-run`) - live readings, no cache involved.

## 4. Implemented

- 2026-09-05 (commit 7a1fc58, superseded): `monitor_lib.py` imported
  `collect_sensors` and delegated to `collect_sensors.collect()` /
  `collect_sensors.run_sensors()`. Rejected: coupled `monitor_lib` to a
  cache-file module.
- 2026-09-05 (commit 68546b6, superseded): moved the cache-free live helpers
  into `monitor_lib.py`; `collect_sensors.py` became a thin cache daemon.
  Rejected: still wrote the cache file.
- 2026-09-05 (next commit): removed **all** cache-file read/write.
  `collect_sensors.py` is now an on-demand reference tool (no file I/O, no
  loop) importing from `monitor_lib.py`, which reads live `sensors` data on
  demand. No conky here - the conky/cache design was reference only.
