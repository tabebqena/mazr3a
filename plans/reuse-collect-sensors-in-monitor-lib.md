# Reuse collect_sensors.py parsing in monitor_lib.py

## 1. Problem

The conky sensor pipeline ([`scripts/collect_sensors.py`](../scripts/collect_sensors.py))
and the Telegram monitoring scripts share the same lm-sensors parsing needs
(CPU cores/package, plus Alienware GPU/fan, NVMe, RAM temps). Parsing was
duplicated inside [`scripts/monitor_lib.py`](../scripts/monitor_lib.py)
(`get_cpu_temp_max()` regex logic), risking drift.

Requirement (user): monitoring scripts must **not** use `CACHE_FILE`
(`/dev/shm/conky_sensors_cache`) at all - they must **read live `sensors`
data on demand**. Only the conky cache daemon may touch the cache file.

## 2. Plan (revised - cache-free shared helpers in the library)

- Host the live, cache-free helpers in the shared library
  [`scripts/monitor_lib.py`](../scripts/monitor_lib.py):
  - `run_sensors()` - runs the `sensors` binary on demand, returns stdout
    lines (no cache read/write; errors go to stderr, returns `[]`).
  - `find_value()` - whitespace-field extraction mirroring `grep | awk`.
  - `collect()` - parses lines into ordered `(key, value)` pairs
    (CPU_PACK, CORE_*, GPU_TEMP, CPU_FAN, GPU_FAN, NVME_SSD, RAM_TEMP).
  - `get_cpu_temp_max()` - hottest live CPU temp (`int °C | None`), built on
    `run_sensors()`/`collect()`, preserving the contract used by
    `machine-monitor.py` and `machine-status.py`.
- [`scripts/collect_sensors.py`](../scripts/collect_sensors.py) becomes a thin
  conky-cache daemon: imports `run_sensors()`/`collect()` from `monitor_lib`
  and writes only the parsed values to `CACHE_FILE` every `POLL_INTERVAL`.
  No parser duplication; `CACHE_FILE` appears nowhere in `monitor_lib.py`.
- Public APIs of both scripts unchanged for their consumers.

## 3. Verification

- `python3 -m py_compile scripts/monitor_lib.py scripts/collect_sensors.py`
- Confirm `monitor_lib.py` contains no `CACHE_FILE` / `/dev/shm` reference.
- Parse simulation: feed representative `sensors` output (coretemp /
  alienware_wmi / nvme-pci / dell_ddv blocks) to `collect()` and check
  `get_cpu_temp_max()` returns the expected value (live on-demand read).
- Confirm `collect_sensors.py` imports cleanly from `monitor_lib`.
- Move to the remote host (ssh.mazr3a.garden, user `ai`) and re-run the two
  cron consumers (`machine-monitor.py --dry-run`, `machine-status.py
  --dry-run`) plus the conky daemon sanity check.

## 4. Implemented

- 2026-09-05 (commit 7a1fc58, superseded): first pass - `monitor_lib.py`
  imported `collect_sensors` and delegated to `collect_sensors.collect()` /
  `collect_sensors.run_sensors()`. Rejected: coupled `monitor_lib` to a
  cache-file module.
- 2026-09-05 (next commit): revised - moved the live cache-free helpers
  (`run_sensors`/`find_value`/`collect`) into `monitor_lib.py` and turned
  `collect_sensors.py` into a thin conky-cache daemon importing them from the
  library. `monitor_lib.get_cpu_temp_max()` reads live sensors on demand;
  `CACHE_FILE` exists only in `collect_sensors.py`.
