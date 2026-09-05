# Reuse collect_sensors.py parsing in monitor_lib.py

## 1. Problem

[`scripts/collect_sensors.py`](../scripts/collect_sensors.py) (new, untracked)
is the canonical lm-sensors parser/cache writer for the conky dashboard. It
exposes clean helpers (`run_sensors`, `find_value`, `collect`).

[`scripts/monitor_lib.py`](../scripts/monitor_lib.py) currently **duplicates**
lm-sensors parsing inside `get_cpu_temp_max()` with its own regex/`subprocess`
logic (`_SENSOR_KEYS`, `_TEMP_RE`). This risks the two parsers drifting apart
and adds a stray debug `print(e)` (uncommitted WIP) that pollutes stdout.

## 2. Plan

- Import `collect_sensors` from `monitor_lib` (same directory; add
  `LIB_DIR` to `sys.path` first so the import works regardless of how
  `monitor_lib` itself is loaded).
- Rewrite `get_cpu_temp_max()` to delegate to
  `collect_sensors.run_sensors()` + `collect_sensors.collect()`, keeping only
  the CPU fields (`CPU_PACK`, `CORE_*`) and preserving the existing
  `int °C | None` return contract used by
  `machine-monitor.py` and `machine-status.py`.
- Drop the now-unused `re`/`subprocess` imports and the debug `print(e)`
  (sensor errors are reported to stderr by `collect_sensors.run_sensors()`).
- Keep the public API unchanged so consumers need no edits.

## 3. Verification

- `python3 -m py_compile scripts/monitor_lib.py scripts/collect_sensors.py`
- Simulate parsing: feed representative `sensors` output (alienware_wmi /
  nvme-pci / dell_ddv blocks) to `collect()` and check `get_cpu_temp_max()`
  returns the expected value.
- Move to the remote host (ssh.mazr3a.garden, user `ai`) and re-run the two
  cron consumers (`machine-monitor.py --dry-run`, `machine-status.py
  --dry-run`) where `sensors` and `config/telegram.conf` exist.

## 4. Implemented

- 2026-09-05: Refactored `monitor_lib.py` to import and reuse
  `collect_sensors.py`; `get_cpu_temp_max()` delegates to
  `collect_sensors.collect(collect_sensors.run_sensors())`. API unchanged.
