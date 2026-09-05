# Merge monitor_lib.py into collect_sensors.py (single monitoring module)

## 1. Problem

The host monitoring stack had two tightly coupled modules:

- `scripts/monitor_lib.py` - shared helpers (Telegram config/send + lm-sensors
  parsing).
- `scripts/collect_sensors.py` - originally a conky cache daemon (reference
  only - no conky here), then an on-demand reader importing from `monitor_lib`.

The user wants a **single consolidated module** that reads **live `sensors`
data on demand** with **no cache file** (no `/dev/shm/conky_sensors_cache`,
no conky).

## 2. Plan / final architecture

- [`scripts/collect_sensors.py`](../scripts/collect_sensors.py) becomes the one
  monitoring module. It absorbs the whole old `monitor_lib.py`:
  - Telegram/config: `load_conf`, `ensure_creds`, `esc_html`,
    `send_telegram`, `send_telegram_photo`.
  - Live cache-free sensors: `run_sensors`, `find_value`, `collect`.
  - `get_cpu_temp_max` (int °C | None) - live on-demand read.
  - Plus an on-demand CLI: running the file directly prints the current
    `KEY=VALUE` readings to stdout (no file I/O, no loop).
- `scripts/monitor_lib.py` is **deleted**.
- Consumers updated to `import collect_sensors as lib`:
  - `scripts/machine-monitor.py`, `scripts/machine-status.py`,
    `scripts/firewatch.py` (docstrings/comments updated too).
- `scripts/deploy_firewatch.sh` bundles `collect_sensors.py` instead of
  `monitor_lib.py`.

## 3. Verification

- `python3 -m py_compile scripts/collect_sensors.py scripts/machine-monitor.py
  scripts/machine-status.py scripts/firewatch.py`
- Confirm nothing references `monitor_lib` anymore (repo-wide scan, excluding
  .venv/.git) and no cache file (`CACHE_FILE`/`/dev/shm`) anywhere.
- Import smoke: load `machine-monitor`, `machine-status` through the new
  `collect_sensors` module.
- Parse simulation + on-demand `collect_sensors.py` output.
- Move to the remote host (ssh.mazr3a.garden, user `ai`): deploy
  `scripts/collect_sensors.py` (and updated machine/firewatch scripts where
  used), then run `machine-monitor.py --dry-run` and `machine-status.py
  --dry-run`.

## 4. Implemented

- 2026-09-05 (commit 7a1fc58, superseded): `monitor_lib.py` imported
  `collect_sensors` and delegated parsing to it.
- 2026-09-05 (commit 68546b6, superseded): moved cache-free live helpers into
  `monitor_lib.py`; `collect_sensors.py` was a cache daemon.
- 2026-09-05 (commit 95d4ad6, superseded): removed all cache-file read/write;
  `collect_sensors.py` was an on-demand reader importing from `monitor_lib`.
- 2026-09-05 (next commit): merged `monitor_lib.py` **into**
  `collect_sensors.py` (single module, live on-demand, no cache), deleted
  `monitor_lib.py`, and repointed all consumers/deploy scripts at
  `collect_sensors.py`.
