#!/usr/bin/env python3
"""scenereader.py - cross-camera episode narrator for the Frigate stack.

Replaces the stopped `scenewatch` sweep+VLM service (plans/scene-description.md).
The required output is a story stitched across cameras, e.g.

    "Ali entered Field 1 (cam04) at 19:36, stayed 12 min; then went to Store
     (cam07) at 19:49, stayed 15 min with Ahmad; left at 20:04."

What runs here (module map; see plans/event-scene-reader.md):
  * frigate.py  - read Frigate's clips on disk + event metadata (exact-id join)
  * store.py    - the WAL text store (events, episodes, visits, aliases)
  * describe.py - the deterministic per-event sentence + importance tier
  * episodes.py - L1 places, L2 linking, L3 episodes, L4 narrative
  * models.py   - the captioner backends (small GGUF resident by default)

Design notes
------------
* EVENT-DRIVEN, NEVER RE-DETECTING. We do not sweep frames or re-derive motion;
  Frigate already detected. We only read what it produced.
* FRAMES ARE NEVER COPIED. They are read in place from Frigate's own media tree;
  the only new files are the text DB + a status/trigger/names file.
* LAZY BY WHEN, NOT BY RESIDENCY. Captioning (the only expensive step) runs only
  during an idle-gated drain, bounded by MAX_EVENTS_PER_RUN/MAX_RUN_SECONDS; the
  model itself is loaded once and kept resident (MODEL_KEEP_LOADED=true) because
  for a small model the reload churn would cost CPU for no real RAM gain.
* BOUNDED + EXCEPTION-ISOLATED. The while-loop is the only unbounded loop; every
  scan/drain pass is guarded; SIGTERM/SIGINT set a stop flag checked each tick;
  the store/status writes are best-effort and never stop the loop.

Usage:
  python scenereader.py                  # run the scheduler loop
  python scenereader.py --check          # config + places + model load probe
  python scenereader.py --scan-only      # harvest + rebuild episodes, exit
  python scenereader.py --drain          # force a caption batch, exit
  python scenereader.py --rebuild-episodes
  python scenereader.py --once           # scan + rebuild + drain, exit
  python scenereader.py --dry-run        # like --once but stores nothing
  python scenereader.py --status
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time

import captioners
import describe
import episodes
import frigate
import store

CONF_PATH = os.environ.get("SCENEREADER_CONF", "/config/scenereader.conf")
PLACES_PATH = os.environ.get("PLACES_CONF", "/config/places.conf")
_STOP = [False]


def LOG(*args):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *args, flush=True)


def _sig_stop(_signum, _frame):
    _STOP[0] = True


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def _raw_conf(path):
    cfg = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                cfg[key.strip()] = value.strip()
    except OSError:
        return {}
    return cfg


def _get(raw, key, default=""):
    return str(os.environ.get(key) or raw.get(key) or default).strip()


def _getf(raw, key, default):
    try:
        return float(_get(raw, key, str(default)))
    except ValueError:
        return default


def _geti(raw, key, default):
    try:
        return int(float(_get(raw, key, str(default))))
    except ValueError:
        return default


def _getb(raw, key, default):
    return _get(raw, key, "true" if default else "false").lower() in (
        "1", "true", "yes", "on")


def _list(raw, key, default):
    return [t.strip() for t in _get(raw, key, ",".join(default)).split(",") if t.strip()]


class S:
    """Resolved settings (declared so attributes are statically known)."""

    def __init__(self):
        self.cameras = []
        self.enabled = True
        self.api = "http://frigate:5000"
        self.media_dir = "/media"
        self.snapshot_dirs = ["clips"]
        self.snapshot_glob = "*.jpg"
        self.metadata_source = "db"
        self.frigate_db = "/config/frigate.db"
        self.scan_every = 60.0
        self.scan_limit = 400
        self.re_enrich_per_run = 50
        self.rescan_overlap_s = 300.0
        self.drain_every = 300.0
        self.episodes_every = 300.0
        self.max_events_per_run = 8
        self.max_run_seconds = 240.0
        self.events_retention_days = 0.0
        self.max_loadavg = 3.0
        self.resume_loadavg = 1.5
        self.max_cpu_temp_c = 78.0
        self.resume_cpu_temp_c = 70.0
        self.force_allow_hot = False
        self.host_proc = "/proc"
        self.host_hwmon = "/sys/class/hwmon"
        self.store_dir = "/media/events"
        self.store_db = "events.db"
        self.store_enabled = True
        self.trigger_file = "/media/events/.drain_request"
        self.status_file = "/media/events/reader_status.json"
        self.names_file = "/media/events/names.json"
        self.heartbeat_s = 300.0
        self.tick_min = 0.5
        # model
        self.model_backend = "llamacpp"
        self.model_file = "/models/scene/smolvlm2-500m/smolvlm2-500m-q8.gguf"
        self.mmproj_file = "/models/scene/smolvlm2-500m/mmproj-smolvlm2-500m-f16.gguf"
        self.llama_server_bin = "llama-server"
        self.llama_cli_bin = "llama-mtmd-cli"
        self.openvino_dir = "/models/scene"
        self.openvino_device = "CPU"
        self.openvino_threads = 2
        self.model_keep_loaded = True
        self.idle_unload_s = 0.0
        self.vlm_n_threads = 2
        self.vlm_ctx = 4096
        self.vlm_max_tokens = 64
        self.vlm_prompt = captioners.DEFAULT_PROMPT
        self.only_person_captions = False
        # episodes
        self.episode_labels = ["person"]
        self.episode_gap_s = 600.0
        self.reid_max_gap_s = 90.0
        self.visit_min_s = 0.0
        self.narrative_tz_offset_h = 0.0


def resolve_settings(raw):
    s = S()
    s.cameras = _list(raw, "CAMERAS", [])
    s.enabled = _getb(raw, "ENABLED", True)
    s.api = _get(raw, "FRIGATE_API", "http://frigate:5000").rstrip("/")
    s.media_dir = _get(raw, "FRIGATE_MEDIA_DIR", "/media").rstrip("/")
    s.snapshot_dirs = _list(raw, "FRIGATE_SNAPSHOT_DIRS", ["clips"])
    s.snapshot_glob = _get(raw, "FRIGATE_SNAPSHOT_GLOB", "*.jpg")
    s.metadata_source = _get(raw, "FRIGATE_METADATA_SOURCE", "db").lower()
    s.frigate_db = _get(raw, "FRIGATE_DB", "/config/frigate.db")
    s.scan_every = max(5.0, _getf(raw, "SCAN_EVERY_S", 60))
    s.scan_limit = max(1, _geti(raw, "SCAN_LIMIT", 400))
    s.re_enrich_per_run = max(0, _geti(raw, "RE_ENRICH_PER_RUN", 50))
    s.rescan_overlap_s = max(0.0, _getf(raw, "RESCAN_OVERLAP_S", 300))
    s.drain_every = max(30.0, _getf(raw, "DRAIN_EVERY_S", 300))
    s.episodes_every = max(30.0, _getf(raw, "EPISODES_EVERY_S", 300))
    s.max_events_per_run = max(1, _geti(raw, "MAX_EVENTS_PER_RUN", 8))
    s.max_run_seconds = max(5.0, _getf(raw, "MAX_RUN_SECONDS", 240))
    s.events_retention_days = max(0.0, _getf(raw, "EVENTS_RETENTION_DAYS", 30))
    s.max_loadavg = max(0.1, _getf(raw, "MAX_LOADAVG", 3.0))
    s.resume_loadavg = max(0.0, _getf(raw, "RESUME_LOADAVG", 1.5))
    s.max_cpu_temp_c = _getf(raw, "MAX_CPU_TEMP_C", 78.0)
    s.resume_cpu_temp_c = _getf(raw, "RESUME_CPU_TEMP_C", 70.0)
    s.force_allow_hot = _getb(raw, "FORCE_ALLOW_HOT", False)
    s.host_proc = _get(raw, "HOST_PROC", "/proc").rstrip("/")
    s.host_hwmon = _get(raw, "HOST_HWMON", "/sys/class/hwmon").rstrip("/")
    s.store_dir = _get(raw, "STORE_DIR", "/media/events").rstrip("/")
    s.store_db = _get(raw, "STORE_DB", "events.db")
    s.store_enabled = _getb(raw, "STORE_ENABLED", True)
    s.trigger_file = _get(raw, "TRIGGER_FILE", os.path.join(s.store_dir, ".drain_request"))
    s.status_file = _get(raw, "STATUS_FILE", os.path.join(s.store_dir, "reader_status.json"))
    s.names_file = _get(raw, "NAMES_FILE", os.path.join(s.store_dir, "names.json"))
    s.heartbeat_s = max(30.0, _getf(raw, "HEARTBEAT_S", 300))
    s.model_backend = _get(raw, "MODEL_BACKEND", "llamacpp").lower()
    s.model_file = _get(raw, "MODEL_FILE", s.model_file)
    s.mmproj_file = _get(raw, "MMPROJ_FILE", s.mmproj_file)
    s.llama_server_bin = _get(raw, "LLAMA_SERVER_BIN", "llama-server")
    s.llama_cli_bin = _get(raw, "LLAMA_CLI_BIN", "llama-mtmd-cli")
    s.openvino_dir = _get(raw, "OPENVINO_DIR", "/models/scene")
    s.openvino_device = _get(raw, "OPENVINO_DEVICE", "CPU")
    s.openvino_threads = max(1, _geti(raw, "OPENVINO_THREADS", s.vlm_n_threads))
    s.model_keep_loaded = _getb(raw, "MODEL_KEEP_LOADED", True)
    s.idle_unload_s = max(0.0, _getf(raw, "IDLE_UNLOAD_S", 0))
    s.vlm_n_threads = max(1, _geti(raw, "VLM_N_THREADS", 2))
    s.vlm_ctx = max(512, _geti(raw, "VLM_CTX", 4096))
    s.vlm_max_tokens = max(1, _geti(raw, "VLM_MAX_TOKENS", 64))
    s.vlm_prompt = _get(raw, "VLM_PROMPT", captioners.DEFAULT_PROMPT)
    s.only_person_captions = _getb(raw, "ONLY_PERSON_CAPTIONS", False)
    s.episode_labels = [x.lower() for x in _list(raw, "EPISODE_LABELS", ["person"])]
    s.episode_gap_s = max(1.0, _getf(raw, "EPISODE_GAP_S", 600))
    s.reid_max_gap_s = max(0.0, _getf(raw, "REID_MAX_GAP_S", 90))
    s.visit_min_s = max(0.0, _getf(raw, "VISIT_MIN_S", 0))
    s.narrative_tz_offset_h = _getf(raw, "NARRATIVE_TZ_OFFSET_H", 0)
    return s


# ---------------------------------------------------------------------------
# idle governor (load + CPU temperature)
# ---------------------------------------------------------------------------
def loadavg1(proc_dir="/proc"):
    """1-minute load average from the host /proc (0.0 when unreadable)."""
    try:
        with open(os.path.join(proc_dir, "loadavg"), encoding="utf-8") as fh:
            return float(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def cpu_temp_c(hwmon_dir="/sys/class/hwmon"):
    """Hottest CPU package/core temperature in C (None when unreadable)."""
    best = None
    try:
        entries = sorted(os.listdir(hwmon_dir))
    except OSError:
        return None
    for entry in entries:
        base = os.path.join(hwmon_dir, entry)
        try:
            with open(os.path.join(base, "name"), encoding="utf-8") as fh:
                if fh.read().strip() not in ("coretemp", "k10temp", "cpu_thermal"):
                    continue
        except OSError:
            continue
        for fname in os.listdir(base):
            if not (fname.startswith("temp") and fname.endswith("_input")):
                continue
            try:
                with open(os.path.join(base, fname), encoding="utf-8") as fh:
                    value = float(fh.read().strip()) / 1000.0
            except (OSError, ValueError):
                continue
            if 0 < value < 200 and (best is None or value > best):
                best = value
    return best


def governor_state(s):
    """(ok, reason) - may a caption batch run right now?"""
    load = loadavg1(s.host_proc)
    temp = cpu_temp_c(s.host_hwmon)
    if load > s.max_loadavg:
        return False, "loadavg1 {:.2f} > {:.2f}".format(load, s.max_loadavg)
    if temp is not None and temp >= s.max_cpu_temp_c:
        return False, "cpu {:.1f}C >= {:.1f}C".format(temp, s.max_cpu_temp_c)
    return True, "loadavg1 {:.2f} temp {}".format(
        load, "n/a" if temp is None else "{:.1f}C".format(temp))


# ---------------------------------------------------------------------------
# paths / status
# ---------------------------------------------------------------------------
def db_path(s):
    return os.path.join(s.store_dir, s.store_db)


def clips_dir(s):
    return os.path.join(s.media_dir, s.snapshot_dirs[0]) if s.snapshot_dirs else s.media_dir


def _write_json(path, payload):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        pass


def read_status(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _bounds(conn):
    row = conn.execute("SELECT MAX(start_time) AS mx, COUNT(*) AS n FROM events").fetchone()
    return (float(row["mx"] or 0.0), int(row["n"] or 0))


def _pending_count(conn):
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM events WHERE (vlm_at IS NULL OR vlm_at = 0)"
        " AND (vlm_status IS NULL OR vlm_status = '')"
        " AND frame_path IS NOT NULL AND frame_path != ''").fetchone()
    return int(row["n"] or 0)


def write_status(s, conn, cap=None, extra=None):
    payload = dict(extra or {})
    try:
        _max_ts, total = _bounds(conn)
        payload.update({"events_total": total, "pending": _pending_count(conn),
                        "episodes_total": conn.execute(
                            "SELECT COUNT(*) FROM episodes").fetchone()[0]})
    except Exception:  # noqa: BLE001 - status is informational
        pass
    if cap is not None:
        payload.update({"model": getattr(cap, "name", ""),
                        "backend": getattr(cap, "backend", ""),
                        "model_loaded": True,
                        "alive": cap.alive()})
    payload["updated_at"] = time.time()
    payload["pid"] = os.getpid()
    _write_json(s.status_file, payload)
    return payload


# ---------------------------------------------------------------------------
# L0 scan: harvest clips + enrich metadata + describe
# ---------------------------------------------------------------------------
def _merge_clip_with_meta(item, meta):
    """Clip-derived frame paths win; metadata supplies label/zones/times."""
    merged = dict(item)
    if meta:
        merged.update({k: v for k, v in meta.items() if v is not None})
        # never let metadata blank the paths we already resolved from disk
        merged["frame_path"] = item["frame_path"]
        merged["frame_clean_path"] = item.get("frame_clean_path")
    merged.setdefault("label", "")
    return merged


def harvest(s, conn, places, dry=False, log=LOG):
    """One scan pass: new clips -> store (enriched + described). Returns counts."""
    since, _total = _bounds(conn)
    since = max(0.0, since - s.rescan_overlap_s) if since else 0.0
    items = frigate.scan_clips(clips_dir(s), since=since, limit=s.scan_limit)
    if not items:
        return {"seen": 0, "new": 0, "enriched": 0}

    db_conn = frigate.open_frigate_db(s.frigate_db) if s.metadata_source in ("db", "auto") else None
    added = enriched = 0
    try:
        for item in items:
            try:
                meta = frigate.event_metadata(
                    item["frigate_event_id"], db_conn=db_conn, api_base=s.api,
                    source=s.metadata_source)
                row = _merge_clip_with_meta(item, meta)
                camera = row.get("camera") or "?"
                place = places.resolve(camera, row.get("zones"))
                zone_list = store.json_col(row.get("zones"), [])
                row["place"] = place
                if meta:
                    row["description_meta"] = describe.describe_event(
                        row, place=place, tz_offset_h=s.narrative_tz_offset_h)
                    importance, tier = describe.score_importance(
                        row.get("label"), row.get("score"),
                        place_mapped=bool(place) and place != camera)
                    row["importance"] = importance
                    row["tier"] = tier
                if dry:
                    continue
                _row_id, inserted = _safe_upsert(conn, row)
                added += 1 if inserted else 0
                enriched += 1 if meta else 0
            except Exception as exc:  # noqa: BLE001 - per-event isolation
                log("scan: event {} failed: {}".format(item.get("frigate_event_id"), exc))
        if not dry:
            conn.commit()
    finally:
        if db_conn is not None:
            try:
                db_conn.close()
            except Exception:  # noqa: BLE001
                pass
    return {"seen": len(items), "new": added, "enriched": enriched}


def _safe_upsert(conn, row):
    return store.upsert_event(conn, row)


def re_enrich(s, conn, places, log=LOG):
    """Retry rows still missing metadata (provisional rows from an earlier pass)."""
    if s.re_enrich_per_run <= 0:
        return 0
    rows = conn.execute(
        "SELECT frigate_event_id, camera FROM events WHERE meta_json IS NULL"
        " ORDER BY start_time DESC LIMIT ?", (s.re_enrich_per_run,)).fetchall()
    if not rows:
        return 0
    db_conn = frigate.open_frigate_db(s.frigate_db) if s.metadata_source in ("db", "auto") else None
    fixed = 0
    try:
        for row in rows:
            try:
                meta = frigate.event_metadata(
                    row["frigate_event_id"], db_conn=db_conn, api_base=s.api,
                    source=s.metadata_source)
                if not meta:
                    continue
                meta = dict(meta)
                camera = meta.get("camera") or row["camera"]
                place = places.resolve(camera, meta.get("zones"))
                meta["place"] = place
                meta["description_meta"] = describe.describe_event(
                    meta, place=place, tz_offset_h=s.narrative_tz_offset_h)
                importance, tier = describe.score_importance(
                    meta.get("label"), meta.get("score"),
                    place_mapped=bool(place) and place != camera)
                meta["importance"] = importance
                meta["tier"] = tier
                # keep the paths already stored for this event
                meta.pop("frame_path", None)
                meta.pop("frame_clean_path", None)
                store.upsert_event(conn, meta)
                fixed += 1
            except Exception as exc:  # noqa: BLE001
                log("re-enrich: {} failed: {}".format(row["frigate_event_id"], exc))
        if fixed:
            conn.commit()
    finally:
        if db_conn is not None:
            try:
                db_conn.close()
            except Exception:  # noqa: BLE001
                pass
    return fixed


# ---------------------------------------------------------------------------
# L4 drain: caption capturable events within an idle window
# ---------------------------------------------------------------------------
def _frame_for(conn_row):
    """Prefer the un-annotated frame; fall back to the annotated one."""
    clean = conn_row["frame_clean_path"]
    if clean and os.path.isfile(clean):
        return clean
    path = conn_row["frame_path"]
    if path and os.path.isfile(path):
        return path
    return None


def drain(s, conn, cap, dry=False, force=False, log=LOG):
    """Caption up to MAX_EVENTS_PER_RUN pending events within MAX_RUN_SECONDS."""
    if cap is None:
        return {"captioner": 0, "missing": 0, "skipped": "no-model"}
    ok, why = governor_state(s)
    if not ok and not (force and s.force_allow_hot):
        log("drain skipped: {}".format(why))
        return {"captioner": 0, "missing": 0, "skipped": "idle-gate"}
    rows = store.pending_events(conn, limit=s.max_events_per_run,
                                only_person=s.only_person_captions)
    if not rows:
        return {"captioner": 0, "missing": 0, "skipped": "nothing-pending"}
    started = time.monotonic()
    done = missing = failed = 0
    for row in rows:
        if _STOP[0] or (time.monotonic() - started) > s.max_run_seconds:
            break
        image = _frame_for(row)
        if image is None:
            if not dry:
                store.mark_vlm(conn, row["id"], "missing")
            missing += 1
            continue
        if dry:
            log("dry-run would caption {} ({})".format(
                row["frigate_event_id"], image))
            done += 1
            continue
        try:
            prompt = s.vlm_prompt
            if row["description_meta"]:
                prompt = prompt + " Facts: " + row["description_meta"]
            text, ms = cap.caption(image, prompt)
            store.mark_vlm(conn, row["id"], "ok" if text else "empty",
                           text=text or None, model=cap.name, latency_ms=ms)
            done += 1 if text else 0
            log("caption {} ({} ms): {}".format(row["frigate_event_id"], ms, text))
        except Exception as exc:  # noqa: BLE001 - one bad frame must not stop a batch
            store.mark_vlm(conn, row["id"], "error")
            failed += 1
            log("caption {} FAILED: {}".format(row["frigate_event_id"], exc))
    if not dry:
        conn.commit()
    return {"captioner": done, "missing": missing, "failed": failed,
            "gate": why, "elapsed_s": round(time.monotonic() - started, 1)}


def rebuild_episodes(s, conn, places, dry=False, log=LOG):
    """Re-derive episodes + narratives from `events` (honours names.json)."""
    if not s.store_enabled or dry:
        return 0
    if os.path.isfile(s.names_file):
        try:
            store.merge_names_file(conn, s.names_file)
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            log("names import failed: {}".format(exc))
    count = episodes.rebuild(conn, places, store, labels=s.episode_labels,
                             reid_max_gap_s=s.reid_max_gap_s,
                             episode_gap_s=s.episode_gap_s, visit_min_s=s.visit_min_s,
                             tz_offset_h=s.narrative_tz_offset_h)
    log("episodes rebuilt: {}".format(count))
    return count


def prune(s, conn, log=LOG):
    if s.events_retention_days <= 0:
        return 0
    removed = store.prune_events(conn, s.events_retention_days)
    if removed:
        conn.commit()
        log("pruned {} event(s) older than {:g} days".format(
            removed, s.events_retention_days))
    return removed


# ---------------------------------------------------------------------------
# run modes
# ---------------------------------------------------------------------------
def _open(s):
    conn = store.open_writer(db_path(s))
    places = episodes.Places.load(PLACES_PATH)
    return conn, places


def runtime_problems(cap):
    """Missing shared libraries of the captioner's binaries, in one pass.

    Running the binary only ever reveals the FIRST missing library, which turns
    a dependency fix into several round trips (exactly what happened with
    libllama.so then libgomp.so.1). `ldd` lists them all at once.
    """
    problems = []
    for label, path in (("server", getattr(cap, "_server_bin", None)),
                        ("cli", getattr(cap, "_cli_bin", None))):
        if not path or not os.path.isfile(path):
            problems.append("{}: {} not found".format(label, path or "(unset)"))
            continue
        try:
            out = subprocess.run(["ldd", path], capture_output=True, text=True,
                                 timeout=20)
        except (OSError, subprocess.SubprocessError) as exc:
            # ldd may be absent in a slim image - not fatal, just unreportable
            problems.append("{}: could not run ldd ({})".format(label, exc))
            continue
        missing = [ln.strip() for ln in (out.stdout or "").splitlines()
                   if "not found" in ln]
        if missing:
            problems.append("{} {}: {}".format(
                label, os.path.basename(path), "; ".join(missing)))
    return problems


def _consume_trigger(s):
    """True when the portal asked for a batch now (clears the flag file)."""
    if os.path.isfile(s.trigger_file):
        try:
            os.remove(s.trigger_file)
        except OSError:
            pass
        return True
    return False


def run_forever(s, conn, places, cap):
    LOG("started: backend {} model {} | scan {:g}s drain {:g}s episodes {:g}s | "
        "gate loadavg<={:g} temp<={:g}C | persist {} | retention {:g}d".format(
            cap.backend if cap is not None else s.model_backend,
            cap.name if cap is not None else "(not loaded yet)",
            s.scan_every, s.drain_every, s.episodes_every,
            s.max_loadavg, s.max_cpu_temp_c,
            "resident" if s.model_keep_loaded else "load-per-batch",
            s.events_retention_days))
    if places.unmapped(s.cameras):
        LOG("NOTE: cameras with no place mapping in {}: {}".format(
            PLACES_PATH, ",".join(places.unmapped(s.cameras))))
    signal.signal(signal.SIGTERM, _sig_stop)
    signal.signal(signal.SIGINT, _sig_stop)
    next_scan = next_drain = next_episodes = time.monotonic()
    last_hb = time.monotonic()
    last_prune = 0.0
    while not _STOP[0]:
        if not s.enabled:
            LOG("disabled (ENABLED=false) - sleeping 60s")
            _sleep(60.0)
            continue
        now = time.monotonic()
        try:
            if now >= next_scan:
                res = harvest(s, conn, places)
                LOG("scan: {} seen, {} new, {} enriched".format(
                    res["seen"], res["new"], res["enriched"]))
                fixed = re_enrich(s, conn, places)
                if fixed:
                    LOG("re-enriched {} event(s)".format(fixed))
                if res["new"]:
                    next_episodes = min(next_episodes, time.monotonic())
                next_scan = now + s.scan_every
            if now >= next_episodes:
                rebuild_episodes(s, conn, places)
                next_episodes = time.monotonic() + s.episodes_every
            triggered = _consume_trigger(s)
            # The captioner may be missing (the model files are a deploy
            # prerequisite, NOT shipped by git). Scanning, descriptions and
            # episodes keep running regardless; the model is picked up as soon as
            # it appears, so a fresh deploy never crash-loops on a missing model.
            if cap is None and (triggered or now >= next_drain):
                try:
                    cap = captioners.make_captioner(s)
                    LOG("captioner loaded: {} {}".format(cap.backend, cap.name))
                except Exception as exc:  # noqa: BLE001 - keep narrating
                    LOG("captioner still unavailable: {}".format(exc))
            if triggered or now >= next_drain:
                res = drain(s, conn, cap, force=triggered)
                if res.get("captioner") or res.get("missing") or triggered:
                    LOG("drain: {}".format(res))
                next_drain = time.monotonic() + s.drain_every
        except Exception as exc:  # noqa: BLE001 - a pass must never kill the loop
            LOG("pass error: {}: {}".format(type(exc).__name__, exc))
        now = time.monotonic()
        if now - last_prune >= 3600.0:
            last_prune = now
            try:
                prune(s, conn)
            except Exception as exc:  # noqa: BLE001
                LOG("prune error: {}".format(exc))
        if now - last_hb >= s.heartbeat_s:
            last_hb = now
            payload = write_status(s, conn, cap)
            LOG("heartbeat: {} events, {} pending, {} episodes, {}".format(
                payload.get("events_total"), payload.get("pending"),
                payload.get("episodes_total"), governor_state(s)[1]))
        _sleep(max(s.tick_min, min(1.0, next_scan - time.monotonic())))


def _sleep(secs):
    end = time.monotonic() + max(0.0, secs)
    while not _STOP[0]:
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(0.5, left))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = set(sys.argv[1:])
    raw = _raw_conf(CONF_PATH)
    if not raw:
        LOG("ERROR: no config at {} - aborting".format(CONF_PATH))
        return 2
    try:
        socket.setdefaulttimeout(30)
    except (OSError, ValueError):
        pass

    s = resolve_settings(raw)
    dry = "--dry-run" in args

    if "--status" in args:
        print(json.dumps(read_status(s.status_file), indent=2, sort_keys=True))
        return 0

    try:
        conn, places = _open(s)
    except Exception as exc:  # noqa: BLE001
        LOG("ERROR: cannot open the store at {}: {}".format(db_path(s), exc))
        return 2

    # An explicit model-requiring FLAG must fail loudly, but the long-running
    # service must NOT: a missing model only disables captions, not the narrator.
    strict_model = bool({"--check", "--drain", "--once", "--dry-run"} & args)
    attempt_model = strict_model or not args
    cap = None
    if attempt_model:
        try:
            cap = captioners.make_captioner(s)
            LOG("captioner: backend {} model {} ({} threads)".format(
                cap.backend, cap.name, s.vlm_n_threads))
        except Exception as exc:  # noqa: BLE001
            if strict_model:
                LOG("ERROR: {}".format(exc))
                return 2
            LOG("WARNING: captioner unavailable: {}".format(exc))
            LOG("WARNING: scanning, descriptions and episodes WILL run; captions "
                "start once the model is present (retried automatically).")
            cap = None

    try:
        if "--check" in args:
            ok, why = governor_state(s)
            LOG("idle gate: {} ({})".format("OPEN" if ok else "CLOSED", why))
            LOG("store: {} | clips: {} | places: {}".format(
                db_path(s), clips_dir(s), PLACES_PATH))
            unmapped = places.unmapped(s.cameras)
            if unmapped:
                LOG("WARN unmapped cameras: {}".format(",".join(unmapped)))
            # The model files and the RUNTIME are separate prerequisites: a GGUF
            # alone cannot caption without the llama.cpp binary, and that would
            # otherwise show up only as empty captions much later. Say it here.
            if cap is not None and getattr(cap, "backend", "") == "llamacpp" \
                    and not cap.alive():
                LOG("WARNING: no llama.cpp runtime found (looked for {}). Captions "
                    "will be EMPTY until it is fetched: `bash "
                    "dev_scripts/prep_scene_model_llamacpp.sh --bin "
                    "--llamacpp-tag b10900`".format(s.llama_server_bin))
            probe = None
            rows = conn.execute(
                "SELECT frame_clean_path, frame_path FROM events"
                " WHERE frame_path IS NOT NULL ORDER BY start_time DESC LIMIT 5").fetchall()
            for row in rows:
                candidate = row["frame_clean_path"] or row["frame_path"]
                if candidate and os.path.isfile(candidate):
                    probe = candidate
                    break
            if cap is None:
                LOG("ERROR: no captioner was built")
                return 2
            if getattr(cap, "backend", "") == "llamacpp":
                for line in runtime_problems(cap):
                    LOG("PREFLIGHT -> {}".format(line))
                server = getattr(cap, "_proc", None)
                server_state = "ready" if (server is not None and server.poll() is None) \
                    else ("failed: " + (getattr(cap, "last_error", "") or "not started")
                          if getattr(cap, "_server_bin", None) else "not found")
                LOG("runtime: server {} | cli {}".format(
                    server_state, getattr(cap, "_cli_bin", None) or "not found"))
            if probe:
                text, ms = cap.caption(probe)
                LOG("probe caption ({} ms): {}".format(ms, text or "<empty>"))
                if not text:
                    LOG("WARNING: the captioner returned NO text - treat captions "
                        "as unproven until this probe prints a sentence.")
                    reason = getattr(cap, "last_error", "")
                    if reason:
                        LOG("  reason: {}".format(
                            " | ".join(reason.splitlines())[-400:]))
                    raw = getattr(cap, "last_raw", "")
                    if raw:
                        LOG("  path: {} | raw output: {}".format(
                            getattr(cap, "last_path", "?"),
                            " | ".join(raw.splitlines())[-400:]))
            else:
                LOG("no stored frame on disk yet - skipped the caption probe")
            return 0

        if "--rebuild-episodes" in args:
            rebuild_episodes(s, conn, places, dry=dry)
            write_status(s, conn, cap if s.model_keep_loaded else None)
            return 0

        if "--scan-only" in args:
            LOG("scan: {}".format(harvest(s, conn, places, dry=dry)))
            rebuild_episodes(s, conn, places, dry=dry)
            write_status(s, conn, cap if cap else None)
            return 0

        if "--drain" in args:
            LOG("drain: {}".format(drain(s, conn, cap, dry=dry, force=True)))
            if not dry:
                rebuild_episodes(s, conn, places)
            write_status(s, conn, cap)
            return 0

        if "--once" in args or dry:
            LOG("scan: {}".format(harvest(s, conn, places, dry=dry)))
            rebuild_episodes(s, conn, places, dry=dry)
            LOG("drain: {}".format(drain(s, conn, cap, dry=dry, force=True)))
            if not dry:
                rebuild_episodes(s, conn, places)
            write_status(s, conn, cap)
            return 0

        run_forever(s, conn, places, cap)
        return 0
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        if cap is not None:
            cap.close()


if __name__ == "__main__":
    sys.exit(main())
