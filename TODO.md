# TODO

Loose follow-ups that are not yet a full plan.

> Plans live in `plans/` (git-ignored / local-only, per [`.gitignore`](.gitignore)).
> This file IS tracked, so it is the place for short, shared follow-ups.

## Follow-ups

- **firewatch false positives — cam01 dogs scored as FIRE.** The last 4 firewatch alerts
  (2026-09-10 ids 98/105/109/114, all cam01) are dogs: v4 scores raw fire 0.54–0.73 and
  the +0.15 `MOTION_BONUS` then confirms via `MIN_HITS=3/4` (verified with a COCO `dog`
  cross-check). Fix options — data fix (dog negatives → v5 fine-tune, recommended),
  animal-overlap guard, per-cam01 stop-gap — evidence + method in
  [`plans/firewatch-dog-false-positives.md`](plans/firewatch-dog-false-positives.md) and
  re-scorer [`dev_scripts/analyze_fire_alert_frames.py`](dev_scripts/analyze_fire_alert_frames.py).
  **No production change made — awaiting explicit permission** before implementing.
- we swap the model to **YOLO11s**, make sure to check the machine monitor after time.

<!--
Context for the line above:
Frigate's COCO detector is now models/coco/yolo11s.onnx running on the Intel
UHD 630 iGPU (commit "feat(frigate): upgrade COCO detector to YOLO11s on the
iGPU"; see plans/better-coco-model-on-igpu.md).

What to check, after a few days of running:
  * media/watchdog/watchdog_baseline_<tag>.csv -> detector inference_ms,
    detection_fps vs process_fps, plus the new gpu_usage_pct column.
  * machine-monitor Telegram alerts / stdout -> live iGPU usage (and GPU temp
    where the host exposes one).
Accept: sum(detection_fps) still tracks sum(process_fps) on all online cameras
and inference_speed stays under the ~100 ms frame budget. If it does not, revert
the model paths in config/config.yaml to /models/coco/yolo11n.onnx.
-->

- **cam01 signal-surface exploration — revisit & harvest.** Two completed recon docs hold
  capabilities we have not used: [`plans/cam01-websocket-exploration.md`](plans/cam01-websocket-exploration.md)
  (camera WS `WsSubscription` status/snapshot push, the `UpServer` dock WS client, and
  ONVIF **PullPoint** as a standardised event fallback) and
  [`plans/snmp-camera-recon.md`](plans/snmp-camera-recon.md) (SNMPv3 health/inventory —
  "up but silent" + reboot detection; traps proven unusable on this firmware).
  Re-read both and pick what actually pays off. Candidate first steps: **(a)** consume the
  already-working HTTP `Event/Subscription` push with
  [`scripts/cam_event_listener.py`](scripts/cam_event_listener.py) and bridge it to MQTT
  `frigate/cameras/cam01/events` for the portal; **(b)** ONVIF `PullMessages` health/event
  poll (outbound-only, no inbound host port); **(c)** camera-health poller
  (`sysUpTime` + eth0 egress rate) beside `machine-monitor.py`. Each needs its own plan and,
  where it writes camera config, explicit permission. **Do the housekeeping first:** cam01
  still has the temporary **SNMP / WebSockets / UpServer** test settings enabled.
  - *Listener output reviewed (2026-09-12, live ~10 h run; §14 of the cam01 plan):*
    the push is **richer than "motion only"** — besides an image-free motion `On/Off`
    stream (~226/h), every **person line crossing** arrives as a 396 B
    `LineDetectorCrossed` alarm **plus** a `System/Event/Notification/Structure`
    body carrying a **1920×1080 JPEG**, the person box (`0..10000` coords), the line
    rule geometry and the device id (47 crossings that day, 49 person entries;
    2 re-sent alarms). That is the payload **(a)** would bridge; the WS mirror is
    status-only. Listener still running (pid `1520455`, port 50235) and still temporary.

## scenereader — cross-camera episode narrator (in progress)

Replaces `scenewatch`, which was stopped for over-subscribing the CPU, filling RAM,
raising host temperature and producing low-accuracy captions. Full design and
rationale: `plans/event-scene-reader.md` (git-ignored working doc).

**Mission.** Turn Frigate's captures into **episodes plus a narrative**, e.g.
*"Person A entered Field 1, stayed ~12 min, then went to the Store with Person B."*
Anonymous IDs first; real names later via assisted labeling, then an optional face
gallery.

**Decided.**
- Event-driven from Frigate only — no `latest.jpg` sweep, no in-house motion.
- Frames are read **in place** from Frigate's `media/clips/` — never copied; we add
text tables only (`events`, `episodes`, `episode_events`, `person_aliases`).
- Metadata-first descriptions (deterministic, zero hallucination); the VLM only
enriches.
- The small model (SmolVLM2-500M GGUF via llama.cpp) is kept **resident**
(`MODEL_KEEP_LOADED=true`); load/unload is **deferred to an optimization**.
- The existing 2B OpenVINO export is **kept and never re-downloaded**; switching is
a config flip (`MODEL_BACKEND=llamacpp|openvino`). Model prep is add-only.
- Idle-gated drain (loadavg + CPU temp), plus a portal **Process now** button and a
CLI `--drain`.

**Host-verified 2026-09-10** (read-only as `ai`): event snapshots live in
`media/clips/` (there is **no** `media/snapshots/`), named
`<camera>-<event_id>.jpg` with an un-annotated `-clean.webp` sibling; `clips/`
subdirs `previews,thumbs,export,review,cache` must be skipped; `config/frigate.db`
`event` table opens **`mode=ro`** while Frigate runs and the filename's id joins it
**exactly**; `score`/`top_score`/`box` columns are NULL — the values live in the
`data` JSON; corpus 4 001 events (person 3 709, cow 149, motorcycle 129, truck 10,
car 2, dog 2); `zones = []` (none defined yet).

### Done
- [x] Host reconnaissance (layout, filename convention, DB read-only, exact-id join).
- [x] `scenereader/store.py` — WAL store: events (UNIQUE `frigate_event_id` upsert),
   episodes + `episode_events` visits, `person_aliases`, guarded migrations,
   rebuild-safe `clear_episodes()`, retention prune. *(commit 30a4fc3)*
- [x] `scenereader/frigate.py` — `scan_clips()` (top level only, exact event-id
   parse, `-clean.webp` pairing, cursor) + `open_frigate_db()` (mode=ro) +
   `lookup_event()` (score/box from `data` JSON) + `/api/events/<id>` fallback.
   *(commit 3b0d4e8)*

### Next — Phase 1
- [ ] L0 metadata description builder + importance/tier scoring.
- [ ] Captioner backend interface: `llamacpp` (default) + retained `openvino-genai`.
- [ ] Idle governor (loadavg + CPU temp) and the scan/drain/reconcile scheduler.
- [ ] CLI flags (`--check`, `--once`, `--scan-only`, `--drain`,
   `--rebuild-episodes`, `--dry-run`, `--status`) + trigger/status/names files.
- [ ] L1 place naming — scaffold `config/places.conf` + resolver (zone beats camera).
- [ ] L2 anonymous entity resolution — cross-camera linking by gap + adjacency with a
   stored confidence.
- [ ] L3 episode builder — visits, durations, episode-gap close, co-presence.
- [ ] L4 narrative composer — deterministic template (optional tiny text LLM later).
- [ ] `docker-compose.yml` — add `scenereader`, remove `scenewatch` (`./media` for
   in-place frames + rw text store, `./config` ro for `frigate.db`; `cpus 1.0`,
   `mem_limit 1.5g`, `oom_score_adj 200`).
- [ ] `config/scenereader.conf` (`FRIGATE_METADATA_SOURCE=db`,
   `FRIGATE_SNAPSHOT_DIRS=clips`, `MODEL_BACKEND`, `MODEL_KEEP_LOADED=true`,
   `IDLE_UNLOAD_S=0`) + scaffolded `config/places.conf`.
- [ ] `scenereader/Dockerfile` + `requirements.txt` (prebuilt llama.cpp binary plus
   the openvino-genai runtime retained for the 2B IR; no torch).
- [ ] `dev_scripts/prep_scene_model_llamacpp.sh` — add-only GGUF + mmproj fetcher
   into its own subdir under `models/scene/`.
- [ ] Update `models/scene/README.md` + `VERSIONS.md` (both models documented; the
   2B IR stays on disk untouched).
- [ ] Remove/deprecate scenewatch files (`scenewatch/`, `config/scenewatch.conf`,
   `config/stores/scenewatch.conf`) and update `.gitignore`.
- [ ] Portal: `/api/episodes*`, `/api/scenelog*`, status, drain; Episodes timeline +
   Scene log + **name this person**; reuse the existing
   `/api/events/{id}/snapshot.jpg`; bump `APP_VERSION`.
- [ ] Update `SYSTEM_SUMMARY.md` (services/map/ports/stores/config/RAM) +
   `README.md`; bump the summary version.
- [ ] Local verification (unit tests + a synthetic end-to-end episode assert).

### Phase 2
- [ ] Person attributes (CLIP zero-shot on the `data.box` crop → `description_attr`;
   daylight-only `ATTR_NIGHT_LUMA` gate) + `ATTR_*` keys.
- [ ] Use attribute agreement as an L2 link-confidence bonus; include attributes in
   the narrative.
- [ ] Appearance ReID as a cross-camera link tie-breaker.
- [ ] Tiny text LLM narrative polish.
- [ ] Compare the small GGUF vs the retained 2B OpenVINO on real captures; pick via
   `MODEL_BACKEND`.
- [ ] Farm dataset + LoRA fine-tune of SmolVLM2-500M, GGUF export, eval harness and
   promote script.

### Deferred
- [ ] Residency optimization (`MODEL_KEEP_LOADED=false` + `IDLE_UNLOAD_S`) only if
   RAM becomes contended, e.g. with the larger retained model.
- [ ] Face recognition names (after assisted labeling), action recognition, gait.

### Needs from the operator
- [ ] Fill `config/places.conf`: `CAMERA_PLACES` (cam01…cam10 → human place names),
   optional `ZONE_PLACES`, and `ADJACENCY` (the plausible walking routes).
- [ ] Decide whether to add Frigate **zones** (config/config.yaml) for finer places.
