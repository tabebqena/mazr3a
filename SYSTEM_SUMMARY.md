# System Summary — mazr3a Edge Security System

> **Purpose.** A single, maintained inventory of the whole system: every service
> in the stack, where its development files live, the host crontab, the required
> host settings, storage/cleanup wiring, networking, secrets and the deploy
> workflow. [`README.md`](README.md) is the *how-to* / narrative; this file is
> the *map*.
>
> **MAINTENANCE IS MANDATORY — see [§11](#11-maintaining-this-file).** Whenever a
> service, config, cron entry, port, mount or host requirement changes, update
> this file in the same change/commit. Enforced by
> [`.roo/rules/system-summary.md`](.roo/rules/system-summary.md).

| Field | Value |
|---|---|
| Summary version | `v16` |
| Last updated | 2026-09-11 |
| Repo | `https://github.com/tabebqena/mazr3a` (branch `master`) |
| Portal `APP_VERSION` | `0.4.0` (see [`portal/app.py`](portal/app.py:40)) — bump on every portal change |

---

## 1. Overview

A farm CCTV / security stack built on **Frigate NVR** as the ingest + detection
core, with purpose-built side services around it:

- **Frigate** decodes 10 IP cameras' low-res detect substreams, runs an OpenVINO
  detector, records **event-only** clips, exposes REST/UI and an embedded
  **go2rtc** live restream, and publishes MQTT events.
- **firewatch** runs an out-of-band fire/smoke model on Frigate's already-decoded
  detect frames and sends Telegram photo alerts + stores evidence.
- **scenereader** (replaced `scenewatch`) does NOT re-detect anything. It reads
  the capture events Frigate already produced — snapshots read **in place** from
  Frigate's media tree, joined by **exact event id** to the `event` table in
  `config/frigate.db` (read-only) — writes a deterministic per-camera sentence,
  links cameras into cross-camera **person episodes** using a place map +
  adjacency, and optionally captions frames with a **small (≤500M) GGUF** model
  during an **idle-gated, bounded** batch.
- **telegram-bot** answers `/status`, `/help`, `/start` live from Telegram.
- **logs** is a read-only Docker-logs sidecar for the portal's admin Debug tab.
- **portal** is an authenticated FastAPI + vanilla-JS SPA (login, live view,
  events, fire alerts, scene descriptions) published via a Cloudflare Tunnel.
- **mqtt** (Mosquitto) is the event broker.
- Host **cron** runs the unified disk heartbeat, a CPU-temp + iGPU watchdog, a
  daily health report and a state sampler.

Roadmap context: this is **Phase 0/1a plus the event-driven episode narrator**.
Cross-camera episode narratives now ship via the `scenereader` service (§3.7);
the **portal Episodes view** and a cron/Telegram digest are still **deferred**,
as are person attributes, face names and gait (see
[§12](#12-deferred--future-phases)).

---

## 2. Host / environment

| Item | Value |
|---|---|
| Host OS | Debian Linux (SSH) |
| Deploy (owner) user | `dr` — owns `/home/dr/frigate` (the git clone) |
| AI edit user | `ai` (see [`.roo/rules/sshuser.md`](.roo/rules/sshuser.md)) |
| SSH endpoint | `ssh.mazr3a.garden` (interactive/`ai` password; `dr` for deploy) |
| Deploy dir | `/home/dr/frigate` — a git clone of `origin/master` |
| CPU | Intel Core i7-9700 (8 cores) |
| RAM | ~7.5 GiB |
| GPU / iGPU | Intel UHD 630 (`/dev/dri/renderD128`) — OpenVINO detector device: `GPU` |
| Cameras | 10 × `192.168.1.200`–`192.168.1.209` (9 UNV/VCP + 1 Hikvision `cam08`) |
| Public access | Cloudflare Tunnel + Access on `live.mazr3a.garden` → host `:8080` (portal) |

---

## 3. Services (Docker Compose)

Defined in [`docker-compose.yml`](docker-compose.yml). Start/stop the whole stack
with `docker compose up -d` / `docker compose down`.

### 3.1 `frigate` — NVR + detector + go2rtc + MQTT publisher
| | |
|---|---|
| Container | `frigate` |
| Image | `ghcr.io/blakeblackshear/frigate:stable` |
| Purpose | Camera ingest, OpenVINO detection, event-only recording, snapshots, live restream (go2rtc), MQTT event publish |
| Dev files | [`config/config.yaml`](config/config.yaml) (Frigate 0.17 config), [`models/coco/`](models/coco/) (ONNX + labelmap), [`docker-compose.yml`](docker-compose.yml) |
| Actual active model | `models/coco/yolo11s.onnx` (COCO 80-class, `yolo-generic`, 640×640, iGPU); `yolo11n.onnx` is the rollback |
| Credentials | `.env` → `{FRIGATE_*}` placeholders (git-ignored) |
| Ports | `5000` HTTP UI/API · `8971` TLS UI · `8554` RTSP restream · `8555` tcp+udp WebRTC |
| Volumes | `./config:/config` · `./media:/media/frigate` · `./models:/models:ro` · tmpfs `/tmp/cache` (1 GB) |
| Device / limits | `/dev/dri/renderD128`; `shm_size: 256mb` |
| Zones | **11 zone polygons** defined per camera (cam01 diwan1/diwan2/estraha_door/estraha_front/estraha_road; cam02 solar_panels; cam03 zone_a3829a54/hoash_door_1/estraha_road_2; cam06 housh_2_door; cam09 field_west). Drawn in the Frigate UI. `scenereader` consumes them through `ZONE_PLACES` in [`config/places.conf`](config/places.conf) so a sentence names the sub-place, and each event carries them in its `zones` array. |
| Notes | Event-only recording (detect substream); software decode (`ffmpeg.hwaccel_args: []`). **`config/config.yaml` is the CANONICAL copy of the host file** — the Frigate UI rewrites it via `ruamel.yaml` (comments are preserved, values may be re-wrapped), so after ANY UI edit it must be re-adopted verbatim into the repo and committed, or the next `git pull --ff-only` refuses to run ("local changes would be overwritten"). |

### 3.2 `mqtt` — Mosquitto broker
| | |
|---|---|
| Container | `mqtt` |
| Image | `eclipse-mosquitto:2` |
| Purpose | MQTT broker; Frigate publishes events under `frigate/#` |
| Dev files | [`mosquitto/config/mosquitto.conf`](mosquitto/config/mosquitto.conf) |
| Ports | `1883` |
| Volumes | `./mosquitto/config` · `./mosquitto/data` · `./mosquitto/log` (git-ignored) |
| Notes | LAN-only, `allow_anonymous true` (Phase 0) |

### 3.3 `firewatch` — fire/smoke watchdog
| | |
|---|---|
| Container | `firewatch` |
| Image | **built** from [`firewatch/Dockerfile`](firewatch/Dockerfile) (`python:3.11-slim` + `openvino`, `numpy`, `Pillow`); unprivileged uid 1000 |
| Purpose | Polls `/api/<cam>/latest.jpg`, motion-gated fire/smoke inference, Telegram photo alerts, SQLite evidence store |
| Dev files | [`firewatch/firewatch.py`](firewatch/firewatch.py) (watcher), [`firewatch/requirements.txt`](firewatch/requirements.txt), shared [`scripts/telegram_notify.py`](scripts/telegram_notify.py), model [`models/fire/`](models/fire/) |
| Config | [`config/firewatch.conf`](config/firewatch.conf) (`/config/firewatch.conf`), [`config/telegram.conf`](config/telegram.conf) (git-ignored) |
| Evidence store | `./media` (`<STORE_DIR>/<cam>/*.jpg` + `<STORE_DIR>/firewatch.db`) |
| Volumes (ro) | `./firewatch:/firewatch` · `./scripts:/scripts` · `./config:/config` · `./models:/models` · `./media:/media/firewatch` (rw) |
| Ports | none (outbound only) |
| Notes | Code/config edits need only `docker compose restart firewatch`. Active model = fire v4 (YOLO26-S). |

### 3.4 `telegram-bot` — on-demand command responder
| | |
|---|---|
| Container | `telegram-bot` |
| Image | `python:3.11-slim` (no build, stdlib only) |
| Purpose | Long-polls Bot API; answers `/status`, `/help`, `/start` in the asking chat |
| Dev files | [`scripts/telegram_bot.py`](scripts/telegram_bot.py), shared [`scripts/telegram_notify.py`](scripts/telegram_notify.py) |
| Config | [`config/telegram.conf`](config/telegram.conf) (git-ignored) |
| Volumes (ro) | `./scripts:/scripts` · `./config:/config` · `/proc:/host/proc` · `./media:/media` |
| Ports | none |
| Notes | `getUpdates` long-poll must be the ONLY consumer (no webhook/manual curl). Edits need `docker compose restart telegram-bot`. |

### 3.5 `logs` — read-only Docker logs sidecar
| | |
|---|---|
| Container | `logs` |
| Image | `python:3.11-slim` (no build, stdlib only) |
| Purpose | Idle HTTP server exposing container list + log tails to the portal Debug tab |
| Dev files | [`scripts/container_logs.py`](scripts/container_logs.py) |
| Volumes | `./scripts:/scripts:ro` · `/var/run/docker.sock:/var/run/docker.sock` |
| Ports | internal `8090` only (not published) |
| Notes | Docker READ calls only; runs as root to open the root-owned socket |

### 3.6 `portal` — authenticated web portal
| | |
|---|---|
| Container | `portal` |
| Image | **built** from [`portal/Dockerfile`](portal/Dockerfile) (`python:3.11-slim` + `fastapi`, `uvicorn`, `httpx`, `websockets`); uid 1000 |
| Purpose | Login (PBKDF2), live view (MSE-over-WebSocket primary, HLS fallback), events/detections, fire alerts (cards/lightbox also show portal-derived **motion** + burst **hits** badges inferred in [`portal/firestore.py`](portal/firestore.py)), **scenereader Episodes** (the cross-camera person stories: each card shows the narrative **in English AND Arabic**, the visits as thumbnails, and the link-confidence; plus a **Scenes** tab of the adaptive L1 scenes — the coexisting objects and which of them actually MOVED — and a **Scene log** of every capture with its tier — all read-only via [`portal/eventstore.py`](portal/eventstore.py), with a **Process now** button that asks the service for a caption batch), admin Debug tab |
| Dev files | [`portal/app.py`](portal/app.py) (`APP_VERSION` here), [`portal/auth.py`](portal/auth.py), [`portal/config.py`](portal/config.py), [`portal/frigate.py`](portal/frigate.py), [`portal/firestore.py`](portal/firestore.py), [`portal/eventstore.py`](portal/eventstore.py), [`portal/genpass.py`](portal/genpass.py), [`portal/static/`](portal/static/) (SPA: `index.html`, `app.js`, `style.css`, `favicon.svg`, `vendor/hls.min.js`), [`portal/requirements.txt`](portal/requirements.txt) |
| Config | [`config/portal.conf`](config/portal.conf) (git-ignored; template [`config/portal.conf.example`](config/portal.conf.example)) |
| Ports | `8080` (internal host port for the Cloudflare Tunnel) |
| Volumes | `./portal:/srv/app/portal:ro` · `./config:/config:ro` · `./media:/media` (rw for SQLite WAL read) |
| Env | `PORTAL_CONF`, `PORTAL_LOGS_API=http://logs:8090` |
| Notes | **Cache-busting policy:** bump `APP_VERSION` + use `{{ ASSET_* }}` tokens (see [`.roo/rules/portal-cache-busting.md`](.roo/rules/portal-cache-busting.md)). Edits need `docker compose restart portal`. |

### 3.7 `scenereader` — cross-camera episode narrator

See [`plans/event-scene-reader.md`](plans/event-scene-reader.md). It **replaced
the stopped `scenewatch`** ([`plans/scene-description.md`](plans/scene-description.md)),
which over-subscribed the CPU with a 15 s per-camera sweep and held a 2B VLM
resident. The **adaptive scene layer** (movement ownership) is specified in
[`plans/adaptive-scene-narrative.md`](plans/adaptive-scene-narrative.md).

| | |
|---|---|
| Container | `scenereader` |
| Image | **built** from [`scenereader/Dockerfile`](scenereader/Dockerfile) (`python:3.11-slim` + `openvino-genai`, `numpy`, `Pillow`); unprivileged uid 1000 |
| Purpose | Read Frigate's own captures (never re-detecting), write a deterministic per-camera sentence, then link cameras into anonymous **person episodes** with a narrative, e.g. *"Person A entered Field 1 (cam04) at 22:36, stayed 12 min; then went to Store (cam07) at 22:49; left at 22:51."* **One STAY = one visit**: Frigate re-fires detection, so consecutive captures of the same place within `VISIT_MERGE_GAP_S` are merged into a single stay (span, not the sum), and the narrative only says *"returned to X"* for a real return after being elsewhere. **Adaptive scenes (L1) + movement ownership:** events are grouped into per-camera scenes bounded by `SCENE_GAP_S` of inactivity; each object's OWN trajectory (`data.path_data` → `events.motion_disp`) decides who MOVED, so a moving dog/cow/truck in a crowded frame is narrated as *"dog moved; person, cow present"* instead of the person *"entering / going to"* a place. Deterministic EN + AR, no model. |
| Dev files | [`scenereader/scenereader.py`](scenereader/scenereader.py) (service), [`frigate.py`](scenereader/frigate.py), [`store.py`](scenereader/store.py), [`episodes.py`](scenereader/episodes.py), [`scenes.py`](scenereader/scenes.py) (L1/L2), [`describe.py`](scenereader/describe.py), [`captioners.py`](scenereader/captioners.py), models [`models/scene/`](models/scene/) (git-ignored) |
| Config | [`config/scenereader.conf`](config/scenereader.conf) (`/config/scenereader.conf`) + [`config/places.conf`](config/places.conf) (`/config/places.conf`). Episode shape: `EPISODE_GAP_S`, `REID_MAX_GAP_S`, `VISIT_MIN_S`, `VISIT_MERGE_GAP_S`; scenes/movement: `SCENE_ENABLED`, `SCENE_GAP_S`, `SCENE_MAX_S`, `MOVE_MIN_DISP`, `SCENE_KEYFRAMES`; caption cost: `VLM_TIMEOUT_S`, `VLM_MAX_IMAGE_PX`, `MAX_EVENTS_PER_RUN`. |
| Frame source | **Read in place, NEVER copied**: host `./media` IS Frigate's `/media/frigate`; event snapshots are `clips/<camera>-<event_id>.jpg` with a `-clean.webp` sibling (the un-annotated frame is the one captioned). Verified 2026-09-10 — there is **no** `media/snapshots/`. |
| Metadata source | `FRIGATE_METADATA_SOURCE=db` (default): Frigate's `config/frigate.db`, **read-only**, single denormalised `event` table, joined by the **exact** event id in the filename. `api` / `auto` fall back to `/api/events/<id>`. `score`/`top_score`/`box` columns are NULL → the live values are parsed from the `data` JSON; the object's own movement is derived from `data.path_data` into `events.motion_disp` / `events.motion_pts`. |
| Model | **`MODEL_BACKEND` switch**: `llamacpp` (default) = a small **SmolVLM2-500M GGUF + mmproj** served by a **resident** `llama-server` (~0.5–0.7 GB); `openvino` = the **RETAINED** Qwen2-VL-2B INT4 IR that was already on the host (~2 GB). Switching is config-only — no re-download, no rebuild. |
| Model prerequisite | **NOT deployed by git.** Small GGUF: [`dev_scripts/prep_scene_model_llamacpp.sh`](dev_scripts/prep_scene_model_llamacpp.sh) (**add-only**; never touches the retained IR). Retained OpenVINO IR: [`dev_scripts/prep_scene_model.sh`](dev_scripts/prep_scene_model.sh) (kept so the 2B never needs re-downloading) — see [`models/scene/README.md`](models/scene/README.md) |
| Store | `./media/events/` — **text only**: `events.db` (WAL: events + episodes + visits + aliases + **scenes** + **scene_events**), `reader_status.json`, `.drain_request`, `names.json`, `.scenereader-{scheduler,cli}.lock` (flock single-instance guards; harmless when stale). **No image copies anywhere.** |
| Volumes | `./scenereader:/scenereader:ro` · `./config:/config:ro` · `./models:/models:ro` · `./media:/media` (rw) |
| Ports | none (outbound only) |
| Concurrency | **Short locks, one writer, by construction.** The CLI is a real `argparse` parser (`--help` works; an unknown flag exits 2) and a flock guard in the store dir refuses a SECOND `scheduler` (exit 3, with the holder's pid); two manual commands may not overlap either (a manual command may still run **beside** the daemon). The store's write lock is held for **milliseconds**: every ITEM is committed as it is written (`_commit`), a `release_txn()` net runs after each loop pass, and `prune()` commits unconditionally — a leaked transaction (or a batch that committed only at the end) was what made every other writer report `database is locked`. The one-shot CLI reports `exit 4` with a "store busy" message instead of a traceback. `--check`/`--status` open the store **read-only** and are safe next to the live service. |
| Notes | Code/config edits need only `docker compose restart scenereader`. No host cron entry — the in-container scheduler does everything. It scans `clips/` every `SCAN_EVERY_S` (cheap disk work) and rebuilds episodes every `EPISODES_EVERY_S`; both are **rebuildable** from `events` so a `places.conf`/gap change needs no re-capture. The **only** expensive step — the VLM caption batch — runs at most every `DRAIN_EVERY_S` and ONLY while the **idle governor** is open (host `loadavg1 ≤ MAX_LOADAVG` and CPU temp `< MAX_CPU_TEMP_C`, read from the container's native `/proc` and `/sys/class/hwmon`), bounded by `MAX_EVENTS_PER_RUN` + `MAX_RUN_SECONDS`. The model is **loaded once and kept** (`MODEL_KEEP_LOADED=true`); unload-after-idle is a **deferred optimization**. Every event is scored 0-100 into a `tier` (high/normal/low). |

### 3.8 Service → development-file map (quick lookup)
| Service | Primary code / config in repo |
|---|---|
| `frigate` | [`config/config.yaml`](config/config.yaml), [`models/coco/`](models/coco/), `docker-compose.yml`, `.env` |
| `mqtt` | [`mosquitto/config/mosquitto.conf`](mosquitto/config/mosquitto.conf) |
| `firewatch` | [`firewatch/firewatch.py`](firewatch/firewatch.py), [`config/firewatch.conf`](config/firewatch.conf), [`models/fire/`](models/fire/), [`scripts/telegram_notify.py`](scripts/telegram_notify.py) |
| `scenereader` | [`scenereader/`](scenereader/) (service + modules), [`config/scenereader.conf`](config/scenereader.conf), [`config/places.conf`](config/places.conf), [`models/scene/`](models/scene/) (git-ignored), [`dev_scripts/prep_scene_model_llamacpp.sh`](dev_scripts/prep_scene_model_llamacpp.sh) |
| `telegram-bot` | [`scripts/telegram_bot.py`](scripts/telegram_bot.py), [`scripts/telegram_notify.py`](scripts/telegram_notify.py) |
| `logs` | [`scripts/container_logs.py`](scripts/container_logs.py) |
| `portal` | [`portal/`](portal/) + [`config/portal.conf`](config/portal.conf) |

### 3.9 Resource limits (CPU / memory / OOM priority)

Set in [`docker-compose.yml`](docker-compose.yml). `cpus` is a hard CFS quota and
`mem_limit` a hard memory ceiling; **neither reserves anything** — a limit is only
a maximum, so a container using less is unaffected. Host: 8 cores, ~7.5 GiB.

| Service | `cpus` | `mem_limit` | `oom_score_adj` | Rationale |
|---|---|---|---|---|
| `frigate` | *(uncapped)* | 3 GiB | **-500** | Critical path — CFS throttling can make it drop decoded frames, so only its **memory** is ceilinged. Its cgroup also holds the 1 GB tmpfs `/tmp/cache` + 256 MB `/dev/shm`, hence the generous ceiling. |
| `scenereader` | 1 | 1.5 GiB | **200** | ~0.5–0.7 GB resident with the small GGUF default; raise towards 3 GiB **only** if `MODEL_BACKEND=openvino` (the retained 2B) is selected. The preferred OOM victim. |
| `firewatch` | 2 | 1 GiB | *(0)* | Small IR model at a low cadence (~0.35–0.5 GB). |
| `portal` | 1 | 512 MiB | *(0)* | HTTP + proxying; HLS is streamed, not transcoded. |
| `mqtt` | 0.5 | 256 MiB | *(0)* | Broker. |
| `logs` | 0.5 | 256 MiB | *(0)* | Idle sidecar. |
| `telegram-bot` | 0.5 | 256 MiB | *(0)* | Idle long-poll. |

- **OOM priority is the real safety mechanism.** `oom_score_adj` (lower = killed
  later) makes `scenereader` the preferred victim, so memory pressure restarts the
  episode narrator rather than the NVR.
- **The ceilings deliberately sum to more than the host's RAM** (~8.25 GiB). They
  are backstops against runaway growth, **not** a hard partition: a strict
  partition would have to be tight enough to OOM-kill Frigate during a recording
  burst, which is the one failure worth avoiding. To make it a hard partition,
  lower the values so the total fits under ~6.5 GiB.
- **The non-Frigate CPU caps sum to ~7.4 of 8 cores**, so Frigate always retains
  headroom without being quota-throttled.
- The old `scenewatch` (4 cpus / 3 GiB / a resident 2B VLM on a 15 s sweep) was
  stopped by the operator for heating the host. `scenereader` is 1 cpu / 1.5 GiB
  and its expensive work is idle-gated and bounded by construction.
- Inspect: `docker stats`; `docker inspect -f '{{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}' <container>`.
- Validated with the real parser: `docker compose config --format json`.

---

## 4. Host scripts (`scripts/`, deployed to & run on the host)

| File | Runs as | Purpose |
|---|---|---|
| [`heartbeat_cleanup.py`](scripts/heartbeat_cleanup.py) | root cron (15 min) | **Unified disk heartbeat** — per-store compliance + global-cap escalation across `config/stores/*.conf`. Modes: `--check`, `--dry-run` |
| [`cleanup_firewatch_store.py`](scripts/cleanup_firewatch_store.py) | in-container worker | firewatch DB-aware evidence cleanup (invoked by the heartbeat's `TYPE=docker-exec` store) |
| [`cleanup_media.sh`](scripts/cleanup_media.sh) | — | **SUPERSEDED** (replaced by the heartbeat) |
| [`collect_sensors.py`](scripts/collect_sensors.py) | library | lm-sensors reading helpers + Intel iGPU usage/temp/freq readers |
| [`machine-monitor.py`](scripts/machine-monitor.py) | `dr` cron (1 min) | CPU-temp watchdog (CRITICAL + WARM tiers) → Telegram; also reports live iGPU usage/temp (best-effort) |
| [`machine-status.py`](scripts/machine-status.py) | `dr` cron (daily 08:00) | Daily host + Frigate health report → Telegram |
| [`watchdog_baseline.py`](scripts/watchdog_baseline.py) | `dr` cron (1 min, `--cron`) | Machine/Frigate state sampler → size-capped CSV (incl. `gpu_usage_pct`, `gpu_temp_c`) |
| [`telegram_notify.py`](scripts/telegram_notify.py) | library | Shared Telegram Bot API helpers (all senders) |
| [`telegram_bot.py`](scripts/telegram_bot.py) | `telegram-bot` service | On-demand `/status` command responder |
| [`container_logs.py`](scripts/container_logs.py) | `logs` service | Read-only Docker-logs sidecar API |
| [`diagnose_detection.py`](scripts/diagnose_detection.py) | manual (host) | Read-only detection diagnosis from `/api/stats|events|config` + `frigate.log` |
| [`verify_remote.py`](scripts/verify_remote.py) | post-deploy / manual | Confirms effective `objects.track` per camera from `/api/config` |
| [`crontab.sample`](scripts/crontab.sample) | template | `dr` crontab lines |
| [`crontab.root.sample`](scripts/crontab.root.sample) | template | root crontab line (heartbeat) |

---

## 5. Host crontab (required)

Install once on the host (from `/home/dr/frigate`):

```bash
sudo crontab scripts/crontab.root.sample   # ROOT  — disk heartbeat
crontab scripts/crontab.sample             # dr    — monitor / report / sampler
crontab -l; sudo crontab -l                # confirm both
```

### 5.1 root crontab — [`scripts/crontab.root.sample`](scripts/crontab.root.sample)
| Schedule | Command | Purpose |
|---|---|---|
| `*/15 * * * *` | `/usr/bin/python3 /home/dr/frigate/scripts/heartbeat_cleanup.py` | Unified disk heartbeat (needs root to delete root-owned Frigate media) |

### 5.2 `dr` crontab — [`scripts/crontab.sample`](scripts/crontab.sample)
| Schedule | Command | Purpose |
|---|---|---|
| `* * * * *` | `/usr/bin/python3 /home/dr/frigate/scripts/machine-monitor.py` | CPU-temp + iGPU watchdog (every-minute sampling is required for ~1-min hot spikes) |
| `0 8 * * *` | `/usr/bin/python3 /home/dr/frigate/scripts/machine-status.py` | Daily health report |
| `* * * * *` | `/usr/bin/python3 /home/dr/frigate/scripts/watchdog_baseline.py --cron --tag baseline_pre` | State sampler → bounded CSV |

> Adjust `/home/dr/frigate` if the host deploy dir differs. Each job logs/alerts
> itself, so cron output is sent to `/dev/null`.

---

## 6. Required host settings

Install / verify on the Debian host:

```bash
# Docker + Compose plugin
sudo apt update && sudo apt install -y docker.io docker-compose-plugin
sudo systemctl enable --now docker

# lm-sensors (machine-monitor.py / machine-status.py read `sensors`)
sudo apt install -y lm-sensors && sudo sensors-detect --auto
# GPU usage needs NO extra tool: the i915 RC6 idle counter in sysfs is
# world-readable, so the watchdog needs no root and NO intel_gpu_top
# (that tool reads the i915 PMU and would need CAP_PERFMON).

# git + python3 are required (host scripts + deploy clone)
sudo apt install -y git python3
```

| Requirement | Why / where |
|---|---|
| `docker` + `docker compose` (v2 plugin), service enabled | Runs the whole stack |
| Host user `dr` owns `/home/dr/frigate` | git refuses "dubious ownership" otherwise; deploy runs as the clone owner |
| `/dev/dri/renderD128` exists | iGPU passthrough to Frigate (OpenVINO detector device: `GPU`) |
| `shm_size: 256mb` (compose) | FFmpeg decode buffers |
| `lm-sensors` (`sensors`) | `scripts/machine-monitor.py`, `scripts/machine-status.py` |
| i915 RC6 sysfs counter (`/sys/class/drm/card0/gt/gt0/rc6_residency_ms`, world-readable) | GPU usage % for the watchdog. **No `intel_gpu_top` / CAP_PERFMON needed.** This host has no GPU temp sensor, so `gpu_temp_c` is blank |
| `python3` on host | Cron scripts |
| root cron capability | disk heartbeat must delete root-owned media |
| Read access to `/var/run/docker.sock` | `logs` sidecar (runs as root in-container) |
| Cloudflare Tunnel (`cloudflared`) | Maps `live.mazr3a.garden` → `host:8080` (portal) under an Access policy |
| SSH `ssh.mazr3a.garden` | Deploy + remote diagnosis. **Never use `sshpass`** — use the SSH_ASKPASS pattern ([`.roo/rules/ssh-password.md`](.roo/rules/ssh-password.md)) |
| `git` remote `origin` = `https://github.com/tabebqena/mazr3a` (branch `master`) | Git-based deploy source |

---

## 7. Configuration files inventory

### 7.1 Git-tracked config (safe to commit — no secrets)
| File | Consumer |
|---|---|
| [`config/config.yaml`](config/config.yaml) | Frigate 0.17 (cameras, **zones**, go2rtc, model, detector, record, motion). Kept byte-identical to the host file — see the `frigate` service Notes for the re-adopt-after-UI-edit rule. |
| [`config/firewatch.conf`](config/firewatch.conf) | firewatch tunables (motion gate, thresholds, store) |
| [`config/scenereader.conf`](config/scenereader.conf) | scenereader tunables (Frigate access, scan/drain timing, idle governor, model backend incl. `VLM_TIMEOUT_S`, episodes incl. `VISIT_MERGE_GAP_S`, adaptive scenes incl. `SCENE_GAP_S`/`MOVE_MIN_DISP`, store) |
| [`config/places.conf`](config/places.conf) | **The human layer**: camera/zone → place names + the `ADJACENCY` routes that link one person across cameras (edit this to name the farm) |
| [`config/heartbeat.conf`](config/heartbeat.conf) | Disk heartbeat global cap (`FS_PATH`, `MIN_FREE_GB`, `RELIEF_FREE_GB`, …) |
| [`config/stores/*.conf`](config/stores/) | Per-service cleanup profiles |
| [`mosquitto/config/mosquitto.conf`](mosquitto/config/mosquitto.conf) | MQTT broker |
| [`config/portal.conf.example`](config/portal.conf.example) | Portal template (users/secret/tunables) |
| [`config/telegram.conf.example`](config/telegram.conf.example) | Telegram template (creds + tunables) |
| [`config/cleanup_firewatch.conf`](config/cleanup_firewatch.conf) | firewatch store cleanup tunables (worker) |
| [`config/cleanup_media.conf`](config/cleanup_media.conf) | **SUPERSEDED** reference only |
| [`config/frigate.yml`](config/frigate.yml) | Deprecated name (Frigate 0.17 reads `config.yaml`) |

### 7.2 Git-ignored config / state (never committed — see [`.gitignore`](.gitignore))
| Path | Contents |
|---|---|
| `.env` | Camera RTSP credentials (`FRIGATE_*`) |
| `config/telegram.conf` | Bot token + `CHAT_ID`(s) + tunables |
| `config/portal.conf` | Portal users (PBKDF2) + `SECRET_KEY` |
| `media/` | Frigate recordings/clips/snapshots + firewatch evidence + the scenereader **text** store (`media/events/`) + watchdog CSVs |
| `mosquitto/data/`, `mosquitto/log/` | Broker runtime state |
| `models/fire/versions/` | Versioned fire-model archive |
| `models/scene/` | **TWO** git-ignored models: the small SmolVLM2-500M GGUF + mmproj (`smolvlm2-500m/`) and the RETAINED Qwen2-VL-2B OpenVINO INT4 IR. Neither is ever deleted or re-downloaded; `MODEL_BACKEND` selects which runs. |
| `heartbeat-cleanup.log`, `machine-monitor.state`, `frigate.log` | Host runtime logs/state |
| `plans/`, `prompt.txt`, `notebooks/`, `fire-model-training/*` | Local working docs / datasets |

### 7.3 Credentials — how they work
- **Camera creds:** `.env` → `env_file` on the `frigate` container → `{FRIGATE_*}` placeholders in `config/config.yaml`. Frigate 0.17 has **no `!env_var` tag** — only `FRIGATE_`-prefixed substitution. `cam08` (Hikvision) uses separate `FRIGATE_HIK_RTSP_USER/PASS`.
- **Telegram:** shared git-ignored `config/telegram.conf`, read by all senders + the bot. `CHAT_ID` accepts a comma-separated recipient list.
- **Portal:** git-ignored `config/portal.conf`; add a user with `python portal/genpass.py --username NAME`.

---

## 8. Storage layout & disk cleanup (heartbeat)

All persistent data lives under the deploy root; a **single cleaner** (root cron,
15 min) keeps the disk from filling:

- **Global cap** — [`config/heartbeat.conf`](config/heartbeat.conf): `FS_PATH={DEPLOY}`, triggers `MIN_FREE_GB` / `MAX_USED_PCT`, relief target `RELIEF_FREE_GB`. On escalation, stores are freed oldest-first in `PRIORITY` order.
- **Per-service profiles** — [`config/stores/`](config/stores/):

| Profile | Type | Root | MAX_AGE_DAYS | MAX_SIZE_GB | Priority |
|---|---|---|---|---|---|
| [`logs.conf`](config/stores/logs.conf) | dir | `{DEPLOY}` (top-level `*.log`) | 30 | 0 | 5 |
| [`frigate.conf`](config/stores/frigate.conf) | dir | `{DEPLOY}/media` (`recordings,clips,snapshots,cache,exports`) | 0 | 20 | 10 |
| [`watchdog.conf`](config/stores/watchdog.conf) | dir | `{DEPLOY}/media/watchdog` (`*.csv`) | 14 | 0 | 20 |
| [`mosquitto.conf`](config/stores/mosquitto.conf) | dir | `{DEPLOY}/mosquitto` (`data,log`) | 7 | 0 | 30 |
| [`firewatch.conf`](config/stores/firewatch.conf) | docker-exec | in-container worker | 90 | 2 | 50 |

- firewatch evidence is **DB-aware**: the profile delegates to `cleanup_firewatch_store.py` in the container (SQLite is the source of truth; only DB-referenced JPEGs are removed — never Frigate media, which shares the `./media` tree).
- scenereader adds **no** image store: its frames are Frigate's own (governed by `frigate.conf` above), so it needs **no cleanup profile**. Its text DB lives in `media/events/` and is hard-protected from deletion (`_HARD_PROTECT` covers `*.db`/`*.db-wal`/`*.db-shm`). Row expiry is owned by the service (`EVENTS_RETENTION_DAYS`).
- Inspect: `sudo /usr/bin/python3 scripts/heartbeat_cleanup.py --check` (permission/usage table) and `--dry-run` (preview deletions). Log: `heartbeat-cleanup.log`.

---

## 9. Networking & ports

| Port (host) | Service | Exposure |
|---|---|---|
| `5000` | frigate | HTTP UI/API (LAN) |
| `8971` | frigate | HTTPS UI (LAN) |
| `8554` | frigate | RTSP restream (LAN) |
| `8555/tcp`,`8555/udp` | frigate | WebRTC (LAN) |
| `1883` | mqtt | MQTT broker (LAN) |
| `8080` | portal | Internal host port — fronted by the Cloudflare Tunnel |
| `8090` | logs | Internal compose network only (not published) |

**Public access:** the whole host sits behind a **Cloudflare Tunnel + Access** on
`live.mazr3a.garden`; the tunnel maps the portal root → `host:8080` under the
Access policy, and the portal login gates the dashboard/JSON API on top. Live
view is **same-origin** (MSE over WebSocket primary, HLS fallback) proxied
server-side through the portal, so no extra tunnel paths are needed. Frigate's
REST/DB and the firewatch WAL DB are reached **server-side only**.

---

## 10. Deploy workflow

Git-based, one orchestrator: [`dev_scripts/deploy_all.sh`](dev_scripts/deploy_all.sh).

Developer scripts of note (local, not deployed):

| File | Purpose |
|---|---|
| [`dev_scripts/deploy_all.sh`](dev_scripts/deploy_all.sh) | SSH to host, `git pull --ff-only`, `docker compose up -d --build` + restart ALL services, verify |
| [`dev_scripts/run_ssh.sh`](dev_scripts/run_ssh.sh) | Run ONE read-only remote command (SSH_ASKPASS, no `sshpass`) |
| [`dev_scripts/prep_fire_model.sh`](dev_scripts/prep_fire_model.sh) | `best.pt` → OpenVINO IR (`models/fire/`) |
| [`dev_scripts/prep_scene_model_llamacpp.sh`](dev_scripts/prep_scene_model_llamacpp.sh) | **ADD-ONLY** fetch of the small GGUF + mmproj into `models/scene/smolvlm2-500m/` (files picked by pattern; skips what exists; `--force` replaces only its own files). `--bin --llamacpp-tag <tag>` also fetches llama.cpp into `models/scene/bin/`. Never touches the retained IR. |
| [`dev_scripts/prep_scene_model.sh`](dev_scripts/prep_scene_model.sh) | Fetch the **RETAINED** OpenVINO VLM (Qwen2-VL-2B int4) into `models/scene/` with `curl`; refuses to clobber without `--force`. Kept so the 2B never needs re-downloading. |
| [`dev_scripts/promote_fire_model.sh`](dev_scripts/promote_fire_model.sh) | Promote a versioned checkpoint to ACTIVE |
| [`dev_scripts/test_fire_model.py`](dev_scripts/test_fire_model.py) | Local fire-model benchmark |
| Other `dev_scripts/*` | dataset build / analysis helpers |

Workflow (**the orchestrator runs on the DEV MACHINE, not the host** — it SSHes to
`ssh.mazr3a.garden` and does everything there as `dr`, the clone owner):
1. Edit + commit locally, then `git push origin master`.
2. Run `./dev_scripts/deploy_all.sh` **from the dev machine**. Remotely it does
   `git pull --ff-only origin master` → `docker compose up -d --build` →
   `docker compose restart`. Needs `DEPLOY_SSH_USER`/`DEPLOY_SSH_PASS` (prompts if
   unset) and never pushes for you.
3. The script verifies stack state, effective Frigate config, `firewatch.py --check` and a `--dry-run` pass.

> **A dirty host file blocks the pull.** `git pull --ff-only` refuses when an
> incoming commit touches a file that is also modified on the host. Editing
> Frigate in its UI rewrites `config/config.yaml` on the host, so that file must
> be backed up and brought in line before deploying (see
> [`plans/event-scene-reader.md`](plans/event-scene-reader.md) §20.0/§20.6).
>
> **Model prep and runtime checks run ON THE HOST** (they write to / read the
> bind-mounted `models/`): `dev_scripts/prep_scene_model_llamacpp.sh`,
> `dev_scripts/prep_scene_model.sh`, and the `docker compose exec … --check` probes.

> **scenereader deploy prerequisite:** the models are **not** in git. On the host
> fetch the small default (`bash dev_scripts/prep_scene_model_llamacpp.sh`) plus
> the llama.cpp binaries it needs (`--bin --llamacpp-tag <tag>`), then point
> `MODEL_FILE`/`MMPROJ_FILE`/`LLAMA_*_BIN` at what it prints. The retained 2B IR
> only needs fetching if it is ever lost (`prep_scene_model.sh`). Verify with
> `docker compose exec scenereader python /scenereader/scenereader.py --check`
> (reports the idle gate, the scheduler/lock state, the store/clips paths and any
> camera missing from `config/places.conf`, then captions one stored frame — it
> opens the store **read-only**, so it is safe while the service runs, and it says
> how long the probe may take). `--help` lists every mode; a second scheduler or a
> second simultaneous manual command is refused with exit 3.
>
> **`config/places.conf` is an operator input, not a default:** without
> `CAMERA_PLACES` + `ADJACENCY` the narrator still works but describes cameras
> rather than places and cannot link a person across cameras.

Ownership rules: deploy/git handoff runs as **`dr`** (owner of `/home/dr/frigate`);
the AI must not run git on the host as `ai` nor pull from the host side. The AI
applies local edits and asks the user to push + deploy.

---

## 11. Maintaining this file

**Rule: any change that adds/removes/modifies a service (or its config, port,
mount, cron entry or host requirement) MUST update this file in the same commit.**
This is enforced by [`.roo/rules/system-summary.md`](.roo/rules/system-summary.md).

When adding a **new service**, update all of:
1. [§3 Services](#3-services-docker-compose) — add a subsection + the
   [service → dev-file map](#37-service--development-file-map-quick-lookup).
2. [§4 Host scripts](#4-host-scripts-scripts-deployed-to--run-on-the-host) — if it ships a script under `scripts/`.
3. [§5 Crontab](#5-host-crontab-required) — if it needs a cron entry (and the matching `crontab*.sample`).
4. [§6 Required settings](#6-required-host-settings) — new apt packages / devices / mounts.
5. [§7 Configuration inventory](#7-configuration-files-inventory) — new tracked vs git-ignored config.
6. [§8 Storage & cleanup](#8-storage-layout--disk-cleanup-heartbeat) — add a `config/stores/<service>.conf` profile if it writes files.
7. [§9 Ports](#9-networking--ports) — new published/internal ports.
8. [§10 Deploy](#10-deploy-workflow) — if deploy/verify steps change.
9. Bump **Summary version / Last updated** in the header table, and bump `APP_VERSION` for any portal change.

---

## 12. Deferred / future phases

- **Phase 1b** — native in-Frigate fire/smoke detection via a fire+smoke+person+car+animal union model (so fire appears in the Frigate UI / MQTT / recorded clips).
- **Phase 2** — *partly done:* cross-camera episode narratives ship via `scenereader` (§3.7) with a deterministic sentence and anonymous person identity; the SQLite event/episode store is included. **Still deferred:** the portal **Episodes** view + "name this person" assisted labeling, the cron/Telegram digest, **person attributes** (CLIP zero-shot clothing, daylight-only), appearance ReID as a link tie-breaker, a tiny text LLM to polish the narrative, and a fine-tuned ≤500M captioner trained on the farm's own captures.
- **Phase 3** — **real names** via face recognition (InsightFace SCRFD + ArcFace against a named gallery, assisted by manual naming first) and gait. Gait is not feasible on the 640×360 detect substream (it needs video + resolution the host cannot sustain without pulling the main streams).
