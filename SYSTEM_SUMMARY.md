# System Summary — mazr3a Edge Security System

> **Purpose.** A **top-level map** of the whole system: the services in the
> stack, where their development files live, the host crontab, the required host
> settings, the config inventory, storage/cleanup wiring, networking and the
> deploy workflow. [`README.md`](README.md) is the *how-to* / narrative.
>
> **This is an overview, NOT a changelog.** Do not record per-change detail,
> dated verification notes or incident history here — that belongs in the
> `plans/*.md` docs (git-ignored) and in the code. See
> [§11 Maintaining this file](#11-maintaining-this-file).

| Field | Value |
|---|---|
| Summary version | `v40` |
| Last updated | 2026-09-12 |
| Repo | `https://github.com/tabebqena/mazr3a` (branch `master`) |
| Portal `APP_VERSION` | `0.11.5` (see [`portal/app.py`](portal/app.py:44)) — bump on every portal change |
| Portal PWA | Installable Android app (per-request manifest + fingerprinted icons; **no caching service worker by design**). Needs a Cloudflare Access **Bypass** for the manifest + `/static/icons/*` — see [`plans/portal-android-pwa.md`](plans/portal-android-pwa.md) |
| Portal notifications | Server-side notification feed + **Notifications** tab with unread badge, browser-Notification gate, and a notification-only `/sw.js` (no fetch/cache handler). See [`plans/portal-notifications.md`](plans/portal-notifications.md) |

---

## 1. Overview

A farm CCTV / security stack built on **Frigate NVR** as the ingest + detection
core, with purpose-built side services around it:

- **Frigate** decodes 10 IP cameras' low-res detect substreams, runs an OpenVINO
  detector, records **event-only** clips, exposes REST/UI and an embedded
  **go2rtc** live restream, and publishes MQTT events.
- **firewatch** runs an out-of-band fire/smoke model on Frigate's already-decoded
  detect frames and sends Telegram photo alerts + stores evidence.
- **scenereader** (replaced `scenewatch`) reads the capture events Frigate
  already produced — snapshots read **in place** from Frigate's media tree,
  joined by **exact event id** to `config/frigate.db` (read-only) — writes a
  deterministic per-camera sentence, links cameras into cross-camera **person
  episodes** using a place map + adjacency, and optionally captions frames with
  a **small (≤500M) GGUF** model during an **idle-gated, bounded** batch.
- **telegram-bot** answers `/status`, `/help`, `/start` live from Telegram.
- **logs** is a read-only Docker-logs sidecar for the portal's admin Debug tab.
- **portal** is an authenticated FastAPI + vanilla-JS SPA (login, live view,
  events, fire alerts, scene descriptions) published via a Cloudflare Tunnel.
- **mqtt** (Mosquitto) is the event broker.
- Host **cron** runs the unified disk heartbeat, a CPU-temp + iGPU watchdog, a
  daily health report and a state sampler.

Roadmap context: this is **Phase 0/1a plus the event-driven episode narrator**
(`scenereader`, §3.7). Still deferred: the portal Episodes digest, person
attributes, face names and gait (see [§12](#12-deferred--future-phases)).

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
| Active model | `models/coco/yolo11s.onnx` (COCO 80-class, 640×640, iGPU); `yolo11n.onnx` is the rollback |
| Credentials | `.env` → `{FRIGATE_*}` placeholders (git-ignored) |
| Ports | `5000` HTTP UI/API · `8971` TLS UI · `8554` RTSP restream · `8555` tcp+udp WebRTC |
| Volumes | `./config:/config` · `./media:/media/frigate` · `./models:/models:ro` · tmpfs `/tmp/cache` (1 GB) |
| Device / limits | `/dev/dri/renderD128`; `shm_size: 256mb` |
| Zones | **11 zone polygons** across cam01/cam02/cam03/cam06/cam09, drawn in the Frigate UI. `scenereader` maps them to sub-place names via `ZONE_PLACES` in [`config/places.conf`](config/places.conf). |
| Notes | Event-only recording (detect substream); software decode. **`config/config.yaml` is the CANONICAL copy of the host file** — the Frigate UI rewrites it, so after ANY UI edit it must be re-adopted verbatim into the repo, or the next `git pull --ff-only` refuses to run. |

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
| Store ownership | `firewatch.db` + its `-wal`/`-shm` sidecars must be writable by **uid 1000** (the container user). NEVER open the live DB from the host (not even `mode=ro`) — a foreign WAL sidecar wedges every write while alerts still fire. Read it via `docker exec firewatch …`. See [`.roo/rules/firewatch-store-ownership.md`](.roo/rules/firewatch-store-ownership.md) and [`plans/firewatch-store-readonly-recovery.md`](plans/firewatch-store-readonly-recovery.md). |
| Store recovery | [`scripts/backfill_firewatch_store.py`](scripts/backfill_firewatch_store.py) re-registers evidence JPEGs with no `frames` row so missed alerts reappear in the portal (dry-run by default, `--commit` to write). |
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
| Purpose | Login (PBKDF2); **Live view** (MSE-over-WebSocket primary, HLS fallback); **Events** (large player fed by a clip strip with playlist auto-advance); **fire alerts** (cards/lightbox with portal-derived motion + burst-hits badges); **scenereader Episodes / Scenes / Scene log** (read-only via [`portal/eventstore.py`](portal/eventstore.py), plus a **Process now** caption trigger); admin **Debug** tab; runtime account management and per-user bandwidth **usage**. |
| Dev files | [`portal/app.py`](portal/app.py) (`APP_VERSION` here), [`portal/auth.py`](portal/auth.py), [`portal/config.py`](portal/config.py), [`portal/frigate.py`](portal/frigate.py), [`portal/firestore.py`](portal/firestore.py), [`portal/eventstore.py`](portal/eventstore.py), [`portal/usage.py`](portal/usage.py), [`portal/notifstore.py`](portal/notifstore.py), [`portal/userstore.py`](portal/userstore.py), [`portal/genpass.py`](portal/genpass.py), [`portal/static/`](portal/static/) (SPA: `index.html`, `app.js`, `style.css`, `sw.js`, `favicon.svg`, `vendor/hls.min.js`), [`portal/requirements.txt`](portal/requirements.txt) |
| Config | [`config/portal.conf`](config/portal.conf) (git-ignored; template [`config/portal.conf.example`](config/portal.conf.example)) |
| Ports | `8080` (internal host port for the Cloudflare Tunnel) |
| Volumes | `./portal:/srv/app/portal:ro` · `./config:/config:ro` · `./media:/media` (rw: SQLite WAL read + the portal's own `media/portal/{usage,users,notifications}.db`) |
| Env | `PORTAL_CONF`, `PORTAL_LOGS_API=http://logs:8090`; `PORTAL_USERS_DB`, `PORTAL_USAGE_DB`, `PORTAL_NOTIF_DB` (defaults under `/media/portal/`) |
| Access model | Tabs are a set of grantable `tab_*` permissions; an **admin** has every tab implicitly. Accounts are **runtime** records in `media/portal/users.db` — changes apply immediately, **no restart** — seeded once from any `user` line in `portal.conf`. |
| Quota | Each account carries a per-user egress **quota** (`quota_bytes`, default **5 GiB**, `0` = unlimited), editable in the admin **Manage** tab and shown as a used/quota bar (footer on large screens, collapsed menu on phones). |
| PWA / Notifications | See the header table; details in [`plans/portal-android-pwa.md`](plans/portal-android-pwa.md) and [`plans/portal-notifications.md`](plans/portal-notifications.md). |
| Notes | **Cache-busting policy:** bump `APP_VERSION` + use `{{ ASSET_* }}` tokens (see [`.roo/rules/portal-cache-busting.md`](.roo/rules/portal-cache-busting.md)). The event-clip proxy forwards `Range` (relays `206`) so the player can seek. Responsive UI: phones collapse the nav into a `#nav-toggle` hamburger; landscape has dedicated Live/Events layouts; centre play/stop overlays on the video. |

### 3.7 `scenereader` — cross-camera episode narrator

See [`plans/event-scene-reader.md`](plans/event-scene-reader.md) (replaced the
stopped `scenewatch`); the adaptive scene layer is in
[`plans/adaptive-scene-narrative.md`](plans/adaptive-scene-narrative.md).

| | |
|---|---|
| Container | `scenereader` |
| Image | **built** from [`scenereader/Dockerfile`](scenereader/Dockerfile) (`python:3.11-slim` + `openvino-genai`, `numpy`, `Pillow`); unprivileged uid 1000 |
| Purpose | Read Frigate's own captures (never re-detecting), write a deterministic per-camera sentence, then link cameras into anonymous **person episodes** with an EN + AR narrative. Groups events into per-camera **adaptive scenes** (L1) and uses each object's own trajectory to decide who MOVED. |
| Dev files | [`scenereader/scenereader.py`](scenereader/scenereader.py) (service), [`frigate.py`](scenereader/frigate.py), [`store.py`](scenereader/store.py), [`episodes.py`](scenereader/episodes.py), [`scenes.py`](scenereader/scenes.py), [`describe.py`](scenereader/describe.py), [`captioners.py`](scenereader/captioners.py), models [`models/scene/`](models/scene/) (git-ignored) |
| Config | [`config/scenereader.conf`](config/scenereader.conf) + [`config/places.conf`](config/places.conf) (episode shape, scene/movement tuning, caption cost, model backend) |
| Frame source | Read **in place, never copied** — host `./media` IS Frigate's `/media/frigate`; snapshots are `clips/<camera>-<event_id>.jpg` with a `-clean.webp` sibling (no `media/snapshots/`). |
| Metadata source | `FRIGATE_METADATA_SOURCE=db` (default): `config/frigate.db`, **read-only**, joined by the exact event id. `api`/`auto` fall back to `/api/events/<id>`. |
| Model | **`MODEL_BACKEND` switch**: `llamacpp` (default) = small **SmolVLM2-500M GGUF + mmproj** served by a resident `llama-server` (~0.5–0.7 GB, single server slot); `openvino` = the **retained** Qwen2-VL-2B INT4 IR (~2 GB). Config-only switch. |
| Model prerequisite | **NOT deployed by git.** Small GGUF: [`dev_scripts/prep_scene_model_llamacpp.sh`](dev_scripts/prep_scene_model_llamacpp.sh) (add-only). Retained IR: [`dev_scripts/prep_scene_model.sh`](dev_scripts/prep_scene_model.sh). See [`models/scene/README.md`](models/scene/README.md). |
| Store | `./media/events/` — **text only**: `events.db` (WAL: events + episodes + visits + aliases + scenes + scene_events), `reader_status.json`, `.drain_request`, `names.json`, `.scenereader-*.lock`. **No image copies anywhere.** |
| Volumes | `./scenereader:/scenereader:ro` · `./config:/config:ro` · `./models:/models:ro` · `./media:/media` (rw) |
| Ports | none (outbound only) |
| Concurrency | Short locks, one writer: a flock guard refuses a second `scheduler` (exit 3) and overlapping manual commands; the store write lock is held for milliseconds (per-item commit); `--check`/`--status` open the store read-only. |
| Notes | Edits need only `docker compose restart scenereader`. No host cron — the in-container scheduler scans `clips/` and rebuilds episodes on fixed intervals (both rebuildable from `events`); the only expensive step (the VLM caption batch) is **idle-gated** (host loadavg + CPU temp) and bounded by `MAX_EVENTS_PER_RUN`/`MAX_RUN_SECONDS`. Every event is scored 0-100 into a `tier`. |

### 3.8 Service → development-file map (quick lookup)
| Service | Primary code / config in repo |
|---|---|
| `frigate` | [`config/config.yaml`](config/config.yaml), [`models/coco/`](models/coco/), `docker-compose.yml`, `.env` |
| `mqtt` | [`mosquitto/config/mosquitto.conf`](mosquitto/config/mosquitto.conf) |
| `firewatch` | [`firewatch/firewatch.py`](firewatch/firewatch.py), [`config/firewatch.conf`](config/firewatch.conf), [`models/fire/`](models/fire/), [`scripts/telegram_notify.py`](scripts/telegram_notify.py) |
| `scenereader` | [`scenereader/`](scenereader/) (service + modules), [`config/scenereader.conf`](config/scenereader.conf), [`config/places.conf`](config/places.conf), [`models/scene/`](models/scene/) (git-ignored), [`dev_scripts/prep_scene_model_llamacpp.sh`](dev_scripts/prep_scene_model_llamacpp.sh) |
| `telegram-bot` | [`scripts/telegram_bot.py`](scripts/telegram_bot.py), [`scripts/telegram_notify.py`](scripts/telegram_notify.py) |
| `logs` | [`scripts/container_logs.py`](scripts/container_logs.py) |
| `portal` | [`portal/`](portal/) + [`config/portal.conf`](config/portal.conf); Android-PWA icons via [`dev_scripts/make_portal_pwa_icons.sh`](dev_scripts/make_portal_pwa_icons.sh) → [`portal/static/icons/`](portal/static/icons/) |

### 3.9 Resource limits (CPU / memory / OOM priority)

Set in [`docker-compose.yml`](docker-compose.yml). `cpus` is a hard CFS quota and
`mem_limit` a hard memory ceiling; **neither reserves anything**. Host: 8 cores,
~7.5 GiB.

| Service | `cpus` | `mem_limit` | `oom_score_adj` | Rationale |
|---|---|---|---|---|
| `frigate` | *(uncapped)* | 3 GiB | **-500** | Critical path — CFS throttling can drop decoded frames, so only memory is ceilinged. Its cgroup also holds the 1 GB tmpfs + 256 MB `/dev/shm`. |
| `scenereader` | 1 | 1.5 GiB | **200** | ~0.5–0.7 GB resident with the small GGUF default; raise towards 3 GiB **only** if `MODEL_BACKEND=openvino` is selected. The preferred OOM victim. |
| `firewatch` | 2 | 1 GiB | *(0)* | Small IR model at a low cadence. |
| `portal` | 1 | 512 MiB | *(0)* | HTTP + proxying; HLS is streamed, not transcoded. |
| `mqtt` | 0.5 | 256 MiB | *(0)* | Broker. |
| `logs` | 0.5 | 256 MiB | *(0)* | Idle sidecar. |
| `telegram-bot` | 0.5 | 256 MiB | *(0)* | Idle long-poll. |

- **OOM priority is the real safety mechanism**: `oom_score_adj` makes
  `scenereader` the preferred victim, so memory pressure restarts the episode
  narrator rather than the NVR. The ceilings deliberately sum to more than the
  host's RAM (backstops, not a hard partition).
- Inspect: `docker stats`; validated with `docker compose config --format json`.

---

## 4. Host scripts (`scripts/`, deployed to & run on the host)

| File | Runs as | Purpose |
|---|---|---|
| [`heartbeat_cleanup.py`](scripts/heartbeat_cleanup.py) | root cron (15 min) | **Unified disk heartbeat** — per-store compliance + global-cap escalation across `config/stores/*.conf`. Modes: `--check`, `--dry-run` |
| [`cleanup_firewatch_store.py`](scripts/cleanup_firewatch_store.py) | in-container worker | firewatch DB-aware evidence cleanup (invoked by the heartbeat's `TYPE=docker-exec` store) |
| [`backfill_firewatch_store.py`](scripts/backfill_firewatch_store.py) | in-container worker (manual) | **Recovery**: registers evidence JPEGs that have no `frames` row so missed fire alerts reappear in the portal. Dry-run by default; `--commit`, `--alert-times FILE` |
| [`cleanup_media.sh`](scripts/cleanup_media.sh) | — | **SUPERSEDED** (replaced by the heartbeat) |
| [`collect_sensors.py`](scripts/collect_sensors.py) | library | lm-sensors + Intel iGPU usage/temp/freq readers |
| [`machine-monitor.py`](scripts/machine-monitor.py) | `dr` cron (1 min) | CPU-temp watchdog (CRITICAL + WARM tiers) → Telegram; reports live iGPU usage/temp |
| [`machine-status.py`](scripts/machine-status.py) | `dr` cron (daily 08:00) | Daily host + Frigate health report → Telegram |
| [`watchdog_baseline.py`](scripts/watchdog_baseline.py) | `dr` cron (1 min, `--cron`) | Machine/Frigate state sampler → size-capped CSV |
| [`telegram_notify.py`](scripts/telegram_notify.py) | library | Shared Telegram Bot API helpers (all senders) |
| [`telegram_bot.py`](scripts/telegram_bot.py) | `telegram-bot` service | On-demand `/status` command responder |
| [`container_logs.py`](scripts/container_logs.py) | `logs` service | Read-only Docker-logs sidecar API |
| [`diagnose_detection.py`](scripts/diagnose_detection.py) | manual (host) | Read-only detection diagnosis from `/api/stats|events|config` + `frigate.log` |
| [`verify_remote.py`](scripts/verify_remote.py) | post-deploy / manual | Confirms effective `objects.track` per camera from `/api/config` |
| [`cam_event_listener.py`](scripts/cam_event_listener.py) | manual (host, **temporary — stopped**) | cam01 (UNV) alarm/event listener: HTTP `Event/Subscription` + WebSocket status mirror, with person **line-crossing** images. Recon only; see [`plans/cam01-websocket-exploration.md`](plans/cam01-websocket-exploration.md) and [`TODO.md`](TODO.md) |
| [`crontab.sample`](scripts/crontab.sample) / [`crontab.root.sample`](scripts/crontab.root.sample) | template | Crontab lines (`dr` / root) |

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
# world-readable, so the watchdog needs no root and NO intel_gpu_top.

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
| i915 RC6 sysfs counter (`/sys/class/drm/card0/gt/gt0/rc6_residency_ms`) | GPU usage % for the watchdog. **No `intel_gpu_top` / CAP_PERFMON needed.** This host exposes no GPU temp sensor |
| `python3` on host | Cron scripts |
| root cron capability | disk heartbeat must delete root-owned media |
| Read access to `/var/run/docker.sock` | `logs` sidecar (runs as root in-container) |
| Cloudflare Tunnel (`cloudflared`) | Maps `live.mazr3a.garden` → `host:8080` (portal) under an Access policy |
| Cloudflare Access **Bypass** for `/manifest.webmanifest`, `/static/icons/*`, `/apple-touch-icon.png`, `/favicon.ico` (**only** these) | **Required for PWA install** — WebAPK minting fetches them unauthenticated. `/` and `/api/*` must stay Access-protected. See [`plans/portal-android-pwa.md`](plans/portal-android-pwa.md) |
| `media/firewatch.db` (+ `-wal`/`-shm`) owned by **uid 1000**, group `dr`, mode `664` | firewatch writes the evidence DB as uid 1000 and SQLite WAL sidecars are writable only by their creator's uid. Repair: `sudo chown 1000:1000 media/firewatch.db*`. See [`.roo/rules/firewatch-store-ownership.md`](.roo/rules/firewatch-store-ownership.md) |
| SSH `ssh.mazr3a.garden` | Deploy + remote diagnosis. **Never use `sshpass`** — use the SSH_ASKPASS pattern ([`.roo/rules/ssh-password.md`](.roo/rules/ssh-password.md)) |
| `git` remote `origin` = `https://github.com/tabebqena/mazr3a` (branch `master`) | Git-based deploy source |

---

## 7. Configuration files inventory

### 7.1 Git-tracked config (safe to commit — no secrets)
| File | Consumer |
|---|---|
| [`config/config.yaml`](config/config.yaml) | Frigate 0.17 (cameras, zones, go2rtc, model, detector, record, motion). Kept byte-identical to the host file. |
| [`config/firewatch.conf`](config/firewatch.conf) | firewatch tunables (motion gate, thresholds, store) |
| [`config/scenereader.conf`](config/scenereader.conf) | scenereader tunables (Frigate access, scan/drain timing, idle governor, model backend, episodes, adaptive scenes, store) |
| [`config/places.conf`](config/places.conf) | **The human layer**: camera/zone → place names + the `ADJACENCY` routes that link a person across cameras |
| [`config/heartbeat.conf`](config/heartbeat.conf) | Disk heartbeat global cap (`FS_PATH`, `MIN_FREE_GB`, `RELIEF_FREE_GB`, …) |
| [`config/stores/*.conf`](config/stores/) | Per-service cleanup profiles |
| [`mosquitto/config/mosquitto.conf`](mosquitto/config/mosquitto.conf) | MQTT broker |
| [`config/portal.conf.example`](config/portal.conf.example) | Portal template (secret + tunables; `user` lines are a first-run seed) |
| [`config/telegram.conf.example`](config/telegram.conf.example) | Telegram template (creds + tunables) |
| [`config/cleanup_firewatch.conf`](config/cleanup_firewatch.conf) | firewatch store cleanup tunables (worker) |
| [`config/cleanup_media.conf`](config/cleanup_media.conf) | **SUPERSEDED** reference only |
| [`config/frigate.yml`](config/frigate.yml) | Deprecated name (Frigate 0.17 reads `config.yaml`) |

### 7.2 Git-ignored config / state (never committed — see [`.gitignore`](.gitignore))
| Path | Contents |
|---|---|
| `.env` | Camera RTSP credentials (`FRIGATE_*`) |
| `config/telegram.conf` | Bot token + `CHAT_ID`(s) + tunables |
| `config/portal.conf` | Portal `SECRET_KEY` + tunables (the `user` seed lines are optional now) |
| `media/` | Frigate recordings/clips/snapshots + firewatch evidence + the scenereader **text** store (`media/events/`) + the portal's own `media/portal/{usage,users,notifications}.db` + watchdog CSVs |
| `mosquitto/data/`, `mosquitto/log/` | Broker runtime state |
| `models/fire/versions/` | Versioned fire-model archive |
| `models/scene/` | **TWO** git-ignored models: the small SmolVLM2-500M GGUF + mmproj and the RETAINED Qwen2-VL-2B OpenVINO INT4 IR; `MODEL_BACKEND` selects which runs |
| `heartbeat-cleanup.log`, `machine-monitor.state`, `frigate.log` | Host runtime logs/state |
| `plans/`, `prompt.txt`, `notebooks/`, `fire-model-training/*` | Local working docs / datasets |

### 7.3 Credentials — how they work
- **Camera creds:** `.env` → `env_file` on the `frigate` container → `{FRIGATE_*}`
  placeholders in `config/config.yaml` (Frigate 0.17 has no `!env_var` tag).
  `cam08` (Hikvision) uses separate `FRIGATE_HIK_RTSP_USER/PASS`.
- **Telegram:** shared git-ignored `config/telegram.conf`; `CHAT_ID` accepts a
  comma-separated recipient list.
- **Portal:** git-ignored `config/portal.conf` holds `SECRET_KEY` + tunables.
  Accounts are **runtime** records in `media/portal/users.db`, managed from the
  portal **Account**/**Manage** tabs (no restart) and seeded once from a `user`
  line (`python portal/genpass.py --username NAME`). Each account carries a
  per-user egress quota (default 5 GiB, `0` = unlimited).

---

## 8. Storage layout & disk cleanup (heartbeat)

All persistent data lives under the deploy root; a **single cleaner** (root cron,
15 min) keeps the disk from filling:

- **Global cap** — [`config/heartbeat.conf`](config/heartbeat.conf): triggers on
  `MIN_FREE_GB` / `MAX_USED_PCT`, relief target `RELIEF_FREE_GB`; on escalation,
  stores are freed oldest-first in `PRIORITY` order.
- **Per-service profiles** — [`config/stores/`](config/stores/):

| Profile | Type | Root | MAX_AGE_DAYS | MAX_SIZE_GB | Priority |
|---|---|---|---|---|---|
| [`logs.conf`](config/stores/logs.conf) | dir | `{DEPLOY}` (top-level `*.log`) | 30 | 0 | 5 |
| [`frigate.conf`](config/stores/frigate.conf) | dir | `{DEPLOY}/media` (`recordings,clips,snapshots,cache,exports`) | 0 | 20 | 10 |
| [`watchdog.conf`](config/stores/watchdog.conf) | dir | `{DEPLOY}/media/watchdog` (`*.csv`) | 14 | 0 | 20 |
| [`mosquitto.conf`](config/stores/mosquitto.conf) | dir | `{DEPLOY}/mosquitto` (`data,log`) | 7 | 0 | 30 |
| [`firewatch.conf`](config/stores/firewatch.conf) | docker-exec | in-container worker | 90 | 2 | 50 |
| [`portal.conf`](config/stores/portal.conf) | dir | `{DEPLOY}/media/portal` (usage + users + notifications DBs) | 0 | 0 | 90 |

- firewatch evidence is **DB-aware**: the profile delegates to
  `cleanup_firewatch_store.py` in the container (SQLite is the source of truth).
- scenereader and the portal add **no image store**; both write only small
  SQLite DBs that are hard-protected from deletion, so their row expiry is
  owned in-app (`EVENTS_RETENTION_DAYS`, `USAGE_RETENTION_DAYS`,
  `NOTIFY_RETENTION_DAYS`).
- Inspect: `sudo /usr/bin/python3 scripts/heartbeat_cleanup.py --check` and
  `--dry-run`. Log: `heartbeat-cleanup.log`.

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
server-side through the portal. Frigate's REST/DB and the firewatch WAL DB are
reached **server-side only**.

---

## 10. Deploy workflow

Git-based, one orchestrator: [`dev_scripts/deploy_all.sh`](dev_scripts/deploy_all.sh).

Developer scripts of note (local, not deployed):

| File | Purpose |
|---|---|
| [`dev_scripts/deploy_all.sh`](dev_scripts/deploy_all.sh) | SSH to host, `git pull --ff-only`, `docker compose up -d --build` + restart ALL services, verify |
| [`dev_scripts/run_ssh.sh`](dev_scripts/run_ssh.sh) | Run ONE read-only remote command (SSH_ASKPASS, no `sshpass`) |
| [`dev_scripts/prep_fire_model.sh`](dev_scripts/prep_fire_model.sh) | `best.pt` → OpenVINO IR (`models/fire/`) |
| [`dev_scripts/prep_scene_model_llamacpp.sh`](dev_scripts/prep_scene_model_llamacpp.sh) | **ADD-ONLY** fetch of the small GGUF + mmproj into `models/scene/smolvlm2-500m/` (`--bin --llamacpp-tag <tag>` also fetches llama.cpp). Never touches the retained IR. |
| [`dev_scripts/prep_scene_model.sh`](dev_scripts/prep_scene_model.sh) | Fetch the **RETAINED** OpenVINO VLM (Qwen2-VL-2B int4) into `models/scene/`; refuses to clobber without `--force` |
| [`dev_scripts/promote_fire_model.sh`](dev_scripts/promote_fire_model.sh) | Promote a versioned checkpoint to ACTIVE |
| [`dev_scripts/test_fire_model.py`](dev_scripts/test_fire_model.py) | Local fire-model benchmark |
| Other `dev_scripts/*` | dataset build / analysis helpers |

Workflow (**the orchestrator runs on the DEV MACHINE, not the host** — it SSHes
to `ssh.mazr3a.garden` and does everything there as `dr`, the clone owner):
1. Edit + commit locally, then `git push origin master`.
2. Run `./dev_scripts/deploy_all.sh` **from the dev machine**. Remotely it does
   `git pull --ff-only origin master` → `docker compose up -d --build` →
   `docker compose restart`. Needs `DEPLOY_SSH_USER`/`DEPLOY_SSH_PASS` (prompts
   if unset) and never pushes for you.
3. The script verifies stack state, effective Frigate config,
   `firewatch.py --check` and a `--dry-run` pass.

> **A dirty host file blocks the pull.** `git pull --ff-only` refuses when an
> incoming commit touches a file also modified on the host (e.g. the Frigate UI
> rewriting `config/config.yaml`).
>
> **Model prep and runtime checks run ON THE HOST** (they write to / read the
> bind-mounted `models/`).
>
> **scenereader deploy prerequisite:** the models are **not** in git — fetch the
> small default + llama.cpp binaries on the host and point the config at them.
> Verify with `docker compose exec scenereader python /scenereader/scenereader.py --check`.
>
> **`config/places.conf` is an operator input:** without `CAMERA_PLACES` +
> `ADJACENCY` the narrator describes cameras instead of places.

Ownership rules: deploy/git handoff runs as **`dr`** (owner of `/home/dr/frigate`);
the AI must not run git on the host as `ai` nor pull from the host side. The AI
applies local edits and asks the user to push + deploy.

---

## 11. Maintaining this file

**This file is a top-level overview, not a changelog.** Keep it short and
structural. Do **not** add per-change detail, dated verification notes, or
incident history — those go in the relevant `plans/*.md` doc and the code. If a
cell grows past a line or two, move the detail out and leave a pointer.

Update this file **in the same commit** whenever a change alters the *structure*
of the system:
1. [§3 Services](#3-services-docker-compose) — add/remove a service or change its
   image, ports, mounts, volumes or key config.
2. [§4 Host scripts](#4-host-scripts-scripts-deployed-to--run-on-the-host) — a
   script under `scripts/` is added or retired.
3. [§5 Crontab](#5-host-crontab-required) — a cron entry is added/removed (mirror
   it in `scripts/crontab.sample` or `scripts/crontab.root.sample`).
4. [§6 Required settings](#6-required-host-settings) — a new apt package, device,
   mount or daemon.
5. [§7 Configuration inventory](#7-configuration-files-inventory) — new tracked
   config vs new git-ignored secret.
6. [§8 Storage & cleanup](#8-storage-layout--disk-cleanup-heartbeat) — a new
   `config/stores/<service>.conf` profile.
7. [§9 Ports](#9-networking--ports) — a new published or internal port.
8. [§10 Deploy](#10-deploy-workflow) — `dev_scripts/deploy_all.sh` steps or
   verification change.
9. Bump **Summary version / Last updated** in the header table, and bump
   `APP_VERSION` for any portal change (see
   [`.roo/rules/portal-cache-busting.md`](.roo/rules/portal-cache-busting.md)).

---

## 12. Deferred / future phases

- **Phase 1b** — native in-Frigate fire/smoke detection via a fire+smoke+person+car+animal union model (so fire appears in the Frigate UI / MQTT / recorded clips).
- **Phase 2** — *partly done:* cross-camera episode narratives ship via `scenereader` (§3.7) with a deterministic sentence and anonymous person identity; the SQLite event/episode store is included. **Still deferred:** the portal **Episodes** digest + "name this person" assisted labeling, **person attributes** (CLIP zero-shot clothing, daylight-only), appearance ReID as a link tie-breaker, a tiny text LLM to polish the narrative, and a fine-tuned ≤500M captioner trained on the farm's own captures.
- **Phase 3** — **real names** via face recognition (InsightFace SCRFD + ArcFace against a named gallery, assisted by manual naming first) and gait. Gait is not feasible on the 640×360 detect substream.
