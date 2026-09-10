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
| Summary version | `v2` |
| Last updated | 2026-09-10 |
| Repo | `https://github.com/tabebqena/mazr3a` (branch `master`) |
| Portal `APP_VERSION` | `0.3.14` (see [`portal/app.py`](portal/app.py:40)) — bump on every portal change |

---

## 1. Overview

A farm CCTV / security stack built on **Frigate NVR** as the ingest + detection
core, with purpose-built side services around it:

- **Frigate** decodes 10 IP cameras' low-res detect substreams, runs an OpenVINO
  detector, records **event-only** clips, exposes REST/UI and an embedded
  **go2rtc** live restream, and publishes MQTT events.
- **firewatch** runs an out-of-band fire/smoke model on Frigate's already-decoded
  detect frames and sends Telegram photo alerts + stores evidence.
- **telegram-bot** answers `/status`, `/help`, `/start` live from Telegram.
- **logs** is a read-only Docker-logs sidecar for the portal's admin Debug tab.
- **portal** is an authenticated FastAPI + vanilla-JS SPA (login, live view,
  events, fire alerts) published via a Cloudflare Tunnel.
- **mqtt** (Mosquitto) is the event broker.
- Host **cron** runs the unified disk heartbeat, a CPU-temp watchdog, a daily
  health report and a state sampler.

Roadmap context: this is **Phase 0/1a**. Ollama VLM descriptions, daily LLM
summaries and person/gait/face profiling are **deferred** (see
[§10](#10-deferred--future-phases)).

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
| Actual active model | `models/coco/yolo11n.onnx` (COCO 80-class, `yolo-generic`, 640×640) |
| Credentials | `.env` → `{FRIGATE_*}` placeholders (git-ignored) |
| Ports | `5000` HTTP UI/API · `8971` TLS UI · `8554` RTSP restream · `8555` tcp+udp WebRTC |
| Volumes | `./config:/config` · `./media:/media/frigate` · `./models:/models:ro` · tmpfs `/tmp/cache` (1 GB) |
| Device / limits | `/dev/dri/renderD128`; `shm_size: 256mb` |
| Notes | Event-only recording (detect substream); software decode (`ffmpeg.hwaccel_args: []`) |

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
| Purpose | Login (PBKDF2), live view (MSE-over-WebSocket primary, HLS fallback), events/detections, fire alerts, admin Debug tab |
| Dev files | [`portal/app.py`](portal/app.py) (`APP_VERSION` here), [`portal/auth.py`](portal/auth.py), [`portal/config.py`](portal/config.py), [`portal/frigate.py`](portal/frigate.py), [`portal/firestore.py`](portal/firestore.py), [`portal/genpass.py`](portal/genpass.py), [`portal/static/`](portal/static/) (SPA: `index.html`, `app.js`, `style.css`, `favicon.svg`, `vendor/hls.min.js`), [`portal/requirements.txt`](portal/requirements.txt) |
| Config | [`config/portal.conf`](config/portal.conf) (git-ignored; template [`config/portal.conf.example`](config/portal.conf.example)) |
| Ports | `8080` (internal host port for the Cloudflare Tunnel) |
| Volumes | `./portal:/srv/app/portal:ro` · `./config:/config:ro` · `./media:/media` (rw for SQLite WAL read) |
| Env | `PORTAL_CONF`, `PORTAL_LOGS_API=http://logs:8090` |
| Notes | **Cache-busting policy:** bump `APP_VERSION` + use `{{ ASSET_* }}` tokens (see [`.roo/rules/portal-cache-busting.md`](.roo/rules/portal-cache-busting.md)). Edits need `docker compose restart portal`. |

### 3.7 Service → development-file map (quick lookup)
| Service | Primary code / config in repo |
|---|---|
| `frigate` | [`config/config.yaml`](config/config.yaml), [`models/coco/`](models/coco/), `docker-compose.yml`, `.env` |
| `mqtt` | [`mosquitto/config/mosquitto.conf`](mosquitto/config/mosquitto.conf) |
| `firewatch` | [`firewatch/firewatch.py`](firewatch/firewatch.py), [`config/firewatch.conf`](config/firewatch.conf), [`models/fire/`](models/fire/), [`scripts/telegram_notify.py`](scripts/telegram_notify.py) |
| `telegram-bot` | [`scripts/telegram_bot.py`](scripts/telegram_bot.py), [`scripts/telegram_notify.py`](scripts/telegram_notify.py) |
| `logs` | [`scripts/container_logs.py`](scripts/container_logs.py) |
| `portal` | [`portal/`](portal/) + [`config/portal.conf`](config/portal.conf) |

---

## 4. Host scripts (`scripts/`, deployed to & run on the host)

| File | Runs as | Purpose |
|---|---|---|
| [`heartbeat_cleanup.py`](scripts/heartbeat_cleanup.py) | root cron (15 min) | **Unified disk heartbeat** — per-store compliance + global-cap escalation across `config/stores/*.conf`. Modes: `--check`, `--dry-run` |
| [`cleanup_firewatch_store.py`](scripts/cleanup_firewatch_store.py) | in-container worker | firewatch DB-aware evidence cleanup (invoked by the heartbeat's `TYPE=docker-exec` store) |
| [`cleanup_media.sh`](scripts/cleanup_media.sh) | — | **SUPERSEDED** (replaced by the heartbeat) |
| [`collect_sensors.py`](scripts/collect_sensors.py) | library | lm-sensors reading helpers |
| [`machine-monitor.py`](scripts/machine-monitor.py) | `dr` cron (1 min) | CPU-temp watchdog (CRITICAL + WARM tiers) → Telegram |
| [`machine-status.py`](scripts/machine-status.py) | `dr` cron (daily 08:00) | Daily host + Frigate health report → Telegram |
| [`watchdog_baseline.py`](scripts/watchdog_baseline.py) | `dr` cron (1 min, `--cron`) | Machine/Frigate state sampler → size-capped CSV |
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
| `* * * * *` | `/usr/bin/python3 /home/dr/frigate/scripts/machine-monitor.py` | CPU-temp watchdog (every-minute sampling is required for ~1-min hot spikes) |
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
| [`config/config.yaml`](config/config.yaml) | Frigate 0.17 (cameras, go2rtc, model, detector, record, motion) |
| [`config/firewatch.conf`](config/firewatch.conf) | firewatch tunables (motion gate, thresholds, store) |
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
| `media/` | Frigate recordings/clips/snapshots + firewatch evidence + watchdog CSVs |
| `mosquitto/data/`, `mosquitto/log/` | Broker runtime state |
| `models/fire/versions/` | Versioned fire-model archive |
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
| [`dev_scripts/promote_fire_model.sh`](dev_scripts/promote_fire_model.sh) | Promote a versioned checkpoint to ACTIVE |
| [`dev_scripts/test_fire_model.py`](dev_scripts/test_fire_model.py) | Local fire-model benchmark |
| Other `dev_scripts/*` | dataset build / analysis helpers |

Workflow:
1. Edit + commit locally, then `git push origin master`.
2. Run `./dev_scripts/deploy_all.sh` (host syncs to `origin/master`; every service is restarted).
3. The script verifies stack state, effective Frigate config, `firewatch.py --check` and a `--dry-run` pass.

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
- **Phase 2** — Ollama VLM scene descriptions + SQLite event log + cron daily summary.
- **Phase 3** — person attributes, gait and identity profiling (DeepFace / YOLOv8-Pose).
