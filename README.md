# Frigate NVR — Edge Security System (Phase 0)

Baseline deployment of **Frigate NVR** + **Mosquitto MQTT** on a Debian host with
10 IP cameras (9 UNV/VCP + 1 Hikvision). Fire/smoke detection is provided by an
external **firewatch** service with instant Telegram photo alerts
([`plans/fire-detection.md`](plans/fire-detection.md)). The remaining AI features
(Ollama descriptions, daily summaries, face/gait profiling, a native in-Frigate
fire model) stay deferred to later phases.

> **System map:** [`SYSTEM_SUMMARY.md`](SYSTEM_SUMMARY.md) is the maintained
> inventory of every service, its development-file locations, the host crontab,
> required host settings and the deploy workflow. Update it whenever a service
> changes (see [`.roo/rules/system-summary.md`](.roo/rules/system-summary.md)).

## Hardware / Environment

| Item | Value |
|---|---|
| Host | Debian (SSH), user `dr`, deploy dir `~/frigate` |
| CPU | Intel Core i7-9700 (8 cores) |
| RAM | 7.5 GiB |
| Detector | OpenVINO on the iGPU (device: GPU, Intel UHD 630) — offloads inference off the CPU cores (2026-09-09) |
| Decode | Software decode (VAAPI/QSV disabled — it caused "Invalid data" on the main record stream) |
| Cameras | 10 × `192.168.1.200` – `192.168.1.209` |

## File Layout

```
.
├── docker-compose.yml          # Frigate + Mosquitto + firewatch services
├── .env.example                # RTSP credentials template (commit this)
├── .env                        # RTSP credentials (git-ignored; copy from .env.example)
├── config/
│   ├── config.yaml             # Frigate config (0.17 name): 10 cameras, OpenVINO, MQTT
│   ├── heartbeat.conf          # Unified disk heartbeat GLOBAL cap (root cron cleaner)
│   ├── stores/                 # Per-service cleanup profiles (one <service>.conf each)
│   │   ├── frigate.conf        #   Frigate media (recordings/clips/snapshots/cache)
│   │   ├── firewatch.conf      #   firewatch evidence (DB-aware, docker-exec delegate)
│   │   ├── watchdog.conf       #   watchdog baseline CSVs (media/watchdog/)
│   │   ├── mosquitto.conf      #   mosquitto data/log
│   │   └── logs.conf           #   host deploy-root runtime logs
│   ├── cleanup_firewatch.conf  # firewatch in-container DB cleanup tunables (worker)
│   ├── cleanup_media.conf      # SUPERSEDED 2026-09-09 (was cleanup_media.sh tunables)
│   ├── firewatch.conf          # Fire-watch tunables (cameras, cadence, thresholds)
│   └── telegram.conf           # Telegram bot creds (git-ignored; example in repo)
├── scripts/                    # HOST scripts - deployed to & run on the Frigate host
│   ├── heartbeat_cleanup.py    # UNIFIED disk heartbeat (per-service + global cap; root cron)
│   ├── cleanup_firewatch_store.py # firewatch DB-aware cleanup worker (in-container, delegated)
│   ├── cleanup_media.sh        # SUPERSEDED 2026-09-09 (replaced by heartbeat_cleanup.py)
│   ├── collect_sensors.py      # lm-sensors reading helpers (host scripts)
│   ├── telegram_notify.py      # Shared Telegram helpers (host scripts + firewatch)
│   ├── crontab.sample          # Sample cron lines to install on the host (root + dr)
│   ├── machine-monitor.py      # CPU-temp watchdog (host cron)
│   ├── machine-status.py       # Daily health report (host cron)
│   ├── diagnose_detection.py   # On-host detection diagnosis (read-only)
│   └── verify_remote.py        # Post-deploy config verification on the host
├── dev_scripts/                # LOCAL scripts - dev/debug + deploy orchestrators only
│   ├── deploy_all.sh           # git-based FULL deploy (configs + firewatch + model)
│   ├── prep_fire_model.sh      # best.pt -> OpenVINO IR (models/fire)
│   ├── promote_fire_model.sh   # Promote a versioned checkpoint to ACTIVE
│   ├── test_fire_model.py      # Local fire-model benchmark
│   └── ...                     # dataset/build/analyze helpers (see plans)
├── firewatch/
│   ├── Dockerfile              # firewatch runtime image (deps only)
│   ├── requirements.txt        # openvino + numpy + Pillow
│   └── firewatch.py            # Fire-watch watcher (runs in the firewatch container)
├── models/
│   ├── coco/                   # Frigate COCO detector ONNX (git-tracked; ACTIVE yolo11n)
│   └── fire/                   # Fire/smoke ACTIVE OpenVINO IR model (git-tracked; versions/ ignored)
├── mosquitto/
│   └── config/mosquitto.conf   # MQTT broker config
├── media/                      # Frigate recordings & snapshots (auto-created)
└── README.md
```

## Camera Stream Map

| Camera | IP | Brand | Main stream | Substream (detect) |
|---|---|---|---|---|
| cam01 | 192.168.1.200 | UNV | `/media/video1` (3200x1800@20) | `/media/video2` (H.264 640x360@12) |
| cam02 | 192.168.1.201 | UNV | `/media/video1` | `/media/video2` (H.264 640x360@12) |
| cam03 | 192.168.1.202 | UNV | `/media/video1` | `/media/video2` (H.264 640x360@12) |
| cam04 | 192.168.1.203 | UNV | `/media/video1` | `/media/video2` (H.264 640x360@12) |
| cam05 | 192.168.1.204 | UNV | `/media/video1` | `/media/video2` (H.264 640x360@12) |
| cam06 | 192.168.1.205 | UNV | `/media/video1` | `/media/video2` (H.264 640x360@12) |
| cam07 | 192.168.1.206 | UNV | `/media/video1` | `/media/video2` (H.264 640x360@12) |
| cam08 | 192.168.1.207 | Hikvision | `/Streaming/Channels/101` (2560x1440@20) | `/Streaming/Channels/102` (H.264 640x360@12) |
| cam09 | 192.168.1.208 | UNV | `/media/video1` | `/media/video2` (H.264 640x360@12) |
| cam10 | 192.168.1.209 | UNV | `/media/video1` | `/media/video2` (H.264 640x360@12) |

> **Note (2026-08-27):** all 10 substreams were switched from H.265/HEVC to **H.264 with
> U-Code (smart coding) disabled** so Frigate can decode them cleanly (no more
> `VPS 0 does not exist` / `Invalid data` / `Discarding`). This raises LAN bandwidth
> slightly (negligible — same network); remote/mobile viewers use a separate
> low-bandwidth transcoded stream instead (see Path B in
> [`plans/improve-camera-substreams.md`](plans/improve-camera-substreams.md)).

## Prerequisites (Debian host)

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-plugin
sudo systemctl enable --now docker
```

## Deploy (git-based)

Deploys are **git-based** (2026-09-05): the repo lives at
`https://github.com/tabebqena/mazr3a` (branch `master`), and the host deploy dir
`/home/dr/frigate` is a clone of it. Local edits are committed → pushed to `origin`,
and the deploy script SSHes to the host and runs `git pull --ff-only`. Git handles
adds/edits/moves/deletes, and the ACTIVE fire model
(`models/fire/best.xml/best.bin/best.pt/labelmap.txt`) is git-tracked so it rides
the pull too. One orchestrator, [`dev_scripts/deploy_all.sh`](dev_scripts/deploy_all.sh),
runs every deploy step on the host (the old `deploy_config.sh` /
`deploy_firewatch.sh` were removed - no more scopes):

```bash
# from this workspace - one-time: add origin (usually already set)
git remote add origin https://github.com/tabebqena/mazr3a

# one-time ONLY on a fresh host (done by YOU, not the deploy script):
# make /home/dr/frigate a clone of origin/master - git init + remote add
# origin + fetch + hard reset. This must NOT touch the git-ignored host
# state (.env, config/telegram.conf, media/, mosquitto data/log,
# models/fire/versions/).

# normal deploy: commit locally, then run the FULL deploy (no subcommands):
./dev_scripts/deploy_all.sh            # configs + firewatch + model
```

### 1. Set credentials in `.env` (host, git-ignored)

Frigate 0.17 reads the camera credentials from **`FRIGATE_`-prefixed** environment
variables, referenced as `{FRIGATE_...}` placeholders in
[`config/config.yaml`](config/config.yaml). See
[RTSP credentials — how it works](#rtsp-credentials--how-it-works-frigate-017).
Copy the template and set the real camera password on the host:

```bash
cd ~/frigate
cp .env.example .env
nano .env    # set FRIGATE_RTSP_USER and FRIGATE_RTSP_PASS
```

No secrets ever live in [`config/config.yaml`](config/config.yaml) or in git — only
in the git-ignored `.env`.

### 2. First-time start on the host (after bootstrap)

```bash
cd ~/frigate
rm -f config/frigate.yml   # Frigate 0.17 reads config.yaml only; drop the deprecated name
mkdir -p media
docker compose config          # validate the compose + env files
docker compose up -d           # start Frigate and Mosquitto
sudo crontab scripts/crontab.root.sample  # ROOT crontab: unified disk heartbeat (every 15 min)
crontab scripts/crontab.sample            # dr crontab: temp watchdog + daily report + sampler
crontab -l; sudo crontab -l               # confirm both crontabs
# sanity-check the heartbeat can reach every store dir (root):
sudo /usr/bin/python3 /home/dr/frigate/scripts/heartbeat_cleanup.py --check
docker compose logs -f frigate # watch startup; Ctrl+C to stop following
```

## RTSP credentials — how it works (Frigate 0.17)

Frigate **0.17 removed the old `!env_var` YAML tag**. Environment substitution is
done by pydantic's `str.format()` on the camera **ffmpeg input path**, using only
variables whose names start with **`FRIGATE_`** (confirmed in the v0.17.x source:
[`frigate/config/env.py`](https://github.com/blakeblackshear/frigate/blob/v0.17.2/frigate/config/env.py)
→ `EnvString`, applied to `CameraInput.path`).

Every UNV camera path in [`config/config.yaml`](config/config.yaml) therefore uses:

```yaml
- path: rtsp://{FRIGATE_RTSP_USER}:{FRIGATE_RTSP_PASS}@192.168.1.200:554/media/video2
```

The `FRIGATE_RTSP_USER` / `FRIGATE_RTSP_PASS` values come from `.env`, which
[`docker-compose.yml`](docker-compose.yml) loads into the `frigate` container via
`env_file: .env`.

**cam08 (Hikvision) uses separate credentials.** Its admin password differs from
the UNV cameras, so `cam08`'s path in `config/config.yaml` uses
`{FRIGATE_HIK_RTSP_USER}` / `{FRIGATE_HIK_RTSP_PASS}` instead of the shared pair.
Without these set to the real Hikvision password, ffmpeg fails with
`DESCRIBE failed: 401 Unauthorized` and cam08 crash-loops. Set both variables in
`.env` (see `.env.example`).

**Do not regress to any of these:**

| Wrong form | Why it breaks |
|---|---|
| `rtsp://admin:CHANGE_ME@...` (plaintext in `config.yaml`) | Commits a secret; Frigate substitutes nothing, and a real password leaks into git history |
| `rtsp://!env_var{RTSP_USER}:...` | 0.17 does not support the `!env_var` tag — the literal text `!env_var{RTSP_USER}` lands in the URL and ffmpeg fails with 401 |
| `rtsp://{RTSP_USER}:{RTSP_PASS}@...` (no `FRIGATE_` prefix) | Frigate only substitutes `FRIGATE_`-prefixed names; the unknown key makes config validation fail (`KeyError`) and cameras never come up |

To change credentials: edit `.env`, then recreate the container with
`docker compose up -d --force-recreate frigate`. Remember cam08 uses the separate
`FRIGATE_HIK_RTSP_USER` / `FRIGATE_HIK_RTSP_PASS` variables.

## Recording: event-only video clips + disk-cap cleanup

Recording is **event-only**: Frigate writes a video clip from the low-res 640×360
detect substream **only when motion / a tracked object is detected**
(`record.enabled: true`, `retain.mode: active_objects`, `pre_capture: 3` in
[`config/config.yaml`](config/config.yaml)). No 24/7 continuous recording, and the
high-res main streams are **not** pulled — they overwhelmed the CPU for a record role
and produced invalid segments. Detection, live view, snapshots, and event clips all
work from the detect stream.

Clips expire two ways:
- **By age** — Frigate `record.retain.default: 7` days (snapshots keep 14).
- **By disk cap** — the unified disk heartbeat
  ([`scripts/heartbeat_cleanup.py`](scripts/heartbeat_cleanup.py), host root cron every
  15 min) replaces the old `cleanup_media.sh`. Each locally-writing service declares its
  own store profile under [`config/stores/`](config/stores/) (`frigate.conf`,
  `firewatch.conf`, `watchdog.conf`, `mosquitto.conf`, `logs.conf`) telling the heartbeat
  how to clean its files — by `MAX_SIZE_GB` (oldest-first while over the cap) and/or
  `MAX_AGE_DAYS` (delete older than N days), scoped to that service's own dirs so no store
  ever touches another's files. A **global cap** in
  [`config/heartbeat.conf`](config/heartbeat.conf) (`MIN_FREE_GB` / `MAX_USED_PCT` +
  `RELIEF_FREE_GB`) makes the heartbeat escalate across stores in `PRIORITY` order when the
  disk is nearly full. firewatch evidence stays DB-aware: its profile is a `docker-exec`
  delegate that calls the in-container `cleanup_firewatch_store.py` worker (the SQLite DB
  is the source of truth). Run `scripts/heartbeat_cleanup.py --check` for a per-store
  permission/usage table, `--dry-run` to preview deletions.

**If high-res (3200×1800) event clips are wanted later** (needs hardware decode to
keep CPU low):
1. Add the main stream back to each camera with `roles: [record]` (UNV `/media/video1`, Hikvision `/Streaming/Channels/101`)
2. Keep `record.enabled: true` — event clips then come from the main stream
3. Add `hwaccel_args: preset-vaapi` per camera and keep the `devices:` block in `docker-compose.yml`

## Detector: OpenVINO model input size (required)

The bundled OpenVINO model `ssdlite_mobilenet_v2.xml` expects a **300×300** input tensor.
Three things in [`config/config.yaml`](config/config.yaml) must all be correct or the
detect toggles in the UI break:

```yaml
version: 0.17-0          # 1) REQUIRED — see below

model:                   # 2) REQUIRED — top-level model block
  path: /openvino-model/ssdlite_mobilenet_v2.xml
  labelmap_path: /labelmap.txt
  width: 300
  height: 300

detectors:
  ov:
    type: openvino
    device: CPU
    model_path: /openvino-model/ssdlite_mobilenet_v2.xml   # 3) flat model_path (optional override)
```

**1) `width`/`height` MUST live in the top-level `model:` block — NOT under the detector.**
Frigate 0.17 explicitly discards a per-detector `model:` block
(`if detector_config.model: detector_config.model = None` — "users should not set model
themselves" in `frigate/config/config.py`) and builds the detector's model from the
top-level `model:` block, with the detector's flat `model_path:` only overriding the path.
If `width`/`height` are missing, Frigate defaults the resize to 320×320 and the first detect
frame crashes the detector with:

```
ValueError: could not broadcast input array from shape (1,320,320,3) into shape (1,300,300,3)
[INFO] Detection appears to have stopped. Exiting Frigate...
```

Frigate then crash-loops: detection never stays enabled, the detect toggles stay unchecked,
and clicking one freezes/blackens the page while the container restarts. When a custom model
is used later (see `plans/animal-motion-detection.md`), set `width`/`height` to **that**
model's input size (top-level `model:`).

**2) `version: 0.17-0` prevents the config migration from rewriting the file.** Frigate's
`migrate_frigate_config` reads `config.get("version", "0.13")`; with no `version` it assumes
the file is 0.13 and runs the full 0.13→0.17 migration chain on every fresh start,
**rewriting `config.yaml` and dropping fields it doesn't carry forward**. Keeping
`version: 0.17-0` makes Frigate log "frigate config does not need migration" and leave the
file untouched.

**3) The `record:` block must use the 0.17 schema.** Frigate 0.17 rejects the old
`record.retain` / `record.events` keys ("Extra inputs are not permitted") and falls back to
safe mode with a default config (no cameras). Use `record.alerts.retain.days` /
`record.alerts.pre_capture` and `record.detections.retain.days` /
`record.detections.pre_capture` as in [`config/config.yaml`](config/config.yaml).

Verify after deploying: `curl -s http://<host-ip>:5000/api/config | python3 -m json.tool`
shows `detectors.ov.model.width == 300` and `detectors.ov.model.height == 300`, and
`docker compose logs frigate` shows neither "Your config file is not valid!" nor
"Detection appears to have stopped. Exiting Frigate...".

**4) `detect.enabled` defaults to `false` in Frigate 0.17** (`DetectConfig.enabled:
bool = Field(default=False)`), so each camera block must set
`detect: enabled: true` (as in [`config/config.yaml`](config/config.yaml)) or the
cameras will never send frames to the detector and no events/recordings will be
produced. To detect objects, a tracked class (e.g. `person`, `car` in `objects.track`)
must actually appear in view; detection on a quiet scene stays at `detection_fps=0`.

**5) Software decode must be forced.** Frigate 0.17 auto-detects VAAPI and applies it
even without `hwaccel_args`; on this host hardware-decoding the HEVC camera streams
crashes ffmpeg (`VPS 0 does not exist` / `Failed to sync surface` /
`Failed to download frame: -5`), breaking the detect streams → zero motion/events.
The global `ffmpeg: hwaccel_args: []` in the config forces software decode (fine for
10 substreams at 5 fps on the i7-9700).

## Verification Checklist

- [ ] `docker compose ps` shows both `frigate` and `mqtt` as `Up`
- [ ] Frigate UI loads at `http://<host-ip>:5000` (allow ~1–2 min warm-up)
- [ ] All 10 cameras show live video with detection boxes for person/car
- [ ] No literal placeholders in URLs:
      `docker compose logs frigate | grep -c '{FRIGATE_'` returns `0`
- [ ] Trigger an event (walk in front of a camera): a **video clip** appears in the UI
      Timeline and under `media/recordings` on the host
- [ ] No persistent `ERROR` lines in `docker compose logs frigate`
- [ ] MQTT events published (optional): on the host run
      `mosquitto_sub -h localhost -p 1883 -t 'frigate/#'`
      then walk in front of a camera — event JSON should appear
- [ ] Disk heartbeat works: `scripts/heartbeat_cleanup.py --check` (as root) shows
      every `config/stores/*.conf` dir readable/writable; `--dry-run` reports what a run
      would delete; lowering `MAX_SIZE_GB` in `config/stores/frigate.conf` (or
      `MIN_FREE_GB` in `config/heartbeat.conf`) deletes oldest files until usage is under
      the cap. `heartbeat-cleanup.log` logs each run.

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI shows no cameras | Frigate 0.17 reads `config/config.yaml` only and ignores `frigate.yml`. If the log says "No config file found, saving default config", the file is misnamed. Fix: `mv config/frigate.yml config/config.yaml && docker compose restart frigate` |
| ffmpeg shows `rtsp://!env_var ...` / `rtsp://{FRIGATE_...}` / 401 / "Invalid data" | Frigate 0.17 substitutes only `{FRIGATE_...}` placeholders from `FRIGATE_`-prefixed env vars — `!env_var` is not supported and non-prefixed names are not substituted. Ensure `.env` defines `FRIGATE_RTSP_USER` / `FRIGATE_RTSP_PASS`, `config/config.yaml` uses `rtsp://{FRIGATE_RTSP_USER}:{FRIGATE_RTSP_PASS}@...`, then recreate: `docker compose up -d --force-recreate frigate` |
| A camera shows "no video" | Verify stream from the host:
`ffprobe -rtsp_transport tcp -v error -show_entries stream=codec_name,width,height -of csv "rtsp://admin:PASS@IP:554/media/video1"` |
| Hikvision cam08 crash-loops with `DESCRIBE failed: 401 Unauthorized` | The Hikvision admin password differs from the UNV cameras — the shared `FRIGATE_RTSP_PASS` is rejected. Set the real Hikvision credentials as `FRIGATE_HIK_RTSP_USER` / `FRIGATE_HIK_RTSP_PASS` in `.env` (cam08 already uses these in `config/config.yaml`), then `docker compose up -d --force-recreate frigate`. Verify first from the host: `ffprobe -rtsp_transport tcp -v error -show_entries stream=codec_name,width,height -of csv "rtsp://admin:HIK_PASS@192.168.1.207:554/Streaming/Channels/102"` |
| `Invalid data found when processing input` / `Invalid or missing video stream in segment ... Discarding` for cam01/02/03/09/10 only | **Resolved (2026-08-27):** the failing cameras' HEVC substreams sent the VPS parameter set in-band, which ffmpeg could not read (`VPS 0 does not exist`). Fixed camera-side by setting each substream `/media/video2` to **H.264 with U-Code disabled** (all 10 cameras now report `codec_name=h264`). If it recurs, check a camera's substream is H.264 (not H.265), U-Code off, enabled and streaming, then `docker compose restart frigate`. Verify from the host: `ffprobe -rtsp_transport tcp -v error -show_entries stream=codec_name,width,height -of csv "rtsp://admin:PASS@192.168.1.20X:554/media/video2"` |
| "Invalid data found when processing input" / discarded recordings | Known with `preset-vaapi` on this setup. Software decode is now the default (no `hwaccel_args` in `config/config.yaml`). If you want to retry QSV later, add `hwaccel_args: preset-vaapi` per camera and keep the `devices:` block in `docker-compose.yml` |
| Detect toggles in the UI never stay checked; clicking one freezes/blackens the page; log shows `ValueError: could not broadcast input array from shape (1,320,320,3) into shape (1,300,300,3)` then `Detection appears to have stopped. Exiting Frigate...` | The model input size is wrong. The bundled `ssdlite_mobilenet_v2` needs `300`×`300`. In `config/config.yaml` put `width: 300` / `height: 300` in the **top-level `model:` block** (Frigate 0.17 discards a per-detector `model:` block), keep `version: 0.17-0` (prevents migration rewriting the file), and use flat `model_path:` on the detector. Also ensure the `record:` block uses the 0.17 `alerts`/`detections` schema (the old `retain`/`events` keys make the config invalid → safe mode with no cameras). Then `docker compose up -d --force-recreate frigate` and verify `detectors.ov.model.width == 300` via `/api/config` |
| High CPU during decode | Software decode of 10 substreams at 5fps is fine on the i7-9700; the high-res main streams are only decoded on demand. If CPU is high, reduce per-camera `detect.fps` from 5 to 3 in `config/config.yaml` |
| MQTT errors in Frigate logs | Confirm the `mqtt` container is running (`docker compose ps`) |
| "`--- Logging error ---` … `BrokenPipeError: [Errno 32] Broken pipe`" spam on Ctrl+C / shutdown | Cosmetic. During shutdown Frigate's internal log queue (a multiprocessing pipe) is closed while camera-maintainer threads still write to it, so each queued record raises `BrokenPipeError` and Python prints `--- Logging error ---`. Harmless — it never affects recordings or shutdown. Stop with `docker compose down` instead of Ctrl+C on `docker compose up`; deploy detached (`docker compose up -d`). A `docker compose pull` to the latest `:stable` image may remove it in newer releases. The related `resource_tracker: ... leaked semaphore objects` warning is also harmless cleanup noise. |

## Fire detection — `firewatch` service (Phase 1a)

Fire/smoke detection runs as an **out-of-band watcher**, not inside Frigate's detector
(this avoids regressing the live person/car/animal detection and does not pull the
high-res main streams). See [`plans/fire-detection.md`](plans/fire-detection.md) for the
full design.

How it works:
- The `firewatch` container polls each camera's **already-decoded detect frame** via the
  Frigate REST API (`http://frigate:5000/api/<cam>/latest.jpg`) - no extra ffmpeg decode,
  so the 640x360 substream is sufficient for detection.
- It runs a dedicated **fire/smoke YOLOv8n** model (OpenVINO IR in
  [`models/fire/`](models/fire/README.md)) and sends a **Telegram photo alert** (via
  `send_telegram_photo()` in [`scripts/telegram_notify.py`](scripts/telegram_notify.py))
  after `MIN_HITS` consecutive frames above `SCORE_THRESHOLD`, then cools down per camera.
- The watcher script lives with its image in
  [`firewatch/firewatch.py`](firewatch/firewatch.py); Telegram creds are shared with the
  host-monitoring scripts via the git-ignored [`config/telegram.conf`](config/telegram.conf).
  `CHAT_ID` accepts a **comma-separated list** of recipients (private user ids, group ids,
  channel ids), so every alert/report/photo goes to all of them (see the template
  [`config/telegram.conf.example`](config/telegram.conf.example)).

Operate:
```bash
# build + start (first build downloads pip deps)
docker compose up -d --build firewatch
docker compose logs -f firewatch          # watch polls / alerts
docker compose exec firewatch python /firewatch/firewatch.py --dry-run   # one live pass
docker compose restart firewatch          # apply firewatch.conf / code edits
```

### Live status on demand — `telegram-bot`

A stdlib-only [`telegram-bot`](scripts/telegram_bot.py) compose service long-polls the
Bot API and answers `/status`, `/help` and `/start` **in the chat that asked** — so you
can get a live host + Frigate report in Telegram without waiting for the daily cron or
an alert. It reuses the shared [`scripts/telegram_notify.py`](scripts/telegram_notify.py)
and reads host health through read-only bind mounts (`/proc`, `./media`) plus the
container's native read-only sysfs coretemp (`/sys/class/hwmon` — a bare bind of host
`/sys/class/hwmon` is unusable in-container: its relative hwmon symlinks resolve outside
the mount, see `plans/telegram-bot-cpu-temp-n-a.md`) and the Frigate REST API. Start it
once (no build — stock python image):

```bash
docker compose up -d telegram-bot
docker compose logs -f telegram-bot       # watch command replies
docker compose restart telegram-bot       # apply code / config edits
```

Then in the group (or privately) send `/status@mazr3a_garden_bot`. Commands are
answered only for the configured recipients; restrict to specific user ids with
`ALLOWED_USER_IDS` in [`config/telegram.conf`](config/telegram.conf). Long-polling
must be the only `getUpdates` consumer — no manual `curl` loops while it runs.

Verification steps are in the plan file's checklist.

### Web portal — `portal`

An authenticated **FastAPI + vanilla-JS portal** that consumes the Frigate REST API and the
Firewatch evidence store ([`portal/`](portal/__init__.py)):

- **Login** (multi-user) — credentials live in the git-ignored
  [`config/portal.conf`](config/portal.conf) (PBKDF2 hashes). Copy the committed
  [`config/portal.conf.example`](config/portal.conf.example) and add a user with
  `python portal/genpass.py --username NAME`. Signed HttpOnly session cookie.
- **Live view** — one camera by default with a switcher and a loading spinner while the
  stream connects. Primary: **HLS** playback proxied same-origin through the portal
  (`GET /api/live/<cam>/hls/stream.m3u8` → Frigate's embedded go2rtc HLS, which requires the
  cameras registered in go2rtc via a `go2rtc.streams` block in
  [`config/config.yaml`](config/config.yaml) with `{FRIGATE_*}` credentials), played with
  hls.js (vendored under `portal/static/vendor/`; native HLS on Safari). go2rtc in this
  Frigate build has no MSE — HLS is its reliable TCP/tunnel-friendly live transport. The
  latest detect frame is shown as a poster while HLS connects. If HLS cannot start, the camera
  is classified from Frigate's online flag: an **offline camera** shows a clean "offline"
  notice (no spinner) and auto-recovers when it returns; an **online camera** that is just slow
  keeps auto-retrying HLS (no ~1 fps snapshot feed). An **idle time watch** stops the stream
  after a timeout configured by an admin in [`config/portal.conf`](config/portal.conf)
  (`STREAM_IDLE_TIMEOUT_S`, default 30 s; there is **no in-UI control**); any user activity
  (mouse/touch/keyboard) resets the clock, and the stream also stops if the browser tab is
  hidden. This releases the go2rtc camera pull when nobody is watching. The Live view
  **fills the screen**, remembers the last camera opened, and lets you **swipe/drag** left/right
  to switch cameras.
- **Events & detections** — Frigate events with snapshots/clips, filterable by camera/class
  with **numbered pagination** (Prev/Next + "Page X of Y") and a **time filter** (quick
  presets 1 h/6 h/24 h/7 d/30 d/All or a custom From/To). Filters **auto-refresh**; the
  compact Refresh button reloads manually. (Events are fetched as one bounded set - up to
  5000 - and paged client-side, since the Frigate API has no offset/total.)
- **Fire alerts** — Firewatch evidence frames with a detection-box overlay and an **alerted**
  badge for frames that produced a Telegram alert (the `frames.alerted` flag added by
  `firewatch.py`). The view **defaults to alerts only** (uncheck "alerts only" to browse all
  detections), has **numbered pagination** and the same preset/custom **time filter**, and
  auto-refreshes when any filter changes.

**Public access:** the whole host sits behind a **Cloudflare Tunnel + Access** on
`live.mazr3a.garden`; the tunnel maps the portal root → `host:8080` under the Access policy,
and the portal login gates the dashboard/API on top. Live HLS is same-origin — the portal
proxies Frigate's `/api/go2rtc/*` server-side — so it needs no extra tunnel path mapping and
flows through the portal's own Access/session boundary. Frigate REST/DB and the Firewatch
WAL DB (`./media/firewatch.db`) are reached only server-side by the portal.

```bash
docker compose up -d --build portal
docker compose logs -f portal
docker compose restart portal        # apply code / config edits
```

The portal API is self-documented at `/docs` (OpenAPI). Start the container after creating
`config/portal.conf` on the host (a missing file means login is impossible).

## Deferred (future phases)

- **Phase 1b** — Native in-Frigate fire/smoke detection by swapping the sole detector
  for a fire + smoke + person + car + animal **union** model (combine with the animal
  Part-2 model work) so fire appears in the Frigate UI / MQTT events / recorded clips.
- **Phase 2** — Ollama VLM scene descriptions + SQLite event log + cron daily summary
- **Phase 3** — Person attributes, gait, and identity profiling (DeepFace / YOLOv8-Pose)

These will reuse this same compose stack, the `frigate/events` MQTT topic, and the
event-based snapshots already produced by Frigate.
