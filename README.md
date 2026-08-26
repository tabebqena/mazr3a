# Frigate NVR — Edge Security System (Phase 0)

Baseline deployment of **Frigate NVR** + **Mosquitto MQTT** on a Debian host with
10 IP cameras (9 UNV/VCP + 1 Hikvision). AI features (fire detection, Ollama
descriptions, daily summaries, face/gait profiling) are **deferred** to later
phases and are not part of this baseline.

## Hardware / Environment

| Item | Value |
|---|---|
| Host | Debian (SSH), user `dr`, deploy dir `~/frigate` |
| CPU | Intel Core i7-9700 (8 cores) |
| RAM | 7.5 GiB |
| Detector | OpenVINO on CPU (no GPU/Coral) |
| Decode | Software decode (VAAPI/QSV disabled — it caused "Invalid data" on the main record stream) |
| Cameras | 10 × `192.168.1.200` – `192.168.1.209` |

## File Layout

```
.
├── docker-compose.yml          # Frigate + Mosquitto services
├── .env.example                # RTSP credentials template (commit this)
├── .env                        # RTSP credentials (git-ignored; copy from .env.example)
├── config/
│   ├── config.yaml             # Frigate config (0.17 name): 10 cameras, OpenVINO, MQTT
│   └── cleanup_media.conf      # Cleanup-cron tunables (disk cap, min free space, age)
├── scripts/
│   ├── cleanup_media.sh        # Deletes oldest media when over the disk cap (host cron)
│   └── crontab.sample          # Sample cron line to install on the host
├── mosquitto/
│   └── config/mosquitto.conf   # MQTT broker config
├── media/                      # Frigate recordings & snapshots (auto-created)
└── README.md
```

## Camera Stream Map

| Camera | IP | Brand | Main stream | Substream (detect) |
|---|---|---|---|---|
| cam01 | 192.168.1.200 | UNV | `/media/video1` (3200x1800@20) | `/media/video2` (640x360@12) |
| cam02 | 192.168.1.201 | UNV | `/media/video1` | `/media/video2` |
| cam03 | 192.168.1.202 | UNV | `/media/video1` | `/media/video2` |
| cam04 | 192.168.1.203 | UNV | `/media/video1` | `/media/video2` |
| cam05 | 192.168.1.204 | UNV | `/media/video1` | `/media/video2` |
| cam06 | 192.168.1.205 | UNV | `/media/video1` | `/media/video2` |
| cam07 | 192.168.1.206 | UNV | `/media/video1` | `/media/video2` |
| cam08 | 192.168.1.207 | Hikvision | `/Streaming/Channels/101` (2560x1440@20) | `/Streaming/Channels/102` |
| cam09 | 192.168.1.208 | UNV | `/media/video1` | `/media/video2` |
| cam10 | 192.168.1.209 | UNV | `/media/video1` | `/media/video2` |

## Prerequisites (Debian host)

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-plugin
sudo systemctl enable --now docker
```

## Deploy

### 1. Set credentials in `.env`

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

### 2. Transfer files from this workspace to the host

```bash
# run on your local machine (this workspace)
scp -r docker-compose.yml .env.example config scripts mosquitto dr@debian:~/frigate/
# .env is created on the host in step 1 (do not transfer a local .env; it is git-ignored)
```

### 3. Start on the host

```bash
cd ~/frigate
rm -f config/frigate.yml   # Frigate 0.17 reads config.yaml only; drop the deprecated name
mkdir -p media
docker compose config          # validate the compose + env files
docker compose up -d           # start Frigate and Mosquitto
chmod +x scripts/cleanup_media.sh
crontab scripts/crontab.sample # install disk-cap cleanup (every 30 min)
crontab -l                     # confirm the cleanup entry
docker compose logs -f frigate # watch startup; Ctrl+C to stop following
```

## RTSP credentials — how it works (Frigate 0.17)

Frigate **0.17 removed the old `!env_var` YAML tag**. Environment substitution is
done by pydantic's `str.format()` on the camera **ffmpeg input path**, using only
variables whose names start with **`FRIGATE_`** (confirmed in the v0.17.x source:
[`frigate/config/env.py`](https://github.com/blakeblackshear/frigate/blob/v0.17.2/frigate/config/env.py)
→ `EnvString`, applied to `CameraInput.path`).

Every camera path in [`config/config.yaml`](config/config.yaml) therefore uses:

```yaml
- path: rtsp://{FRIGATE_RTSP_USER}:{FRIGATE_RTSP_PASS}@192.168.1.200:554/media/video2
```

The `FRIGATE_RTSP_USER` / `FRIGATE_RTSP_PASS` values come from `.env`, which
[`docker-compose.yml`](docker-compose.yml) loads into the `frigate` container via
`env_file: .env`.

**Do not regress to any of these:**

| Wrong form | Why it breaks |
|---|---|
| `rtsp://admin:CHANGE_ME@...` (plaintext in `config.yaml`) | Commits a secret; Frigate substitutes nothing, and a real password leaks into git history |
| `rtsp://!env_var{RTSP_USER}:...` | 0.17 does not support the `!env_var` tag — the literal text `!env_var{RTSP_USER}` lands in the URL and ffmpeg fails with 401 |
| `rtsp://{RTSP_USER}:{RTSP_PASS}@...` (no `FRIGATE_` prefix) | Frigate only substitutes `FRIGATE_`-prefixed names; the unknown key makes config validation fail (`KeyError`) and cameras never come up |

To change credentials: edit `.env`, then recreate the container with
`docker compose up -d --force-recreate frigate`.

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
- **By disk cap** — a host cron runs
  [`scripts/cleanup_media.sh`](scripts/cleanup_media.sh) every 30 min and deletes the
  **oldest** media files whenever usage exceeds the configurable cap in
  [`config/cleanup_media.conf`](config/cleanup_media.conf) (`MAX_MEDIA_GB`,
  `MIN_FREE_GB`, `MAX_AGE_DAYS`). Edit the values and re-deploy the file to the host.

**If high-res (3200×1800) event clips are wanted later** (needs hardware decode to
keep CPU low):
1. Add the main stream back to each camera with `roles: [record]` (UNV `/media/video1`, Hikvision `/Streaming/Channels/101`)
2. Keep `record.enabled: true` — event clips then come from the main stream
3. Add `hwaccel_args: preset-vaapi` per camera and keep the `devices:` block in `docker-compose.yml`

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
- [ ] Cleanup cron works: `scripts/cleanup_media.sh` logs a run to
      `media-cleanup.log`; temporarily lowering `MAX_MEDIA_GB` in
      `config/cleanup_media.conf` deletes the oldest files until usage drops back under
      the cap

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI shows no cameras | Frigate 0.17 reads `config/config.yaml` only and ignores `frigate.yml`. If the log says "No config file found, saving default config", the file is misnamed. Fix: `mv config/frigate.yml config/config.yaml && docker compose restart frigate` |
| ffmpeg shows `rtsp://!env_var ...` / `rtsp://{FRIGATE_...}` / 401 / "Invalid data" | Frigate 0.17 substitutes only `{FRIGATE_...}` placeholders from `FRIGATE_`-prefixed env vars — `!env_var` is not supported and non-prefixed names are not substituted. Ensure `.env` defines `FRIGATE_RTSP_USER` / `FRIGATE_RTSP_PASS`, `config/config.yaml` uses `rtsp://{FRIGATE_RTSP_USER}:{FRIGATE_RTSP_PASS}@...`, then recreate: `docker compose up -d --force-recreate frigate` |
| A camera shows "no video" | Verify stream from the host:
`ffprobe -rtsp_transport tcp -v error -show_entries stream=codec_name,width,height -of csv "rtsp://admin:PASS@IP:554/media/video1"` |
| Hikvision cam08 no feed | Check substream: `ffprobe ... -of csv "rtsp://admin:PASS@192.168.1.207:554/Streaming/Channels/102"` |
| "Invalid data found when processing input" / discarded recordings | Known with `preset-vaapi` on this setup. Software decode is now the default (no `hwaccel_args` in `config/config.yaml`). If you want to retry QSV later, add `hwaccel_args: preset-vaapi` per camera and keep the `devices:` block in `docker-compose.yml` |
| High CPU during decode | Software decode of 10 substreams at 5fps is fine on the i7-9700; the high-res main streams are only decoded on demand. If CPU is high, reduce per-camera `detect.fps` from 5 to 3 in `config/config.yaml` |
| MQTT errors in Frigate logs | Confirm the `mqtt` container is running (`docker compose ps`) |
| "`--- Logging error ---` … `BrokenPipeError: [Errno 32] Broken pipe`" spam on Ctrl+C / shutdown | Cosmetic. During shutdown Frigate's internal log queue (a multiprocessing pipe) is closed while camera-maintainer threads still write to it, so each queued record raises `BrokenPipeError` and Python prints `--- Logging error ---`. Harmless — it never affects recordings or shutdown. Stop with `docker compose down` instead of Ctrl+C on `docker compose up`; deploy detached (`docker compose up -d`). A `docker compose pull` to the latest `:stable` image may remove it in newer releases. The related `resource_tracker: ... leaked semaphore objects` warning is also harmless cleanup noise. |

## Deferred (future phases)

- **Phase 1** — Fire/smoke YOLOv8 detection + instant Telegram/webhook alert with high-res snapshot
- **Phase 2** — Ollama VLM scene descriptions + SQLite event log + cron daily summary
- **Phase 3** — Person attributes, gait, and identity profiling (DeepFace / YOLOv8-Pose)

These will reuse this same compose stack, the `frigate/events` MQTT topic, and the
event-based snapshots already produced by Frigate.
