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
| Hikvision cam08 crash-loops with `DESCRIBE failed: 401 Unauthorized` | The Hikvision admin password differs from the UNV cameras — the shared `FRIGATE_RTSP_PASS` is rejected. Set the real Hikvision credentials as `FRIGATE_HIK_RTSP_USER` / `FRIGATE_HIK_RTSP_PASS` in `.env` (cam08 already uses these in `config/config.yaml`), then `docker compose up -d --force-recreate frigate`. Verify first from the host: `ffprobe -rtsp_transport tcp -v error -show_entries stream=codec_name,width,height -of csv "rtsp://admin:HIK_PASS@192.168.1.207:554/Streaming/Channels/102"` |
| `Invalid data found when processing input` / `Invalid or missing video stream in segment ... Discarding` for cam01/02/03/09/10 only | Camera-side, not a Frigate config bug (cam04–07 work with the identical config). In each failing camera's web UI check the substream `/media/video2`: ensure it is enabled, set to H.264 (not H.265), and actively streaming; then `docker compose restart frigate`. Verify from the host: `ffprobe -rtsp_transport tcp -v error -show_entries stream=codec_name,width,height -of csv "rtsp://admin:PASS@192.168.1.20X:554/media/video2"` |
| "Invalid data found when processing input" / discarded recordings | Known with `preset-vaapi` on this setup. Software decode is now the default (no `hwaccel_args` in `config/config.yaml`). If you want to retry QSV later, add `hwaccel_args: preset-vaapi` per camera and keep the `devices:` block in `docker-compose.yml` |
| Detect toggles in the UI never stay checked; clicking one freezes/blackens the page; log shows `ValueError: could not broadcast input array from shape (1,320,320,3) into shape (1,300,300,3)` then `Detection appears to have stopped. Exiting Frigate...` | The model input size is wrong. The bundled `ssdlite_mobilenet_v2` needs `300`×`300`. In `config/config.yaml` put `width: 300` / `height: 300` in the **top-level `model:` block** (Frigate 0.17 discards a per-detector `model:` block), keep `version: 0.17-0` (prevents migration rewriting the file), and use flat `model_path:` on the detector. Also ensure the `record:` block uses the 0.17 `alerts`/`detections` schema (the old `retain`/`events` keys make the config invalid → safe mode with no cameras). Then `docker compose up -d --force-recreate frigate` and verify `detectors.ov.model.width == 300` via `/api/config` |
| High CPU during decode | Software decode of 10 substreams at 5fps is fine on the i7-9700; the high-res main streams are only decoded on demand. If CPU is high, reduce per-camera `detect.fps` from 5 to 3 in `config/config.yaml` |
| MQTT errors in Frigate logs | Confirm the `mqtt` container is running (`docker compose ps`) |
| "`--- Logging error ---` … `BrokenPipeError: [Errno 32] Broken pipe`" spam on Ctrl+C / shutdown | Cosmetic. During shutdown Frigate's internal log queue (a multiprocessing pipe) is closed while camera-maintainer threads still write to it, so each queued record raises `BrokenPipeError` and Python prints `--- Logging error ---`. Harmless — it never affects recordings or shutdown. Stop with `docker compose down` instead of Ctrl+C on `docker compose up`; deploy detached (`docker compose up -d`). A `docker compose pull` to the latest `:stable` image may remove it in newer releases. The related `resource_tracker: ... leaked semaphore objects` warning is also harmless cleanup noise. |

## Deferred (future phases)

- **Phase 1** — Fire/smoke YOLOv8 detection + instant Telegram/webhook alert with high-res snapshot
- **Phase 2** — Ollama VLM scene descriptions + SQLite event log + cron daily summary
- **Phase 3** — Person attributes, gait, and identity profiling (DeepFace / YOLOv8-Pose)

These will reuse this same compose stack, the `frigate/events` MQTT topic, and the
event-based snapshots already produced by Frigate.
