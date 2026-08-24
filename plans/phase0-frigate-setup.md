# Phase 0 — Frigate NVR Baseline Deployment

## 1. Goal

Stand up a working **Frigate NVR** deployment on the Debian host (`dr@debian`, dir `~/frigate`) that ingests all **10 cameras** with dual RTSP streams, detects motion/objects via the **OpenVINO CPU detector**, and publishes events to **Mosquitto MQTT** — with **no AI features yet**.

All later phases (fire/smoke YOLOv8, Ollama LLaVA descriptions, daily summary, face/gait profiling) are explicitly **deferred** per the user's direction. The stack and MQTT plumbing are chosen now so those phases can be added without reworking the base.

## 2. Context & Confirmed Facts

| Item | Value |
|---|---|
| Host CPU | Intel Core i7-9700 (8 cores) |
| Host RAM | 7.5 GiB total, ~7.0 GiB available |
| Detector hardware | None dedicated → **OpenVINO on CPU** |
| Cameras | 10 total, IPs `192.168.1.200` – `192.168.1.209` |
| 9 UNV/VCP cameras | Main `/media/video1` = HEVC 3200x1800 @ 20fps; Sub `/media/video2` = HEVC 640x360 @ 12fps |
| Hikvision cam08 (`.207`) | Main `/Streaming/Channels/101` = HEVC 2560x1440 @ 20fps (verified via ffprobe); Sub `/Streaming/Channels/102` (assumed, auto-detected) |
| Credentials | Same user/pass for all cameras; stored in `.env`, never hard-coded in YAML |
| Deployment model | Files built in local workspace `mazr3a`, transferred over SSH to `~/frigate` on the Debian host |

### RAM consideration
7.5 GiB is tight. Phase 0 (Frigate + Mosquitto) fits comfortably. When a VLM is added later, it must be a small quantized model (e.g. Qwen2-VL-based `minicpm-v:2.6` 3B or `moondream` 2B), configurable via `.env`. This is recorded here so the future phase does not over-provision.

## 3. Architecture (Phase 0)

```mermaid
flowchart LR
    A[UNV Cameras cam01-cam10] --> B[RTSP Substream video2 640x360 H265]
    A --> C[RTSP Main video1 3200x1800 H265]
    B --> E[Frigate NVR Docker]
    C --> E
    E --> F[OpenVINO CPU Detector]
    E --> G[Frigate Web UI port 5000]
    E --> H[Mosquitto MQTT port 1883]
    E --> I[media volume Recordings and Snapshots]
```

Note: cam08 Hikvision uses `Streaming/Channels/101` and `102` for main/sub instead of `/media/video1|video2`.

## 4. Deliverables (files to create in this workspace)

| File | Content |
|---|---|
| `docker-compose.yml` | Frigate service (`ghcr.io/blakeblackshear/frigate:stable`, ports `5000`/`8971`, `config/` + `media/` mounts, `env_file: .env`, `restart: unless-stopped`) + Mosquitto service (`eclipse-mosquitto:2`, port `1883`, volume for config/persist) |
| `config/config.yaml` | MQTT block (`host: mqtt`, `port: 1883`); `detectors` → OpenVINO on CPU; `objects` → person, car; `record` + `snapshots` on events only; 10 camera blocks. Named `config.yaml` because Frigate 0.17+ ignores `frigate.yml` |
| `.env` | `RTSP_USER`, `RTSP_PASS` placeholders; camera IP list documented in comments |
| `.gitignore` | Ignore `.env`, `media/`, `*.db` |
| `README.md` | SSH transfer to `~/frigate`, `docker compose up -d`, verification checklist, optional QSV note |

### config.yaml camera block shape (UNV)
```yaml
cam01:
  enabled: true
  ffmpeg:
    inputs:
      - path: rtsp://!env_var RTSP_USER:!env_var RTSP_PASS@192.168.1.200:554/media/video1
        roles: [record]
      - path: rtsp://!env_var RTSP_USER:!env_var RTSP_PASS@192.168.1.200:554/media/video2
        roles: [detect]
  detect:
    width: 640
    height: 360
    fps: 5
  objects:
    track:
      - person
      - car
```
cam08 uses `/Streaming/Channels/101` (main) and `/Streaming/Channels/102` (detect, no fixed size → auto-detected).

## 5. Design Decisions

1. **OpenVINO CPU detector** — only option without GPU/Coral; sufficient for 10 low-res substreams at 5fps.
2. **`!env_var` credentials** — Frigate supports env substitution; keeps secrets out of YAML and git.
3. **Event-only record + snapshots** — no continuous recording; low disk/CPU. Main stream read only on demand.
4. **Detect fps capped at 5** — reduces CPU load; 12fps source is fine.
5. **Software decode first** — if CPU saturates, add Intel QSV via `/dev/dri` (documented as optional follow-up, not in initial config).
6. **Mosquitto included now** — it is the transport Frigate uses for events and the foundation for Phases 1–3; adding it later would require touching the compose stack anyway.

## 6. Deployment Steps (on the Debian host, via SSH)

```bash
# from the local machine, transfer the project
scp -r docker-compose.yml .env config/ dr@debian:~/frigate/

# on the host
cd ~/frigate
mkdir -p media
docker compose config          # validate YAML
docker compose up -d           # start Frigate + Mosquitto
docker compose logs -f frigate # watch startup
```

## 7. Verification Checklist

- [ ] `docker compose config` exits without errors
- [ ] `docker compose ps` shows both `frigate` and `mosquitto` running
- [ ] Frigate UI loads at `http://<host-ip>:5000`
- [ ] All 10 cameras show live detection frames with person/car boxes (allowing ~1 min warm-up)
- [ ] No persistent `ERROR` lines in `docker compose logs frigate` for stream decode
- [ ] (Optional) Mosquitto reachable: `mosquitto_sub -h localhost -p 1883 -t 'frigate/#'` on the host shows event JSON when motion is triggered
- [ ] (Optional) If cam08 shows no feed, verify substream: `ffprobe -rtsp_transport tcp -v error -show_entries stream=codec_name,width,height -of csv "rtsp://admin:PASS@192.168.1.207:554/Streaming/Channels/102"`

## 8. Deferred (future phases, per user direction)

- **Phase 1:** Fire/smoke YOLOv8 detection + instant Telegram/webhook alert with high-res snapshot
- **Phase 2:** Ollama VLM event descriptions + SQLite logging + cron daily summary
- **Phase 3:** Person attributes, gait, and identity profiling (DeepFace / YOLOv8-Pose)

These will reuse the same compose stack, MQTT topic (`frigate/events`), and snapshot output established in Phase 0.
