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
├── .env                        # RTSP credentials (git-ignored, EDIT THIS)
├── config/
│   └── config.yaml             # Frigate config (0.17 name): 10 cameras, OpenVINO, MQTT
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

### 1. Set credentials (inline in config.yaml)

Frigate 0.17 does **not** substitute `!env_var` tags, so the camera credentials
are written inline in [`config/config.yaml`](config/config.yaml). After
transferring the file, set the real camera password on the host:

```bash
cd ~/frigate
sed -i 's/CHANGE_ME/YOUR_REAL_PASSWORD/' config/config.yaml
grep -c CHANGE_ME config/config.yaml   # expect: 0
```

### 2. Transfer files from this workspace to the host

```bash
# run on your local machine (this workspace)
scp -r docker-compose.yml .env config mosquitto dr@debian:~/frigate/
```

### 3. Start on the host

```bash
cd ~/frigate
rm -f config/frigate.yml   # Frigate 0.17 reads config.yaml only; drop the deprecated name
mkdir -p media
docker compose config          # validate the compose + env files
docker compose up -d           # start Frigate and Mosquitto
docker compose logs -f frigate # watch startup; Ctrl+C to stop following
```

## Phase 0 scope: recording disabled

To keep CPU usage low on the 8-core i7-9700, Phase 0 **pulls only the 640×360 detect substreams** and sets `record.enabled: false` in [`config/config.yaml`](config/config.yaml). The high-res main streams (3200×1800 HEVC @20fps) overwhelmed the CPU for the record role and produced invalid segments. Detection, live view, and snapshots (from the detect stream) all still work.

**To re-enable recording later** (ideally with hardware decode to keep CPU low):
1. Add the main stream back to each camera with `roles: [record]` (UNV `/media/video1`, Hikvision `/Streaming/Channels/101`)
2. Set `record.enabled: true` (and optionally `retain.days`)
3. If CPU is high, add `hwaccel_args: preset-vaapi` per camera and keep the `devices:` block in `docker-compose.yml`

## Verification Checklist

- [ ] `docker compose ps` shows both `frigate` and `mqtt` as `Up`
- [ ] Frigate UI loads at `http://<host-ip>:5000` (allow ~1–2 min warm-up)
- [ ] All 10 cameras show live video with detection boxes for person/car
- [ ] No persistent `ERROR` lines in `docker compose logs frigate`
- [ ] MQTT events published (optional): on the host run
      `mosquitto_sub -h localhost -p 1883 -t 'frigate/#'`
      then walk in front of a camera — event JSON should appear

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI shows no cameras | Frigate 0.17 reads `config/config.yaml` only and ignores `frigate.yml`. If the log says "No config file found, saving default config", the file is misnamed. Fix: `mv config/frigate.yml config/config.yaml && docker compose restart frigate` |
| ffmpeg shows `rtsp://!env_var ...` / 401 / "Invalid data" | Frigate 0.17 does not substitute `!env_var` — the literal tag went into the URL. Put the credentials inline in `config/config.yaml` (`rtsp://admin:PASS@...`) and recreate: `docker compose up -d --force-recreate frigate` |
| A camera shows "no video" | Verify stream from the host:
`ffprobe -rtsp_transport tcp -v error -show_entries stream=codec_name,width,height -of csv "rtsp://admin:PASS@IP:554/media/video1"` |
| Hikvision cam08 no feed | Check substream: `ffprobe ... -of csv "rtsp://admin:PASS@192.168.1.207:554/Streaming/Channels/102"` |
| "Invalid data found when processing input" / discarded recordings | Known with `preset-vaapi` on this setup. Software decode is now the default (no `hwaccel_args` in `config/config.yaml`). If you want to retry QSV later, add `hwaccel_args: preset-vaapi` per camera and keep the `devices:` block in `docker-compose.yml` |
| High CPU during decode | Software decode of 10 substreams at 5fps is fine on the i7-9700; the high-res main streams are only decoded on demand. If CPU is high, reduce per-camera `detect.fps` from 5 to 3 in `config/config.yaml` |
| MQTT errors in Frigate logs | Confirm the `mqtt` container is running (`docker compose ps`) |

## Deferred (future phases)

- **Phase 1** — Fire/smoke YOLOv8 detection + instant Telegram/webhook alert with high-res snapshot
- **Phase 2** — Ollama VLM scene descriptions + SQLite event log + cron daily summary
- **Phase 3** — Person attributes, gait, and identity profiling (DeepFace / YOLOv8-Pose)

These will reuse this same compose stack, the `frigate/events` MQTT topic, and the
event-based snapshots already produced by Frigate.
