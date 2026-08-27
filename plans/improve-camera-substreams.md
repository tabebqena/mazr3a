# Two-Path Architecture: H.264 Substream for Frigate + Low-Bandwidth Remote Stream

## 1. Goal

Split camera streaming into two independent paths:

- **Path A — Frigate readability (detection + event recording):** change each camera's
  **substream to H.264 and disable U-Code (smart coding)** so Frigate can decode it
  cleanly with no `VPS 0 does not exist` / `Invalid data` / `Discarding` errors. LAN
  bandwidth between camera and Frigate is on the same network, so the higher H.264
  bitrate is **accepted** (negligible cost).
- **Path B — remote/mobile live view:** create a **separate, low-bandwidth stream** by
  transcoding on the fly (decode the H.264 substream → re-encode smaller) so remote
  users over WAN/4G/5G use little bandwidth. The transcoder may be Frigate's bundled
  go2rtc or another low-RAM/low-CPU option. The mobile client is chosen later.

```mermaid
flowchart TD
    subgraph LAN - same network, bandwidth not a concern
        C[10 cameras] -->|substream /media/video2 H264 640x360 U-Code off| F[Frigate]
        F --> D[OpenVINO detection + event-only recording]
    end
    F -->|Path B on-the-fly transcode on demand| R[Low-bandwidth stream e.g. 320x180 H264 5-10fps ~150-250kbps]
    R --> M[Mobile / remote client - TBD]
```

## 2. Evidence (from live probing + logs)

- **All 10 substreams are currently HEVC (H.265) 640×360.** UNV at 12 fps; Hikvision
  cam08 nominal 20 fps, measured avg ~12 fps. Bitrate is VBR: cam03 ~226 kbps,
  cam08 ~258 kbps, quiet cams ~66–90 kbps. See
  [`camera-substream-report.md`](../camera-substream-report.md).
- **Live `ffprobe` printed `[hevc] VPS 0 does not exist` for cam01/02/03/09/10** — and
  NOT for cam04/05/06/07/08.
- **`frigate.log` shows the same camera set** failing: OpenCV
  `Unable to read codec parameters from stream (Invalid data found when processing input)`,
  `record.maintainer` `Invalid or missing video stream ... Discarding`, and `watchdog`
  `Ffmpeg process crashed unexpectedly` for cam01/02/03 with `[hevc] VPS 0 does not
  exist`, `[AVHWFramesContext] Failed to sync surface`, `[hwdownload] Failed to download
  frame: -5` (VAAPI-on-HEVC crash loop).

**Root cause:** the 5 UNV cameras send HEVC **VPS/SPS/PPS parameter sets in-band only**,
so ffmpeg cannot initialize the decoder cleanly. Switching the substream to **H.264**
(H.264 has no VPS; SPS/PPS are far more robustly handled) and **disabling U-Code**
eliminates this at the source.

## 3. Bandwidth rationale (why H.264 on LAN + transcode for remote)

| Segment | Codec / stream | Bitrate | Concern? |
|---|---|---|---|
| Camera ↔ Frigate (LAN) | H.264 640×360 @ 12 fps (after Path A) | ~150–450 kbps/cam, ~2–3.5 Mbps total | **No** — same network, negligible |
| Remote/mobile (WAN) | Transcoded low stream (Path B) | ~150–250 kbps/viewer | **Yes** — this is the bandwidth to protect |

H.264 uses ~1.5–2× the bitrate of H.265 at equal quality, but that only affects the LAN
leg which is not a constraint. For remote users, Path B re-compresses to a far smaller
H.264 stream (universally decodable by phones) regardless of the LAN-side codec.

## 4. Path A — Camera-side: substream → H.264, disable U-Code (primary)

Change **each camera's substream** (NOT the main stream) in the camera web UI:

| Cameras | IPs | Substream | Change |
|---|---|---|---|
| cam01–cam07, cam09, cam10 | .200–.206, .208, .209 | `/media/video2` | Codec → **H.264**; **disable U-Code**; keep 640×360 @ ~12 fps |
| cam08 (Hikvision) | .207 | `/Streaming/Channels/102` | Codec → **H.264** (substream); disable smart coding; keep 640×360 |

UNV / VCP web UI → Video → Stream (sub-stream `/media/video2`):
1. **Video Encoding / Codec → H.264** (Main profile is fine at 640×360).
2. **U-Code / smart coding → OFF** (plain H.264; also keeps motion detection intact —
   U-Code's static-region frame-dropping can blunt detection).
3. Keep resolution **640×360**, frame rate **~10–12 fps**, GOP/I-frame interval **≤ 2 s**
   (≈ 24 frames) so ffmpeg syncs instantly on connect.
4. **Do not touch the main stream** (`/media/video1`, 3200×1800) — not pulled.

Hikvision cam08: Configuration → Video/Audio → Video → Stream Type **Sub Stream**:
- Video Encoding → **H.264**, Resolution 640×360, Frame Rate 10–12, smart/H.264+ coding off.

> Frigate-side: keep `ffmpeg.hwaccel_args: []` (software decode) — do NOT re-enable
> VAAPI/QSV; the historical crash loop was VAAPI-on-HEVC. H.264 software decode is
> lightweight and reliable. Keep `version: 0.17-0`, 300×300 model, `detect.fps: 5`,
> event-only recording.

## 5. Path B — Remote low-bandwidth stream (on-the-fly transcode)

Purpose: a **separate stream** that remote/mobile users consume, kept small regardless of
the LAN-side codec. Design:

1. **Source:** the (now H.264) substream per camera — lightest input, no extra camera
   session needed beyond what Frigate already uses.
2. **Transcode target (recommended default):** H.264, **320×180 @ 5–10 fps,
   ~150–250 kbps** (tunable). H.264 is chosen because every mobile device decodes it
   natively (no HEVC/royalty concerns on phones).
3. **Engine — two options to decide at implementation:**
   - **Option 1 (recommended): Frigate's bundled `go2rtc`** (already running, v1.9.10).
     go2rtc can define an extra per-camera stream with an ffmpeg transcode, and it
     transcodes **on demand only while a viewer is connected** → near-zero idle CPU/RAM.
     This is the low-RAM/low-CPU option already in the stack.
   - **Option 2: a dedicated lightweight transcoder/service** if go2rtc's profile cannot
     meet the latency/quality needs. Only consider if Option 1 falls short.
4. **Delivery protocol (affects client choice):** expose the low stream as **HLS**
   (most universal for mobile) and optionally **WebRTC** (Frigate-native, lowest
   latency). RTSP is possible but needs an RTSP-capable app.
5. **Concurrency/CPU:** on-demand transcode means CPU scales with active viewers, not
   with cameras. A few concurrent mobile viewers is trivial for the i7-9700; revisit
   QSV/hardware encode only if many concurrent viewers become a requirement.
6. **Authentication/access:** the low stream must be secured (go2rtc can be placed
   behind Frigate's auth / a reverse proxy) so remote users cannot reach raw streams.

### Path B implementation sketch (config-level, exact go2rtc syntax validated in code mode)

In [`config/config.yaml`](../config/config.yaml), add a `go2rtc:` block defining a
transcoded stream per camera (e.g. `cam01_low`) that reads the substream and re-encodes
to low-bitrate H.264, and expose the appropriate HLS/WebRTC endpoint for the chosen
mobile client. Exact ffmpeg/go2rtc filter syntax and the client URL scheme are confirmed
during implementation against the running go2rtc 1.9.10.

## 6. Files to change

| File | Change |
|---|---|
| [`config/config.yaml`](../config/config.yaml) | Add `go2rtc:` transcode stream block for Path B (all 10 cams). No Frigate camera input changes (substream path/roles stay the same). |
| [`docker-compose.yml`](../docker-compose.yml) | Only if an HLS/WebRTC port or an auth proxy needs exposing for Path B (evaluate at implementation; 8554/8555 already published). |
| [`README.md`](../README.md:261) | Update troubleshooting: substreams are now H.264 + U-Code off (Path A); document the low-bandwidth remote stream (Path B). |
| [`camera-substream-report.md`](../camera-substream-report.md) | Re-probe after Path A; refresh table (codec → h264, note U-Code off). |

Camera web-UI changes (Path A) are manual operator actions; the `config.yaml` /
`docker-compose.yml` / docs edits are done in this repo and deployed to `/home/dr/frigate`
per the sshuser rule.

## 7. Verification

```bash
# On host (ai@ssh.mazr3a.garden)
# Path A - confirm codec changed to h264, no VPS errors
ffprobe -v error -rtsp_transport tcp -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate -of default=noprint_wrappers=1 \
  "rtsp://USER:PASS@192.168.1.200:554/media/video2"     # expect codec_name=h264

# Restart Frigate
docker compose -f /home/dr/frigate/docker-compose.yml up -d --force-recreate frigate

# Confirm no residual decode errors (counts should be 0)
docker logs frigate 2>&1 | grep -cE "VPS 0 does not exist|Invalid or missing video stream|Ffmpeg process crashed"

# Path B - confirm the low-bandwidth stream plays and its bitrate is small
# (measure from the mobile/HLS endpoint, or ffprobe the transcoded stream URL)
# Confirm all cameras live + detection + CPU/RAM headroom
docker logs --since 5m frigate | grep -E "ERROR|WARNING"
docker stats --no-stream
```

Acceptance criteria:
- All 10 substreams `codec_name=h264`; zero `VPS`/`Discarding`/crash lines after a few hours.
- All 10 cameras live with detection; an event still produces a clip/snapshot.
- Path B low stream plays on a mobile device at ~150–250 kbps with acceptable latency;
  server idle CPU/RAM unchanged (transcode only when a viewer is connected).

## 8. Rollback

- **Path A:** set substream codec back to H.265 / re-enable U-Code in the camera UI (no
  repo change). Frigate resumes tolerating the (known) HEVC quirks.
- **Path B:** remove the `go2rtc:` block; revert compose port/auth changes.
- All repo changes are additive and git-tracked; worst case Frigate restarts with the
  previous config.
