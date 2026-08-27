# Camera Substream Report — 10 Cameras (H.264)

> **⚠️ IMPORTANT — READ FIRST**
> The features below (codec, resolution, fps, bitrate) are **configured on the cameras
> themselves** (in each camera's web UI), NOT in
> [`config/config.yaml`](config/config.yaml). Frigate only *consumes* the stream.
> **If any camera's substream settings are changed, these results become stale** and
> must be re-probed. Bitrate in particular is **variable (VBR)** and depends on scene
> motion/activity at the moment of measurement.

- **Last probed:** 2026-08-27 (host time, local `Asia/Riyadh`) — after switching all
  10 substreams from H.265/HEVC to **H.264 with U-Code disabled**.
- **Method:** live probe from the NVR host (`ssh.mazr3a.garden`, user `ai`), using the
  real RTSP credentials from the running `frigate` container
  (`docker exec frigate printenv FRIGATE_*`)
- **Resolution/codec/fps:** `ffprobe -rtsp_transport tcp -select_streams v:0 ...`
- **Bitrate:** 8-second `ffmpeg` capture to a temp file, then `bytes * 8 / seconds`
  (RTSP SDP reports `bit_rate=N/A`, so it cannot be read from the stream header)

---

## Per-camera substream features

| Cam | IP | Substream path | Codec | Resolution | Nominal fps | Measured fps | Bitrate (8s avg) |
|---|---|---|---|---|---|---|---|
| cam01 | 192.168.1.200 | `/media/video2` | H.264 | 640×360 | 12 | 12 | ~120 kbps |
| cam02 | 192.168.1.201 | `/media/video2` | H.264 | 640×360 | 12 | 12 | ~101 kbps |
| cam03 | 192.168.1.202 | `/media/video2` | H.264 | 640×360 | 12 | 12 | ~150 kbps |
| cam04 | 192.168.1.203 | `/media/video2` | H.264 | 640×360 | 12 | 12 | ~149 kbps |
| cam05 | 192.168.1.204 | `/media/video2` | H.264 | 640×360 | 12 | 12 | ~143 kbps |
| cam06 | 192.168.1.205 | `/media/video2` | H.264 | 640×360 | 12 | 12 | ~120 kbps |
| cam07 | 192.168.1.206 | `/media/video2` | H.264 | 640×360 | 12 | 12 | ~127 kbps |
| cam08 | 192.168.1.207 | `/Streaming/Channels/102` | H.264 | 640×360 | 20 | ~12 (avg) | ~229 kbps |
| cam09 | 192.168.1.208 | `/media/video2` | H.264 | 640×360 | 12 | 12 | ~174 kbps |
| cam10 | 192.168.1.209 | `/media/video2` | H.264 | 640×360 | 12 | 12 | ~109 kbps |

All 10 cameras use pixel format `yuvj420p` and also carry an audio track on the
substream (`pcm_mulaw`, 8 kHz mono), which Frigate ignores for the detect role.

---

## Key findings

1. **All 10 substreams are now H.264 (changed from HEVC on 2026-08-27), with U-Code
   (smart coding) disabled.** This was done so Frigate can decode them cleanly — the
   previous H.265/HEVC substreams on cam01/02/03/09/10 sent the VPS parameter set
   in-band (`VPS 0 does not exist`), causing `Invalid data found when processing input`
   and `Invalid or missing video stream ... Discarding`. All 10 now probe with
   `codec_name=h264` and **zero** VPS warnings.
2. **Resolution is 640×360 on all 10**, consistent with the `detect.width`/`detect.height`
   overrides in the config.
3. **FPS:** all cameras deliver ~12 fps measured (the Hikvision cam08 advertises 20 fps
   nominal but averages ~12). Frigate's `detect.fps: 5` in
   [config/config.yaml](config/config.yaml:301) throttles actual *detection* to 5 fps
   regardless of the stream fps.
4. **Bitrate is scene-dependent (VBR).** Under H.264, per-camera 8s-average bitrates
   range ~101–229 kbps; cam08 is the highest (~229 kbps) and cam02 the lowest
   (~101 kbps). Total substream network load across all 10 is roughly **~1.4 Mbps** —
   still negligible on the same LAN.
5. All 10 streams were reachable and healthy at probe time — no auth failures or dead
   streams.

> **Note on bandwidth vs HEVC:** H.264 uses ~1.5–2× the bitrate of H.265 at equal
> quality, which is why the total rose from ~1.2 Mbps (HEVC) to ~1.4 Mbps (H.264).
> Since cameras and Frigate share the same network this is irrelevant. For remote/mobile
> viewers, a separate low-bandwidth transcoded stream is used instead — see Path B in
> [`plans/improve-camera-substreams.md`](plans/improve-camera-substreams.md).

---

## How to re-probe (when camera settings change)

From the host:

```bash
# 1) resolution / codec / fps (e.g. cam01)
docker exec frigate printenv FRIGATE_RTSP_USER
docker exec frigate printenv FRIGATE_RTSP_PASS   # UNV cams
docker exec frigate printenv FRIGATE_HIK_RTSP_PASS  # Hikvision cam08

ffprobe -v error -rtsp_transport tcp -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,r_frame_rate,pix_fmt \
  -of default=noprint_wrappers=1 \
  "rtsp://USER:PASS@192.168.1.200:554/media/video2"        # cam01–07, 09–10
  "rtsp://USER:PASS@192.168.1.207:554/Streaming/Channels/102"  # cam08

# 2) actual bitrate (8-second sample)
ffmpeg -hide_banner -loglevel info -rtsp_transport tcp \
  -i "rtsp://USER:PASS@IP:554/media/video2" -t 8 -an -c copy -y /tmp/br.ts
# read the final "Lsize=... bitrate=..." line from the output
```

> **Note:** camera-side changes (codec, resolution, fps, bitrate caps, U-Code) made
> through each camera's web UI will change these values. Re-run the probe above to
> refresh this report.
