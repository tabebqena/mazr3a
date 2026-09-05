# Reduce CPU Load — Use the Processor's Integrated GPU (iGPU)

> **Status:** **Option A (OpenVINO detector on the iGPU) is IMPLEMENTED and deployed.**
> **Option B (VAAPI hwaccel) is intentionally NOT applied** (software decode kept; see §5).
>
> - ✅ **Applied:** [`config/config.yaml`](../config/config.yaml:108) `device: GPU` — verified live on
>   the host (`/home/dr/frigate/config/config.yaml:108`) and confirmed healthy on 2026-09-05.
> - ❌ **Not applied:** [`config/config.yaml`](../config/config.yaml:72) still `hwaccel_args: []`.
>
> **Date examined:** 2026-08-27 (baseline) · **Date re-verified:** 2026-09-05 ·
> **Host:** `ai@ssh.mazr3a.garden` · **Deploy dir:** `/home/dr/frigate`
>
> **Re-verification summary (2026-09-05):** detector `inference_speed` **9.85 ms** (was 13.58 ms),
> container CPU **23.67%** (was ~309–456%), host load avg **0.14** (was ~10). No OpenVINO/GPU/VAAPI
> errors in logs. **Caveat:** 7 of 10 cameras (cam04–10) were unreachable (`No route to host`) during
> re-verification, so the CPU drop is partly due to fewer active cameras and the full-load
> "no dropped detect frames" criterion is not yet provable. Re-run §6 once cam04–10 are back online.

## 1. Why this matters (live host state)

| Metric | Value | Verdict |
|---|---|---|
| CPU | Intel Core i7-9700, 8 cores / 8 threads (no SMT) | — |
| Load average | **10.02 / 9.73 / 9.53** on 8 cores | Oversubscribed |
| CPU idle | **0.0% idle** (`73.9 us / 23.9 sy`) | Pegged |
| Frigate container CPU | **~309–456%** (main `frigate+` process ≈ 309%, per-camera `frigate.process:*` ≈ 9% each) | Dominant consumer |
| Detector | OpenVINO **2025.3.0**, `device: CPU`, `inference_speed ≈ 13.58 ms` | Runs on CPU only |
| RAM | 7.5 GiB total, ~2.8 GiB used, no swap pressure | Fine |

The OpenVINO object-detection inference is the single biggest CPU load. It currently runs
entirely on the CPU because **the CPU's integrated GPU is enabled in hardware, mounted into
the container, and recognized by OpenVINO — but never used.**

## 2. The feature: Intel UHD Graphics 630 (iGPU inside the CPU)

The i7-9700 contains an integrated **Intel UHD Graphics 630** (24 EU, Gen9.5). It is a
"feature in the processor" that can offload processing from the CPU. Verified on the host:

- `/dev/dri/renderD128` **exists and is already mounted** into the `frigate` container
  ([`docker-compose.yml`](../docker-compose.yml:41) → `devices: - /dev/dri/renderD128:/dev/dri/renderD128`).
- VAAPI driver works: `Intel iHD` `24.3.4`, VA-API `1.22`.
- OpenVINO sees it as a usable device:
  ```
  devices: ['CPU', 'GPU']
  GPU -> Intel(R) UHD Graphics 630 (iGPU)
  ```
- Current Frigate (`ghcr.io/blakeblackshear/frigate:stable`, OpenVINO 2025.3.0) exposes
  **only `type` + `device`** for the OpenVINO detector — there is **no**
  `num_threads` / `num_streams` knob in this build. The iGPU (`device: GPU`) is the
  relevant lever.

## 3. Verified on the host (non-disruptive tests)

### 3.1 OpenVINO inference: CPU vs GPU (ssdlite_mobilenet_v2 @ 300x300)

```
CPU: compile=0.19s  inference=5.00ms avg
GPU: compile=2.90s  inference=7.57ms avg   (only at startup)
```

- GPU inference is **slightly slower** (7.57 ms vs 5.00 ms) but **well within budget**:
  live detector reports 13.58 ms and the aggregate detect load is ≈31.5 fps.
- Enabling `device: GPU` moves the dominant CPU load off the CPU for a modest latency cost.

### 3.2 VAAPI hardware decode (cam01 substream, H.264 640x360, 4 s capture)

- **Software and VAAPI decode both ran cleanly** on the current H.264 substreams
  (only benign `SEI type 5 ... truncated` H.264 warnings).
- The historical "Invalid data / VPS 0 does not exist" crash loop was **VAAPI-on-HEVC**
  (see [`plans/improve-camera-substreams.md`](improve-camera-substreams.md)); it does **not**
  apply to the current H.264 detect streams. `VAProfileH264High VLD` is available.

## 4. How to enable it

> **Option A is already applied and deployed** (verified 2026-09-05). Options B and C below are
> the remaining levers if further CPU relief is wanted.

### Option A — OpenVINO inference on the iGPU (biggest CPU relief) ✅ APPLIED

[`config/config.yaml`](../config/config.yaml:108) — the detector device was changed to `GPU`:

```yaml
detectors:
  ov:
    type: openvino
    device: GPU            # was CPU — now applied/deployed
    model_path: /openvino-model/ssdlite_mobilenet_v2.xml
```

### Option B — VAAPI hardware decode (offload ffmpeg)

[`config/config.yaml`](../config/config.yaml:71) — replace the empty hwaccel list:

```yaml
ffmpeg:
  hwaccel_args: preset-vaapi
```

### Option C — Both (A + B)

Apply both snippets together for maximum CPU relief. The `/dev/dri/renderD128` mount is
already present in the compose file, so **no `docker-compose.yml` change is required**.

## 5. Risks / considerations

- **GPU inference is slower** than CPU for this tiny model (7.57 vs 5.00 ms). If the iGPU
  can't keep up once all 10 cameras fire simultaneously, `detection_fps` could drop below
  `process_fps`. Watch the stats after enabling; if it regresses, revert to `CPU`.
- The iGPU shares system memory bandwidth; with 10 concurrent decode + inference it may
  contend. Test Option B (decode) and Option A (inference) incrementally, not both at once,
  to isolate any regression.
- Changing the detector requires a **Frigate restart** (brief detection gap). Plan a short
  maintenance window; Frigate is currently healthy and recording.
- VAAPI-on-HEVC historically crashed this host, but the detect streams are **H.264** now and
  VAAPI H.264 decode was verified working — Option B is low risk on current streams.

## 6. Verification after enabling (run on the host)

```bash
cd /home/dr/frigate && docker compose restart frigate && sleep 15
python3 scripts/verify_remote.py
curl -s http://localhost:5000/api/stats | python3 -m json.tool   # inference_speed, per-cam detection_fps vs process_fps
top -bn1 | head -20                                               # frigate container CPU% should drop
docker compose logs --since 5m frigate 2>&1 | grep -iE "error|invalid|device|openvino|vaapi|gpu" | tail -30
```

Accept criteria:
- `sum(detection_fps)` still tracks `sum(process_fps)` (no new dropped detect frames).
- Detector `inference_speed` stays comfortably below the frame budget.
- Container CPU drops materially vs the current ~309–456%.

### Observed results — re-verified 2026-09-05 on the live host

| Metric | Baseline (2026-08-27) | 2026-09-05 | Verdict |
|---|---|---|---|
| Detector device | `CPU` | `GPU` (deployed) | ✅ |
| `inference_speed` | 13.58 ms | **9.85 ms** | ✅ below budget |
| Frigate container CPU | ~309–456% | **23.67%** | ✅ material drop* |
| Host load average | ~10 | **0.14** | ✅* |
| `detection_fps` vs `process_fps` | tracks | cam01–03: det 0.0 / proc ~1.1; cam04–10 offline | ⚠️ see caveat |
| OpenVINO/GPU/VAAPI log errors | — | none | ✅ |

> **Caveat (\*):** during re-verification, cam04–10 were unreachable (`No route to host`,
> e.g. `192.168.1.203–209`), so only 3 cameras were streaming. The CPU/load drop is therefore
> **partly** explained by the reduced camera count and cannot be fully attributed to the iGPU.
> Detection was idle (`detection_fps=0.0`) at the time of sampling, so the "no dropped detect
> frames under full 10-camera load" criterion remains unproven. Re-run the checks above once
> cam04–10 are back online to close this out.

## 7. Related context

- Current config: [`config/config.yaml`](../config/config.yaml:105) (detector), [`config/config.yaml`](../config/config.yaml:71) (hwaccel).
- Compose device mount: [`docker-compose.yml`](../docker-compose.yml:41).
- Baseline diagnosis: [`plans/improve-person-detection.md`](improve-person-detection.md) (Phase 1 findings).
- Why VAAPI was historically disabled: [`plans/improve-camera-substreams.md`](improve-camera-substreams.md).
