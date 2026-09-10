# TODO

Loose follow-ups that are not yet a full plan.

> Plans live in `plans/` (git-ignored / local-only, per [`.gitignore`](.gitignore)).
> This file IS tracked, so it is the place for short, shared follow-ups.

## Follow-ups

- we swap the model to **YOLO11s**, make sure to check the machine monitor after time.

<!--
Context for the line above:
Frigate's COCO detector is now models/coco/yolo11s.onnx running on the Intel
UHD 630 iGPU (commit "feat(frigate): upgrade COCO detector to YOLO11s on the
iGPU"; see plans/better-coco-model-on-igpu.md).

What to check, after a few days of running:
  * media/watchdog/watchdog_baseline_<tag>.csv -> detector inference_ms,
    detection_fps vs process_fps, plus the new gpu_usage_pct column.
  * machine-monitor Telegram alerts / stdout -> live iGPU usage (and GPU temp
    where the host exposes one).
Accept: sum(detection_fps) still tracks sum(process_fps) on all online cameras
and inference_speed stays under the ~100 ms frame budget. If it does not, revert
the model paths in config/config.yaml to /models/coco/yolo11n.onnx.
-->
