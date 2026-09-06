# Track COCO detector model in git — self-contained deploy

## Goal

Make the git-based deploy (`dev_scripts/deploy_all.sh`) fully self-contained so a fresh
host clone needs no manual model files. The COCO ONNX artifacts that Frigate loads from
`/config/coco/` were previously git-ignored in `models/coco/` and scp'd by hand to host
`config/coco/` — the git-deploy has no such copy step, so a fresh clone would have **no
detector model** (Frigate would fail to start detection).

## Root findings (why this surfaced)

- The new `deploy_all.sh` changed-set only handles `models/fire/*`, scripts, configs and
  compose — it never ships `config/coco/*.onnx`, and those files were git-ignored.
- Host `config/coco/` (verified live on `ssh.mazr3a.garden`) is byte-identical
  (md5 `acafa5d1…` / `44de45dc…` / `0c544af3…`) to the local `models/coco/` files.

## Change (implemented 2026-09-06)

1. **Move canonical model location** from `models/coco/` → **`config/coco/`** (the runtime
   dir Frigate mounts at `/config/coco`), matching the `config/config.yaml`
   `model.path`/`labelmap_path`/`detector.model_path`:
   - `config/coco/yolo11n.onnx` (ACTIVE, 10.7 MB), `config/coco/yolov8s.onnx` (44.9 MB,
     alternative), `config/coco/labelmap.txt` (80-line COCO order) — **all git-tracked**.
   - `config/coco/README.md` = new provenance doc (replaces the old `models/coco` README).
2. `models/coco/` reduced to a pointer `README.md` (kept so historical doc links resolve);
   no binaries remain there.
3. `.gitignore` — removed the `models/coco/*.onnx` / `*.pt` ignore rules; `config/coco/*`
   is tracked and must stay tracked.
4. `dev_scripts/prep_coco_model.sh` — export target changed `models/coco` → `config/coco`;
   notes updated (artifacts tracked; commit + deploy via git, no scp).
5. `dev_scripts/deploy_all.sh` — added `config/coco/*` to the changed-set as
   `FRIGATE_CFG_CHANGED=1` so a model file change restarts `frigate` (model reload).
6. `config/config.yaml` — comment updated to say the model dir is git-tracked.
7. `plans/replace-coco-with-yolo-on-igpu.md` — top-of-file update note pointing to the new
   canonical location; old scp steps marked historical.

## Verified

- Local md5 of moved files == host `config/coco/` md5 (committing exactly the live model).
- `git check-ignore config/coco/yolo11n.onnx …` → not ignored (will be tracked).
- `config.yaml` path unchanged: `/config/coco/yolo11n.onnx` + `/config/coco/labelmap.txt`.

## Commits / deploy

- Local commit(s) for this plan.
- Deploy: host fresh clone + `docker compose up` (self-contained). No scp of model files.
- Follow-up still open: firewatch Telegram no-alert diagnosis (cam01 hits never reached
  `MIN_HITS=3`; transient `HTTP 500` window right after the 2026-09-06 11:54 UTC host reboot;
  `cam04` offline "No route to host").
