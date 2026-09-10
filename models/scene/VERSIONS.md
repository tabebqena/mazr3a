# Scene-description model — version registry

Registry of every **SmolVLM** OpenVINO IR export installed under this
directory for the `scenewatch` service. Unlike
[`models/fire/VERSIONS.md`](../fire/VERSIONS.md) there is no promote/archive
helper: `models/scene/` is **git-ignored** and holds exactly one ACTIVE export
(see [`README.md`](README.md)). Re-running
[`dev_scripts/prep_scene_model.sh`](../../dev_scripts/prep_scene_model.sh)
replaces it.

**Rule:** after every `prep_scene_model.sh` run, append the line the script
prints (it computes the date, `openvino_model.bin` md5 and on-disk size) and
mark the superseded row `SUPERSEDED`.

| Date (UTC) | Model id | Precision | Size | md5 (`openvino_model.bin`) | Status |
|---|---|---|---|---|---|
| _none yet_ | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | int4 | — | — | not exported |

## Fields

| Column | Meaning |
|---|---|
| Date (UTC) | When the export was produced |
| Model id | Hugging Face repo id passed to `optimum-cli export openvino --model` |
| Precision | `--weight-format` used (`int4`, `int8`, `fp16`) |
| Size | `du -sh models/scene/` |
| md5 | md5 of `openvino_model.bin` (weights) |
| Status | `ACTIVE` (currently in `models/scene/`) or `SUPERSEDED` |

## Notes

- **Precision vs RAM.** `int4` (default) is the ~500 MB target that fits the
  host's ~7.5 GiB budget alongside Frigate + firewatch. `int8`/`fp16` roughly
  double/quadruple the resident size — only use them if the host RAM headroom
  is measured to be sufficient.
- **Device.** Whether the export is loaded on CPU or the iGPU is a runtime
  choice (`MODEL_DEVICE` in [`config/scenewatch.conf`](../../config/scenewatch.conf)),
  not an export property — the same IR works for both. The default is `CPU`
  so the OpenVINO detector keeps the iGPU.
