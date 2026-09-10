# Scene-description model — version registry

Registry of the VLM export installed under this directory for the `scenewatch`
service. Unlike [`models/fire/VERSIONS.md`](../fire/VERSIONS.md) there is no
promote/archive helper: `models/scene/` is **git-ignored** and holds exactly one
ACTIVE export (see [`README.md`](README.md)). Re-running
[`dev_scripts/prep_scene_model.sh`](../../dev_scripts/prep_scene_model.sh) with
`--force` replaces it.

**Rule:** after every `prep_scene_model.sh` run, append the line the script prints
(date, source, how obtained, on-disk size, md5 of the largest `.bin`) and mark the
superseded row `SUPERSEDED`.

| Date (UTC) | Source | How | Architecture | Size | md5 (largest `.bin`) | Status |
|---|---|---|---|---|---|---|
| _none yet_ | `helenai/Qwen2-VL-2B-Instruct-ov-int4` | download (`--repo 2b`) | `qwen2_vl` | ~1.76 GB | — | not fetched |

## Fields

| Column | Meaning |
|---|---|
| Date (UTC) | When the export was fetched/built |
| Source | Upstream repo id (`--repo`), i.e. what `prep_scene_model.sh` printed |
| How | `download` (pre-converted OV export via curl) or `export` (`optimum-cli`) |
| Architecture | `config.json` `model_type` — **must** be one the runtime implements |
| Size | `du -sh models/scene/` |
| md5 | md5 of the largest `*.bin` (the language decoder — the bulk of the model) |
| Status | `ACTIVE` (currently in `models/scene/`) or `SUPERSEDED` |

## Notes

- **Architecture is the hard constraint.** `openvino_genai.VLMPipeline` supports
  only `llava`, `qwen2_vl`, `qwen2_5_vl`, `gemma3`, `minicpm`, `phi3_v`,
  `phi4mm`. A model whose `config.json` says anything else (e.g. `smolvlm`)
  fails at load with `Unsupported '<type>' VLM model type` — which is exactly
  why the first SmolVLM attempt never started.
  `prep_scene_model.sh` now checks `model_type` and refuses to install an
  unsupported export.
- **Size vs RAM.** The default ~1.76 GB export needs ~2 GB resident. The host
  has ~7.5 GiB shared with Frigate + firewatch + portal, so watch
  `machine-status.py` / `docker stats` after the first caption burst; if it is
  tight, the escape hatches are fewer captions (raise `CAPTION_COOLDOWN_S` /
  `BASELINE_EVERY_S`) rather than a smaller model, since no smaller supported
  VLM exists.
- **Device.** CPU vs iGPU is a runtime choice (`MODEL_DEVICE`), not an export
  property. The default is `CPU` so the OpenVINO detector keeps the iGPU.
