# Scene-caption models — version registry

Registry of the models installed under this directory for the **`scenereader`**
service. `models/scene/` is **git-ignored** and now holds **two** models that
coexist deliberately (see [`README.md`](README.md)):

| Backend (`MODEL_BACKEND`) | Model | Fetched by | Notes |
|---|---|---|---|
| `llamacpp` (**default**) | SmolVLM2-500M GGUF + mmproj in `smolvlm2-500m/` | [`dev_scripts/prep_scene_model_llamacpp.sh`](../../dev_scripts/prep_scene_model_llamacpp.sh) | ADD-ONLY: skips existing files; `--force` replaces only what it fetches |
| `openvino` (**retained**) | Qwen2-VL-2B-Instruct OpenVINO INT4 IR, directly in `models/scene/` | [`dev_scripts/prep_scene_model.sh`](../../dev_scripts/prep_scene_model.sh) | **KEPT — never deleted, never re-downloaded**; `--force` required to replace |

**Rule:** after a successful fetch, append the line the script prints (date,
source, how obtained, size, md5 of the largest weight file) and mark a replaced
row `SUPERSEDED`. Do not delete a `SUPERSEDED` row — it is the rollback record.

## 1. Small model — llama.cpp backend (DEFAULT)

| Date (UTC) | Source repo | Files | Size | md5 (model `.gguf`) | Status |
|---|---|---|---|---|---|
| _none yet_ | `ggml-org/SmolVLM2-500M-Video-Instruct-GGUF` | `<model>.gguf` + `mmproj*.gguf` | ~0.4–0.6 GB | — | not fetched |

The script picks the files **by pattern** at run time (prefers `Q8_0`, then
`Q6_K`/`Q4_K_M`; `mmproj` prefers `f16`), so a quant rename upstream does not
break it. Record the exact filenames it reported.

## 2. Retained larger model — OpenVINO backend

| Date (UTC) | Source | How | Architecture | Size | md5 (largest `.bin`) | Status |
|---|---|---|---|---|---|---|
| _none yet_ | `helenai/Qwen2-VL-2B-Instruct-ov-int4` | download (`--repo 2b`) | `qwen2_vl` | ~1.76 GB | — | not fetched |

## Fields

| Column | Meaning |
|---|---|
| Date (UTC) | When the files were fetched/built |
| Source | Upstream repo id (what the prep script printed) |
| How | `download` (pre-converted export via curl) or `export` (`optimum-cli`) |
| Architecture | OpenVINO only: `config.json` `model_type` — **must** be one the runtime implements |
| Size | `du -sh` of the model's directory |
| md5 | md5 of the largest weight file (the bulk of the model) |
| Status | `ACTIVE` (in use) or `SUPERSEDED` (kept for rollback) |

## Notes

- **Architecture is the hard constraint for the OpenVINO path only.**
  `openvino_genai.VLMPipeline` supports just `llava`, `qwen2_vl`, `qwen2_5_vl`,
  `gemma3`, `minicpm`, `phi3_v`, `phi4mm`. Anything else fails at load with
  `Unsupported '<type>' VLM model type` — the failure that took the first
  SmolVLM attempt down. The llama.cpp path has no such list, which is why the
  small model lives there.
- **Coexistence is intentional.** The 2B IR is a fallback the operator may
  reassess; removing it would force a 1.76 GB re-download, so both stay.
- **Device.** OpenVINO CPU vs iGPU is a runtime choice (`OPENVINO_DEVICE`), not
  an export property. The default is `CPU` so Frigate's OpenVINO detector keeps
  the iGPU.
- **Resident cost.** Small ≈0.5–0.7 GB; retained 2B ≈2 GB. `mem_limit` in
  compose is 1500m for the small default — raise it towards 3g before switching
  `MODEL_BACKEND=openvino`.
