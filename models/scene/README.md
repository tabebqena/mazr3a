# Scene-description model — `models/scene/`

Holds the **SmolVLM** vision-language model used by the `scenewatch` service
(`scenewatch/scenewatch.py`, see [`plans/scene-description.md`](../../plans/scene-description.md)).
The directory is mounted **read-only** at `/models/scene` in the `scenewatch` container.

## CHOSEN model (default)

**SmolVLM2-500M-Video-Instruct**, exported to **OpenVINO INT4 IR** and run on
**CPU** (`MODEL_DEVICE=CPU` in [`config/scenewatch.conf`](../../config/scenewatch.conf)).
The iGPU is deliberately left to Frigate's OpenVINO detector — see
[`plans/scene-description.md`](../../plans/scene-description.md).

| Candidate | Params | INT4 size (≈) | Role |
|---|---|---|---|
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 500M | ~0.5 GB | **default** |
| `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` | 256M | ~0.3 GB | RAM fallback (`--model 256m` in the prep script) |

Both are members of the **SmolVLM2** family (SigLIP vision encoder + SmolLM2
language decoder) and are the checkpoints OpenVINO GenAI documents for
`VLMPipeline`, so only the directory contents change, not the loader. "Video-
Instruct" refers to the training mixture; single-image captioning is the mode
`scenewatch` uses.

## Why this model

- **Small enough to stay resident in RAM** for the process lifetime — `scenewatch`
  loads it once at startup and keeps it compiled, it is never reloaded per caption.
- **~512 px vision input** matches the 640x360 detect substream (
  [`plans/event-only-recording.md`](../../plans/event-only-recording.md)), so no
  higher-resolution source is needed.
- **OpenVINO-native**: the same runtime family already used by Frigate and
  firewatch, so no second inference stack (no llama.cpp/Ollama) is introduced.

## Required files (after export)

| File | Purpose |
|---|---|
| `openvino_model.xml` | OpenVINO IR graph |
| `openvino_model.bin` | OpenVINO IR weights |
| `openvino_tokenizer.xml` / `.bin` | OpenVINO tokenizer |
| `openvino_detokenizer.xml` / `.bin` | OpenVINO detokenizer |
| `config.json` | HF model config (processor/tokenizer needed by `openvino-genai`) |
| `preprocessor_config.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja` | tokenizer/processor assets |

The exact filenames are produced by the prep script below; `scenewatch.py` only
requires that `MODEL_DIR` contains a `config.json` + an `openvino_model.xml`.

## GIT POLICY — this model is **git-ignored**

Unlike the small models ([`models/fire/`](../fire/README.md), 20 MB,
[`models/coco/`](../coco/README.md)) which ride `git pull`, **`models/scene/` is
NOT tracked** — a ~500 MB binary must not live in git history. It is produced on
the machine that needs it by [`dev_scripts/prep_scene_model.sh`](../../dev_scripts/prep_scene_model.sh)
and fetched onto the host as a **deploy prerequisite** (the same pattern as the
git-ignored `models/fire/versions/` archive). [`../../.gitignore`](../../.gitignore)
ignores `models/scene/*` but keeps this README.

## Produce / refresh the export

```bash
# on the dev machine (or on the host if it has the space + network)
./dev_scripts/prep_scene_model.sh                 # default: SmolVLM-500M-Instruct
./dev_scripts/prep_scene_model.sh --model 256m    # RAM fallback
```

The script writes into `models/scene/` and prints an md5 + provenance line to
append to [`VERSIONS.md`](VERSIONS.md).

## Registry

[`VERSIONS.md`](VERSIONS.md) records each export (model id, precision, size,
md5, date) so the deployed model is identifiable without inspecting the host.
