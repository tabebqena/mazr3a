# Scene-description model — `models/scene/`

Holds the **SmolVLM2** vision-language model used by the `scenewatch` service
(`scenewatch/scenewatch.py`, see [`plans/scene-description.md`](../../plans/scene-description.md)).
The directory is mounted **read-only** at `/models/scene` in the `scenewatch` container.

## CHOSEN model (default)

**SmolVLM2-500M-Video-Instruct**, pre-converted to an **OpenVINO IR** export and
run on **CPU** (`MODEL_DEVICE=CPU` in [`config/scenewatch.conf`](../../config/scenewatch.conf)).
The iGPU is deliberately left to Frigate's OpenVINO detector — see
[`plans/scene-description.md`](../../plans/scene-description.md).

`prep_scene_model.sh` picks between four **curated, already-converted** OpenVINO
exports, so no conversion toolchain is needed on the host:

| `--repo` | Hugging Face repo | ~Size | Why |
|---|---|---|---|
| `int4` *(default)* | [`circulus/SmolVLM2-500M-ov-sym-int4`](https://huggingface.co/circulus/SmolVLM2-500M-ov-sym-int4) | 356 MB | Smallest, **and** ships explicit OpenVINO `openvino_tokenizer`/`openvino_detokenizer` IRs → widest `openvino-genai` compatibility. Undocumented third-party export (optimum-intel 1.25.3, int4 symmetric) — weights only, no remote code. |
| `int8` | [`echarlaix/SmolVLM2-500M-Video-Instruct-openvino-8bit-woq`](https://huggingface.co/echarlaix/SmolVLM2-500M-Video-Instruct-openvino-8bit-woq) | 509 MB | 8-bit weight-only export from an **HF/optimum maintainer** (trusted provenance). Ships `tokenizer.json` instead of detokenizer IRs. |
| `fp16` | [`echarlaix/SmolVLM2-500M-Video-Instruct-openvino`](https://huggingface.co/echarlaix/SmolVLM2-500M-Video-Instruct-openvino) | ~2.0 GB | Same trusted source, full precision. Only if RAM/disk allow. |
| `256m` | [`echarlaix/SmolVLM2-256M-Video-Instruct-openvino`](https://huggingface.co/echarlaix/SmolVLM2-256M-Video-Instruct-openvino) | ~1.0 GB | Smaller model at full precision (not a RAM saving vs `int4`). |

Any other pre-converted OpenVINO VLM repo works with `--repo <user/model>`.
Use `--list` to print this table.

All are members of the **SmolVLM2** family (SigLIP vision encoder + SmolLM2
language decoder) and are the checkpoint family OpenVINO GenAI documents for
`VLMPipeline`. "Video-Instruct" refers to the training mixture; single-image
captioning is the mode `scenewatch` uses.

## Required files (what `VLMPipeline` loads)

| File | Purpose |
|---|---|
| `config.json` | HF model config (also what `scenewatch.py --check` verifies) |
| `openvino_language_model.xml` / `.bin` | language decoder IR (**the big file**) |
| `openvino_vision_embeddings_model.xml` / `.bin` | SigLIP vision encoder IR |
| `openvino_text_embeddings_model.xml` / `.bin` | token-embedding IR |
| `openvino_tokenizer.xml` / `.bin` | OpenVINO tokenizer IR (**or** `tokenizer.json`) |
| `openvino_detokenizer.xml` / `.bin` | OpenVINO detokenizer IR (**or** `tokenizer.json`) |
| `preprocessor_config.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.*` | processor / chat-template assets |

`prep_scene_model.sh` verifies this layout after downloading and **fails loudly**
if the language/vision/text-embedding IRs or a tokenizer source are missing, so a
broken download cannot reach the container unnoticed.

## GIT POLICY — this model is **git-ignored**

Unlike the small models ([`models/fire/`](../fire/README.md), 20 MB,
[`models/coco/`](../coco/README.md)) which ride `git pull`, **`models/scene/` is
NOT tracked** — a several-hundred-MB binary must not live in git history. It is
fetched on the machine that needs it by
[`dev_scripts/prep_scene_model.sh`](../../dev_scripts/prep_scene_model.sh) (the
same pattern as the git-ignored `models/fire/versions/` archive).
[`../../.gitignore`](../../.gitignore) ignores `models/scene/*` but keeps this
README and [`VERSIONS.md`](VERSIONS.md).

## Fetch the model

**On the host (recommended — needs only `curl` + `python3`, no optimum, no torch):**

```bash
cd ~/frigate
bash dev_scripts/prep_scene_model.sh              # default: --repo int4 (~356 MB)
bash dev_scripts/prep_scene_model.sh --repo int8  # trusted-source 8-bit (~509 MB)
bash dev_scripts/prep_scene_model.sh --list       # show the curated repos
```

**Building it yourself instead** (dev machine; needs `optimum[openvino]`, which
pulls torch ~2 GB):

```bash
./dev_scripts/prep_scene_model.sh --export --repo HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

Either path writes into `models/scene/` and prints an md5 + a ready-to-paste
provenance line for [`VERSIONS.md`](VERSIONS.md).

## Registry

[`VERSIONS.md`](VERSIONS.md) records each export (repo id, how it was obtained,
size, md5, date) so the deployed model is identifiable without inspecting the host.
