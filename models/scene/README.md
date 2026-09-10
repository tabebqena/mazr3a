# Scene-description model — `models/scene/`

Holds the **vision-language model** used by the `scenewatch` service
(`scenewatch/scenewatch.py`, see [`plans/scene-description.md`](../../plans/scene-description.md)).
The directory is mounted **read-only** at `/models/scene` in the `scenewatch`
container, and `VLMPipeline` is pointed at it via `MODEL_DIR`.

## CHOSEN model

**Qwen2-VL-2B-Instruct**, pre-converted to **OpenVINO INT4 IR** and run on
**CPU** (`MODEL_DEVICE=CPU` in [`config/scenewatch.conf`](../../config/scenewatch.conf)).
The iGPU is deliberately left to Frigate's OpenVINO detector.

| | |
|---|---|
| Default repo | [`helenai/Qwen2-VL-2B-Instruct-ov-int4`](https://huggingface.co/helenai/Qwen2-VL-2B-Instruct-ov-int4) |
| Architecture | `qwen2_vl` (2B params, INT4) |
| On disk | **~1.76 GB** |
| Resident RAM | ~2 GB (loaded once, kept for the process lifetime) |

## Why not a 500 MB SmolVLM — the runtime decides

`openvino_genai.VLMPipeline` implements a **closed list of VLM architectures**.
Verified against the shipped runtime (openvino-genai **2026.3.1**, the newest
release on PyPI) by inspecting `libopenvino_genai.so` for architecture strings:

> `llava` · `qwen2_vl` · `qwen2_5_vl` · `gemma3` · `minicpm` · `phi3_v` · `phi4mm`

**SmolVLM is not on that list.** Its export fails at load with:

```
Unsupported 'smolvlm' VLM model type
```

Relabelling the export does not help either, because SmolVLM's parent
architecture `idefics3` is absent too. Upgrading cannot help (2026.3.1 is
already the newest release). So the small-SmolVLM idea is simply not available
through OpenVINO GenAI, and **Qwen2-VL-2B is the smallest VLM that runtime can
actually load**.

> The alternatives considered were llama.cpp + SmolVLM2-500M GGUF (520 MB, a
> second runtime) and converting a 2B model ourselves with `optimum-cli` (pulls
> torch, ~2 GB). OpenVINO's own org publishes **only 7B** VLMs (Qwen2-VL-7B,
> Qwen2.5-VL-7B, LLaVA-1.6-7B) — far too heavy for a 7.5 GB host.

Because the architecture list is the real constraint, **any model from that
list works** — only `MODEL_DIR` (and possibly the prompt) changes.

## Cost profile

CPU is governed mostly by the **caption rate**, not by model size: the motion
gate plus `CAPTION_COOLDOWN_S` and `BASELINE_EVERY_S` yield a handful of
captions per hour rather than a continuous stream, and `INFERENCE_NUM_THREADS`
caps the cores the model may use. Between captions the process is idle.

## Required files

`prep_scene_model.sh` verifies all of these and **fails loudly** if any is missing:

| File | Purpose |
|---|---|
| `config.json` | HF model config — its `model_type` must be a supported architecture |
| `openvino_language_model.xml` / `.bin` | language decoder IR (**the big file**) |
| `openvino_vision_embeddings_model.xml` / `.bin` | vision encoder IR |
| `openvino_vision_embeddings_merger_model.xml` / `.bin` | Qwen2-VL vision→text merger |
| `openvino_text_embeddings_model.xml` / `.bin` | token-embedding IR (**or** the merger above) |
| `openvino_tokenizer.xml` / `.bin` | OpenVINO tokenizer IR (**or** `tokenizer.json`) |
| `openvino_detokenizer.xml` / `.bin` | OpenVINO detokenizer IR (**or** `tokenizer.json`) |
| `preprocessor_config.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.*` | processor / chat-template assets |

## GIT POLICY — this model is **git-ignored**

Unlike the small models ([`models/fire/`](../fire/README.md) 20 MB,
[`models/coco/`](../coco/README.md)) which ride `git pull`, **`models/scene/` is
NOT tracked** — a ~1.76 GB binary must not live in git history, and the host
pulls its own copy. [`../../.gitignore`](../../.gitignore) ignores
`models/scene/*` but keeps this README and [`VERSIONS.md`](VERSIONS.md).

## Fetch the model

**On the host (needs only `curl` + `python3` — no optimum, no torch):**

```bash
cd ~/frigate
bash dev_scripts/prep_scene_model.sh                     # Qwen2-VL-2B int4 (~1.76 GB)
bash dev_scripts/prep_scene_model.sh --list              # curated alternatives
bash dev_scripts/prep_scene_model.sh --force             # replace an existing export
```

Replacing a previously downloaded model (e.g. the earlier SmolVLM attempt)
requires `--force`.

**Building it yourself instead** (dev machine; needs `optimum[openvino]`, which
pulls torch ~2 GB):

```bash
./dev_scripts/prep_scene_model.sh --export --model Qwen/Qwen2-VL-2B-Instruct
```

## Registry

[`VERSIONS.md`](VERSIONS.md) records the installed export (repo, size, md5,
date) so the deployed model is identifiable without inspecting the host.
