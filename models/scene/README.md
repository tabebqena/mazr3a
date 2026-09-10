# Scene-caption models — `models/scene/`

Holds the vision-language models used by the **`scenereader`** service
([`scenereader/scenereader.py`](../../scenereader/scenereader.py), see
[`plans/event-scene-reader.md`](../../plans/event-scene-reader.md)).

The directory is mounted **read-only** at `/models` in the container
(`/models/scene`), and it holds **TWO models on purpose**:

| # | Model | Backend (`MODEL_BACKEND`) | Role | On disk | Resident |
|---|---|---|---|---|---|
| 1 | **SmolVLM2-500M** GGUF + mmproj, under `smolvlm2-500m/` | `llamacpp` | **DEFAULT** — the smallest useful model; start here | ~0.4–0.6 GB | ~0.5–0.7 GB |
| 2 | **Qwen2-VL-2B-Instruct** OpenVINO INT4 IR, directly here | `openvino` | **RETAINED fallback** — kept, never deleted, never re-downloaded | ~1.76 GB | ~2 GB |

`MODEL_BACKEND` in [`config/scenereader.conf`](../../config/scenereader.conf)
selects which one runs. Switching between them is a **config change only** —
no code change, no re-download, no image rebuild.

## Why the previous design failed (and what changed)

The retired `scenewatch` service held the 2B model resident **and** ran a 15 s
per-camera motion sweep, which filled RAM and drove the host hot; it was stopped
by the operator. `scenereader` keeps the *event-driven, no-re-detection* design:
Frigate's own captures are read from disk, and the model runs **only during an
idle-gated batch**. See [`plans/event-scene-reader.md`](../../plans/event-scene-reader.md) §1.

## 1. The DEFAULT small model (llama.cpp)

**SmolVLM2-500M-Video-Instruct** converted to **GGUF** (+ its `mmproj` vision
projector), served by a **resident `llama-server`** child process. Chosen first
because the goal is the smallest useful model.

**Fetch it on the host** (plain `curl` + `python3`, no pip/optimum/torch):

```bash
cd ~/frigate
bash dev_scripts/prep_scene_model_llamacpp.sh            # GGUF + mmproj
bash dev_scripts/prep_scene_model_llamacpp.sh --list     # inspect the repo first
bash dev_scripts/prep_scene_model_llamacpp.sh --force    # replace what is there
# llama.cpp itself (the server/CLI) as reproducible pinned binaries:
bash dev_scripts/prep_scene_model_llamacpp.sh --bin --llamacpp-tag <tag>
```

The script is **ADD-ONLY**: it skips every file that already exists (even a
`--force` only replaces what it fetches), and it **never touches the retained
OpenVINO IR** below.

The llama.cpp binaries are **not baked into the image** (no unverifiable release
URL at build time); they land in `models/scene/bin/` and
`config/scenereader.conf` points `LLAMA_SERVER_BIN` / `LLAMA_CLI_BIN` at them.

## 2. The RETAINED OpenVINO model (`MODEL_BACKEND=openvino`)

| | |
|---|---|
| Repo | [`helenai/Qwen2-VL-2B-Instruct-ov-int4`](https://huggingface.co/helenai/Qwen2-VL-2B-Instruct-ov-int4) |
| Architecture | `qwen2_vl` (2B params, INT4) |
| Why it is still here | the operator may **reassess a larger model later**; deleting it would force a 1.76 GB re-download |
| Refresh/re-install it | `bash dev_scripts/prep_scene_model.sh` (refuses to clobber without `--force`) |

### Why the small model is NOT downloadable through OpenVINO GenAI

`openvino_genai.VLMPipeline` implements a **closed list** of architectures —
`llava`, `qwen2_vl`, `qwen2_5_vl`, `gemma3`, `minicpm`, `phi3_v`, `phi4mm`
(verified by inspecting the shipped `libopenvino_genai.so`). **SmolVLM is not on
that list**, so it cannot load there at all:

```
Unsupported 'smolvlm' VLM model type
```

Relabelling does not help (SmolVLM's parent `idefics3` is absent too) and
upgrading cannot help. That is exactly why the **small** path uses a second, tiny
runtime (llama.cpp) — and why the OpenVINO path exists only to keep what is
already on the host usable. `prep_scene_model.sh` therefore asserts
`config.json`'s `model_type` and refuses anything unsupported.

## Cost profile

CPU is governed by **how many frames are captioned**, not by model size: the
harvest only reads disk, and the caption batch runs only while the **idle
governor** (host loadavg + CPU temp) is open, bounded by `MAX_EVENTS_PER_RUN`
and `MAX_RUN_SECONDS`. Both backends load **once and stay resident**
(`MODEL_KEEP_LOADED=true`): for a ≤500M model the reload churn would burn CPU for
no real RAM gain, so unload-after-idle is a **deferred optimization**
(`MODEL_KEEP_LOADED=false`, `IDLE_UNLOAD_S`).

## GIT POLICY — the models are **git-ignored**

Far too large for git history (unlike the 20 MB [`models/fire/`](../fire/README.md)
ACTIVE set). [`../../.gitignore`](../../.gitignore) ignores `models/scene/*` but
keeps **this README and [`VERSIONS.md`](VERSIONS.md)**.

## Registry

[`VERSIONS.md`](VERSIONS.md) records what is installed per backend (repo, size,
md5, date) so the deployed models are identifiable without inspecting the host.
