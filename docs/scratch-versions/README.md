# Scratch model campaign — version records

Tracked, human-readable training records for the fire-model scratch campaign. Each entry mirrors one
version folder under the git-ignored / `.rooignore`d `model-training/scratch-model/` tree, where the
actual checkpoints, `args.yaml`, `results.csv` and run logs live. The authoritative strategy/plan is
[`plans/fire-model-v3-class-expansion.md`](../../plans/fire-model-v3-class-expansion.md).

| Version | Dir | Record |
|---|---|---|
| v1 — fire-only from scratch | `scratch-v1` | [`scratch-v1/TRAINING.md`](scratch-v1/TRAINING.md) |
| v2 — v1 continued on D-Fire | `scratch-v1-dfire` | [`scratch-v1-dfire/TRAINING.md`](scratch-v1-dfire/TRAINING.md) |
| v3 — smoke + other head expansion | `scratch-v3` | [`scratch-v3/TRAINING.md`](scratch-v3/TRAINING.md) |

Each `TRAINING.md` records the training path, original (base) model, classes, datasets, training
parameters, result metrics and checkpoint md5.
