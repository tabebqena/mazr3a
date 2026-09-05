# Fine-tune `models/fire/best.pt` on the large `ready_fire_smoke` dataset via a Google Colab notebook

**Status:** ✅ scaffolding + local dataset prep done (2026-09-05). ⏳ GPU run pending —
the user runs the provided Colab notebook, then drops the weights back here for local eval +
versioning (same workflow as [`plans/fire-model-abonia-finetune.md`](fire-model-abonia-finetune.md)).

**Goal:** domain-adapt the HF YOLO26-S checkpoint [`models/fire/best.pt`](../models/fire/best.pt)
(v1, active) by **further fine-tuning on the LARGE local fire/smoke data** — the 12,799-image
`ready_fire_smoke_dataset.yolov8` **train** split plus the 430 curated pure-background negatives
(`dataset/default-other/`) — producing a **separate candidate** checkpoint for later evaluation
against v1 (and the Abonia fine-tune v2). Training runs **on the user's GPU/Google Colab** (this
machine is CPU-only).

**User request (2026-09-05):** the user created a new notebook in Google Colab and asked to use
it to further train the model on:
- `dataset/ready_fire_smoke_dataset.yolov8` (large labeled export, train split only present)
- `dataset/default-others` (they wrote the plural; the actual dir is
  [`dataset/default-other/`](../dataset/default-other) — 430 curated background negatives)

Deliverables chosen by the user:
1. A **ready-to-import Colab notebook** (`.ipynb`) — [`notebooks/fire-large-finetune-colab.ipynb`](../notebooks/fire-large-finetune-colab.ipynb)
2. A **local dataset-prep script** — [`dev_scripts/prep_fire_large_dataset.py`](../dev_scripts/prep_fire_large_dataset.py)
3. A generated **`data.yaml`** (part of the prep output, under the git-ignored `dataset/large_finetune/`)
4. This **plan file**

---

## 1. Contract (must not break)

- [`models/fire/best.pt`](../models/fire/best.pt) is **NOT overwritten**. The fine-tuned result
  is produced in the Colab run and later archived under `models/fire/versions/` as a new
  candidate (per [`plans/model-versioning.md`](model-versioning.md)), never as the active file.
- The checkpoint must keep **class order `fire(0)/other(1)/smoke(2)`**. The dataset indices in the
  `ready_fire_smoke` export are already `fire(0)/other(1)/smoke(2)` — **labels need NO remap**
  (verified, see [`dataset/eval/class_map.json`](../dataset/eval/class_map.json): `indices_aligned: true`).
  Only the `data.yaml` **names strings** are set to the model's names so `.names` stays
  `fire/other/smoke` after fine-tuning.
- `other`/`default` (index 1) is trained but ignored in production, exactly like today.
- `default-other/` images are **pure background (no boxes)** → added to the training set as
  **empty-label** samples (Ultralytics keeps empty-label images as "backgrounds"), consistent
  with how [`dev_scripts/build_fire_negatives_eval.py`](../dev_scripts/build_fire_negatives_eval.py) treats
  them. Their purpose is false-positive reduction.
- Local prep output lives under **`dataset/*` (git-ignored)**; only the notebook, prep script and
  plan are committed.

> **Context/caveat:** [`plans/fire-model-dataset-test.md`](fire-model-dataset-test.md) previously
> recommended *not* fine-tuning on this export because it overlaps the HF checkpoint's own
> Roboflow training pool (so improvements are partly circular). The user has decided to proceed
> with a full large-data fine-tune anyway; results must be judged by independent eval (Abonia
> test split, on-camera `--dry-run`), not just the overlapping val split.

---

## 2. Data facts (confirmed by inspection 2026-09-05)

| Item | Value |
|---|---|
| [`dataset/ready_fire_smoke_dataset.yolov8/`](../dataset/ready_fire_smoke_dataset.yolov8) | Roboflow YOLOv8 export (CC BY 4.0); only `train/` split present (12,799 imgs, all 640×640) |
| Labeled (`≥1 box`) | 12,357 imgs / 30,275 boxes |
| Empty-label (`0 boxes`) | 442 imgs (background) |
| Per-class boxes | fire `0` = 14,145 · other `1` = 3,796 · smoke `2` = 12,334 |
| Per-class images | fire = 8,921 · other = 2,198 · smoke = 10,190 |
| [`dataset/default-other/`](../dataset/default-other) | 430 curated pure-background images, no labels, 640×640 (FP-reduction negatives) |
| Names in export `data.yaml` | lost (`['0','1','2']`) → relabel only, no reorder |

---

## 3. Dataset prep (local, `dev_scripts/prep_fire_large_dataset.py`)

Builds a self-contained training layout under **`dataset/large_finetune/`** (git-ignored):

```
dataset/large_finetune/
  train/images/  train/labels/      # ready_fire labeled split + negatives + empty-label ready imgs
  val/images/    val/labels/        # held-out labeled subset (mAP / early stopping)
  data.yaml                          # names fire/other/smoke (see below)
  prep_report.txt                    # split/class counts
```

Prep logic:
1. **Split** the 12,357 labeled `ready_fire` train images into `train`/`val` (default `--val-frac 0.05`,
   `--seed 0`), deterministic. Val is drawn **only from labeled** images so mAP/early stopping are
   meaningful; every val image has GT.
2. **Negatives** (`default-other`, 430) and the 442 **empty-label** ready images go into `train`
   with empty `.txt` (background) — `--no-negatives` disables adding `default-other`.
3. Copy images + labels (no re-mapping of class ids). Write `data.yaml` + `prep_report.txt`.
4. Optional `--zip` → a single archive for easy Colab upload
   (default `dataset/large_finetune_colab.zip`).

`data.yaml` (generated):

```yaml
# Fine-tune HF YOLO26-S best.pt on ready_fire_smoke + default-other negatives.
# Class order matches the checkpoint: fire=0, other=1, smoke=2 (no remap; names only).
path: /absolute/path/to/dataset/large_finetune     # notebook rewrites this for Colab
train: train/images
val: val/images
nc: 3
names:
  0: fire
  1: other
  2: smoke
```

---

## 4. Colab notebook — [`notebooks/fire-large-finetune-colab.ipynb`](../notebooks/fire-large-finetune-colab.ipynb)

Ready-to-import; cells do the following. **Where the data/weights come from is configurable at the
top of the notebook**: (a) Google Drive (recommended), or (b) `files.upload()` of the zip from §3.

1. Runtime setup — `nvidia-smi`, install a **current `ultralytics`** (YOLO26 needs ≥ the version
   used locally, 8.4.x).
2. **Get dataset** — mount Drive / upload zip → extract to `/content/large_finetune`; rewrite the
   `path:` line of `data.yaml` to that location.
3. **Get base model** — copy `models/fire/best.pt` into the session (Drive / upload). Verify
   `.names` is `fire/other/smoke` before training.
4. **Train** (from v1 `best.pt`):

```python
m = YOLO("best.pt")                 # models/fire/best.pt uploaded by the user
m.train(data="/content/large_finetune/data.yaml",
        epochs=EPOCHS, imgsz=640, batch=BATCH, device=0,   # adjust BATCH to VRAM
        project="fire_large_finetune", name="run1",
        freeze=FREEZE, lr0=0.001, patience=PATIENCE,
        plots=True, exist_ok=True, verbose=True)
```

Default tunables (edit at top): `EPOCHS=50`, `BATCH=16` (Colab T4 ~ adjust if OOM),
`FREEZE=10` (keep backbone early layers → faster, less forgetting), `PATIENCE=15`. Rationale
mirrors [`plans/fire-model-abonia-finetune.md`](fire-model-abonia-finetune.md) §3.3 but this
dataset is ~14× larger, so more epochs are warranted; `freeze` can be relaxed to `0` if GPU time
allows for full adaptation.
5. **Save & retrieve** the fine-tuned weights: `fire_large_finetune/run1/weights/best.pt`
   → renamed `best_finetuned_large.pt`, zipped/downloaded back (or copied to Drive). The user then
   copies it into this repo as `dataset/large_finetune/best_finetuned_large.pt` and we run the local
   eval (§5).

---

## 5. Post-training eval (local, after user drops the weights back)

```bash
# (a) quick validity / class-order check + ready val metrics
.venv/bin/python - <<'PY'
from ultralytics import YOLO
m = YOLO("dataset/large_finetune/best_finetuned_large.pt")
print("names:", m.names)   # must be {0: fire, 1: other, 2: smoke}
PY

# (b) same harness as the other benchmarks, on the Abonia TEST split (independent)
.venv/bin/python dev_scripts/test_fire_model.py dataset/large_finetune/best_finetuned_large.pt \
    dataset/abonia_eval/eval/data.yaml --out dataset/large_finetune/eval_abonia_test \
    --conf 0.5 --annotate

# (c) FP audit on the 430 pure-background negatives (deploy view)
.venv/bin/python dev_scripts/test_fire_model.py dataset/large_finetune/best_finetuned_large.pt \
    dataset/eval/negatives/data.yaml --out dataset/large_finetune/eval_negatives --conf 0.5
```

Compare against: v1 baseline (Abonia test all mAP@50 **0.405**), v2 Abonia fine-tune (0.948), and
the ready-dataset deploy view numbers. **Do not promote blindly** — promotion + OpenVINO export +
on-host verification is a separate explicit step ([`dev_scripts/prep_fire_model.sh`](../dev_scripts/prep_fire_model.sh),
`.roo/rules/sshuser.md`).

---

## 6. Implementation checklist

- [x] Write plan (this file).
- [x] Write [`dev_scripts/prep_fire_large_dataset.py`](../dev_scripts/prep_fire_large_dataset.py).
- [x] Run prep locally → verify `dataset/large_finetune/` structure + counts + class coverage (§3).
- [x] Write + validate the Colab notebook
      [`notebooks/fire-large-finetune-colab.ipynb`](../notebooks/fire-large-finetune-colab.ipynb).
- [x] Commit scaffolding (plan + script + notebook) locally.
- [ ] User runs the notebook on GPU/Colab, downloads `best_finetuned_large.pt`, drops it here.
- [ ] Local eval + record results in this plan §6a.2; archive as a versioned candidate; decide on
      promotion + deploy (separate follow-up).

---

## 6a. Implementation log (scaffolding, filled 2026-09-05)

- **2026-09-05 — scaffolding (Code):** wrote this plan,
  [`dev_scripts/prep_fire_large_dataset.py`](../dev_scripts/prep_fire_large_dataset.py),
  [`dev_scripts/build_fire_large_colab_nb.py`](../dev_scripts/build_fire_large_colab_nb.py) and generated
  [`notebooks/fire-large-finetune-colab.ipynb`](../notebooks/fire-large-finetune-colab.ipynb)
  (11 cells, valid nbformat-4). Committed locally (git log).
- **2026-09-05 — prep run (Code):** ran the prep script (val-frac 0.05, seed 0) → built
  `dataset/large_finetune/` + `dataset/large_finetune_colab.zip` (596 MB). **Key finding:** the 430
  `default-other` negatives are a subset of the 442 empty-label images already inside the ready
  `train` split, so `copy_if_missing` dedupes them — all 442 background images are in `train`
  (counts verified below). Class ids need **no remap** (already fire/other/smoke at idx 0/1/2);
  names enforced in `data.yaml`.

### 6a.1 Prep report (from `dataset/large_finetune/prep_report.txt`)

```
Large fire/smoke fine-tune prep (large_finetune)
============================================================
source ready      : .../dataset/ready_fire_smoke_dataset.yolov8 (train split)
source negatives  : .../dataset/default-other
seed / val-frac   : 0 / 0.05

ready total imgs   : 12799
  labeled (>=1 box): 12357
  empty-label      : 442

TRAIN images (actual files) = 12170
  of which empty-label/background .txt = 442
  negatives new to ready (only if outside train split): 0
  per-class boxes (train labeled): fire=8448, other=2118, smoke=9656
VAL images (actual files, all labeled) = 629
  per-class boxes (val): fire=473, other=80, smoke=534

data.yaml -> .../dataset/large_finetune/data.yaml
WARNING: val has no boxes for class(es): none
```

Sanity check: `check_det_dataset("dataset/large_finetune/data.yaml")` resolves
`train/val` dirs and returns `names {0: fire, 1: other, 2: smoke}`, `nc 3`. Notebook code cells
(all but the `!`-magic cell 1) compile; cell 1's `!nvidia-smi`/`!pip install` are valid IPython
magics, not `ast`-parseable — expected.

### 6a.2 GPU run results (to be filled after the user runs the Colab notebook)

_Placeholder — user drops `dataset/large_finetune/best_finetuned_large.pt` here, we evaluate per
§5 and record mAP/P/R vs v1/v2 baselines + FP audit here._

---

## 7. Risks / caveats

- **Training/eval overlap:** the ready split largely overlaps the HF checkpoint's original
  training pool, so val metrics are optimistic. Judge strictly by independent evals (§5b/§5c) and
  an on-camera pilot.
- **Large data → long run:** 12,799 imgs × 50 epochs on a Colab T4 ≈ many hours; a T4
  session may time out. Use A100 if available, `resume=True` if interrupted, and rely on
  `patience`/val early stopping. `BATCH` may need lowering if OOM.
- **Empty-label backgrounds:** Ultralytics keeps them as "backgrounds"; a small minority of the
  442 empty-label ready images may actually contain fire/smoke (unverified) — treated as background
  purely because they were exported unlabelled. The curated `default-other` negatives were
  user-triaged, so they are trustworthy.
- **Class-order drift:** verify `.names` is still `fire/other/smoke` after training (the §3 names
  strings enforce this on save).
- **Not overwriting active model:** promotion is a separate, explicit step that also regenerates
  the OpenVINO IR + `labelmap.txt` and deploys/tests on `ssh.mazr3a.garden`.

---

## 8. Git & remote handoff

- Commit plan + script + notebook after each logical change (per `.roo/rules/Agents.md`);
  artifacts under `dataset/*` stay untracked.
- No production/firewatch change in this phase → **no remote deploy**. Model promotion + OpenVINO
  export + on-host verification is a separate follow-up plan if the numbers justify it (and then
  per `.roo/rules/sshuser.md` the changed files get moved to `ssh.mazr3a.garden` and verified).
