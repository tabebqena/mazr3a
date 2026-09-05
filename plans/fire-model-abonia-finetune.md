0l,0# Fine-tune `models/fire/best.pt` on the Abonia `fire-8` dataset

**Status:** planning. Related prior work:
[`plans/fire-model-abonia-benchmark.md`](fire-model-abonia-benchmark.md) (baseline
benchmark on the Abonia `fire-8` test split, commit `34dafc3`).

**Goal:** domain-adapt the HF YOLO26-S checkpoint [`models/fire/best.pt`](../models/fire/best.pt)
by fine-tuning on the Abonia `fire-8` **train** split (877 images), producing a
**separate candidate** model, then re-benchmarking it on the untouched **test** split
(55 images) to measure improvement over the baseline.

**User decisions:**
1. Do a **small CPU smoke run** (~5-10 epochs) first to confirm the pipeline and measure
   time/epoch; the user then decides whether to scale up.
2. The **full training run is executed by the user on GPU/Colab** using the training
   `data.yaml` + exact command below — this machine is CPU-only.

**Contract (must not break):**
- [`models/fire/best.pt`](../models/fire/best.pt) is **NOT overwritten**; the fine-tuned
  result lives under the git-ignored `dataset/abonia_eval/finetune/` as a candidate.
- The fine-tuned checkpoint must keep the **class order `fire(0)/other(1)/smoke(2)`**
  (verified on `best.pt`). The Abonia indices (`Fire`=0, `default`=1, `smoke`=2) already
  align, so labels need **no remap** — only the `data.yaml` **names strings** are set to
  the model's names so `.names` stays `fire/other/smoke` after fine-tuning.
- `other`/`default` (index 1) is trained but ignored in production, exactly like today.

---

## 1. Baseline to beat (Abonia `fire-8` test split, 55 imgs / 57 boxes)

| Class (idx) | P | R | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| all | 0.525 | 0.416 | **0.405** | **0.142** |
| fire (0) | 0.561 | 0.514 | **0.462** | 0.144 |
| smoke (2) | 0.489 | 0.318 | **0.349** | 0.140 |

Deploy view (conf 0.5): fire image recall 24/35 (68.6 %), smoke 7/21 (33.3 %), 0 FP.
See [`dataset/abonia_eval/report.md`](../dataset/abonia_eval/report.md).

**Success target (indicative):** fine-tuned model on the SAME test split should exceed
baseline fire mAP@50 (~0.46) and image-level fire recall (~68.6 %) while keeping 0 FP;
smoke recall should also improve. These are the reference numbers — the fine-tune is on
the dataset's train split, so test-set lift is the honest measure.

---

## 2. Training `data.yaml`

**To be created** (Code mode) at **`dataset/abonia_eval/finetune/data.yaml`** (git-ignored
under `dataset/*`). The repo's own
[`datasets/fire-8/data.yaml`](../dataset/abonia_repo/datasets/fire-8/data.yaml:1) is NOT
reused as-is because its Colab-relative paths (`train: fire-8/train/images`,
`test: ../test/images`) do not resolve against the committed layout, and its names
(`Fire/default/smoke`) would overwrite the checkpoint's class contract on save. The copy
below fixes both (absolute paths to the real splits + model names `fire/other/smoke`).

```yaml
# Fine-tune HF YOLO26-S best.pt on Abonia fire-8 (Roboflow export, CC BY 4.0).
# Class order matches the checkpoint: fire=0, other=1, smoke=2.
# Abonia labels: Fire(0), default(1), smoke(2) -> names below are relabels only (no reorder).
path: /mnt/Main/Others/Programming/mazr3a/dataset/abonia_repo/datasets/fire-8
train: train/images
val: valid/images
test: test/images
nc: 3
names:
  0: fire
  1: other
  2: smoke
```

> **Colab note:** after uploading the `fire-8` folder, edit the `path:` line to the Colab
> location (e.g. `/content/fire-8`).

---

## 3. Commands

### 3.1 Local CPU smoke run (this repo, `.venv/`)

```bash
.venv/bin/python - <<'PY'
from ultralytics import YOLO
m = YOLO("models/fire/best.pt")          # start from the production checkpoint
m.train(data="dataset/abonia_eval/finetune/data.yaml",
        epochs=5, imgsz=640, batch=8, device="cpu",
        project="dataset/abonia_eval/finetune", name="smoke",
        freeze=10, lr0=0.001, plots=False, workers=2,
        cache=False, exist_ok=True, verbose=True)
PY
```

Expected outcome: clean run, ~`time/epoch` printed each epoch (this is the metric the
user wants to measure), and `dataset/abonia_eval/finetune/smoke/weights/best.pt`.

### 3.2 Promote smoke candidate + quick test-split eval

```bash
cp dataset/abonia_eval/finetune/smoke/weights/best.pt dataset/abonia_eval/finetune/best_finetuned_abonia.pt
.venv/bin/python scripts/test_fire_model.py dataset/abonia_eval/finetune/best_finetuned_abonia.pt \
    dataset/abonia_eval/eval/data.yaml --out dataset/abonia_eval/finetune/eval_smoke --conf 0.5 --annotate
```

Compares directly against the baseline numbers in §1 (same harness, same test split).

### 3.3 Full run — GPU / Google Colab (user)

```python
!pip install -q ultralytics
from ultralytics import YOLO
m = YOLO("best.pt")                      # upload models/fire/best.pt
# upload the fire-8 dataset and fix `path:` in data.yaml to the Colab dir
m.train(data="data.yaml",
        epochs=100, imgsz=640, batch=16, device=0,   # adjust batch to VRAM
        project="abonia_finetune", name="run1",
        freeze=10, lr0=0.001, patience=20,
        plots=True, exist_ok=True, verbose=True)
```

Then download `abonia_finetune/run1/weights/best.pt` → copy into this repo as
`dataset/abonia_eval/finetune/best_finetuned_abonia.pt` and run the §3.2 eval command
(pointing `--out` to `eval_full`).

**Hyperparameter rationale:**
- `freeze=10` — keeps the early backbone features, big CPU speed-up, less forgetting of
  the original fire/smoke distribution; relax to `freeze=0` only if GPU time allows and
  if the user wants full adaptation.
- `lr0=0.001` — lower than the from-scratch default (0.01) for a pretrained fine-tune.
- `patience=20` — stop early on the Abonia `valid` split to limit overfitting to the
  small domain.
- `imgsz=640` — matches the checkpoint/production input. (Dataset is native 608×608.)

---

## 4. Implementation checklist

- [ ] Create `dataset/abonia_eval/finetune/data.yaml` (§2) — Code mode.
- [ ] Run the CPU smoke run (§3.1) — confirm pipeline + measure time/epoch (Code mode).
- [ ] Promote smoke `best.pt` and evaluate on the test split (§3.2) (Code mode).
- [ ] Record smoke-run results + measured time/epoch in this plan's §6.
- [ ] Commit plan + any script/yaml scaffolding (local-only).
- [ ] User runs the full GPU/Colab training (§3.3), drops the weights here, we evaluate
      and compare vs §1 baseline, then decide whether to promote + deploy (separate step).

---

## 5. Risks / caveats

- **Small domain:** 877 training images from fire/smoke videos/web — domain-shift vs the
  original HF data and the farm cameras; fine-tuning can overfit or reduce generality.
  Mitigations: early stopping on `valid`, modest `freeze`, low LR; judge strictly by the
  **test split**, not `valid` (both come from the same source pool).
- **Overlap:** Abonia test may partly share source material with the HF checkpoint's
  training pool, so even the improved numbers stay optimistic for the farm cameras. A
  `--dry-run` on-camera pilot remains the real acceptance test.
- **Class-order drift:** verify the fine-tuned `.names` is still `fire/other/smoke` after
  training (the §2 names strings enforce this).
- **Do not promote blindly:** promoting would change the deployed model — a separate,
  explicit step that must also regenerate the OpenVINO IR + `labelmap.txt`
  ([`scripts/prep_fire_model.sh`](../scripts/prep_fire_model.sh)) and deploy/test on
  `ssh.mazr3a.garden` (per `.roo/rules/sshuser.md`).

---

## 6. Implementation log (filled as work completes)

- *(to be appended: smoke-run outcome, time/epoch, eval numbers)*

## 7. Git & remote handoff

- Commit plan + yaml scaffolding + scripts after each logical change (per
  `.roo/rules/Agents.md`); artifacts under `dataset/*` stay untracked.
- No production/firewatch change in this phase → no remote deploy. Model promotion +
  OpenVINO export + on-host verification is a separate follow-up plan if the numbers
  justify it.
