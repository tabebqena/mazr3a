#!/usr/bin/env python3
"""build_fire_large_colab_nb.py - generate notebooks/fire-large-finetune-colab.ipynb.

A ready-to-import Google Colab notebook that fine-tunes models/fire/best.pt (YOLO26-S,
classes fire/other/smoke) on the large local fire/smoke dataset built by
dev_scripts/prep_fire_large_dataset.py (<out>_colab.zip).

SURVIVES A RUNTIME RESET / DISCONNECT
-------------------------------------
* the DATASET and the BASE checkpoint are read from Google Drive;
* the RUN OUTPUT DIR (`project=`) is on Google Drive, so Ultralytics' per-epoch
  `weights/last.pt` (and `best.pt`, `results.csv`, `args.yaml`) are written there
  and survive when the Colab runtime is recycled;
* a dedicated **RESUME** cell restarts training from
  `<Drive>/runs/<NAME>/weights/last.pt` (Ultralytics restores the full run state
  from the `args.yaml` saved in that same directory).
* the dataset is extracted to LOCAL disk (`/content/...`) for fast reads; the
  extract step is idempotent, so re-running it after a reset is a no-op.

Data entry supports either (a) Google Drive (recommended for the ~600-700 MB zip) or
(b) files.upload(). Set the DRIVE_* variables to "" to trigger an upload dialog instead.

Usage:
    python dev_scripts/build_fire_large_colab_nb.py   # writes notebooks/fire-large-finetune-colab.ipynb
"""
import json
import os

OUT = "notebooks/fire-large-finetune-colab.ipynb"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# --------------------------------------------------------------------------- #
# Cell sources (list of lines, each ending with newline)
# --------------------------------------------------------------------------- #
C_TITLE = md(
    [
        "# Fine-tune the fire/smoke model (large dataset)\n",
        "\n",
        "Fine-tunes a base checkpoint (HF **YOLO26-S**, classes `fire(0)/other(1)/smoke(2)`) "
        "on the dataset zip built by `dev_scripts/prep_fire_large_dataset.py`.\n",
        "\n",
        "**Crash-resilient:** the dataset + base checkpoint are read from **Google Drive**, and the "
        "**training run directory is written to Drive** (`project=DRIVE_RUNS`), so the per-epoch "
        "`weights/last.pt` survives a runtime reset. If the runtime dies, just re-run the setup cells "
        "and the **RESUME** cell.\n",
        "\n",
        "- Plan: [`plans/fire-model-large-finetune-colab.md`](../plans/fire-model-large-finetune-colab.md)\n",
        "- Dataset prep (run once, locally), v5 example:\n",
        "  `python dev_scripts/prep_fire_large_dataset.py --ready fire-model-training/ready_fire_smoke_dataset.yolov8 "
        "--negatives fire-model-training/v5_negatives_dedup_rfs --out fire-model-training/v5_finetune_firerich "
        "--val-frac 0.05 --seed 0 --zip`\n",
        "  → produces `fire-model-training/v5_finetune_firerich_colab.zip` (already `train/`, `val/`, `data.yaml`).\n",
        "- **Class contract:** fire=0 / other=1 / smoke=2. Export is index-aligned; labels are **not** "
        "remapped, only the `data.yaml` names are set. `other` is trained but ignored in production.\n",
    ]
)

C_SETUP = code(
    [
        "# Cell 1 - runtime + install a current ultralytics (YOLO26 needs >= ~8.3)\n",
        "!nvidia-smi\n",
        "!pip install -q --upgrade ultralytics\n",
        "\n",
        "import ultralytics, torch\n",
        "print('ultralytics', ultralytics.__version__)\n",
        "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\n",
    ]
)

C_CONFIG = md(
    [
        "## Configuration\n",
        "\n",
        "All `DRIVE_*` paths live on Google Drive so the dataset, the base checkpoint and - most "
        "importantly - the **run outputs** survive a disconnect. Set either of the first two to `\"\"` "
        "and the matching step pops up an upload dialog instead.\n",
        "\n",
        "`DRIVE_RUNS` is passed as Ultralytics `project=`, so every epoch writes "
        "`DRIVE_RUNS/<NAME>/weights/last.pt` straight to Drive.\n",
    ]
)

C_CONFIG_CODE = code(
    [
        "# Cell 2 - EDIT THIS\n",
        "\n",
        "DRIVE_DIR  = '/content/drive/MyDrive/mazr3a'          # <-- your Drive folder\n",
        "DRIVE_ZIP  = DRIVE_DIR + '/v5_finetune_firerich_colab.zip'   # dataset zip on Drive (or '')\n",
        "DRIVE_BEST = DRIVE_DIR + '/models_fire_best.pt'       # base checkpoint on Drive (or '')\n",
        "DRIVE_RUNS = DRIVE_DIR + '/runs'                      # RUN OUTPUTS ON DRIVE -> survive a reset\n",
        "\n",
        "# --- training hyperparameters (tune freely) ---\n",
        "EPOCHS   = 50     # patience will stop early\n",
        "BATCH    = 16     # reduce to 8 if OOM on a T4; raise on A100/L4\n",
        "IMGSZ    = 640    # matches production input\n",
        "FREEZE   = 10     # keep early backbone (faster, less forgetting); 0 = full fine-tune\n",
        "LR0      = 0.001  # low LR for a pretrained fine-tune\n",
        "PATIENCE = 15     # early stop on the held-out val split\n",
        "NAME     = 'v5'   # run name -> DRIVE_RUNS/<NAME>/weights/...\n",
        "\n",
        "WORKDIR  = '/content/v5_data'   # LOCAL extract (fast reads); idempotent, re-run is a no-op\n",
    ]
)

C_MOUNT = code(
    [
        "# Cell 3 - (only needed if you use Drive paths) mount Drive\n",
        "if DRIVE_ZIP or DRIVE_BEST or DRIVE_RUNS:\n",
        "    from google.colab import drive\n",
        "    drive.mount('/content/drive')\n",
        "else:\n",
        "    print('Drive not required - using file upload dialogs.')\n",
        "os.makedirs(DRIVE_RUNS, exist_ok=True)\n",
        "print('runs will persist in:', DRIVE_RUNS)\n",
    ]
)

C_DATASET = code(
    [
        "# Cell 4 - get + extract the dataset (IDEMPOTENT), then fix data.yaml for Colab\n",
        "import glob, os, shutil, zipfile\n",
        "\n",
        "os.makedirs(WORKDIR, exist_ok=True)\n",
        "yaml_path = os.path.join(WORKDIR, 'data.yaml')\n",
        "\n",
        "if os.path.isfile(yaml_path):\n",
        "    print('dataset already extracted ->', WORKDIR, '(skipping extract)')\n",
        "else:\n",
        "    data_zip = None\n",
        "    if DRIVE_ZIP and os.path.exists(DRIVE_ZIP):\n",
        "        data_zip = DRIVE_ZIP\n",
        "    else:\n",
        "        if DRIVE_ZIP:\n",
        "            print('Drive zip not found - uploading instead.')\n",
        "        from google.colab import files\n",
        "        uploaded = files.upload()\n",
        "        data_zip = next(iter(uploaded))\n",
        "    print('data zip:', data_zip)\n",
        "    with zipfile.ZipFile(data_zip) as z:\n",
        "        z.extractall(WORKDIR)\n",
        "\n",
        "assert os.path.isfile(yaml_path), 'data.yaml not found after extract: ' + yaml_path\n",
        "\n",
        "# rewrite the absolute local `path:` to the Colab location\n",
        "lines = []\n",
        "for ln in open(yaml_path, encoding='utf-8'):\n",
        "    if ln.startswith('path:'):\n",
        "        lines.append(f'path: {WORKDIR}\\n')\n",
        "    else:\n",
        "        lines.append(ln)\n",
        "open(yaml_path, 'w', encoding='utf-8').writelines(lines)\n",
        "\n",
        "print('data.yaml ->', yaml_path)\n",
        "print(open(yaml_path, encoding='utf-8').read())\n",
    ]
)

C_VERIFY_DATASET = code(
    [
        "# Cell 5 - verify the extracted layout + class coverage\n",
        "import collections, os\n",
        "\n",
        "def tally(img_dir, lbl_dir):\n",
        "    n = n_box = 0\n",
        "    cls = collections.Counter()\n",
        "    for f in sorted(os.listdir(img_dir)):\n",
        "        if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):\n",
        "            continue\n",
        "        n += 1\n",
        "        lb = os.path.join(lbl_dir, os.path.splitext(f)[0] + '.txt')\n",
        "        if os.path.isfile(lb):\n",
        "            for row in open(lb, encoding='utf-8'):\n",
        "                if row.split():\n",
        "                    cls[int(float(row.split()[0]))] += 1\n",
        "                    n_box += 1\n",
        "    return n, n_box, cls\n",
        "\n",
        "for split in ('train', 'val'):\n",
        "    n, nb, cls = tally(os.path.join(WORKDIR, split, 'images'), os.path.join(WORKDIR, split, 'labels'))\n",
        "    names = {0: 'fire', 1: 'other', 2: 'smoke'}\n",
        "    print(f'{split}: images={n} boxes={nb} per-class=' +\n",
        "          ', '.join(f'{names[k]}={v}' for k, v in sorted(cls.items())))\n",
    ]
)

C_MODEL = code(
    [
        "# Cell 6 - get the BASE checkpoint and verify class order\n",
        "base_pt = None\n",
        "if DRIVE_BEST and os.path.exists(DRIVE_BEST):\n",
        "    base_pt = DRIVE_BEST\n",
        "else:\n",
        "    if DRIVE_BEST:\n",
        "        print('Drive best.pt not found - uploading instead.')\n",
        "    from google.colab import files\n",
        "    up = files.upload()\n",
        "    base_pt = next(k for k in up if k.endswith('.pt'))\n",
        "print('base:', base_pt)\n",
        "\n",
        "from ultralytics import YOLO\n",
        "m = YOLO(base_pt)\n",
        "assert list(m.names.values()) == ['fire', 'other', 'smoke'], m.names\n",
        "print('class order OK ->', m.names)\n",
    ]
)

C_TRAIN = code(
    [
        "# Cell 7 - FINE-TUNE (GPU). project=DRIVE_RUNS => every epoch is checkpointed to DRIVE.\n",
        "# If this runtime dies, the last.pt in DRIVE_RUNS survives; run Cell 8 to resume.\n",
        "m.train(data=os.path.join(WORKDIR, 'data.yaml'),\n",
        "        epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0,\n",
        "        project=DRIVE_RUNS, name=NAME, exist_ok=True,\n",
        "        freeze=FREEZE, lr0=LR0, patience=PATIENCE,\n",
        "        save_period=1,            # write weights every epoch (survival)\n",
        "        plots=True, verbose=True)\n",
    ]
)

C_RESUME = code(
    [
        "# Cell 8 - RESUME (ONLY if the run is genuinely INCOMPLETE)\n",
        "#\n",
        "# Ultralytics STRIPS the optimizer from last.pt when training finishes, so a\n",
        "# COMPLETED checkpoint is NOT resumable: calling resume on it silently starts a\n",
        "# brand-new training run - on the default coco8.yaml - wasting hours. This cell\n",
        "# inspects the checkpoint first and refuses unless it is truly resumable.\n",
        "import os, torch\n",
        "\n",
        "last = os.path.join(DRIVE_RUNS, NAME, 'weights', 'last.pt')\n",
        "best = os.path.join(DRIVE_RUNS, NAME, 'weights', 'best.pt')\n",
        "print('checkpoint:', last)\n",
        "assert os.path.exists(last), 'no checkpoint found - run the TRAIN cell first: ' + last\n",
        "\n",
        "ckpt = torch.load(last, map_location='cpu', weights_only=False)\n",
        "ta = ckpt.get('train_args') or {}\n",
        "done = int(ckpt.get('epoch', -1)) + 1\n",
        "total = int(ta.get('epochs', 0) or 0)\n",
        "resumable = ckpt.get('optimizer') is not None\n",
        "print(f'epoch {done}/{total}  -  optimizer state: '\n",
        "      f'{\"present\" if resumable else \"STRIPPED (run finished)\"}')\n",
        "\n",
        "if not resumable:\n",
        "    print('=> nothing to resume: this run is COMPLETE.')\n",
        "    print('   Use the best checkpoint for evaluation: ', best)\n",
        "else:\n",
        "    from ultralytics import YOLO\n",
        "    YOLO(last).train(resume=True)\n",
    ]
)

C_AFTER = code(
    [
        "# Cell 9 - post-train: verify .names, keep a clean candidate, bundle + download\n",
        "import os, shutil, zipfile\n",
        "\n",
        "run_dir = os.path.join(DRIVE_RUNS, NAME)\n",
        "best = os.path.join(run_dir, 'weights', 'best.pt')\n",
        "last = os.path.join(run_dir, 'weights', 'last.pt')\n",
        "print('best exists:', os.path.exists(best), '->', best)\n",
        "\n",
        "fin = YOLO(best if os.path.exists(best) else last)\n",
        "print('final .names:', fin.names)\n",
        "assert list(fin.names.values()) == ['fire', 'other', 'smoke']\n",
        "\n",
        "# a stable copy at the Drive folder root (survives even if the run dir is cleaned)\n",
        "cand_drive = os.path.join(DRIVE_DIR, 'best_finetuned.pt')\n",
        "shutil.copy(best if os.path.exists(best) else last, cand_drive)\n",
        "print('candidate saved on Drive ->', cand_drive)\n",
        "\n",
        "cand = 'best_finetuned.pt'\n",
        "shutil.copy(cand_drive, cand)\n",
        "zname = 'fire_finetune_results.zip'\n",
        "with zipfile.ZipFile(zname, 'w', zipfile.ZIP_DEFLATED) as z:\n",
        "    z.write(cand, cand)\n",
        "    for root, _dirs, files in os.walk(run_dir):\n",
        "        for f in files:\n",
        "            full = os.path.join(root, f)\n",
        "            z.write(full, os.path.relpath(full, DRIVE_RUNS))\n",
        "print('results zip ->', zname)\n",
        "\n",
        "from google.colab import files\n",
        "files.download(zname)\n",
    ]
)

C_NEXT = md(
    [
        "## Next steps after the run\n",
        "\n",
        "1. `best_finetuned.pt` is already on Drive (`DRIVE_DIR/best_finetuned.pt`) plus inside the "
        "results zip - pull it back into the repo as "
        "`fire-model-training/<out>/best_finetuned.pt`.\n",
        "2. Run the LOCAL evaluation (FP on `eval/negatives` must drop; fire recall on "
        "`dedup/dfire_clean_eval` must not) - see [`plans/fire-model-large-finetune-colab.md`]"
        "(../plans/fire-model-large-finetune-colab.md) §5.\n",
        "3. If metrics justify it, archive as a versioned candidate (`models/fire/versions/`, per "
        "[`plans/model-versioning.md`](../plans/model-versioning.md)), then "
        "`prep_fire_model.sh` → `promote_fire_model.sh` → on-host `firewatch.py --dry-run`.\n",
    ]
)


def build():
    cells = [
        C_TITLE,
        C_SETUP,
        C_CONFIG,
        C_CONFIG_CODE,
        C_MOUNT,
        C_DATASET,
        C_VERIFY_DATASET,
        C_MODEL,
        C_TRAIN,
        C_RESUME,
        C_AFTER,
        C_NEXT,
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
