#!/usr/bin/env python3
"""build_colab_fireviewer_nb.py - generate the FireViewer v6 Colab bootstrap notebook.

Prepares the dataset ON the Colab machine so that nothing large is uploaded: the corpus is
pulled from HuggingFace at datacenter speed and converted in place, while only a ~83 MB
bundle (our domain negatives + the v4 checkpoint + this converter) is uploaded.

The bundle is expected on Drive at:
    /content/drive/MyDrive/colab-data/v6-training/colab_upload.zip

Flow of the generated notebook:
  1. install deps (ultralytics, pyarrow, huggingface_hub)
  2. mount Drive and unpack the upload bundle (`colab_upload.zip`)
  3. snapshot_download ONLY the parquet it trains on (data/train + data/validation) into
     /content/fvcache  -> accepted directly by prep_fireviewer_dataset.py (plain data/ root)
  4. convert parquet -> YOLO at /content/fv_yolo (train/ + val/, no test split)
  5. add the uploaded domain negatives to TRAIN as background (empty labels); keep the
     audit images (our real fires) OUT of training
  6. delete the parquet cache to reclaim ~25 GB
  7. fine-tune the v4 checkpoint with the run dir on Drive (survives a disconnect) +
     a guarded RESUME cell that refuses to restart a finished run
  8. copy the candidate checkpoint back to Drive

The corpus `test` split is deliberately NOT downloaded: the trained checkpoint is judged
locally against the full 102,257-image YOLO export we already hold.

Usage:
    python dev_scripts/build_colab_fireviewer_nb.py \
        --out model-training/datasets--fireviewer--fire-smoke-detection-corpus-v1/fireviewer-v6-bootstrap-colab.ipynb
"""
import argparse
import json
import os

DEFAULT_OUT = "notebooks/fireviewer-v6-bootstrap-colab.ipynb"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


C_TITLE = md([
    "# Fire model v6 — bootstrap + fine-tune on the FireViewer corpus\n",
    "\n",
    "Downloads the corpus from HuggingFace **on the Colab machine** (nothing large is\n",
    "uploaded), converts it to YOLO, adds our domain negatives, then fine-tunes **v4**\n",
    "(`models/fire/best.pt`).\n",
    "\n",
    "**Before you start:** upload `colab_upload.zip` (built locally by\n",
    "`dev_scripts/pack_colab_upload.sh`, ~83 MB) to\n",
    "`MyDrive/colab-data/v6-training/`.\n",
    "\n",
    "Crash-resilient: the run dir is on Drive, so a disconnect is recoverable.\n",
    "The corpus `test` split is intentionally not downloaded — judge the checkpoint locally.\n",
    "Plan: `plans/fire-model-v6-fireviewer-training.md` §12.\n",
])

C_SETUP = code([
    "# Cell 1 - deps + GPU\n",
    "!nvidia-smi\n",
    "!pip -q install --upgrade ultralytics pyarrow huggingface_hub\n",
    "import ultralytics, torch, pyarrow, huggingface_hub\n",
    "print('ultralytics', ultralytics.__version__, '| pyarrow', pyarrow.__version__)\n",
    "print('torch', torch.__version__, '| cuda', torch.cuda.is_available())\n",
])

C_CONFIG = code([
    "# Cell 2 - config (edit if needed)\n",
    "DRIVE_DIR  = '/content/drive/MyDrive/colab-data/v6-training'\n",
    "HF_REPO    = 'fireviewer/fire-smoke-detection-corpus-v1'\n",
    "CACHE      = '/content/fvcache'    # parquet download (temporary, deleted in Cell 7)\n",
    "YOLO_DS    = '/content/fv_yolo'    # converted YOLO dataset (train/ + val/)\n",
    "UPLOAD     = '/content/upload'     # unpacked colab_upload.zip\n",
    "UPLOAD_ZIP = DRIVE_DIR + '/colab_upload.zip'\n",
    "RUNS       = DRIVE_DIR + '/runs'   # RUN OUTPUTS ON DRIVE -> survive a reset\n",
    "\n",
    "# --- training mix ---\n",
    "# ADD_NEGATIVES=True injects our 841 local frames (cam dog/warm-object FPs + generic\n",
    "# background) into TRAIN. At 841 / 60,981 = 1.4 % they are far too diluted to fix the\n",
    "# cam01 dog FP on their own - this run is mainly about GENERALISATION on the corpus.\n",
    "# They are cheap and target the one documented production failure, so True is a fine\n",
    "# default; set False for a clean corpus-only attribution run and compare (plan §14).\n",
    "ADD_NEGATIVES = True\n",
    "\n",
    "# --- disk ---\n",
    "# Cell 7 deletes the ~25 GB parquet cache before training to free space. Set False if you\n",
    "# intend several runs in one runtime (Run A / Run B): the cache is then RE-USED instead of\n",
    "# re-downloaded; ~49 GB stays on disk (free-tier /content is ~78 GB, so it fits).\n",
    "FREE_CACHE = True\n",
    "\n",
    "# --- training hyperparameters ---\n",
    "EPOCHS   = 15     # ~61k train images/epoch; raise only if time allows\n",
    "BATCH    = 16     # lower to 8 on a T4 if OOM\n",
    "IMGSZ    = 640    # matches production input\n",
    "FREEZE   = 10     # keep early backbone (less forgetting); 0 = full fine-tune\n",
    "LR0      = 0.001  # low LR for a pretrained fine-tune\n",
    "PATIENCE = 5      # early stop on the held-out val split\n",
    "NAME     = 'v6-fireviewer'\n",
])

C_MOUNT = code([
    "# Cell 3 - mount Drive + unpack the (small) upload bundle\n",
    "import os, zipfile, glob, shutil\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "os.makedirs(RUNS, exist_ok=True)\n",
    "\n",
    "if not os.path.isdir(UPLOAD) or not glob.glob(UPLOAD + '/*'):\n",
    "    if os.path.exists(UPLOAD_ZIP):\n",
    "        os.makedirs(UPLOAD, exist_ok=True)\n",
    "        with zipfile.ZipFile(UPLOAD_ZIP) as z:\n",
    "            z.extractall(UPLOAD)\n",
    "    else:\n",
    "        from google.colab import files\n",
    "        up = files.upload()\n",
    "        os.makedirs(UPLOAD, exist_ok=True)\n",
    "        with zipfile.ZipFile(next(iter(up))) as z:\n",
    "            z.extractall(UPLOAD)\n",
    "\n",
    "print('bundle contents:', sorted(os.listdir(UPLOAD)))\n",
    "assert os.path.exists(UPLOAD + '/model/best.pt'), 'best.pt missing from the bundle'\n",
    "print('base checkpoint:', UPLOAD + '/model/best.pt')\n",
    "print('runs will persist in:', RUNS)\n",
])

C_DOWNLOAD = code([
    "# Cell 4 - download ONLY the parquet we train on (train + validation), idempotently\n",
    "from huggingface_hub import snapshot_download\n",
    "\n",
    "if glob.glob(CACHE + '/data/train/*.parquet'):\n",
    "    print('corpus already downloaded ->', CACHE)\n",
    "else:\n",
    "    snapshot_download(HF_REPO, repo_type='dataset', local_dir=CACHE,\n",
    "                      allow_patterns=['data/train/*', 'data/validation/*'])\n",
    "\n",
    "for sp in ('train', 'validation'):\n",
    "    n = len(glob.glob(f'{CACHE}/data/{sp}/*.parquet'))\n",
    "    print(f'  {sp:11s} {n} parquet shard(s)')\n",
    "print('NOTE: the corpus test split is deliberately NOT downloaded.')\n",
])

C_CONVERT = code([
    "# Cell 5 - convert parquet -> YOLO on the Colab disk (reuses the repo converter)\n",
    "import subprocess, sys\n",
    "\n",
    "conv = UPLOAD + '/scripts/prep_fireviewer_dataset.py'\n",
    "if os.path.isfile(YOLO_DS + '/data.yaml'):\n",
    "    print('already converted ->', YOLO_DS)\n",
    "else:\n",
    "    subprocess.run([sys.executable, conv, '--corpus', CACHE, '--out', YOLO_DS,\n",
    "                    '--splits', 'train,validation'], check=True)\n",
    "print(open(YOLO_DS + '/data.yaml').read())\n",
])

C_NEGATIVES = code([
    "# Cell 6 - add our domain negatives to TRAIN (empty label = background)\n",
    "neg = UPLOAD + '/negatives'\n",
    "ti, tl = YOLO_DS + '/train/images', YOLO_DS + '/train/labels'\n",
    "added = 0\n",
    "if ADD_NEGATIVES:\n",
    "    for f in sorted(glob.glob(neg + '/*')):\n",
    "        if os.path.splitext(f)[1].lower() not in ('.jpg', '.jpeg', '.png'):\n",
    "            continue\n",
    "        base = os.path.basename(f)\n",
    "        if not os.path.exists(os.path.join(ti, base)):\n",
    "            shutil.copy2(f, os.path.join(ti, base))\n",
    "        open(os.path.join(tl, os.path.splitext(base)[0] + '.txt'), 'w').close()\n",
    "        added += 1\n",
    "    print('negatives added to TRAIN:', added)\n",
    "else:\n",
    "    print('ADD_NEGATIVES=False -> corpus-only run (no local negatives injected)')\n",
    "print('audit set (our real fires, HELD OUT, never trained):', UPLOAD + '/audit',\n",
    "      len(glob.glob(UPLOAD + '/audit/*')), 'imgs')\n",
])

C_VERIFY = code([
    "# Cell 7 - verify layout + class coverage, then reclaim the parquet cache\n",
    "import collections\n",
    "\n",
    "def tally(split):\n",
    "    idir, ldir = f'{YOLO_DS}/{split}/images', f'{YOLO_DS}/{split}/labels'\n",
    "    n = boxes = 0\n",
    "    cls = collections.Counter()\n",
    "    for f in sorted(os.listdir(idir)):\n",
    "        n += 1\n",
    "        lb = os.path.join(ldir, os.path.splitext(f)[0] + '.txt')\n",
    "        if os.path.isfile(lb):\n",
    "            for row in open(lb):\n",
    "                if row.split():\n",
    "                    cls[int(float(row.split()[0]))] += 1\n",
    "                    boxes += 1\n",
    "    return n, boxes, cls\n",
    "\n",
    "names = {0: 'fire', 1: 'other', 2: 'smoke'}\n",
    "for sp in ('train', 'val'):\n",
    "    n, b, c = tally(sp)\n",
    "    print(f'{sp}: images={n} boxes={b} ' +\n",
    "          ', '.join(f'{names[k]}={v}' for k, v in sorted(c.items())))\n",
    "\n",
    "if FREE_CACHE:\n",
    "    shutil.rmtree(CACHE, ignore_errors=True)\n",
    "    free = shutil.disk_usage('/content').free / 1e9\n",
    "    print(f'parquet cache removed -> {CACHE} | /content free: {free:.1f} GB')\n",
    "else:\n",
    "    print(f'FREE_CACHE=False -> keeping {CACHE} (re-used by a second run)')\n",
])

C_MODEL = code([
    "# Cell 8 - load the v4 base checkpoint and verify the class order\n",
    "from ultralytics import YOLO\n",
    "base = UPLOAD + '/model/best.pt'\n",
    "m = YOLO(base)\n",
    "assert list(m.names.values()) == ['fire', 'other', 'smoke'], m.names\n",
    "print('base:', base, '| class order OK ->', m.names)\n",
])

C_TRAIN = code([
    "# Cell 9 - FINE-TUNE. project=RUNS puts every epoch on Drive -> survives a reset.\n",
    "# If this runtime dies, re-run Cells 1-3 then Cell 10 to resume.\n",
    "m.train(data=YOLO_DS + '/data.yaml',\n",
    "        epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0,\n",
    "        project=RUNS, name=NAME, exist_ok=True,\n",
    "        freeze=FREEZE, lr0=LR0, patience=PATIENCE,\n",
    "        save_period=1, plots=True, verbose=True)\n",
])

C_RESUME = code([
    "# Cell 10 - RESUME (only if the run is genuinely INCOMPLETE)\n",
    "# Ultralytics strips the optimizer from a FINISHED last.pt, and resuming that would\n",
    "# silently start a brand-new run on the default dataset. Inspect first, then decide.\n",
    "import torch\n",
    "last = os.path.join(RUNS, NAME, 'weights', 'last.pt')\n",
    "print('checkpoint:', last)\n",
    "assert os.path.exists(last), 'nothing to resume - run Cell 9 first'\n",
    "ckpt = torch.load(last, map_location='cpu', weights_only=False)\n",
    "ta = ckpt.get('train_args') or {}\n",
    "done, total = int(ckpt.get('epoch', -1)) + 1, int(ta.get('epochs', 0) or 0)\n",
    "resumable = ckpt.get('optimizer') is not None\n",
    "print(f'epoch {done}/{total} - optimizer {\"present\" if resumable else \"STRIPPED (finished)\"}')\n",
    "if resumable:\n",
    "    YOLO(last).train(resume=True)\n",
    "else:\n",
    "    print('run is COMPLETE - use best.pt')\n",
])

C_SAVE = code([
    "# Cell 11 - keep the candidate on Drive (+ md5 for the version registry)\n",
    "import hashlib, shutil\n",
    "run = os.path.join(RUNS, NAME)\n",
    "best, last = os.path.join(run, 'weights', 'best.pt'), os.path.join(run, 'weights', 'last.pt')\n",
    "src = best if os.path.exists(best) else last\n",
    "cand = os.path.join(DRIVE_DIR, 'best_fireviewer_v6.pt')\n",
    "shutil.copy(src, cand)\n",
    "md5 = hashlib.md5(open(cand, 'rb').read()).hexdigest()\n",
    "print('candidate ->', cand)\n",
    "print('md5:', md5, '| bytes:', os.path.getsize(cand))\n",
    "fin = YOLO(cand)\n",
    "assert list(fin.names.values()) == ['fire', 'other', 'smoke'], fin.names\n",
    "print('names OK ->', fin.names)\n",
])

C_NEXT = md([
    "## Next\n",
    "\n",
    "Pull `best_fireviewer_v6.pt` from Drive into\n",
    "`model-training/fireviewer_v6_finetune/best_finetuned.pt`, then judge it **locally**\n",
    "against the full export (v4 baseline in the plan, §11):\n",
    "\n",
    "```bash\n",
    "FV=model-training/datasets--fireviewer--fire-smoke-detection-corpus-v1\n",
    "sed 's#val: val/images#val: test/images#' \\\n",
    "    $FV/fireviewer_v1_yolo/data.yaml > $FV/fireviewer_v1_yolo/data_test.yaml\n",
    ".venv/bin/python dev_scripts/test_fire_model.py <ckpt> \\\n",
    "    $FV/fireviewer_v1_yolo/data_test.yaml --conf 0.5 --out <out>\n",
    "```\n",
    "\n",
    "Then run the recall guard (`dedup/dfire_clean_eval`) and only promote if fire recall holds.\n",
])


def build(out=DEFAULT_OUT):
    cells = [C_TITLE, C_SETUP, C_CONFIG, C_MOUNT, C_DOWNLOAD, C_CONVERT,
             C_NEGATIVES, C_VERIFY, C_MODEL, C_TRAIN, C_RESUME, C_SAVE, C_NEXT]
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
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"wrote {out} ({len(cells)} cells)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="notebook output path (default: %(default)s)")
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
