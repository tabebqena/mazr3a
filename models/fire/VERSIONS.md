# Fire/smoke model version registry — `models/fire/versions/`

**Tracked index** of every fire/smoke checkpoint version. The checkpoints (`model.pt`,
large) and per-version `VERSION.json` manifests are **git-ignored** under
`models/fire/versions/`; the inline tables below are the durable, reviewable record (they
survive even where the `.pt` files are not present).

- **Naming convention + full design:** [`plans/model-versioning.md`](../plans/model-versioning.md)
- **Promote helper:** [`dev_scripts/promote_fire_model.sh`](../dev_scripts/promote_fire_model.sh)
- **Deploy semantics:** the `models/fire/` root `best.*` files are the *active* set that
  `firewatch` loads (`MODEL_DIR=/models/fire`, hardcodes `best.xml`). They are a copy of
  the active version; `versions/` holds the canonical archive.

> **Class order for every version below: `fire`(0) / `other`(1) / `smoke`(2).** Production
> is fire-only (index 0), so alerts are unaffected; keep `labelmap.txt` in this order when
> regenerating IR ([`dev_scripts/prep_fire_model.sh`](../dev_scripts/prep_fire_model.sh)).

---

## Active

| Version | Description |
|---|---|
| `v4-2026-09-07-hf-dfire-cleanft52ep` | **Clean D-Fire fine-tune** (v1 → 6,819 clean imgs = D-Fire clean train + clean Abonia, 52 ep best@27, T4). Promoted to ACTIVE 2026-09-07 (user decision). On the held-out **clean D-Fire test (2,164 imgs)**: all mAP@50 **0.704** (v1 0.182), fire mAP@50 0.643, image-level fire recall @0.5 **83.8 %** (v1 73.5 %), fire FP 11/1440 (v1 51/1440). v1 superseded but retained in the archive. |

> **v6 candidate (2026-09-13)** — FireViewer-corpus fine-tune of v4, row 4 below. Numerically a
> large win on that corpus, but **NOT promoted**: it still needs an honest **held-out** FP audit
> (the 72-image audit set overlaps its own training negatives, so 0/72 is not generalization)
> and a clean-D-Fire recall guard before the on-camera `firewatch.py --dry-run` pilot.

---

## All versions

| # | Version dir (`versions/<dir>/model.pt`) | Date | Base / source | Training set | Key metrics | Status |
|---|---|---|---|---|---|---|
| 1 | `v1-2026-09-05-hf-yolo26s-8939img` | 2026-09-05 | YOLO26-S from `huggingface.co/SalahALHaismawi/yolov26-fire-detection` (MIT) | 8,939 imgs / 100 ep @ 640 (author Roboflow pool) | author-reported mAP@50 **0.949** / mAP@50-95 0.680 / P 0.896 / R 0.888 | superseded (kept) |
| 2 | `v2-2026-09-05-hf-abonia877-ft5ep` | 2026-09-05 | fine-tune of v1 on Abonia `fire-8` **train** (CC BY 4.0) | 877 imgs / 5 ep @ 640, freeze 10, lr0 0.001 (CPU smoke run) | Abonia *test* split (55 imgs): all mAP@50 **0.948** / fire 0.930 / smoke 0.965; vs v1 baseline 0.405 | candidate |
| 3 | `v4-2026-09-07-hf-dfire-cleanft52ep` | 2026-09-07 | fine-tune of v1 on the CLEAN (deduplicated) D-Fire + clean Abonia mix (see `fire-model-training/dedup/dfire_finetune/`) | 6,819 imgs / 52 ep (best@27) @ 640, freeze 10, lr0 0.001, batch 16, T4 | CLEAN D-Fire test (2,164 imgs): all mAP@50 **0.704** / fire 0.643 / smoke 0.764; image-level @0.5 fire recall **83.8 %** / smoke 78.8 %; fire FP 11/1440 | **active** |
| 4 | `v6-2026-09-13-fireviewer-hfcorpus-ft15ep` | 2026-09-13 | fine-tune of **v4** on the FireViewer corpus v1 (in-tree dedup + class-balance) + our domain negatives (Google Colab) | 29,930 imgs = 29,089 corpus (32,609 deduped → balance) + 841 ours / 15 ep, no early stop @ 640, freeze 10, lr0 0.001, batch 16, amp, T4, 83.7 min; md5 `9894c224325d5cdef71828cd18456914`, 20,276,549 B | **held-out FireViewer test sample (1,000 imgs)**: all mAP@50 **0.687** (v4 0.446) / mAP@50-95 0.392 (0.226); fire mAP@50 **0.671** (0.483) / smoke 0.703 (0.409); image-level @0.5 fire recall 81.8 % (82.2 %), smoke recall **63.1 %** (44.3 %), fire FP **3/786** (10/786). Train-val (19,209 imgs): all mAP@50 0.622 | **candidate** |

**v4 provenance/md5 (see its `VERSION.json`):** `model.pt` (also `best.xml`/`best.bin`/`labelmap.txt`
bundled), md5 `4a4ef19540518e27f886c2894ec168f8`, 20,294,341 B — copy of
`fire-model-training/dedup/dfire_finetune/best_finetuned_dfire.pt`. Recipe + full eval in
[`fire-model-training/dedup/dfire_finetune/report.md`](../fire-model-training/dedup/dfire_finetune/report.md).
**v2 provenance:** `model.pt`, md5 `ddc0cd6c3089c0da140017f881673327`, 20,309,061 B — copy of
`fire-model-training/2_Abonia/abonia_eval/finetune/best_finetuned_abonia.pt`; results recorded in
[`plans/fire-model-abonia-finetune.md`](../plans/fire-model-abonia-finetune.md:182).
**v1 md5:** `2fd972183c2ffec0d327ec534c119086`, 20,301,317 B.

> v4's clean-test numbers are on the **uncontaminated** held-out D-Fire test (13.4 % of raw
> D-Fire was removed as near-dup of v1's 8,939 training pool before scoring). Still, the real
> acceptance test for the farm cameras is a `firewatch.py --dry-run` on-camera pilot —
> v4 was promoted by user decision on 2026-09-07 with that caveat recorded.

---

## Add a new version

1. Copy/promote your trained `best.pt` (or `model.pt`) into
   `models/fire/versions/<next-tag>/model.pt`, where `<next-tag>` follows
   `v<N>-<YYYY-MM-DD>-<slug>` (increment `N`; never edit an existing version dir).
2. Write its `VERSION.json` (mirror the schema in the v1/v2 manifests): source, training
   set + hyperparameters, metrics + their eval split, `parent` (if a fine-tune), md5/size,
   `status: candidate`.
3. Add a row to **All versions** above and commit.

## Promote to active

```bash
./dev_scripts/promote_fire_model.sh v2-2026-09-05-hf-abonia877-ft5ep   # exact name or "v2"
```

The script copies the version's `model.pt` → `models/fire/best.pt` (+ bundled IR if any),
then tells you how to regenerate/install the OpenVINO IR and deploy to the host. After
promoting, flip that version's `status` to `active` (and the old one to `superseded`) here
+ in its `VERSION.json`, then:

```bash
./dev_scripts/deploy_all.sh          # full deploy (no subcommands - runs every step on the host)
docker compose exec firewatch python /firewatch/firewatch.py --check
docker compose exec firewatch python /firewatch/firewatch.py --dry-run
```

> Deploys are **git-based** (2026-09-05): the ACTIVE model files
> (`best.xml`/`best.bin`/`labelmap.txt`[/`best.pt`]) are **git-tracked** and ride
> `deploy_all.sh` (full deploy) via `git pull` — the host always matches the repo.
> The git-ignored `versions/` archive never ships. To change the served model, generate
> its IR from the active `.pt`
> ([`dev_scripts/prep_fire_model.sh`](../dev_scripts/prep_fire_model.sh)), commit the ACTIVE set,
> then deploy.
