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
| `v1-2026-09-05-hf-yolo26s-8939img` | **HF YOLO26-S baseline** — deployed + verified on `ssh.mazr3a.garden` (firewatch Up, `--check` OK, `--dry-run` 9 cams, 0 fire) 2026-09-05. v2 is an unevaluated-on-camera candidate. |

---

## All versions

| # | Version dir (`versions/<dir>/model.pt`) | Date | Base / source | Training set | Key metrics | Status |
|---|---|---|---|---|---|---|
| 1 | `v1-2026-09-05-hf-yolo26s-8939img` | 2026-09-05 | YOLO26-S from `huggingface.co/SalahALHaismawi/yolov26-fire-detection` (MIT) | 8,939 imgs / 100 ep @ 640 (author Roboflow pool) | author-reported mAP@50 **0.949** / mAP@50-95 0.680 / P 0.896 / R 0.888 | **active** |
| 2 | `v2-2026-09-05-hf-abonia877-ft5ep` | 2026-09-05 | fine-tune of v1 on Abonia `fire-8` **train** (CC BY 4.0) | 877 imgs / 5 ep @ 640, freeze 10, lr0 0.001 (CPU smoke run) | Abonia *test* split (55 imgs): all mAP@50 **0.948** / fire 0.930 / smoke 0.965; vs v1 baseline 0.405 | candidate |

**v2 provenance/md5 (see its `VERSION.json`):** `model.pt`, md5 `ddc0cd6c3089c0da140017f881673327`,
20,309,061 B — copy of
`dataset/abonia_eval/finetune/best_finetuned_abonia.pt`; results recorded in
[`plans/fire-model-abonia-finetune.md`](../plans/fire-model-abonia-finetune.md:182).
**v1 md5:** `2fd972183c2ffec0d327ec534c119086`, 20,301,317 B.

> v2's test-split lift is on the **same source pool** as its training (optimistic for the
> farm cameras). A `firewatch.py --dry-run` on-camera pilot remains the real acceptance
> test before promotion.

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
./dev_scripts/deploy_all.sh firewatch
docker compose exec firewatch python /scripts/firewatch.py --check
docker compose exec firewatch python /scripts/firewatch.py --dry-run
```

> Deploys are **git-based** (2026-09-05): the ACTIVE model files
> (`best.xml`/`best.bin`/`labelmap.txt`[/`best.pt`]) are **git-tracked** and ride
> `deploy_all.sh firewatch` via `git pull` — the host always matches the repo.
> The git-ignored `versions/` archive never ships. To change the served model, generate
> its IR from the active `.pt`
> ([`dev_scripts/prep_fire_model.sh`](../dev_scripts/prep_fire_model.sh)), commit the ACTIVE set,
> then deploy.
