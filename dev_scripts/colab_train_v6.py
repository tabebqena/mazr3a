#!/usr/bin/env python3
"""colab_train_v6.py - detached, Drive-resilient fine-tune of the fire model (Colab).

WHY A SEPARATE SCRIPT
---------------------
The notebook launches this with `subprocess.Popen(..., start_new_session=True)`, i.e. in its
own session, so it keeps running when you **close VS Code / the Colab tab / the kernel dies**.
The launcher redirects stdout+stderr to a LOCAL log (`--log`), and this script mirrors that
log (plus the run state) onto Drive **every epoch** - writing unbuffered progress bars straight
to Drive's FUSE mount is slow and flaky, so the hot path stays local and Drive gets periodic
whole-file copies.

EVERYTHING THAT MATTERS ENDS UP ON DRIVE
----------------------------------------
    <runs>/<name>/train.log            mirrored log (updated every epoch)
    <runs>/<name>/results.csv          one row per epoch
    <runs>/<name>/args.yaml            the exact recipe (needed for --resume)
    <runs>/<name>/weights/last.pt      per-epoch checkpoint (save_period)
    <runs>/<name>/weights/best.pt
Two modes:
  * `--mode drive`         Ultralytics writes into `<runs>/<name>` directly (simplest, but
                           Drive I/O is slow - many small writes per epoch).
  * `--mode local_mirror`  DEFAULT in the notebook: the run dir is LOCAL (`--local`) and the
                           state above is copied to Drive at every epoch. Faster and much more
                           robust; worst case you lose the current epoch on a recycle.

USAGE
-----
    python colab_train_v6.py --data /content/fv_yolo/data.yaml \
        --base /content/upload/model/best.pt \
        --runs /content/drive/MyDrive/colab-data/v6-training/runs --name v6-fireviewer \
        --mode local_mirror --local /content/runs --log /content/train.log \
        --epochs 15 --batch 16 --imgsz 640 --freeze 10 --lr0 0.001 --patience 5

Resume an INCOMPLETE run (Ultralytics strips the optimizer from a finished one):
    python colab_train_v6.py --resume <runs>/<name>/weights/last.pt
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

# artefacts copied from the run dir to Drive after every epoch
FILES = ("results.csv", "args.yaml", "weights/last.pt", "weights/best.pt")


def mirror(src_dir, dst_dir):
    """Copy the run state to dst_dir atomically (tmp + os.replace), skipping misses."""
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for rel in FILES:
        p = src / rel
        if not p.exists():
            continue
        tgt = dst / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        tmp = tgt.with_name(tgt.name + ".tmp")
        try:
            shutil.copy2(p, tmp)
            os.replace(tmp, tgt)
        except OSError as e:
            print("  ! mirror %s failed: %s" % (rel, e), flush=True)


def copy_log(log_path, dst_dir):
    if not log_path or not os.path.exists(log_path):
        return
    dst = Path(dst_dir) / Path(log_path).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        shutil.copy2(log_path, tmp)
        os.replace(tmp, dst)
    except OSError as e:
        print("  ! mirror log failed: %s" % e, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="YOLO data.yaml (required unless --resume)")
    ap.add_argument("--base", help="base checkpoint .pt (required unless --resume)")
    ap.add_argument("--runs", help="Drive root that holds every run artifact")
    ap.add_argument("--name", default="v6-fireviewer")
    ap.add_argument("--mode", choices=("drive", "local_mirror"), default="local_mirror")
    ap.add_argument("--local", default="/content/runs",
                    help="local project dir used by --mode local_mirror")
    ap.add_argument("--log", default="/content/train.log",
                    help="the local log the launcher redirected stdout to (mirrored each epoch)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--freeze", type=int, default=10)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--save-period", type=int, default=1)
    ap.add_argument("--device", default="0")
    ap.add_argument("--plots", action="store_true",
                    help="write training curves too (skipped by default: many small Drive writes)")
    ap.add_argument("--resume", default=None,
                    help="path to an INCOMPLETE last.pt; ignores the training hyperparameters")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.resume:
        if not os.path.exists(args.resume):
            sys.exit("resume checkpoint not found: %s" % args.resume)
        print("[resume] %s" % args.resume, flush=True)
        YOLO(args.resume).train(resume=True)
        print("[resume] finished", flush=True)
        return

    for need in ("data", "base", "runs"):
        if not getattr(args, need):
            sys.exit("--%s is required (or use --resume)" % need)

    project = args.runs if args.mode == "drive" else args.local
    drive_run = os.path.join(args.runs, args.name)

    model = YOLO(args.base)
    names = list(model.names.values())
    print("[base] %s | classes %s" % (args.base, names), flush=True)
    if names != ["fire", "other", "smoke"]:
        sys.exit("unexpected class order %s (want fire/other/smoke)" % names)

    if args.mode == "local_mirror":
        def _mirror_epoch(trainer):
            mirror(trainer.save_dir, drive_run)
            copy_log(args.log, drive_run)
            print("[mirror] epoch %d -> %s" % (int(getattr(trainer, "epoch", 0)) + 1, drive_run),
                  flush=True)
        model.add_callback("on_fit_epoch_end", _mirror_epoch)

    print("[train] project=%s name=%s epochs=%d batch=%d imgsz=%d freeze=%d lr0=%g patience=%d"
          % (project, args.name, args.epochs, args.batch, args.imgsz, args.freeze, args.lr0,
             args.patience), flush=True)

    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, project=project, name=args.name, exist_ok=True,
                freeze=args.freeze, lr0=args.lr0, patience=args.patience,
                save_period=args.save_period, plots=args.plots, verbose=True)

    save_dir = str(getattr(model.trainer, "save_dir", os.path.join(project, args.name)))
    if Path(save_dir).resolve() != Path(drive_run).resolve():
        mirror(save_dir, drive_run)
    copy_log(args.log, drive_run)
    print("[DONE] save_dir=%s" % save_dir, flush=True)
    print("[DONE] Drive copy -> %s/weights/best.pt" % drive_run, flush=True)


if __name__ == "__main__":
    main()
