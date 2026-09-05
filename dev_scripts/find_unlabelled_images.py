#!/usr/bin/env python3
"""Copy train images that have no label (missing or empty .txt) to dataset/unlabelled-images-dir."""
import shutil
from pathlib import Path

BASE = Path("dataset/ready_fire_smoke_dataset.yolov8/train")
IMAGES = BASE / "images"
LABELS = BASE / "labels"
DEST = Path("dataset/unlabelled-images-dir")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    images = sorted(IMAGES.glob("*"))
    copied = []
    missing_label = []
    empty_label = []

    for img in images:
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label = LABELS / (img.stem + ".txt")
        if not label.exists():
            missing_label.append(img.name)
        elif label.stat().st_size == 0:
            empty_label.append(img.name)
        else:
            continue
        # image has no label -> copy it
        dest = DEST / img.name
        if not dest.exists():
            shutil.copy2(img, dest)
        copied.append(img.name)

    print(f"Images scanned: {len(images)}")
    print(f"Copied to {DEST}: {len(copied)}")
    print(f"  missing label file: {len(missing_label)}")
    print(f"  empty label file:   {len(empty_label)}")


if __name__ == "__main__":
    main()
