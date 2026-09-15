from datasets import load_dataset
from pathlib import Path

out = Path("model-training/places365_sample/images")
out.mkdir(parents=True, exist_ok=True)

# streaming=True + shuffle gives a spread across the 365 classes, not just the
# first shard (which would otherwise be all "airfield / airplane cabin / ...").
ds = load_dataset("ljnlonoljpiljm/places365-256px",
                  split="train", streaming=True)
ds = ds.shuffle(seed=42, buffer_size=20_000)

for i, ex in enumerate(ds.take(500)):
    ex["image"].save(out / f"places_{i:04d}.jpg")

print("saved", len(list(out.glob("*.jpg"))), "images to", out)
