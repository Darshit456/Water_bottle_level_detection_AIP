"""
Step 1: Prepare the images.

For every photo in labels.csv:
  1. open the original
  2. rotate it the right way up (phones store rotation in EXIF)
  3. shrink it so the longest side is 512 pixels
  4. save the small copy in CACHE_DIR (same folder structure)
Your original photos are NOT changed.

At the end, a preview image is saved to results/step1_preview.png.

Run from the project folder:
    python src/step1_prepare_images.py
"""
import time

import matplotlib
matplotlib.use("Agg")                     # save plots to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from config import CACHE_DIR, DATA_DIR, LABELS_CSV, RESULTS_DIR

MAX_SIZE = 512   # longest side of the small copy, in pixels

# Pillow refuses images above ~179 million pixels (protection against
# "decompression bomb" files). Some photos were taken in the phone's
# 200 MP mode, so we allow up to 300 million pixels for our own photos.
Image.MAX_IMAGE_PIXELS = 300_000_000


def load_photo(path):
    """Open a photo the right way up."""
    img = Image.open(path)
    # Decode the JPEG at a smaller size directly (faster, less memory).
    # It stays at least MAX_SIZE * 2, so the final quality is the same.
    img.draft("RGB", (MAX_SIZE * 2, MAX_SIZE * 2))
    img = ImageOps.exif_transpose(img)    # apply the phone's rotation note
    return img.convert("RGB")             # make sure it has 3 colour channels


def main():
    df = pd.read_csv(LABELS_CSV)
    print(f"Photos in labels.csv: {len(df)}")

    # ---- 1. Make the small copies (skip the ones that already exist) ----
    start = time.time()
    made, skipped = 0, 0
    failed = []                           # photos that could not be opened
    for filename in tqdm(df["filename"], desc="Preparing"):
        out_path = CACHE_DIR / filename
        if out_path.exists():
            skipped += 1
            continue
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img = load_photo(DATA_DIR / filename)
            img.thumbnail((MAX_SIZE, MAX_SIZE))   # shrink, keep the shape
            img.save(out_path, quality=90)
            made += 1
        except Exception as error:            # don't stop for one bad photo
            failed.append((filename, str(error)))

    print(f"New small copies: {made}, already existed: {skipped}, failed: {len(failed)}")
    for filename, error in failed:
        print(f"  FAILED: {filename} -> {error}")
    print(f"Time: {time.time() - start:.0f} s")
    print(f"Saved in: {CACHE_DIR}")

    # ---- 2. Compare one original with its small copy ----
    first = df["filename"].iloc[0]
    original = Image.open(DATA_DIR / first)
    small = Image.open(CACHE_DIR / first)
    print(f"\nExample: {first}")
    print(f"  original size  : {original.size[0]} x {original.size[1]} px")
    print(f"  small copy size: {small.size[0]} x {small.size[1]} px")

    # ---- 3. Preview: 2 random photos per bottle with their weight ----
    RESULTS_DIR.mkdir(exist_ok=True)
    sample = df.groupby("bottle_id").sample(2, random_state=42)
    plt.figure(figsize=(12, 7))
    for k, (_, row) in enumerate(sample.iterrows(), start=1):
        plt.subplot(2, 4, k)
        plt.imshow(Image.open(CACHE_DIR / row["filename"]))
        plt.title(f"Bottle {row['bottle_id']}: {row['water_g']} g", fontsize=11)
        plt.axis("off")
    plt.tight_layout()
    preview_path = RESULTS_DIR / "step1_preview.png"
    plt.savefig(preview_path, dpi=100)
    print(f"\nPreview saved: {preview_path}")
    print("Open it and check: are all bottles upright? Do the weights look right?")


if __name__ == "__main__":
    main()
