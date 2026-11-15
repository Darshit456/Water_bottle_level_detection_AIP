"""
Step 2: See how a photo becomes the numbers the model receives.

  Part 1: the photo as Pillow sees it
  Part 2: the photo as a tensor (numbers 0..1)
  Part 3: after normalisation (ImageNet scale)
  Part 4: augmentation -> results/step2_augmentation.png
  Part 5: Dataset + DataLoader -> one batch of 16 photos

Run from the project folder:
    python src/step2_check_data.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from config import CACHE_DIR, LABELS_CSV, RESULTS_DIR
from data import IMG_HEIGHT, IMG_WIDTH, SCALE, BottleDataset, train_transform

torch.manual_seed(42)

df = pd.read_csv(LABELS_CSV)
train_df = df[df["split"] == "train"].reset_index(drop=True)
row = train_df.sample(1, random_state=7).iloc[0]          # one example photo
print(f"Example photo: {row['filename']}  (bottle {row['bottle_id']}, {row['water_g']} g)")

# ---------- Part 1: the photo as Pillow sees it ----------
img = Image.open(CACHE_DIR / row["filename"]).convert("RGB")
w, h = img.size
print("\nPart 1 - Pillow image")
print(f"  size (width x height): {w} x {h}, mode: {img.mode}")
print(f"  centre pixel (R, G, B): {img.getpixel((w // 2, h // 2))}   <- whole numbers 0..255")

# ---------- Part 2: resize + ToTensor ----------
to_tensor = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])
t = to_tensor(img)
print("\nPart 2 - Tensor (after Resize + ToTensor)")
print(f"  shape: {tuple(t.shape)}   <- (colour channels, height, width)")
print(f"  type : {t.dtype}, min {t.min():.3f}, max {t.max():.3f}   <- numbers 0..1")
print(f"  centre pixel (R, G, B): {[round(v, 3) for v in t[:, IMG_HEIGHT // 2, IMG_WIDTH // 2].tolist()]}")
print(f"  numbers per image: {t.numel():,}")

# ---------- Part 3: + Normalize ----------
normalize = train_transform.transforms[-1]                 # the Normalize step
n = normalize(t)
print("\nPart 3 - After Normalize (ImageNet mean/std)")
print(f"  min {n.min():.2f}, max {n.max():.2f}   <- now around 0, roughly -2..+2.6")
print(f"  average per channel (R, G, B): {[round(v, 2) for v in n.mean(dim=(1, 2)).tolist()]}")

# ---------- Part 4: augmentation, shown as pictures ----------
# The first 4 steps of train_transform are the picture changes
# (Resize, Flip, Rotation, ColorJitter). We stop before ToTensor so we can show them.
augment_only = transforms.Compose(train_transform.transforms[:4])
resize_only = train_transform.transforms[0]

RESULTS_DIR.mkdir(exist_ok=True)
plt.figure(figsize=(12, 9))
plt.subplot(2, 4, 1)
plt.imshow(resize_only(img))
plt.title("resized only\n(test photos look like this)", fontsize=10)
plt.axis("off")
for k in range(2, 9):
    plt.subplot(2, 4, k)
    plt.imshow(augment_only(img))                          # new random changes each time
    plt.title(f"augmented #{k - 1}", fontsize=10)
    plt.axis("off")
plt.suptitle(f"Same photo, 7 random versions ({row['water_g']} g, bottle {row['bottle_id']})")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "step2_augmentation.png", dpi=100)
plt.close()
print("\nPart 4 - Augmentation grid saved: results/step2_augmentation.png")

# ---------- Part 5: Dataset + DataLoader ----------
dataset = BottleDataset(train_df, train_transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

image, target = dataset[0]
print("\nPart 5 - Dataset and DataLoader")
print(f"  photos in dataset: {len(dataset)}")
print(f"  one item: image {tuple(image.shape)}, target {target.item():.3f}"
      f"  (= {target.item() * SCALE:.0f} g / {SCALE:.0f})")
print(f"  batches per epoch (16 photos each): {len(loader)}")

images, targets = next(iter(loader))                      # take the first batch
print(f"  one batch: images {tuple(images.shape)}   <- (batch, channels, height, width)")
print(f"             targets {tuple(targets.shape)}")
print(f"  the 16 weights in this batch (g): {[int(round(v * SCALE)) for v in targets.tolist()]}")
