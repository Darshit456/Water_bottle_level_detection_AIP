"""
data.py
Everything about turning photos into model input, in one place.
Reused by step 2, training and evaluation, so all of them
prepare the images in exactly the same way.
"""
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from config import CACHE_DIR

# ---------- Settings ----------
IMG_HEIGHT, IMG_WIDTH = 384, 288   # portrait, same shape (3:4) as the photos
SCALE = 1000.0                     # targets = grams / 1000 -> numbers around 0..1

# The pretrained ResNet learned on ImageNet photos normalised with these
# average colour values (mean) and spreads (std), for R, G and B.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

Image.MAX_IMAGE_PIXELS = 300_000_000   # allow our 200 MP photos


def load_photo(path, max_size=512):
    """Open an ORIGINAL photo upright and small (used for validation photos)."""
    img = Image.open(path)
    img.draft("RGB", (max_size * 2, max_size * 2))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_size, max_size))
    return img


# Training: random changes every time a photo is used (augmentation)
train_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),            # 1. same size for all
    transforms.RandomHorizontalFlip(p=0.5),                # 2. mirror left-right
    transforms.RandomRotation(degrees=8),                  # 3. small tilt
    transforms.ColorJitter(brightness=0.3, contrast=0.3,   # 4. other light/camera
                           saturation=0.3, hue=0.05),
    transforms.ToTensor(),                                 # 5. image -> numbers 0..1
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),     # 6. same scale as ImageNet
])

# Test / validation: no random changes, so results are repeatable
eval_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class BottleDataset(Dataset):
    """
    Tells PyTorch two things:
      __len__     -> how many photos there are
      __getitem__ -> photo number i as (image tensor, target)
    """

    def __init__(self, df, transform):
        self.files = df["filename"].tolist()
        self.targets = (df["water_g"].values / SCALE).astype("float32")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(CACHE_DIR / self.files[i]).convert("RGB")
        return self.transform(img), torch.tensor(self.targets[i])
