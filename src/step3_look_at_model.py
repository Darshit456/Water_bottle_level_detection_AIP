"""
Step 3: Look inside the pretrained ResNet-18 and make a first prediction
(before any training).

  Part 1: load the model, count its parameters, list its parts
  Part 2: follow one batch through the model, layer by layer (shapes)
  Part 3: first prediction - the untrained head gives nonsense
  Part 4: pictures: first-layer filters + feature maps
          -> results/step3_filters.png, results/step3_feature_maps.png

Run from the project folder:
    python src/step3_look_at_model.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader

from config import LABELS_CSV, RESULTS_DIR
from data import IMAGENET_MEAN, IMAGENET_STD, SCALE, BottleDataset, eval_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
RESULTS_DIR.mkdir(exist_ok=True)

# ---------- Part 1: load the model ----------
# pretrained=True -> weights learned on ImageNet (downloaded the first time)
# num_classes=1   -> replace the last layer: 1 output (water / 1000)
model = timm.create_model("resnet18", pretrained=True, num_classes=1)
model = model.to(DEVICE)       # move all weights to the GPU
model.eval()                   # "use" mode, not "learning" mode

cfg = model.pretrained_cfg
print("Part 1 - The model")
print(f"  device: {DEVICE}")
print(f"  pretrained on: {cfg.get('tag', '?')}, mean {cfg['mean']}, std {cfg['std']}")
total = sum(p.numel() for p in model.parameters())
print(f"  parameters (numbers the model can learn): {total:,}")
print("\n  Main parts:")
print(f"  {'name':12s} {'type':22s} {'parameters':>12s}")
for name, module in model.named_children():
    n = sum(p.numel() for p in module.parameters())
    print(f"  {name:12s} {module.__class__.__name__:22s} {n:>12,}")
print(f"\n  The new last layer: {model.fc}")

# ---------- Part 2: follow one batch through the model ----------
df = pd.read_csv(LABELS_CSV)
dataset = BottleDataset(df, eval_transform)          # all photos, no augmentation
loader = DataLoader(dataset, batch_size=16, shuffle=True,
                    generator=torch.Generator().manual_seed(42))
images, targets = next(iter(loader))
images = images.to(DEVICE)

print("\nPart 2 - One batch, layer by layer")
print(f"  {'input':14s}: {tuple(images.shape)}")
with torch.no_grad():                                # no learning here
    x = images
    for name in ["conv1", "bn1", "act1", "maxpool",
                 "layer1", "layer2", "layer3", "layer4", "global_pool", "fc"]:
        x = getattr(model, name)(x)
        print(f"  after {name:8s}: {tuple(x.shape)}")
    step_by_step = x
    all_at_once = model(images)                      # the normal way
print(f"  same result as model(images)? {torch.allclose(step_by_step, all_at_once, atol=1e-5)}")

# ---------- Part 3: first prediction (untrained head) ----------
preds = all_at_once.squeeze(1).cpu() * SCALE         # (16, 1) -> (16,) -> grams
true = targets * SCALE
print("\nPart 3 - First prediction, BEFORE training")
print(f"  {'true (g)':>9s} {'predicted (g)':>14s}")
for t, p in zip(true.tolist(), preds.tolist()):
    print(f"  {t:9.0f} {p:14.1f}")
mae = (preds - true).abs().mean().item()
baseline = (true - df["water_g"].mean()).abs().mean().item()
print(f"\n  MAE of the untrained model     : {mae:7.1f} g")
print(f"  MAE of 'always guess average'  : {baseline:7.1f} g")

# ---------- Part 4a: what the first layer looks for ----------
w = model.conv1.weight.detach().cpu()                # (64 filters, 3 colours, 7, 7)
plt.figure(figsize=(8, 8))
for k in range(64):
    f = w[k]
    f = (f - f.min()) / (f.max() - f.min() + 1e-8)   # stretch to 0..1 to show it
    plt.subplot(8, 8, k + 1)
    plt.imshow(f.permute(1, 2, 0))                   # (3,7,7) -> (7,7,3) for showing
    plt.axis("off")
plt.suptitle("The 64 filters of the first layer (7 x 7 pixels each, learned on ImageNet)")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "step3_filters.png", dpi=100)
plt.close()

# ---------- Part 4b: what the model 'sees' after layer1 ----------
with torch.no_grad():
    one = images[:1]                                 # first photo of the batch
    fm = model.layer1(model.maxpool(model.act1(model.bn1(model.conv1(one)))))
fm = fm[0].cpu()                                     # (64 channels, 96, 72)
strongest = fm.std(dim=(1, 2)).argsort(descending=True)[:11]   # most active channels

mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
photo = (images[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0)   # undo Normalize

plt.figure(figsize=(12, 10))
plt.subplot(3, 4, 1)
plt.imshow(photo)
plt.title(f"input ({true[0]:.0f} g)")
plt.axis("off")
for k, c in enumerate(strongest.tolist(), start=2):
    plt.subplot(3, 4, k)
    plt.imshow(fm[c], cmap="gray")
    plt.title(f"channel {c}")
    plt.axis("off")
plt.suptitle("Feature maps after layer1: each channel reacts to a different pattern")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "step3_feature_maps.png", dpi=100)
plt.close()

print("\nPart 4 - Pictures saved:")
print("  results/step3_filters.png")
print("  results/step3_feature_maps.png")
