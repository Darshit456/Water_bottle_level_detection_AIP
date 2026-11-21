"""
Step 4: Train the model (transfer learning with ResNet-18).

Each epoch:
  1. TRAIN: show all training photos (with augmentation) in batches of 16,
     and after every batch adjust the weights a little
  2. TEST:  predict the test photos (no augmentation) and measure the MAE
  3. SAVE:  keep the weights of the epoch with the lowest test MAE

At the end:
  - results per bottle, test_predictions.csv
  - plots: learning curve, predicted vs true, example photos
  - one row added to results/experiments.csv (your experiment log)

Run from the project folder:
    python src/step4_train.py
"""
import random
import subprocess
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from config import CACHE_DIR, LABELS_CSV, MODELS_DIR, RESULTS_DIR
from data import IMG_HEIGHT, IMG_WIDTH, SCALE, BottleDataset, eval_transform, train_transform

# ---------- Settings (change these for new experiments) ----------
RUN_NAME = "resnet18_v4"       # name for the saved model and the log
MODEL_NAME = "resnet18"
EPOCHS = 30                    # how many times the model sees all training photos
BATCH_SIZE = 16                # photos per weight update
LEARNING_RATE = 1e-4           # how big each weight change is (0.0001)
SEED = 42                      # same seed -> (almost) same result every run
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer):
    """One pass over all training photos. Returns the average error in grams."""
    model.train()                                   # learning mode
    total_error, count = 0.0, 0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        preds = model(images).squeeze(1)            # 1. FORWARD: guess 16 values
        loss = F.l1_loss(preds, targets)            # 2. LOSS: how wrong on average?

        optimizer.zero_grad()                       #    forget last batch's directions
        loss.backward()                             # 3. BACKWARD: direction for every weight
        optimizer.step()                            # 4. UPDATE: small step in that direction

        total_error += loss.item() * len(images)
        count += len(images)
    return total_error / count * SCALE


@torch.no_grad()                                    # predicting only, no learning
def predict(model, loader):
    model.eval()                                    # use mode
    preds = []
    for images, _ in loader:
        preds.append(model(images.to(DEVICE)).squeeze(1).cpu())
    grams = torch.cat(preds).numpy() * SCALE
    return np.clip(grams, 0, None)                  # water can't be negative


def plot_learning_curve(history, baseline):
    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_mae"], "o-", label="train MAE")
    plt.plot(history["epoch"], history["test_mae"], "o-", label="test MAE")
    plt.axhline(baseline, color="gray", ls="--", label=f"baseline ({baseline:.0f} g)")
    plt.xlabel("epoch")
    plt.ylabel("average error (g)")
    plt.title("Learning curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "step4_learning_curve.png", dpi=120)
    plt.close()


def plot_scatter(test_df):
    plt.figure(figsize=(6, 6))
    for b, g in test_df.groupby("bottle_id"):
        plt.scatter(g["water_g"], g["pred_g"], s=18, alpha=0.6, label=f"bottle {b}")
    top = max(test_df["water_g"].max(), test_df["pred_g"].max()) * 1.1
    plt.plot([0, top], [0, top], "k--", lw=1, label="perfect prediction")
    plt.xlabel("true water (g)")
    plt.ylabel("predicted water (g)")
    plt.title("Test photos: predicted vs true")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "step4_test_scatter.png", dpi=120)
    plt.close()


def plot_examples(test_df):
    sample = test_df.groupby("bottle_id").sample(2, random_state=SEED)
    plt.figure(figsize=(12, 8))
    for k, (_, row) in enumerate(sample.iterrows(), start=1):
        plt.subplot(2, 4, k)
        plt.imshow(Image.open(CACHE_DIR / row["filename"]))
        plt.title(f"Bottle {row['bottle_id']}\ntrue {row['water_g']} g | "
                  f"pred {row['pred_g']:.0f} g", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "step4_test_examples.png", dpi=110)
    plt.close()


def log_experiment(row):
    """Append one line to results/experiments.csv."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    row["git_commit"] = commit
    log_path = RESULTS_DIR / "experiments.csv"
    pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


def main():
    set_seed(SEED)
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---------- Data ----------
    df = pd.read_csv(LABELS_CSV)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_loader = DataLoader(BottleDataset(train_df, train_transform),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(SEED))
    test_loader = DataLoader(BottleDataset(test_df, eval_transform),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    baseline = np.abs(test_df["water_g"] - train_df["water_g"].mean()).mean()
    print(f"Device: {DEVICE}")
    print(f"Train photos: {len(train_df)} ({len(train_loader)} batches), test photos: {len(test_df)}")
    print(f"Baseline (always guess {train_df['water_g'].mean():.0f} g): test MAE = {baseline:.1f} g")

    # ---------- Model + optimiser ----------
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ---------- Training loop ----------
    best_mae, best_epoch = float("inf"), 0
    best_path = MODELS_DIR / f"{RUN_NAME}.pt"
    history = {"epoch": [], "train_mae": [], "test_mae": []}

    print(f"\nTraining {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_mae = train_one_epoch(model, train_loader, optimizer)
        test_mae = np.abs(predict(model, test_loader) - test_df["water_g"].values).mean()

        history["epoch"].append(epoch)
        history["train_mae"].append(train_mae)
        history["test_mae"].append(test_mae)

        note = ""
        if test_mae < best_mae:
            best_mae, best_epoch = test_mae, epoch
            torch.save(model.state_dict(), best_path)        # save the weights
            note = "  <- best so far, saved"
        print(f"Epoch {epoch:2d}/{EPOCHS} | train MAE {train_mae:6.1f} g | "
              f"test MAE {test_mae:6.1f} g | {time.time() - start:3.0f} s{note}")

    # ---------- Results with the best weights ----------
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_df["pred_g"] = predict(model, test_loader)
    test_df["error_g"] = (test_df["pred_g"] - test_df["water_g"]).abs()
    test_df.to_csv(RESULTS_DIR / "step4_test_predictions.csv", index=False)

    print(f"\nBest epoch: {best_epoch}")
    print(f"  Baseline  : test MAE {baseline:6.1f} g")
    print(f"  ResNet-18 : test MAE {best_mae:6.1f} g")
    print("\nPer bottle and test level:")
    for (b, w), g in test_df.groupby(["bottle_id", "water_g"]):
        print(f"  {b}: true {w:4d} g | mean prediction {g['pred_g'].mean():6.1f} g"
              f" | MAE {g['error_g'].mean():5.1f} g | {len(g)} photos")

    plot_learning_curve(history, baseline)
    plot_scatter(test_df)
    plot_examples(test_df)
    log_experiment({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "run": RUN_NAME, "model": MODEL_NAME, "image_size": f"{IMG_HEIGHT}x{IMG_WIDTH}",
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "train_photos": len(train_df), "test_photos": len(test_df),
        "baseline_mae": round(baseline, 1), "best_epoch": best_epoch,
        "test_mae": round(best_mae, 1),
    })

    print(f"\nModel saved : {best_path}")
    print("Plots       : results/step4_learning_curve.png, step4_test_scatter.png, step4_test_examples.png")
    print("Log         : results/experiments.csv (one new row)")


if __name__ == "__main__":
    main()