"""
Step 5: Evaluate the trained model on the professor's validation set
(the "final exam": photos the model has never seen, taken by someone else).

  1. Load the best weights saved in step 4
  2. Prepare each validation photo exactly like the training photos
  3. Predict and compare with the true weights
  4. Report MAE: all photos, <= 250 g (our training range), > 250 g
  5. Compare with 'no-photo' baselines (always guess the train mean / median)
  6. Save: predictions CSV, scatter plot, best and worst examples

Run from the project folder:
    python src/step5_evaluate.py
"""
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch

from config import LABELS_CSV, MODELS_DIR, RESULTS_DIR, VAL_DIR
from data import SCALE, eval_transform, load_photo

# ---------- Settings ----------
RUN_NAME = "resnet18_v1"       # which saved model to evaluate (models/<RUN_NAME>.pt)
MODEL_NAME = "resnet18"
SPLIT_AT_G = 250               # our training photos only go up to ~250 g (+ full)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mae(errors):
    return float(np.mean(errors)) if len(errors) else float("nan")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---------- 1. Load the trained model ----------
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)  # empty model
    model.load_state_dict(torch.load(MODELS_DIR / f"{RUN_NAME}.pt", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print(f"Loaded model: models/{RUN_NAME}.pt  (device: {DEVICE})")

    # ---------- 2 + 3. Prepare and predict every validation photo ----------
    val = pd.read_csv(VAL_DIR / "labels.txt", sep=r"\s+", header=None,
                      names=["filename", "water_g"])
    preds = []
    with torch.no_grad():
        for f in val["filename"]:
            img = load_photo(VAL_DIR / f)              # upright + max 512 px (like the cache)
            x = eval_transform(img).unsqueeze(0)       # (3,384,288) -> (1,3,384,288): a batch of 1
            preds.append(max(model(x.to(DEVICE)).item() * SCALE, 0.0))
    val["pred_g"] = preds
    val["error_g"] = (val["pred_g"] - val["water_g"]).abs()
    val["range"] = np.where(val["water_g"] <= SPLIT_AT_G, f"<= {SPLIT_AT_G} g", f"> {SPLIT_AT_G} g")
    val.to_csv(RESULTS_DIR / f"step5_validation_{RUN_NAME}.csv", index=False)

    # ---------- 4 + 5. Results and baselines ----------
    train = pd.read_csv(LABELS_CSV).query("split == 'train'")
    low = val["water_g"] <= SPLIT_AT_G
    rows = []
    for name, guess in [("ResNet-18", val["pred_g"]),
                        (f"always {train['water_g'].mean():.0f} g (train mean)", train["water_g"].mean()),
                        (f"always {train['water_g'].median():.0f} g (train median)", train["water_g"].median())]:
        err = (val["water_g"] - guess).abs()
        rows.append((name, mae(err), mae(err[low]), mae(err[~low])))

    print(f"\nValidation: {len(val)} photos ({low.sum()} <= {SPLIT_AT_G} g, {(~low).sum()} > {SPLIT_AT_G} g)")
    print(f"  {'':32s} {'all':>8s} {'<=' + str(SPLIT_AT_G):>8s} {'>' + str(SPLIT_AT_G):>8s}")
    for name, a, l, h in rows:
        print(f"  {name:32s} {a:7.1f}g {l:7.1f}g {h:7.1f}g")

    print("\nEvery photo (sorted by true weight):")
    print(f"  {'file':>8s} {'true':>6s} {'pred':>7s} {'error':>7s}")
    for _, r in val.sort_values("water_g").iterrows():
        print(f"  {r['filename']:>8s} {r['water_g']:6d} {r['pred_g']:7.0f} {r['error_g']:7.0f}")

    # ---------- 6a. Scatter plot ----------
    plt.figure(figsize=(6.5, 6.5))
    for rng, g in val.groupby("range"):
        plt.scatter(g["water_g"], g["pred_g"], s=35, alpha=0.75, label=rng)
    top = max(val["water_g"].max(), val["pred_g"].max()) * 1.05
    plt.plot([0, top], [0, top], "k--", lw=1, label="perfect prediction")
    plt.axvline(SPLIT_AT_G, color="red", ls=":", lw=1)
    plt.xlabel("true water (g)")
    plt.ylabel("predicted water (g)")
    plt.title(f"Validation (professor's photos): {RUN_NAME}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"step5_validation_scatter_{RUN_NAME}.png", dpi=120)
    plt.close()

    # ---------- 6b. Best 4 and worst 4 predictions ----------
    ordered = val.sort_values("error_g")
    picks = pd.concat([ordered.head(4), ordered.tail(4)])
    plt.figure(figsize=(12, 8))
    for k, (_, r) in enumerate(picks.iterrows(), start=1):
        plt.subplot(2, 4, k)
        plt.imshow(load_photo(VAL_DIR / r["filename"]))
        tag = "BEST" if k <= 4 else "WORST"
        plt.title(f"{tag}: true {r['water_g']} g\npred {r['pred_g']:.0f} g", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"step5_best_worst_{RUN_NAME}.png", dpi=110)
    plt.close()

    # ---------- Log ----------
    log_path = RESULTS_DIR / "validation_log.csv"
    pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "run": RUN_NAME,
        "val_mae_all": round(rows[0][1], 1), "val_mae_low": round(rows[0][2], 1),
        "val_mae_high": round(rows[0][3], 1), "baseline_mean_all": round(rows[1][1], 1),
    }]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    print(f"\nSaved: results/step5_validation_{RUN_NAME}.csv, "
          f"step5_validation_scatter_{RUN_NAME}.png, step5_best_worst_{RUN_NAME}.png")
    print("Log  : results/validation_log.csv (one new row)")


if __name__ == "__main__":
    main()
