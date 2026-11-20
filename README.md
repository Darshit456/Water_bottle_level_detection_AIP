<div align="center">

# 💧 Water Bottle Level Detection

**Estimating the amount of water in a bottle from a single photo using AI**

![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![Type](https://img.shields.io/badge/type-university%20project-blue)
![Python](https://img.shields.io/badge/python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange)

</div>

---

## 📖 About

This project was created as part of the **AI Project** course at OTH Amberg-Weiden. The goal is to build a model that looks at a photo of one of four bottles and predicts **how many grams (≈ ml) of water** are inside.

The professor provides a separate **validation set** (40 photos with the true water weight), which is used as the final test.

---

## 🚀 Progress

| Step | Stage | Status |
| :---: | :--- | :---: |
| – | 📸 Dataset collection | 🟡 14 levels per bottle; upper range of A, B, C still missing |
| – | 🏷️ Data labeling (`labels.csv`) | ✅ Done |
| 1 | 🖼️ Image preparation (rotation + resizing) | ✅ Done |
| 2 | 🔢 Data pipeline (tensors, augmentation, DataLoader) | ✅ Done |
| 3 | 🔍 Model inspection (ResNet-18 layers, filters, feature maps) | ✅ Done |
| 4 | 🧠 Model training | ✅ Done (v1, v2, v3) |
| 5 | 📊 Evaluation on the professor's validation set | ✅ Done (v1, v2, v3) |
| 6 | ✂️ Crop the bottle with YOLO, then train v4 | ⚪ Next |

---

## 🍶 Dataset

Photos of **4 bottles**, each filled with a known amount of water (weighed on a kitchen scale, 1 ml ≈ 1 g).

| ID | Bottle | Size | Material |
| :---: | :--- | :---: | :---: |
| A | Labertaler Stephanie Medium | 1.0 L | PET |
| B | Kondrauer Prickelnd | 0.7 L | Glass |
| C | Staatl. Fachingen Medium | 0.75 L | Glass |
| D | Labertaler Stephanie Mild | 0.5 L | Glass |

**Current state**

- 📷 **2298 photos**, **14 fill levels per bottle**, from empty to full (full = up to the company's fill line)
- 💡 Different backgrounds, lighting, angles and distances (mostly one university lab, newer photos in more varied places)
- 🔀 Train / test split **by fill level** (all photos of one level stay together): **1901 train, 397 test** (2 test levels per bottle, ~140 g and ~310 g)

| Bottle | Photos | Levels | Range | Biggest gap (still missing) |
| :---: | :---: | :---: | :---: | :--- |
| A | 437 | 14 | 0–1000 g | 493 g (507 → 1000 g) |
| B | 837 | 14 | 0–700 g | 181 g (519 → 700 g) |
| C | 648 | 14 | 0–750 g | 207 g (543 → 750 g) |
| D | 376 | 14 | 0–505 g | 89 g (11 → 100 g) |

**Folder structure** (stored outside the repository, path set in `src/config.py`):

```
Bottle_dataset/
├── Bottle_A/
│   ├── 0000gm/   ← water weight in grams
│   ├── 0021gm/
│   └── ...
├── Bottle_B/ ...
├── Bottle_C/ ...
├── Bottle_D/ ...
└── labels.csv    ← created by src/make_labels.py
```

`labels.csv` columns: `filename`, `bottle_id`, `water_g`, `split`.

---

## 🧠 Approach

```
photo → upright + resize → augmentation (train only) → tensor → normalise → ResNet-18 → grams
```

- **Transfer learning:** ResNet-18 pretrained on ImageNet (via `timm`), last layer replaced by **one output** (water in grams ÷ 1000)
- **Input:** 384 × 288 px (portrait), normalised with ImageNet mean/std
- **Augmentation:** horizontal flip, rotation ±8°, colour jitter (never vertical flip)
- **Training:** L1 loss (= MAE), AdamW, learning rate 1e-4, batch size 16, 15–30 epochs, all layers fine-tuned; the epoch with the lowest test MAE is kept
- **Metric:** MAE (mean absolute error) in grams

---

## 🪜 Steps

| Script | What it does |
| :--- | :--- |
| `src/make_labels.py` | Reads the dataset folders and creates `labels.csv` with the train/test split; prints gaps and a baseline |
| `src/step1_prepare_images.py` | Applies the phone's EXIF rotation and saves small copies (max 512 px) in a cache folder |
| `src/step2_check_data.py` | Shows how one photo becomes a tensor, saves an augmentation preview, checks one batch |
| `src/step3_look_at_model.py` | Lists the ResNet-18 parts, traces the tensor shape through every layer, shows first-layer filters and feature maps |
| `src/step4_train.py` | Trains the model, saves the best weights, plots, prints results per bottle and level, and logs the run in `results/experiments.csv` |
| `src/step5_evaluate.py` | Tests a saved model on the professor's validation photos, compares with baselines, saves best/worst examples and logs to `results/validation_log.csv` |
| `src/data.py` | Shared data code: photo loading, transforms and `BottleDataset` (used by steps 2–5) |
| `src/config.py` | All paths in one place |

---

## 🧪 Experiments

Each experiment changes **one thing**, so we know what caused the difference. `RUN_NAME` in `step4_train.py` and `step5_evaluate.py` names the run.

| Run | What changed | Train / test photos | Epochs | Best epoch |
| :---: | :--- | :---: | :---: | :---: |
| v1 | First model, data 0–258 g + full | 1069 / 111 | 15 | 10 |
| v2 | **More data** (14 levels per bottle) | 1901 / 397 | 15 | 13 |
| v3 | **Train longer** (same data as v2) | 1901 / 397 | 30 | 28 |

---

## 📊 Results on Our Test Split

| Run | Baseline (always guess the train mean) | ResNet-18 |
| :---: | :---: | :---: |
| v1 | 98.3 g | **56.9 g** |
| v2 | 48.6 g | 68.9 g |
| v3 | 48.6 g | 59.6 g |

> ⚠️ v1 cannot be compared with v2/v3: the test split changed when new levels were added. v2 and v3 use the **same** 397 test photos.

- v3 is ~9 g better than v2, but on our own test photos the model is **still worse than a constant guess**
- Predictions vary a lot between photos of the **same bottle and level** (spread of 50–97 g)
- v3 learning curve: train MAE kept falling (to 47.5 g) while test MAE stopped improving after ~15 epochs → start of **overfitting**

Plots: `results/step4_learning_curve.png`, `results/step4_test_scatter.png`, `results/step4_test_examples.png`

---

## 🎯 Validation Results (professor's 40 photos)

The validation photos were taken by the professor in other places, so they show how the model works on **truly new** photos. Results are split at 250 g.

| Run | All (40) | ≤ 250 g (15) | > 250 g (25) | Lowest / highest prediction |
| :---: | :---: | :---: | :---: | :---: |
| Baseline v1 (always 244 g) | 233.7 g | 148.8 g | 284.7 g | – |
| Baseline v2/v3 (always 311 g) | 219.7 g | 215.7 g | 222.1 g | – |
| **v1** | 266.7 g | **93.5 g** | 370.5 g | 0 / 403 g |
| **v2** | **151.0 g** | 173.1 g | **137.7 g** | 92 / 588 g |
| **v3** | 154.2 g | 141.1 g | 162.1 g | 79 / 613 g |

**What we learned**

- ✅ **More data was the biggest lever:** v1 → v2 reduced the validation error by **43 %**, and the model can now predict high weights (585 → 587 g, 549 → 531 g)
- ➖ **Training longer did not help:** v2 and v3 are about equal on validation (a few grams on 40 photos is noise)
- ⚠️ **Predictions are squeezed towards the middle:** nearly empty bottles are predicted too high (4 → 306 g in v2), nearly full ones too low (926 → 587 g)
- ⚠️ 13 of the 40 validation photos are above 519 g, the range where A, B and C still have no training levels

**Likely causes:** the bottle is often **tiny in the photo** (after resizing the waterline is only a few pixels), transparent water makes full and empty look alike, missing high levels, and L1 loss pulls uncertain predictions towards the middle.

Plots: `results/step5_validation_scatter_<run>.png`, `results/step5_best_worst_<run>.png`

---

## 📁 Project Structure

```
Water_bottle_level_detection_AIP/
├── src/
│   ├── config.py
│   ├── data.py
│   ├── make_labels.py
│   ├── step1_prepare_images.py
│   ├── step2_check_data.py
│   ├── step3_look_at_model.py
│   ├── step4_train.py
│   └── step5_evaluate.py
├── results/              ← plots, predictions, experiments.csv, validation_log.csv
├── models/               ← trained weights (.pt), not in Git
├── check_gpu.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Run

1. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install PyTorch with CUDA (command from [pytorch.org](https://pytorch.org/get-started/locally/)), then the rest:
   ```bash
   pip install -r requirements.txt
   ```
3. Set the dataset and validation paths in `src/config.py`
4. Run the steps in order:
   ```bash
   python src/make_labels.py
   python src/step1_prepare_images.py
   python src/step2_check_data.py
   python src/step3_look_at_model.py
   python src/step4_train.py
   python src/step5_evaluate.py
   ```

---

## 🔜 Next Steps

- ✂️ **Step 6:** crop the bottle with a pretrained YOLO detector so it fills the whole input image, then train **v4** on the same data as v2/v3 and compare
- 📸 Capture the missing high levels: A 507 → 1000 g (~9 levels), B 519 → 700 g (~3), C 543 → 750 g (~3), at 4–5 different places per level
- 🔥 Grad-CAM heatmaps to show where the model looks
- 📝 Report and presentation

---

## 🛠️ Tech Stack

Python · PyTorch · timm · torchvision · Pillow · pandas · matplotlib · tqdm

---

<div align="center">

🎓 *Made for the AI Project course at OTH Amberg-Weiden*

</div>