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
| – | 📸 Dataset collection | 🟡 B, C, D complete; A still missing 707 → 1000 g |
| – | 🏷️ Data labeling (`labels.csv`) | ✅ Done |
| 1 | 🖼️ Image preparation (rotation + resizing) | ✅ Done |
| 2 | 🔢 Data pipeline (tensors, augmentation, DataLoader) | ✅ Done |
| 3 | 🔍 Model inspection (ResNet-18 layers, filters, feature maps) | ✅ Done |
| 4 | 🧠 Model training | ✅ Done (v1–v4) |
| 5 | 📊 Evaluation on the professor's validation set | ✅ Done (v1–v4) |
| 6 | ✂️ Crop the bottle with YOLO, then train v5 | ⚪ Next |

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

- 📷 **2650 photos**, 14–18 fill levels per bottle, from empty to full (full = up to the company's fill line)
- 💡 Different backgrounds, lighting, angles and distances (mostly one university lab, newer photos in more varied places)
- 🔀 Train / test split **by fill level** (all photos of one level stay together): **2157 train, 493 test** (3 test levels for A, B, C; 2 for D)

| Bottle | Photos | Levels | Range | Biggest gap |
| :---: | :---: | :---: | :---: | :--- |
| A | 543 | 18 | 0–1000 g | 293 g (707 → 1000 g, still missing) |
| B | 925 | 17 | 0–700 g | 60 g |
| C | 806 | 18 | 0–750 g | 69 g |
| D | 376 | 14 | 0–505 g | 89 g |

Level folders must be named like `0123gm` (`make_labels.py` skips any other folder name, so stray folders like `New folder` can never be read as a weight).

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
- **Training:** L1 loss (= MAE), AdamW, learning rate 1e-4, batch size 16, 30 epochs (15 in v1/v2), all layers fine-tuned; the epoch with the lowest test MAE is kept
- **Metric:** MAE (mean absolute error) in grams

---

## 🪜 Steps

| Script | What it does |
| :--- | :--- |
| `src/make_labels.py` | Reads the dataset folders (`0123gm` names only) and creates `labels.csv` with the train/test split; prints gaps and a baseline |
| `src/step1_prepare_images.py` | Applies the phone's EXIF rotation and saves small copies (max 512 px) in a cache folder |
| `src/step2_check_data.py` | Shows how one photo becomes a tensor, saves an augmentation preview, checks one batch |
| `src/step3_look_at_model.py` | Lists the ResNet-18 parts, traces the tensor shape through every layer, shows first-layer filters and feature maps |
| `src/step4_train.py` | Trains the model, saves the best weights, plots, prints results per bottle and level, and logs the run in `results/experiments.csv` |
| `src/step5_evaluate.py` | Tests a saved model on the professor's validation photos, reports MAE by range (0–250 / 251–520 / > 520 g) against baselines, saves best/worst examples and logs to `results/validation_log.csv` |
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
| v4 | **High levels** for A, B, C (same settings as v3) | 2157 / 493 | 30 | 25 |

---

## 📊 Results on Our Test Split

| Run | Baseline (always guess the train mean) | ResNet-18 | Improvement over baseline |
| :---: | :---: | :---: | :---: |
| v1 | 98.3 g | **56.9 g** | +42 % |
| v2 | 48.6 g | 68.9 g | −42 % |
| v3 | 48.6 g | 59.6 g | −23 % |
| v4 | 112.5 g | **75.1 g** | **+33 %** |

> ⚠️ The test split changes whenever new levels are added (v1 → v2 → v4), so test numbers are only comparable within the same data (v2 vs v3). The wider v4 test range (138–588 g) also makes the baseline easier to beat.

**v4 per test level:** middle levels (~310 g) are predicted well (MAE 54–62 g for B, C, D); low levels (~140 g) are predicted too high for every bottle (A: 138 g → 304 g on average); high levels of A and B are predicted too low.

Plots: `results/step4_learning_curve.png`, `results/step4_test_scatter.png`, `results/step4_test_examples.png`

---

## 🎯 Validation Results (professor's 40 photos)

The validation photos were taken by the professor in other places, so they show how the model works on **truly new** photos. This set never changes, so all runs can be compared directly.

| Run | 0–250 g (15) | 251–520 g (12) | > 520 g (13) | **All (40)** | Baseline (mean guess) | Improvement |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| v1 | **93.5 g** | 271.1 g | 462.3 g | 266.7 g | 233.7 g | −14 % |
| v2 | 173.1 g | **80.4 g** | 190.5 g | **151.0 g** | 219.7 g | 31 % |
| v3 | 141.1 g | 135.3 g | 186.7 g | 154.2 g | 219.7 g | 30 % |
| v4 | 210.9 g | 90.4 g | **139.6 g** | 151.7 g | 213.5 g | 29 % |

**What we learned**

- ✅ **More data helps exactly where it is added:** 251–520 g dropped from 271 g (v1) to 80 g (v2); > 520 g dropped to 140 g in v4 (926 → 814 g, 766 → 649 g)
- ➖ **Training longer did not help** (v2 vs v3, same data)
- ⚠️ **The total is stuck at ~151 g (≈ 30 % better than guessing) since v2:** each time one range improves, another gets worse
- ⚠️ **Predictions are pulled towards the training average:** as more high levels were added (average 244 → 353 g), nearly empty bottles were predicted higher and higher (v4: 90 → 686 g, 4 → 362 g)
- 💡 **Conclusion:** the model does not yet read the waterline reliably; it mostly predicts a "typical" value for similar-looking photos. The bottleneck has moved from **data** to **what the model can see**: the bottle is often small in the photo, so after resizing the waterline is only a few pixels

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

- ✂️ **Step 6:** crop the bottle with a pretrained YOLO detector so it fills the whole input image, then train **v5** on the **same data as v4** and compare (main question: does the 0–250 g range improve?)
- 📸 Capture A's last levels (~750, 800, 850, 900, 950 g) in a later data round
- 🔥 Grad-CAM heatmaps to show where the model looks
- 📝 Report and presentation

---

## 🛠️ Tech Stack

Python · PyTorch · timm · torchvision · Pillow · pandas · matplotlib · tqdm

---

<div align="center">

🎓 *Made for the AI Project course at OTH Amberg-Weiden*

</div>