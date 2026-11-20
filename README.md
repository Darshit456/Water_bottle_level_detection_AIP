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
| – | 📸 Dataset collection | 🟡 In progress (0–258 g + full) |
| – | 🏷️ Data labeling (`labels.csv`) | ✅ Done |
| 1 | 🖼️ Image preparation (rotation + resizing) | ✅ Done |
| 2 | 🔢 Data pipeline (tensors, augmentation, DataLoader) | ✅ Done |
| 3 | 🔍 Model inspection (ResNet-18 layers, filters, feature maps) | ✅ Done |
| 4 | 🧠 Model training (first model `resnet18_v1`) | ✅ Done |
| 5 | 📊 Evaluation on the professor's validation set | ✅ Done (v1) |
| – | 📸 Missing levels (258 g → full) + model v2 | ⚪ Next |

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

- 📷 **1180 photos**, 8 fill levels per bottle: empty, 5–6 levels between ~10 g and ~258 g, and full (up to the company's fill line)
- 💡 Different backgrounds, lighting, angles and distances
- 🔀 Train / test split **by fill level** (all photos of one level stay together): 1069 train, 111 test

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
- **Training:** L1 loss (= MAE), AdamW, learning rate 1e-4, batch size 16, 15 epochs, all layers fine-tuned
- **Metric:** MAE (mean absolute error) in grams

---

## 🪜 Steps

| Script | What it does |
| :--- | :--- |
| `src/make_labels.py` | Reads the dataset folders and creates `labels.csv` with the train/test split; prints gaps and a baseline |
| `src/step1_prepare_images.py` | Applies the phone's EXIF rotation and saves small copies (max 512 px) in a cache folder |
| `src/step2_check_data.py` | Shows how one photo becomes a tensor, saves an augmentation preview, checks one batch |
| `src/step3_look_at_model.py` | Lists the ResNet-18 parts, traces the tensor shape through every layer, shows first-layer filters and feature maps |
| `src/step4_train.py` | Trains the model, saves the best weights, plots, and logs the run in `results/experiments.csv` |
| `src/step5_evaluate.py` | Tests a saved model on the professor's validation photos, compares with baselines, saves best/worst examples and logs to `results/validation_log.csv` |
| `src/data.py` | Shared data code: photo loading, transforms and `BottleDataset` (used by steps 2–5) |
| `src/config.py` | All paths in one place |

---

## 📊 First Results (`resnet18_v1`, test split)

| Method | Test MAE |
| :--- | :---: |
| Always guess 244 g (train mean) | 98.3 g |
| Always guess 217 g (train median) | 71.6 g |
| **ResNet-18 (best epoch 10)** | **56.9 g** |

| Bottle | True level | Mean prediction | MAE |
| :---: | :---: | :---: | :---: |
| A | 138 g | 123.4 g | 58.6 g |
| B | 153 g | 126.5 g | 63.4 g |
| C | 150 g | 149.8 g | 43.1 g |
| D | 138 g | 137.2 g | 57.5 g |

**What this tells us**

- ✅ The full pipeline works end-to-end and the model beats the "no-photo" baselines
- ⚠️ The test split has only **one level per bottle** (138–153 g), so the test MAE is a weak estimate
- ⚠️ Predictions are correct **on average** per bottle but scatter a lot between single photos
- ⚠️ No training data between ~258 g and full yet, so larger amounts cannot be predicted well

Plots: `results/step4_learning_curve.png`, `results/step4_test_scatter.png`, `results/step4_test_examples.png`

---

## 🎯 Validation Results (`resnet18_v1`, professor's 40 photos)

The validation photos were taken by the professor in other places, so they show how the model works on **truly new** photos. Results are split at 250 g, because our training data only goes up to ~258 g (plus full bottles).

| Method | All (40) | ≤ 250 g (15) | > 250 g (25) |
| :--- | :---: | :---: | :---: |
| Always guess 244 g (train mean) | 233.7 g | 148.8 g | 284.7 g |
| Always guess 217 g (train median) | 240.4 g | 122.1 g | 311.4 g |
| **ResNet-18 v1** | **266.7 g** | **93.5 g** | **370.5 g** |

**What this tells us**

- ✅ **≤ 250 g (training range):** the model beats both baselines on new photos; some predictions are very close (189 → 187 g, 4 → 0 g, 100 → 109 g)
- ⚠️ **But unreliable:** median error 59 g, with some large misses (137 → 403 g)
- ❌ **> 250 g (no training data):** worse than a constant guess; the model never predicts more than ~400 g
- ❌ The worst errors are nearly full bottles predicted as nearly empty (588 → 51 g, 585 → 43 g): clear water and air look similar, and the model has never seen waterlines in the upper half

**Main causes:** missing fill levels between 258 g and full (62 % of the validation photos are in this range), transparent water making full and empty look alike, and L1 loss pulling uncertain predictions towards the middle of the training range.

Plots: `results/step5_validation_scatter_resnet18_v1.png`, `results/step5_best_worst_resnet18_v1.png`

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

- 📸 Capture the missing fill levels (258 g → full): steps of ~30–50 g, 5–10 photos per level at 4–5 different places
- 🔁 Retrain as `resnet18_v2` and compare with v1 on the test split and the validation set
- ✂️ Crop the bottle with a pretrained YOLO detector before training
- 🔥 Grad-CAM heatmaps to show where the model looks

---

## 🛠️ Tech Stack

Python · PyTorch · timm · torchvision · Pillow · pandas · matplotlib · tqdm

---

<div align="center">

🎓 *Made for the AI Project course at OTH Amberg-Weiden*

</div>