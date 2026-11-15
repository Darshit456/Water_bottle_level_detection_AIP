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
| 5 | 📊 Evaluation on the professor's validation set | ⚪ Next |

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
| `src/data.py` | Shared data code: transforms and `BottleDataset` (used by steps 2–5) |
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
│   └── step4_train.py
├── results/              ← plots, predictions, experiments.csv
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
3. Set the dataset paths in `src/config.py`
4. Run the steps in order:
   ```bash
   python src/make_labels.py
   python src/step1_prepare_images.py
   python src/step2_check_data.py
   python src/step3_look_at_model.py
   python src/step4_train.py
   ```

---

## 🔜 Next Steps

- 📊 **Step 5:** evaluate on the professor's validation set (split into ≤ 250 g and > 250 g)
- 📸 Capture the remaining fill levels (258 g → full) with more levels and varied backgrounds
- ✂️ Crop the bottle with a pretrained YOLO detector before training
- 🔥 Grad-CAM heatmaps to show where the model looks

---

## 🛠️ Tech Stack

Python · PyTorch · timm · torchvision · Pillow · pandas · matplotlib · tqdm

---

<div align="center">

🎓 *Made for the AI Project course at OTH Amberg-Weiden*

</div>