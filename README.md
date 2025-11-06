# HAM10000 Skin Lesion Classification

> **Samsung Innovation Camp AI Capstone Project**  
> Deep Learning for Dermatoscopic Image Classification

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Project Overview

This project implements a deep learning pipeline for classifying skin lesions using the **HAM10000 dataset** (10,015 dermatoscopic images across 7 diagnostic categories). The system features:

- ✅ **ResNet50** transfer learning with ImageNet pre-trained weights
- ✅ **Stratified Group K-Fold** splitting to prevent lesion-level data leakage
- ✅ **Class weights** to handle severe imbalance (nv: 67%, mel: 11%, etc.)
- ✅ **Data augmentation** (flips, rotation) for improved generalization
- ✅ **Grad-CAM visualization** via interactive Streamlit app
- ✅ **DirectML/CUDA** support for GPU acceleration

### 🎯 Dataset Classes (HAM10000)

| Class | Full Name | Count | Description |
|-------|-----------|-------|-------------|
| `nv` | Melanocytic Nevus | ~6,700 | Benign mole (67%) |
| `mel` | Melanoma | ~1,100 | Malignant tumor (11%) |
| `bkl` | Benign Keratosis | ~1,100 | Non-cancerous growth (11%) |
| `bcc` | Basal Cell Carcinoma | ~500 | Common skin cancer (5%) |
| `akiec` | Actinic Keratosis | ~300 | Precancerous lesion (3%) |
| `vasc` | Vascular Lesion | ~140 | Blood vessel growth (1.4%) |
| `df` | Dermatofibroma | ~115 | Benign fibrous nodule (1.2%) |

---

## 🚀 Quick Start

### 1. Environment Setup

**Option A: Conda (Recommended)**
```powershell
# Create environment with GPU support
conda env create -f environment.yml
conda activate ham10000

# Install additional dependencies
pip install -r requirements.txt
```

**Option B: Pip + Virtual Environment**
```powershell
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Dataset Preparation

**Option A: Manual Download**
1. Download from [Kaggle HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
2. Extract to `data/raw/`:
   ```
   data/raw/
   ├── HAM10000_metadata.csv
   ├── HAM10000_images_part_1/  # 5,000 images
   └── HAM10000_images_part_2/  # 5,015 images
   ```

**Option B: Automated Download (Kaggle API)**
```powershell
# Setup Kaggle credentials (first time only)
python -m src.data.download_kaggle --accept

# Download dataset (requires ~/.kaggle/kaggle.json)
python -m src.data.download_kaggle --dataset kmader/skin-cancer-mnist-ham10000 --out data/raw
```

### 3. Data Splitting (Critical Step)

⚠️ **Important:** This step prevents **data leakage** by grouping images from the same lesion:

```powershell
python -m src.data.split --meta data/raw/HAM10000_metadata.csv --out data/processed --seed 42
```

**What it does:**
- Uses **StratifiedGroupKFold** to ensure same `lesion_id` stays in one split
- Creates `train.csv` (~64%), `val.csv` (~16%), `test.csv` (~20%)
- Maintains class distribution across all splits

**Verify splits:**
```powershell
python -m src.data.inspect --splits data/processed
```

### 4. Model Training

**Local Training (Windows/Linux):**
```powershell
python -m src.train --config configs/resnet50.yaml
```

**Google Colab (Fast iteration):**
```powershell
python -m src.train --config configs/resnet50_colab_fast.yaml
# 5 epochs, 160x160 images, ~10 minutes on T4 GPU
```

**Google Colab (Best accuracy):**
```powershell
python -m src.train --config configs/resnet50_colab_long.yaml
# 15 epochs, 224x224 images, ~30 minutes on T4 GPU
```

**Training outputs:**
- Best model saved to: `runs/{config_name}/best.pt`
- Prints train/val accuracy per epoch
- Auto-detects GPU (DirectML → CUDA → CPU)

### 5. Run Streamlit App

```powershell
streamlit run app/main_app.py
```

**App Features:**
- 📤 Upload dermatoscopic images (JPG/PNG)
- 🧠 Real-time classification with confidence scores
- 🔥 Grad-CAM heatmaps showing model attention regions
- 📊 Compare multiple images side-by-side
- ⚙️ Adjustable confidence thresholds

Access at: `http://localhost:8501`

---

## 📂 Project Structure

```
ham10000_capstone/
├── src/
│   ├── train.py                    # Main training script
│   └── data/
│       ├── split.py                # Stratified lesion-level splitting
│       ├── dataset.py              # PyTorch Dataset implementation
│       ├── inspect.py              # Split verification utility
│       └── download_kaggle.py      # Kaggle API downloader
├── configs/
│   ├── resnet50.yaml               # Local training config
│   ├── resnet50_colab_fast.yaml    # Quick Colab run (5 epochs)
│   └── resnet50_colab_long.yaml    # Full Colab run (15 epochs)
├── app/
│   └── main_app.py                 # Streamlit web application
├── data/
│   ├── raw/                        # Original HAM10000 dataset (gitignored)
│   └── processed/                  # Split CSVs (train/val/test)
├── runs/                           # Model checkpoints (gitignored)
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment spec
└── README.md                       # This file
```

---

## 🔧 Configuration Files

All configs support these parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `seed` | Random seed for reproducibility | 42 |
| `model` | Architecture (`resnet50` or `resnet18`) | resnet50 |
| `image_size` | Input resolution (NxN) | 224 |
| `batch_size` | Batch size | 32 |
| `max_epochs` | Training epochs | 15 |
| `lr` | Learning rate (AdamW) | 0.0003 |
| `weight_decay` | L2 regularization | 0.0001 |
| `label_smoothing` | Smoothing factor (0-1) | 0.05 |
| `use_class_weights` | Enable inverse-frequency weighting | true |
| `train_csv` | Path to training split | data/processed/train.csv |
| `val_csv` | Path to validation split | data/processed/val.csv |
| `out_dir` | Output directory for checkpoints | runs/resnet50_baseline |

**Example custom config:**
```yaml
seed: 42
model: resnet50
image_size: 224
batch_size: 32
max_epochs: 20
lr: 0.0001
weight_decay: 0.0001
label_smoothing: 0.1
use_class_weights: true
train_csv: data/processed/train.csv
val_csv: data/processed/val.csv
out_dir: runs/my_experiment
```

---

## 📊 Expected Results

**Baseline Performance (15 epochs, ResNet50):**
- **Training Accuracy:** ~85-90%
- **Validation Accuracy:** ~75-80%
- **Model Size:** ~98 MB (best.pt)
- **Inference Speed:** ~30ms per image (GPU)

**Class-wise Challenges:**
- **High accuracy:** `nv` (abundant data)
- **Lower accuracy:** `vasc`, `df` (rare classes, ~1-2% of dataset)
- **Mitigation:** Class weights, data augmentation, label smoothing

---

## 🛠️ Common Commands Reference

### Data Pipeline
```powershell
# Download dataset (if using Kaggle API)
python -m src.data.download_kaggle --dataset kmader/skin-cancer-mnist-ham10000 --out data/raw

# Create stratified splits (prevents lesion leakage)
python -m src.data.split --meta data/raw/HAM10000_metadata.csv --out data/processed --seed 42

# Verify class distributions
python -m src.data.inspect --splits data/processed
```

### Training
```powershell
# Local training
python -m src.train --config configs/resnet50.yaml

# Colab fast (5 epochs, 160x160)
python -m src.train --config configs/resnet50_colab_fast.yaml

# Colab long (15 epochs, 224x224)
python -m src.train --config configs/resnet50_colab_long.yaml
```

### App Deployment
```powershell
# Run locally
streamlit run app/main_app.py

# Run on specific port
streamlit run app/main_app.py --server.port 8080

# Run with auto-reload (dev mode)
streamlit run app/main_app.py --server.runOnSave true
```

---

## ⚠️ Important Notes

### 1. Data Leakage Prevention
- HAM10000 contains **multiple images per lesion** (same patient, different angles)
- **Never split by image ID** → Use `lesion_id` grouping
- Our `split.py` uses **StratifiedGroupKFold** to ensure all images from one lesion stay together
- Prevents inflated accuracy from seeing same lesion in train and test

### 2. Class Imbalance
- **Severe imbalance:** `nv` (67%) vs `df` (1.2%)
- **Mitigation strategies:**
  - Inverse-frequency class weights (`use_class_weights: true`)
  - Data augmentation (horizontal/vertical flips, rotation)
  - Label smoothing to prevent overconfidence on majority class

### 3. GPU Acceleration
- **DirectML** (Windows AMD/Intel GPUs): Auto-detected
- **CUDA** (NVIDIA GPUs): Auto-detected
- **CPU fallback**: Automatic if no GPU available
- Training time: ~5-10x faster on GPU vs CPU

### 4. Model Checkpoints
- Only `best.pt` saved (based on validation accuracy)
- Located in `runs/{config_name}/best.pt`
- Load in app: `torch.load('runs/resnet50_colab_long/best.pt')`
- **Not version controlled** (see `.gitignore`)

---

## 🐛 Troubleshooting

### Issue: "Model file not found"
**Solution:** Train a model first or check path in `app/main_app.py`:
```python
model, device = load_model("runs/resnet50_colab_long/best.pt")
```

### Issue: "Missing columns in metadata"
**Solution:** Ensure `HAM10000_metadata.csv` has columns: `image_id`, `dx`, `lesion_id`

### Issue: Slow training on CPU
**Solution:** 
- Use smaller config (`resnet50_colab_fast.yaml`)
- Reduce batch size to 16
- Or train on Google Colab (free T4 GPU)

### Issue: Out of memory (GPU)
**Solution:**
- Reduce `batch_size` in config (try 16 or 8)
- Reduce `image_size` to 160 or 128
- Use `resnet18` instead of `resnet50`

---

## 📖 References

1. **Dataset:** Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset. *Scientific Data* 5, 180161 (2018).
2. **Model:** He, K. et al. Deep Residual Learning for Image Recognition. *CVPR* (2016).
3. **Grad-CAM:** Selvaraju et al. Grad-CAM: Visual Explanations from Deep Networks. *ICCV* (2017).

---

## 📄 License

MIT License - See LICENSE file for details.

**⚠️ Medical Disclaimer:** This tool is for **educational purposes only**. Not intended for clinical diagnosis. Always consult qualified healthcare professionals for medical decisions.

---

## 👥 Contributors

Samsung Innovation Camp AI Track - Capstone Team

**Questions?** Create an issue or contact the team.
