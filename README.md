# HAM10000 Capstone Starter (Windows PowerShell)

## Quick start
```powershell
# 1) Create env (conda recommended)
conda env create -f environment.yml
conda activate ham10000

# 2) Kaggle API setup helper
python -m src.data.download_kaggle --accept

# 3) Download dataset
python -m src.data.download_kaggle --dataset kmader/skin-cancer-mnist-ham10000 --out data/raw

# 4) Lesion-aware stratified split
python -m src.data.split --meta data/raw/HAM10000_metadata.csv --out data/processed --seed 42

# 5) Sanity check
python -m src.data.inspect --splits data/processed

# 6) Train baseline
python -m src.train --config configs/resnet50.yaml
```
