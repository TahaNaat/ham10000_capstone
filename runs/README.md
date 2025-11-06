# Model Checkpoints Directory

This directory stores trained model weights after running the training script.

## Expected Structure

```
runs/
├── resnet50_baseline/
│   └── best.pt
├── resnet50_colab_fast/
│   └── best.pt
└── resnet50_colab_long/
    └── best.pt
```

## How to Get Models

### Option 1: Train Yourself (Recommended)
```powershell
# Quick training (5 epochs, ~10 minutes on GPU)
python -m src.train --config configs/resnet50_colab_fast.yaml

# Full training (15 epochs, ~30 minutes on GPU)
python -m src.train --config configs/resnet50_colab_long.yaml
```

### Option 2: Download Pre-trained Model
If your team has shared a pre-trained model:

1. Download `best.pt` from the shared link
2. Create the appropriate directory: `runs/resnet50_colab_long/`
3. Place `best.pt` inside: `runs/resnet50_colab_long/best.pt`
4. Run the app: `streamlit run app/main_app.py`

## Notes

- Model files are **NOT tracked by Git** (they're in `.gitignore`)
- Typical model size: ~98 MB (ResNet50)
- The Streamlit app will automatically detect available models
- If no model is found, the app will show instructions for training

