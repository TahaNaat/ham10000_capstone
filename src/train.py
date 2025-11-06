import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import pandas as pd
from src.data.dataset import HAM10000Dataset, LABEL_ORDER


def build_transforms(size: int):
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf


def class_weights_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    counts = df['dx'].value_counts()
    total = counts.sum()
    weights = []
    for c in LABEL_ORDER:
        w = total / (len(LABEL_ORDER) * counts.get(c, 1))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float)


def get_device():
    try:
        import torch_directml
        dml_device = torch_directml.device()
        _ = torch.empty(1, device=dml_device)
        print("🟣 Using device: DirectML")
        return dml_device
    except Exception as e:
        if torch.cuda.is_available():
            print(f"⚠️ DirectML not available ({e}). Falling back to CUDA.")
            return 'cuda'
        else:
            print(f"⚠️ DirectML and CUDA not available ({e}). Using CPU.")
            return 'cpu'


def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not cfg:
        raise ValueError(f"Empty or invalid config file: {cfg_path}")

    print("✅ Loaded config keys:", list(cfg.keys()))

    torch.manual_seed(cfg['seed'])
    device = get_device()
    train_tf, val_tf = build_transforms(cfg['image_size'])
    train_ds = HAM10000Dataset(cfg['train_csv'], transform=train_tf)
    val_ds = HAM10000Dataset(cfg['val_csv'], transform=val_tf)

    train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                          num_workers=cfg['num_workers'])
    val_dl = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                        num_workers=cfg['num_workers'])

    if cfg['model'].lower() == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, len(LABEL_ORDER))
    elif cfg['model'].lower() == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, len(LABEL_ORDER))
    else:
        raise ValueError('Only resnet18 or resnet50 supported in this baseline.')

    model = model.to(device)

    if cfg.get('use_class_weights', False):
        w = class_weights_from_csv(cfg['train_csv']).to(device)
        criterion = nn.CrossEntropyLoss(weight=w,
                                        label_smoothing=cfg.get('label_smoothing', 0.0))
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get('label_smoothing', 0.0))

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(cfg['lr']),
                                  weight_decay=float(cfg['weight_decay']))

    best_val = 0.0
    out_dir = Path(cfg['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting training for {cfg['max_epochs']} epochs...\n")

    for epoch in range(1, cfg['max_epochs'] + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total

        model.eval()
        tot_v, cor_v = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                cor_v += (logits.argmax(1) == y).sum().item()
                tot_v += y.size(0)

        val_acc = cor_v / tot_v

        print(f"Epoch {epoch:02d}/{cfg['max_epochs']} | "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"💾 Saved new best model (val_acc={val_acc:.3f})")

    print("\n✅ Training complete! Best validation accuracy:", round(best_val, 3))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/resnet50.yaml')
    args = ap.parse_args()
    main(args.config)
