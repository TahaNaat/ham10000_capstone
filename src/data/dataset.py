from pathlib import Path
from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

LABEL_ORDER = ['nv','mel','bkl','bcc','akiec','vasc','df']
LABEL_TO_IDX = {c:i for i,c in enumerate(LABEL_ORDER)}

class HAM10000Dataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row.image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = LABEL_TO_IDX[row.dx]
        return img, label
