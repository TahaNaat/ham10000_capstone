import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LABEL_ORDER = ['nv','mel','bkl','bcc','akiec','vasc','df']

def make_splits(meta_csv: Path, out_dir: Path, seed: int = 42, test_size_fold: int = 5):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(meta_csv)
    required_cols = {'image_id','dx','lesion_id'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns in metadata: need {required_cols}")

    df = df[df['dx'].isin(LABEL_ORDER)].copy()
    df['dx'] = pd.Categorical(df['dx'], categories=LABEL_ORDER)

    root = meta_csv.parent
    def guess_path(image_id: str):
        for sub in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
            p = root / sub / f"{image_id}.jpg"
            if p.exists():
                return p.as_posix()
        return None
    df['image_path'] = df['image_id'].apply(guess_path)
    df = df.dropna(subset=['image_path'])

    X = df[['image_id']]
    y = df['dx']
    groups = df['lesion_id']

    sgkf = StratifiedGroupKFold(n_splits=test_size_fold, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sgkf.split(X, y=y, groups=groups))
    d_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    d_test = df.iloc[test_idx].reset_index(drop=True)

    y_tv = d_trainval['dx']
    g_tv = d_trainval['lesion_id']
    sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr_idx, val_idx = next(sgkf2.split(d_trainval[['image_id']], y=y_tv, groups=g_tv))
    d_train = d_trainval.iloc[tr_idx].reset_index(drop=True)
    d_val = d_trainval.iloc[val_idx].reset_index(drop=True)

    for name, d in [('train', d_train), ('val', d_val), ('test', d_test)]:
        d[['image_id','image_path','dx','lesion_id']].to_csv(out_dir/f"{name}.csv", index=False)
        print(name, d['dx'].value_counts().sort_index())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('data/processed'))
    ap.add_argument('--images-root', type=Path, default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    make_splits(args.meta, args.out, seed=args.seed)
