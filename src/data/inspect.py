import argparse
from pathlib import Path
import pandas as pd
LABEL_ORDER = ['nv','mel','bkl','bcc','akiec','vasc','df']
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', type=Path, default=Path('data/processed'))
    args = ap.parse_args()
    for name in ['train','val','test']:
        df = pd.read_csv(args.splits/f"{name}.csv")
        counts = df['dx'].value_counts().reindex(LABEL_ORDER, fill_value=0)
        print(f"\n{name.upper()} counts:\n{counts}")
