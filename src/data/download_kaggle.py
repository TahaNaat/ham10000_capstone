import argparse, os, subprocess, sys
from pathlib import Path

KAGGLE_DIR = Path.home()/".kaggle"
CFG = KAGGLE_DIR/"kaggle.json"

def ensure_kaggle_cli():
    try:
        subprocess.run([sys.executable, "-m", "kaggle", "-h"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("Installing kaggle CLI ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"]) 

def accept_terms():
    print("âž¡ï¸  Create ~/.kaggle/kaggle.json (Kaggle > Account > Create API Token). Windows does not need chmod.")
    if not CFG.exists():
        print(f"!! Missing {CFG}. See https://www.kaggle.com/docs/api")
    else:
        print("kaggle.json found.")

def download(dataset: str, out: str):
    ensure_kaggle_cli()
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset, "-p", str(outp), "-w"]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--accept', action='store_true')
    ap.add_argument('--dataset', type=str, default='kmader/skin-cancer-mnist-ham10000')
    ap.add_argument('--out', type=str, default='data/raw')
    args = ap.parse_args()

    if args.accept:
        accept_terms()
        sys.exit(0)
    download(args.dataset, args.out)
