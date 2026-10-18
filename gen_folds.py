import pandas as pd
from src.utils import gen_folds
from src.config import Config
from pathlib import Path

DATA_DIR = Path("data")
TRAIN_PATH = Path("data/train.csv")

if __name__ == "__main__":
    if not DATA_DIR.exists():
        raise FileNotFoundError("'data\' folder should exist with train and public test files.")
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"{DATA_DIR} should contain train.csv")

    df = pd.read_csv(TRAIN_PATH)
    df = gen_folds(df, Config.target)

    df.to_csv(DATA_DIR / "train_folds.csv", index=False)
