import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import Config
from src import utils
from argparse import ArgumentParser

PARAMS_DIR  = Path("params")
MAPPINGS_DIR = Path("mappings")
MODELS_DIR = Path("models/tabdpt")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test", required=True, help="Path to test set")
    parser.add_argument("--part", required=True, choices=["public", "private"], default="public")

    args = parser.parse_args()
    test_path = Path(args.test)

    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} file doesn't exist. Check correctness of path to test set")
    if not MAPPINGS_DIR.exists() or not PARAMS_DIR.exists():
        raise FileNotFoundError(f"Firstly run 'tabdpt_hpo.py' to get {MAPPINGS_DIR}/{PARAMS_DIR} and try again")
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Firstly run 'tabdpt_train.py' to get {MODELS_DIR} and try again")

    test = pd.read_csv(test_path)
    if args.part == "public":
        # manual processing of rare 'employment_type' column category
        test.loc[421, "employment_type"] = "unemployed"
    # private test doesn't contain rare category
   
    for col in Config.cat_cols:
        mapping = utils.read_params(MAPPINGS_DIR / f"{col}.json")
        test[col] = test[col].map(mapping)

    with open(PARAMS_DIR / "tabdpt_features.txt", 'r', encoding="utf-8") as f:
        features = list(map(lambda x: x.strip(), f.readlines()))

    x_test = test[features].values

    raw_paths = sorted(MODELS_DIR.iterdir(), key=lambda x: int(x.stem.split('_')[0][-1]))
    test_preds = np.zeros((test.shape[0], len(raw_paths)))
    weights = []
    for i, p in enumerate(raw_paths):
        model = joblib.load(p)
        w, th = tuple(map(float, p.stem.split('_')[1:]))
        weights.append(w)
        probs = model.predict_proba(x_test)[:, 1]
        test_preds[:, i] = (probs >= th).astype(np.int8)

    weights = np.asarray(weights)
    weighted_votes = test_preds @ weights
    final_classes = (weighted_votes >= 0.5).astype(np.int8)

    Path("subs").mkdir(exist_ok=True)
    test["accepted"] = final_classes
    test[["customer_id", "accepted"]].to_csv(
        f"subs/submission_tabdpt_{args.part}.csv", 
        index=False
    )
