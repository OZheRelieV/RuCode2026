import array
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import Config
from src import utils
from tabdpt import TabDPTClassifier
import shutil

DATA_DIR = Path("data")
TRAIN_PATH = Path("data/train_folds.csv")
TEST_PATH = Path("data/public_test.csv")
PARAMS_PATH = Path("params/tabdpt.json")
MAPPINGS_DIR = Path("mappings")

if __name__ == "__main__":
    if not DATA_DIR.exists():
        raise FileNotFoundError("'data\' folder should exist with train and public test files.")
    if not TRAIN_PATH.exists():
        raise FileNotFoundError("Firtsly run 'gen_folds.py' script and try again")
    if not TEST_PATH.exists():
        raise FileNotFoundError("'data\' folder should contain train and public test files.")
    if not PARAMS_PATH.exists() or not MAPPINGS_DIR.exists():
        raise FileNotFoundError(f"Firstly run 'tabdpt_hpo.py' to get {PARAMS_PATH}/{MAPPINGS_DIR} and try again")
   
    df = pd.read_csv(TRAIN_PATH)
    df.drop(Config.drop_cols, axis=1, inplace=True)

    for col in Config.cat_cols:
        mapping = utils.read_params(MAPPINGS_DIR / f"{col}.json")
        df[col] = df[col].map(mapping)

    features = utils.get_features(df, Config.target)
    if not Config.fold_col in features:
        raise ValueError(f"{TRAIN_PATH} should contain {Config.fold_col}. Rerun 'gen_folds.py' script and check up manually persistance of {Config.fold_col}")
    features.remove(Config.fold_col)

    data = df[features].values
    target = df[Config.target].values
    folds = df[Config.fold_col].values

    best_params = utils.read_params(PARAMS_PATH)

    oof_preds = np.zeros(len(df))
    scores, models, score_thresholds = array.array('d'), [], array.array('d')
    for fold in np.unique(folds):
        vl_mask = folds == fold
        x_train, x_val = data[~vl_mask], data[vl_mask]
        y_train, y_val = target[~vl_mask], target[vl_mask]

        clf = TabDPTClassifier(**best_params)
        clf.fit(x_train, y_train)
        models.append(clf)

        preds = clf.predict_proba(x_val)[:, 1]
        oof_preds[vl_mask] = preds

        best_f1, best_thresh = utils.find_threshold(y_val, preds, return_threshold=True)
        print(f"Best Threshold: {best_thresh} with F1: {best_f1}")

        scores.append(best_f1)
        score_thresholds.append(best_thresh)

    weights = np.asarray(scores) / sum(scores)
    print(np.mean(scores), np.std(scores), end="\n\n")

    if Path("models/tabdpt").exists():
        shutil.rmtree("models/tabdpt")
    Path("models/tabdpt").mkdir(exist_ok=True, parents=True)
    for i, (m, w, th) in enumerate(zip(models, weights, score_thresholds)):
        joblib.dump(m, f"models/tabdpt/fold{i}_{str(w)}_{str(th)}.pkl")
