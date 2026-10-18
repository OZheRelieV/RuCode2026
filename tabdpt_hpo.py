import array
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tabdpt import TabDPTClassifier
from functools import partial
import optuna
from src.config import Config
from src import utils

DATA_DIR = Path("data")
TRAIN_PATH = Path("data/train_folds.csv")
TEST_PATH = Path("data/public_test.csv")


def objective(trial, data, target, folds):
    params = {
        "normalizer": trial.suggest_categorical("normalizer", ["standard", "minmax", "robust", "power", "quantile-uniform", "quantile-normal", "log1p", None]),
        "missing_indicators": trial.suggest_categorical("missing_indicators", [False, True]),
        "clip_sigma": trial.suggest_float("clip_sigma", 0.1, 10, log=True),
        "feature_reduction": trial.suggest_categorical("feature_reduction", ["pca", "subsample"]),
        "context_reduction": trial.suggest_categorical("context_reduction", ["retrieval", "subsample", "subsample-balanced"]),
        "faiss_metric": trial.suggest_categorical("faiss_metric", ["ip", "l2"]),
    }

    try:
        scores = array.array('d')
        for x_tr, x_vl, y_tr, y_vl in utils.iter_folds(data, target, folds):
            clf = TabDPTClassifier(**params)
            clf.fit(x_tr, y_tr)

            preds = clf.predict_proba(x_vl)[:, 1]
            score = utils.find_threshold(y_vl, preds)
            scores.append(score)

        return np.mean(scores)
    except Exception as _:
        return 0.0


if __name__ == "__main__":
    if not DATA_DIR.exists():
        raise FileNotFoundError("'data\' folder should exist with train and public test files.")
    if not TRAIN_PATH.exists():
        raise FileNotFoundError("Firtsly run 'gen_folds.py' script and try again")
    if not TEST_PATH.exists():
        raise FileNotFoundError("'data\' folder should contain train and public test files.")
    
    df = pd.read_csv(TRAIN_PATH)
    df.drop(Config.drop_cols, axis=1, inplace=True)

    Path("mappings").mkdir(exist_ok=True)
    for col in Config.cat_cols:
        mapping = {c: i for i, c in enumerate(df[col].unique())}
        with open(f"mappings/{col}.json", 'w', encoding="utf-8") as f:
            json.dump(mapping, f)
        df[col] = df[col].map(mapping)

    features = utils.get_features(df, Config.target)
    if not Config.fold_col in features:
        raise ValueError(f"{TRAIN_PATH} should contain {Config.fold_col}. Rerun 'gen_folds.py' script and check up manually persistance of {Config.fold_col}")
    features.remove(Config.fold_col)

    Path("params").mkdir(exist_ok=True)
    with open("params/tabdpt_features.txt", 'w', encoding="utf-8") as f:
        f.write('\n'.join(features))

    data = df[features].values
    target = df[Config.target].values
    folds = df[Config.fold_col].values

    objective_func = partial(objective, data=data, target=target, folds=folds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=100, n_jobs=4, show_progress_bar=True)

    with open("params/tabdpt.json", 'w', encoding="utf-8") as f:
            json.dump(study.best_params, f)
