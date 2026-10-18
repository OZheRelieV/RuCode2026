import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, metric_at_thresholds
from sklearn.model_selection import StratifiedKFold
from typing import Any, Generator
from pathlib import Path
import orjson


def find_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    return_threshold: bool = False
) -> float | tuple[float, float]:
    scores, thresholds = metric_at_thresholds(y_true, y_score, f1_score)
    best_idx = np.argmax(scores)
    if return_threshold:
        return scores[best_idx], thresholds[best_idx]
    return scores[best_idx]


def get_features(df: pd.DataFrame, target: str) -> list[str]:
    columns = df.columns.tolist()
    if target in columns:
        columns.remove(target)
    return columns


def gen_folds(
    df: pd.DataFrame,
    target: str,
    kfold_params: dict[str, int | bool] | None = None
) -> pd.DataFrame:
    params = {"n_splits": 5, "shuffle": True, "random_state": 42}
    if kfold_params is not None:
        params.update(kfold_params)
    kfold = StratifiedKFold(**params)
    df["fold"] = -1
    for i, (_, vl) in enumerate(kfold.split(df, df[target])):
        df.loc[vl, "fold"] = i
    return df


def read_params(path: Path | str) ->  dict[str, Any] | None:
    if isinstance(path, str):
        path = Path(path)
    if path.exists():
        return orjson.loads(path.read_text(encoding="utf-8"))

def iter_folds(
    data: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray
) -> Generator[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    None,
    None,
]:
    for fold in np.unique(folds):
        val_mask = folds == fold
        yield (
            data[~val_mask],
            data[val_mask],
            target[~val_mask],
            target[val_mask],
        )
