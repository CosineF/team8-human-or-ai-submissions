"""Course experiments on the fixed 2,000-row validation split.

Train and validation AUROC for lr_bpe, sgd_bpe, and gbm_format are computed
here, along with the 3-fold search on the 8,000-row partition. The Kaggle
test file is not read in this module.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from pipeline.config import (
    GBM_GRID,
    LR_CS,
    N_INNER_FOLDS,
    SGD_ALPHAS,
    SPLIT_SEED,
)
from pipeline.data import exact_text_overlap, validation_split
from pipeline.features import (
    FORMAT_COLUMNS,
    build_word_features,
    flatten_text,
    format_matrix,
)
from pipeline.models import make_gbm, make_lr, make_sgd


def _auroc(y_true, scores) -> float:
    return float(roc_auc_score(y_true, scores))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")


def _fit_scores(factory, x_train, y_train, x_val):
    model = factory()
    started = time.perf_counter()
    model.fit(x_train, y_train)
    seconds = time.perf_counter() - started
    return model, model.decision_function(x_train), model.decision_function(x_val), seconds


def _inner_cv(factory, x_matrix, y, seed: int) -> float:
    splitter = StratifiedKFold(N_INNER_FOLDS, shuffle=True, random_state=seed)
    scores = []
    for train_idx, valid_idx in splitter.split(np.zeros(len(y)), y):
        model = factory()
        model.fit(x_matrix[train_idx], y[train_idx])
        scores.append(_auroc(y[valid_idx], model.decision_function(x_matrix[valid_idx])))
    return float(np.mean(scores))


def _mean_diff(x_matrix: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    for index, name in enumerate(FORMAT_COLUMNS):
        column = x_matrix[:, index]
        ai = column[y == 1]
        human = column[y == 0]
        pooled = np.sqrt((ai.var(ddof=1) + human.var(ddof=1)) / 2)
        gap = 0.0 if pooled == 0 else float((ai.mean() - human.mean()) / pooled)
        rows.append(
            {
                "feature": name,
                "mean_human": float(human.mean()),
                "mean_ai": float(ai.mean()),
                "standardized_mean_diff_ai_minus_human": gap,
            }
        )
    rows.sort(key=lambda row: abs(row["standardized_mean_diff_ai_minus_human"]), reverse=True)
    return rows


def run_report(
    train_df: pd.DataFrame,
    metrics_path: Path,
    seed: int = SPLIT_SEED,
) -> dict:
    train_part, val_part = validation_split(train_df, seed=seed)
    y_train = train_part["label"].to_numpy()
    y_val = val_part["label"].to_numpy()
    overlap_ids = int(len(set(train_part["id"]) & set(val_part["id"])))
    copied, copied_casefold = exact_text_overlap(train_part, val_part)
    metrics: dict = {
        "split_seed": seed,
        "model_seed": 42,
        "n_train": int(len(train_part)),
        "n_val": int(len(val_part)),
        "val_human": int((y_val == 0).sum()),
        "val_ai": int((y_val == 1).sum()),
        "id_overlap": overlap_ids,
        "exact_text_overlap": copied,
        "casefold_text_overlap": copied_casefold,
        "val_ids": val_part["id"].astype(int).tolist(),
    }
    if overlap_ids or copied or copied_casefold:
        raise RuntimeError(
            f"Validation split leaked: id overlap {overlap_ids}, exact text overlap {copied}"
        )
    if metrics["val_human"] != 1000 or metrics["val_ai"] != 1000 or metrics["n_val"] != 2000:
        raise RuntimeError("Validation split is not 1000/1000")
    print(
        f"split train={metrics['n_train']} val={metrics['n_val']} "
        f"human/ai={metrics['val_human']}/{metrics['val_ai']}",
        flush=True,
    )

    x_fmt_train = format_matrix(train_part["text"])
    x_fmt_val = format_matrix(val_part["text"])
    metrics["format_feature_gaps_train"] = _mean_diff(x_fmt_train, y_train)

    print("searching gbm_format", flush=True)
    gbm_search = []
    for params in GBM_GRID:
        started = time.perf_counter()
        score = _inner_cv(lambda params=params: make_gbm(**params), x_fmt_train, y_train, seed)
        gbm_search.append({**params, "cv_auroc": score, "seconds": time.perf_counter() - started})
        print(f"  gbm {params} -> {score:.4f}", flush=True)
    metrics["gbm_search"] = gbm_search

    print("searching sgd will start after BPE; fitting wording features on the 8000", flush=True)
    train_flat = flatten_text(train_part["text"]).tolist()
    val_flat = flatten_text(val_part["text"]).tolist()
    _, _, word_matrices, word_info = build_word_features(
        train_flat, {"train": train_flat, "val": val_flat}
    )
    metrics["word_features_supervised"] = word_info
    print(
        f"  vocab {word_info['vocab_size']} features {word_info['n_features']} "
        f"in {word_info['seconds']:.1f}s",
        flush=True,
    )
    x_word_train = word_matrices["train"]
    x_word_val = word_matrices["val"]

    print("searching sgd_bpe alpha", flush=True)
    sgd_search = []
    for alpha in SGD_ALPHAS:
        started = time.perf_counter()
        score = _inner_cv(lambda alpha=alpha: make_sgd(alpha=alpha), x_word_train, y_train, seed)
        sgd_search.append(
            {"alpha": alpha, "cv_auroc": score, "seconds": time.perf_counter() - started}
        )
        print(f"  sgd alpha={alpha} -> {score:.4f}", flush=True)
    metrics["sgd_search"] = sgd_search

    print("searching lr_bpe C", flush=True)
    lr_search = []
    for C in LR_CS:
        started = time.perf_counter()
        score = _inner_cv(lambda C=C: make_lr(C=C), x_word_train, y_train, seed)
        lr_search.append({"C": C, "cv_auroc": score, "seconds": time.perf_counter() - started})
        print(f"  lr C={C} -> {score:.4f}", flush=True)
    metrics["lr_search"] = lr_search
    _write(metrics_path, metrics)

    print("fitting locked single models on the 8000", flush=True)
    locked = {}
    specs = {
        "lr_bpe": (lambda: make_lr(), x_word_train, x_word_val),
        "sgd_bpe": (lambda: make_sgd(), x_word_train, x_word_val),
        "gbm_format": (lambda: make_gbm(), x_fmt_train, x_fmt_val),
    }
    for name, (factory, x_train, x_val) in specs.items():
        model, train_scores, val_scores, seconds = _fit_scores(factory, x_train, y_train, x_val)
        locked[name] = {
            "model": model,
            "train_scores": np.asarray(train_scores, dtype=np.float64),
            "val_scores": np.asarray(val_scores, dtype=np.float64),
            "train_auroc": _auroc(y_train, train_scores),
            "val_auroc": _auroc(y_val, val_scores),
            "train_seconds": seconds,
        }
        print(
            f"  {name} train {locked[name]['train_auroc']:.4f} "
            f"val {locked[name]['val_auroc']:.4f} fit {seconds:.1f}s",
            flush=True,
        )

    model_rows = []
    for name in ("lr_bpe", "sgd_bpe", "gbm_format"):
        model_rows.append(
            {
                "name": name,
                "train_auroc": locked[name]["train_auroc"],
                "val_auroc": locked[name]["val_auroc"],
                "train_seconds": locked[name]["train_seconds"],
            }
        )
    metrics["models"] = model_rows
    _write(metrics_path, metrics)
    print(f"wrote {metrics_path}", flush=True)
    return metrics
