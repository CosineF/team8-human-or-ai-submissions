"""Course experiments on the fixed 2,000-row validation split.

The three single models, the gbm_format learning curve, the sgd_bpe unlabeled
feature comparison, and the gbm_format length bins and errors are computed
here. The Kaggle test file is not read in this module.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from pipeline.config import (
    DECISION_THRESHOLD,
    GBM_GRID,
    LEARNING_CURVE_SIZES,
    LR_CS,
    N_INNER_FOLDS,
    SGD_ALPHAS,
    SPLIT_SEED,
)
from pipeline.data import exact_text_overlap, stratified_prefix, validation_split
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


def _preview(text: str, limit: int = 280) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def run_report(
    train_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    metrics_path: Path,
    figure_path: Path,
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
            "val_proba": model.predict_proba(x_val)[:, 1],
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

    print("learning curve on gbm_format", flush=True)
    curve = []
    for size in LEARNING_CURVE_SIZES:
        subset = stratified_prefix(train_part, size, seed)
        model, train_scores, val_scores, seconds = _fit_scores(
            make_gbm,
            format_matrix(subset["text"]),
            subset["label"].to_numpy(),
            x_fmt_val,
        )
        row = {
            "n": int(size),
            "n_human": int((subset["label"] == 0).sum()),
            "n_ai": int((subset["label"] == 1).sum()),
            "train_auroc": _auroc(subset["label"], train_scores),
            "val_auroc": _auroc(y_val, val_scores),
            "train_seconds": seconds,
        }
        curve.append(row)
        print(f"  n={size} train {row['train_auroc']:.4f} val {row['val_auroc']:.4f}", flush=True)
    metrics["learning_curve_model"] = "gbm_format"
    metrics["learning_curve"] = curve
    _plot_curve(curve, figure_path)

    print("unlabeled feature learning for sgd_bpe", flush=True)
    unlabeled_flat = flatten_text(unlabeled_df["text"]).tolist()
    _, _, unlabeled_matrices, unlabeled_info = build_word_features(
        train_flat + unlabeled_flat,
        {"train": train_flat, "val": val_flat},
    )
    model, train_scores, val_scores, seconds = _fit_scores(
        make_sgd,
        unlabeled_matrices["train"],
        y_train,
        unlabeled_matrices["val"],
    )
    supervised = next(row for row in model_rows if row["name"] == "sgd_bpe")
    metrics["unlabeled"] = {
        "method": "bpe_and_tfidf_fit_on_train_partition_plus_unlabeled",
        "n_unlabeled": int(len(unlabeled_df)),
        "features": unlabeled_info,
        "supervised_train_auroc": supervised["train_auroc"],
        "supervised_val_auroc": supervised["val_auroc"],
        "supervised_seconds": supervised["train_seconds"],
        "with_unlabeled_train_auroc": _auroc(y_train, train_scores),
        "with_unlabeled_val_auroc": _auroc(y_val, val_scores),
        "with_unlabeled_seconds": seconds,
    }
    print(
        f"  supervised {supervised['val_auroc']:.4f} "
        f"with unlabeled {metrics['unlabeled']['with_unlabeled_val_auroc']:.4f}",
        flush=True,
    )

    print("length bins and errors on gbm_format", flush=True)
    words = val_part["text"].str.split().str.len().fillna(0).to_numpy()
    q1, q2 = np.quantile(words, [1 / 3, 2 / 3])
    bins = [
        ("short", words <= q1),
        ("medium", (words > q1) & (words <= q2)),
        ("long", words > q2),
    ]
    bin_rows = []
    val_scores = np.asarray(locked["gbm_format"]["val_proba"], dtype=np.float64)
    for name, mask in bins:
        bin_rows.append(
            {
                "bin": name,
                "rule": {
                    "short": f"word count <= {q1:.0f}",
                    "medium": f"{q1:.0f} < word count <= {q2:.0f}",
                    "long": f"word count > {q2:.0f}",
                }[name],
                "q1": float(q1),
                "q2": float(q2),
                "count": int(mask.sum()),
                "n_human": int((y_val[mask] == 0).sum()),
                "n_ai": int((y_val[mask] == 1).sum()),
                "val_auroc": _auroc(y_val[mask], val_scores[mask]),
                "mean_words": float(words[mask].mean()),
            }
        )
    metrics["length_bins_model"] = "gbm_format"
    metrics["length_bins"] = bin_rows

    predicted_ai = val_scores >= DECISION_THRESHOLD
    false_positive = predicted_ai & (y_val == 0)
    false_negative = (~predicted_ai) & (y_val == 1)
    fp_order = np.argsort(-val_scores)
    fn_order = np.argsort(val_scores)
    errors = {
        "threshold": DECISION_THRESHOLD,
        "score": "gbm_format_proba",
        "false_positives": [],
        "false_negatives": [],
    }
    for index in fp_order:
        if not false_positive[index]:
            continue
        errors["false_positives"].append(_error_row(val_part, words, val_scores, index, "FP"))
        if len(errors["false_positives"]) == 5:
            break
    for index in fn_order:
        if not false_negative[index]:
            continue
        errors["false_negatives"].append(_error_row(val_part, words, val_scores, index, "FN"))
        if len(errors["false_negatives"]) == 5:
            break
    errors["n_fp"] = int(false_positive.sum())
    errors["n_fn"] = int(false_negative.sum())
    errors["n_correct"] = int((predicted_ai == (y_val == 1)).sum())
    metrics["errors"] = errors

    _write(metrics_path, metrics)
    print(f"wrote {metrics_path}", flush=True)
    return metrics


def _error_row(val_part, words, scores, index: int, kind: str) -> dict:
    return {
        "id": int(val_part.iloc[index]["id"]),
        "error_type": kind,
        "label": int(val_part.iloc[index]["label"]),
        "p_ai": float(scores[index]),
        "n_words": int(words[index]),
        "preview": _preview(str(val_part.iloc[index]["text"])),
    }


def _plot_curve(curve: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sizes = [row["n"] for row in curve]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(sizes, [row["train_auroc"] for row in curve], marker="o", color="#19334C", label="Training AUROC")
    ax.plot(sizes, [row["val_auroc"] for row in curve], marker="o", color="#236B82", label="Validation AUROC")
    ax.set_xlabel("Labeled training rows")
    ax.set_ylabel("AUROC")
    ax.set_xticks(sizes)
    scores = [row["train_auroc"] for row in curve] + [row["val_auroc"] for row in curve]
    ax.set_ylim(max(0.5, min(scores) - 0.02), 1.005)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
