"""Classifiers for the three single models.

lr_bpe and sgd_bpe score BPE TF-IDF. gbm_format scores the 21 layout features.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

from pipeline.config import (
    GBM_L2,
    GBM_LEARNING_RATE,
    GBM_MAX_ITER,
    GBM_MAX_LEAF_NODES,
    GBM_MIN_SAMPLES_LEAF,
    LR_C,
    MODEL_SEED,
    SGD_ALPHA,
)


def make_sgd(alpha: float = SGD_ALPHA, seed: int = MODEL_SEED) -> SGDClassifier:
    return SGDClassifier(
        loss="modified_huber",
        alpha=alpha,
        max_iter=5000,
        tol=1e-4,
        random_state=seed,
    )


def make_lr(C: float = LR_C) -> LogisticRegression:
    return LogisticRegression(C=C, solver="liblinear", max_iter=2000)


def make_gbm(
    learning_rate: float = GBM_LEARNING_RATE,
    max_leaf_nodes: int = GBM_MAX_LEAF_NODES,
    l2_regularization: float = GBM_L2,
    seed: int = MODEL_SEED,
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=GBM_MAX_ITER,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=GBM_MIN_SAMPLES_LEAF,
        l2_regularization=l2_regularization,
        random_state=seed,
    )
