"""Train and evaluate the three single-model detectors.

Report mode holds out the fixed validation split and writes train and
validation AUROC for lr_bpe, sgd_bpe, and gbm_format. It also writes the
gbm_format learning curve, the sgd_bpe unlabeled comparison, and the
gbm_format length bins and threshold-0.5 errors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.config import SPLIT_SEED
from pipeline.data import load_split
from pipeline.experiments import run_report


def _data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir is not None:
        return args.data_dir
    if args.train is None:
        raise SystemExit("Pass --data-dir or --train")
    return args.train.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Human-or-AI detectors")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--train", type=Path, default=None, help="train.csv; parent is the data dir")
    parser.add_argument("--unlabeled", type=Path, default=None)
    parser.add_argument("--mode", choices=("report",), default="report")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--metrics-out", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--figure-out", type=Path, default=Path("figures/learning_curve.pdf"))
    args = parser.parse_args()

    data_dir = _data_dir(args)
    train_path = args.train or (data_dir / "train.csv")
    unlabeled_path = args.unlabeled or (data_dir / "unlabeled.csv")
    train_df = load_split(train_path, labeled=True)
    unlabeled_df = load_split(unlabeled_path, labeled=False)
    if args.mode == "report":
        run_report(
            train_df,
            unlabeled_df,
            args.metrics_out,
            args.figure_out,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
