# Group Number 8

**Team members:** fill in names and NetIDs  
**Kaggle team name:** `group-number-8`

Three single models on one fixed split. `lr_bpe` and `sgd_bpe` score TF-IDF on a from-scratch byte-level BPE. `gbm_format` scores 21 layout features with histogram gradient boosting. Hyperparameters were chosen by 3-fold search on the 8,000-row training partition.

The same report writes a `gbm_format` learning curve, compares `sgd_bpe` with and without unlabeled text in the feature fit, and lists length bins and threshold-0.5 errors for `gbm_format`.

## Setup

Tested with `/opt/miniconda3/bin/python3.12` and scikit-learn 1.5.2. From this directory:

```bash
/opt/miniconda3/bin/python3.12 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`--system-site-packages` keeps the conda scikit-learn 1.5.2. `tokenizers==0.20.3` supplies the BPE tokenizer. No pretrained language model is used.

Place the Kaggle files where you like and pass that directory as `--data-dir`. Do not commit them. The command below uses the course CSVs that live next to this repository.

## Train / validate

This holds out the stratified 2,000-row validation set (`random_state=20261023`, 1,000/1,000) and searches hyperparameters with 3-fold CV on the other 8,000 rows. It then writes:

- train and validation AUROC for `lr_bpe`, `sgd_bpe`, and `gbm_format`
- a `gbm_format` learning curve at 500, 1,000, 2,000, 5,000, and 8,000 rows, each a stratified prefix of the 8,000-row partition, scored on that same validation set (`figures/learning_curve.pdf`)
- `sgd_bpe` when BPE and TF-IDF are fit on the training partition, compared with the same classifier after those features are fit on the training partition plus `unlabeled.csv`
- length bins and the strongest false positives and false negatives for `gbm_format`, using probability threshold 0.5

It reads `train.csv` and `unlabeled.csv`. It does not read `test.csv`.

```bash
python train.py \
  --data-dir ../../cornell-tech-cs-5785-combined-2026-ai-vs-human-text \
  --mode report \
  --seed 20261023
```

## Seeds and locked settings

- Validation split seed: `20261023`
- Classifier `random_state`: `42` (`SGDClassifier`, `HistGradientBoostingClassifier`)
- `sgd_bpe`: modified Huber, `alpha=5e-5`
- `lr_bpe`: liblinear, `C=4`
- `gbm_format`: 300 iterations, learning rate 0.05, 15 leaves, minimum 40 samples per leaf, L2 penalty 1
- Learning-curve sizes: `500`, `1000`, `2000`, `5000`, `8000`
- Decision threshold for the `gbm_format` error table only: 0.5 on validation `predict_proba`

The inner-CV grid is written to `results/metrics.json`. `alpha=1e-4` and `C=16` were a few thousandths higher than the locked values. The booster grid was flat (0.9886 to 0.9894). The holdout was not used to pick these values.

## AI assistance

An AI coding assistant helped turn the model definitions into `train.py`. The three models, the locked hyperparameters, and the checks on `gbm_format` were ours. The AUROCs come from the report command above. No tool labeled the test passages.
