# Group Number 8

**Team members:** fill in names and NetIDs  
**Kaggle team name:** `group-number-8`

`ENS_top2` is an equal-weight rank average of `sgd_bpe` and `gbm_format`. `ENS_top3` adds `lr_bpe` at the same weight. After comparing the two averages on the fixed validation split, the file we submit is `ENS_top2`. Length bins and the threshold-0.5 errors use that rank average.

## Setup

Tested with `/opt/miniconda3/bin/python3.12` and scikit-learn 1.5.2. From this directory:

```bash
/opt/miniconda3/bin/python3.12 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`--system-site-packages` keeps the conda scikit-learn 1.5.2. `tokenizers==0.20.3` supplies the BPE tokenizer. `scipy` turns each model's scores into percentile ranks before the average. No pretrained language model is used.

Place the Kaggle files where you like and pass that directory as `--data-dir`. Do not commit them. The commands below use the course CSVs that live next to this repository.

## Train / validate

Report mode holds out the stratified 2,000-row validation set (`random_state=20261023`, 1,000/1,000), searches hyperparameters with 3-fold CV on the other 8,000 rows, and compares `lr_bpe`, `sgd_bpe`, `gbm_format`, `ENS_top2`, and `ENS_top3`. It also writes the `gbm_format` learning curve, the `sgd_bpe` unlabeled comparison, and length bins plus errors for `ENS_top2`. It does not read `test.csv`.

```bash
python train.py \
  --data-dir ../../cornell-tech-cs-5785-combined-2026-ai-vs-human-text \
  --mode report \
  --seed 20261023
```

## Submit

`--mode submit` refits on all 10,000 labels. BPE and TF-IDF are fit on train, unlabeled, and test text. The classifiers see only labeled rows. `submission.csv` is `ENS_top2`. The same run writes `submission_ens_top3.csv` for comparison and saves `model.joblib`. Do not commit those files.

```bash
python train.py \
  --data-dir ../../cornell-tech-cs-5785-combined-2026-ai-vs-human-text \
  --mode submit \
  --out submission.csv
```

`predict.py` reloads `model.joblib` and writes the same two-model file:

```bash
python predict.py \
  --test ../../cornell-tech-cs-5785-combined-2026-ai-vs-human-text/test.csv \
  --model model.joblib \
  --out submission.csv
```

## Seeds and locked settings

- Validation split seed: `20261023`
- Classifier `random_state`: `42` (`SGDClassifier`, `HistGradientBoostingClassifier`)
- `sgd_bpe`: modified Huber, `alpha=5e-5`
- `lr_bpe`: liblinear, `C=4`
- `gbm_format`: 300 iterations, learning rate 0.05, 15 leaves, minimum 40 samples per leaf, L2 penalty 1
- Submitted model: `ENS_top2` (`sgd_bpe` and `gbm_format`)
- Decision threshold for the error tables only: 0.5 on the `ENS_top2` validation rank average

The inner-CV grid is written to `results/metrics.json`. `alpha=1e-4` and `C=16` were a few thousandths higher than the locked values. The booster grid was flat (0.9886 to 0.9894). The holdout was not used to pick these values.

## AI assistance

An AI coding assistant helped turn the model definitions into `train.py` and `predict.py`. The five models and the choice to submit `ENS_top2` were ours. The AUROCs come from the report command above. No tool labeled the test passages.
