# FRESCO Early Job Failure Prediction — Team Solution

IEEE Computer Society 2026 Global Student Challenge.

## Team
- [Your name(s) here]
- [Kaggle username(s) here]

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Project structure
config.yml -- all rules (leakage columns, window %, split dates)
src/
build_features.py -- feature builder for labeled participant/training data
build_judge_features.py -- feature builder for the unlabeled judge set
check_jid_uniqueness.py -- diagnostic: verifies jid is a safe join key
feature_correlation_study.py -- feature validation / leakage-proxy check
make_split.py -- time-respecting train/val split
train_model.py -- per-cluster XGBoost training
build_submission.py -- scores judge set, assembles submission.csv
outputs/
model_S.json, model_C.json, model_Anvil.json -- trained models
model_<cluster>_meta.json -- feature order + category levels
PROJECT_SPEC.md -- full design rationale, leakage rules, official
organizer clarification, improvement backlog
METHODOLOGY.md -- methodology report
## Reproducing the submission

1. Download the competition data (training_data_bundle, unlabeled_judge_data_bundle)
   from Kaggle. Not included in this repo per competition data-use rules.

2. Build training features, per cluster:
```bash
   python src/build_features.py "<path>/training_data" --cluster S --out outputs/features_S_full.csv
   python src/build_features.py "<path>/training_data" --cluster C --out outputs/features_C_full.csv
   python src/build_features.py "<path>/training_data" --cluster Anvil --out outputs/features_Anvil_full.csv
```

3. Build the time-respecting train/val split, per cluster:
```bash
   python src/make_split.py outputs/features_S_full.csv --cluster S
   python src/make_split.py outputs/features_C_full.csv --cluster C
   python src/make_split.py outputs/features_Anvil_full.csv --cluster Anvil
```

4. Train the three cluster-specific models:
```bash
   python src/train_model.py --cluster S
   python src/train_model.py --cluster C
   python src/train_model.py --cluster Anvil
```

5. Build judge-set features, per cluster:
```bash
   python src/build_judge_features.py "<path>/unlabeled_judge_data" --cluster S --out outputs/judge_features_S.csv
   python src/build_judge_features.py "<path>/unlabeled_judge_data" --cluster C --out outputs/judge_features_C.csv
   python src/build_judge_features.py "<path>/unlabeled_judge_data" --cluster Anvil --out outputs/judge_features_Anvil.csv
```

6. Score and assemble the final submission (requires `data/sample_submission.csv`):
```bash
   python src/build_submission.py
```

Output: `outputs/submission.csv`, zip it (containing only that file) for Kaggle upload.

## Design summary

Three independent cluster-specific models (Stampede/Conte/Anvil), no
pooling. Early-window rule: dynamic per-job window, 10% of each job's
own `timelimit`. Full leakage rules, windowing rationale, and the
official organizer clarification on early-horizon rules are documented
in `PROJECT_SPEC.md`.
