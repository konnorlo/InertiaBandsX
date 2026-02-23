# InertiaBands Training UI (PySide6)

This folder contains an offline desktop training tool for learning:

`descriptors -> hyperparameters`

for the InertiaBands clustering hyperparameters:
- `distance_penalty` (linear)
- `norm_rate` (linear)
- `base_penalty_log` (exp transform)
- `penalty_skew` (base+skew split)
- `minimax_strength` (sigmoid transform)

It uses:
- Bayesian optimization (Optuna TPE) per file.
- Degree-3 polynomial ridge regression with K-fold CV.
- DuckDB-backed dataset storage.

## Install

From repo root:

```bash
python3 -m venv .venv-trainer
source .venv-trainer/bin/activate
pip install -r training/requirements.txt
```

## Run

```bash
source .venv-trainer/bin/activate
python -m training.app.main
```

## Workflow

1. `Import + Descriptors`:
   - Add audio files.
   - Extract descriptors.
2. `Bayesian Opt`:
   - Select a file.
   - Run BO trials to estimate best hyperparameters.
3. `Model`:
   - Train degree-3 ridge model on all `(descriptors, best_hyperparams)` rows.
4. `Inference`:
   - Pick a new file.
   - Predict hyperparameters.
   - Optionally run short local BO refinement.
   - Export preset JSON.

## Public Dataset Fetching (Hugging Face)
Use the included script to import public datasets, filter by license, materialize audio locally, and store metadata in DuckDB.

Run from repo root:

```bash
source .venv-trainer/bin/activate
python -m training.scripts.fetch_datasets --dataset-id philgzl/fsd50k --split train --max-items 500
```

Examples:

```bash
# NSynth
python -m training.scripts.fetch_datasets --dataset-id jg583/NSynth --split train --max-items 1000

# GTZAN
python -m training.scripts.fetch_datasets --dataset-id Lehrig/GTZAN-Collection --split train --max-items 500

# Custom license allowlist
python -m training.scripts.fetch_datasets \
  --dataset-id philgzl/fsd50k \
  --split train \
  --allow-license cc0 \
  --allow-license "cc-by"
```

Notes:
- Default license allowlist is `cc0`, `cc-by`.
- Decoded/copied audio goes to `training/data/audio_cache/`.
- Per-file dataset metadata (dataset id/config/split/license/source/tags/raw record) is stored in `audio_metadata`.

## Data layout

- `training/data/training.duckdb`
- `training/models/*.joblib`
- `training/presets/*.json`

## Notes

- This is an offline trainer; it does not run JUCE plugin processing directly.
- The objective is proxy-based and should be iterated with listening feedback.
