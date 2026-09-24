# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Predicts Remaining Useful Life (RUL) of jet engines from the NASA CMAPSS **FD001** dataset, end to end: data prep → feature engineering → model training with MLflow tracking → batch inference → FastAPI + Streamlit serving. Based on the cookiecutter data-science layout. Windows dev environment (conda, Python 3.11).

## Commands

All scripts use paths relative to the repo root, so run everything from the root. Modules import each other as `features.*` (not `src.features.*`), so `src` must be on `PYTHONPATH` for scripts that import across packages.

```bash
pip install -r requirements.txt          # conda-exported; many entries are file:// paths from a conda build

# Tests (pytest.ini sets pythonpath = src)
python -m pytest
python -m pytest tests/test_generate_labels.py::test_generate_failure_labels_basic

flake8 src                               # lint (tox.ini: max-line-length 79, max-complexity 10)

# Pipeline, in order
python src/data/make_dataset.py                                  # raw .txt -> data/processed + data/cleaned
PYTHONPATH=src python src/features/driver_health_indicators.py   # labels + features -> data/features
python src/backup_mlflow.py              # mlflow.db -> backups/mlflow_<date>_<HHMM>.db (or: make backup-mlflow)
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000    # must be running before training/inference
python src/models/train_model.py            # RandomForest baseline
python src/models/train_model_xgb.py
python src/models/train_model_lightgbm.py
python src/models/predict_model.py          # best run -> data/processed/rul_predictions.csv
python src/models/log_top_model.py          # compare best run per experiment

# Serving
uvicorn src.app.app_api:app --reload        # Swagger at http://127.0.0.1:8000/docs
streamlit run src/app/app_dashboard.py

# Streaming simulation (needs Docker Desktop)
PYTHONPATH=src python src/streaming/run_all.py                   # broker, MLflow, consumer, API, dashboard; Ctrl+C stops all
PYTHONPATH=src python src/streaming/producer.py --interval 0.2   # one replay session (refuses unless the consumer is online)
# watch the dashboard's Live tab, or http://127.0.0.1:8000/live; logs in logs/<service>.log
# by hand instead of run_all: docker compose up -d, MLflow, consumer.py, uvicorn, streamlit (in that order)
```

The README's usage section says `scripts/...` and `app/...`. Those paths are out of date; the code lives under `src/`.

## Architecture

The stages are separate scripts that hand data to each other through CSV files on disk. There is no shared config module: every path and constant is hardcoded in each script, so if you rename a file or column, update every stage that uses it.

1. **`src/data/make_dataset.py`**: parses the space-separated `data/raw/{train,test}_FD001.txt`, assigns columns `unit, time_in_cycles, op_setting_1..3, sensor_1..21`, and writes the same frame to both `data/processed/*_FD001.csv` and `data/cleaned/*_FD001_cleaned.csv`.
2. **`src/features/driver_health_indicators.py`** runs two steps:
   - `generate_failure_labels`: `RUL = max(time_in_cycles per unit) - time_in_cycles`, plus `failure_binary = RUL <= 30`. Writes `data/cleaned/*_labeled.csv`. For the test set (truncated series), the driver passes `true_rul_path=data/processed/rul_FD001.csv` (written by `make_dataset.py` from `RUL_FD001.txt`), so each test unit's RUL is offset by its ground truth. The last-cycle RUL then equals RUL_FD001.
   - `engineer_health_indicators`: for a fixed list of 15 informative sensors, computes per-unit baselines (the mean of the first 5 cycles), `<sensor>_baseline`, `<sensor>_degraded` (below 75% of baseline), `<sensor>_rolling_mean`/`_rolling_std` (window 5) and `<sensor>_cycle_change`. Writes `data/features/*_features.csv`. The `health_score` feature was removed on purpose (it's commented out).
3. **Training (`src/models/train_model*.py`)**: each script picks its feature columns with the same rule, `"sensor" in col or "health_score" in col`, excluding `failure_binary`. It runs a small random hyperparameter search with an 80/20 split **grouped by `unit`** (`GroupShuffleSplit`; a random row split leaks cycles of the same engine across the split), and logs MAE/RMSE/R2/Max Error, the model and a feature-importance CSV to MLflow at `http://127.0.0.1:5000` (override with `MLFLOW_TRACKING_URI`, e.g. to retrain against a throwaway server; don't use `localhost`, see Known state). The importance CSVs and feature-column JSONs land in the repo root as side effects. MLflow experiment names:
   - `FD001 RUL Hyperparam Tuning (unit split, seeded)` (sklearn RF, artifact `random_forest_model`)
   - `FD001 RUL XGBoost Hyperparam Tuning (unit split, seeded)` (artifact `xgboost_model`)
   - `FD001 RUL LightGBM Hyperparam Tuning (unit split, seeded)` (artifact `lightgbm_model`)
   - Older experiments: no suffix = leaky row split; `(unit split)` = unseeded search. Don't select from them.
   - **Reproducibility:** each training script has one `SEED = 42`. It drives `GroupShuffleSplit`, the search RNG (a local `random.Random(SEED)`) and the grid's `random_state`, and each run logs `search_seed`. Never use the global `random`/`np.random` functions or unseeded splitters: `tests/test_reproducibility.py` scans `src/` with an AST check and fails on them. If a change alters which runs get selected, log to a new experiment name so old runs aren't mixed in.
4. **`src/models/predict_model.py`**: finds the lowest-RMSE finished run across those three experiments. It infers the loader flavour and artifact name *from the experiment name string*, so experiment and artifact names have to stay in sync with the training scripts. It rebuilds the feature list from the train features CSV, predicts on test features, and assigns a risk level: High < 30, Medium < 100, Low ≥ 100. It writes `data/processed/rul_predictions.csv` (one row per test cycle: `unit, time_in_cycles, RUL, risk_level`). `evaluate_against_truth` scores each unit's last-cycle prediction against the true labels. The run is logged, with `test_MAE/RMSE/R2`, to the `FD001 RUL Inference` experiment.
5. **Serving**: both apps only read CSVs; neither loads a model.
   - `src/app/app_api.py` (FastAPI) serves each unit's latest-cycle prediction from `rul_predictions.csv` at `/`, `/predictions` and `/predictions/{unit_id}`. It has a fallback copy of the risk thresholds.
   - `src/app/app_dashboard.py` (Streamlit, `@st.cache_data` loaders) reads the train features, the predictions (reduced to the latest cycle per unit with `latest_per_unit`) and the test labels for the true RUL. Tabs: Summary, Overview, Unit Analysis, RUL Predictions, Commercial Analysis. The Overview and Unit Analysis tabs plot *training* engines, and Unit Analysis mixes them with *test* predictions that share the same unit number (a known issue).

6. **Streaming (`src/streaming/`, `src/app/stream_gateway.py`)**: a real-time simulation built on top of the batch pipeline. It doesn't replace any part of it.
   - `producer.py` replays `data/raw/test_FD001.txt` as one QoS 1 MQTT message per engine per cycle on `pm/fd001/engine/{unit}/telemetry`. Engines start at seeded offsets (`--stagger`), and each run is a new session announced on the retained `pm/fd001/control/session`.
   - `consumer.py` feeds messages to `StreamProcessor`, which stores the raw readings in SQLite (`data/stream/stream_state.db`, via `state_store.py`). On each micro-batch, `stream_features.py` reruns the batch `engineer_health_indicators()` over each touched engine's full history. The result is **bitwise identical** to batch; a bounded 5-row buffer would drift by about 1e-9, because pandas' rolling sums accumulate from the start of the series.
   - Cycles 1-4 are held back and emitted together at cycle 5, since the batch baseline averages the first 5 cycles.
   - Predictions come from `predict_model.load_best_model()` and are published retained on `.../{unit}/prediction`. MQTT messages are acked only after the SQLite commit.
   - A new session wipes the other sessions' rows unless the producer ran with `--keep-history`.
   - `stream_gateway.py` bridges predictions to `/ws/live` and is mounted on `app_api.py` with its lifespan. The Streamlit "Live" tab iframes the `/live` page, so updates arrive without reruns. `GET /stream/status` reports the broker connection, the session, WebSocket clients and message/drop counters; the gateway also logs a stats line every 10 s while messages flow.
   - Readiness: once its model is loaded and its subscriptions are acknowledged, the consumer publishes a retained `{"state": "online", "pid", "model_run_id", "ts"}` on `pm/fd001/consumer/status`. A last-will and a clean stop set it to `offline`. The producer refuses to start unless it is `online` (`--no-consumer-check` overrides). The consumer warns when telemetry arrives more than 5 s after it was sent.
   - `run_all.py` starts broker → MLflow (only if it isn't already up) → consumer → gateway → dashboard, waiting on a real readiness check for each: MQTT CONNACK, `/health`, a consumer `online` status newer than its start, `/stream/status` showing the broker connected, and `/_stcore/health`. Children run in their own process groups. On Ctrl+C it sends each one Ctrl+Break in reverse order, then force-kills after 10 s. The dashboard is killed directly because it ignores Ctrl+Break for longer than that. A broker or MLflow server that was already running is left running. Children get `PYTHONUNBUFFERED=1`, since piped output is block-buffered, and `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`, since otherwise conda's Intel Fortran runtime aborts on Ctrl+Break before Python's cleanup runs.
   - `tests/test_stream_*.py` hold the parity tests against batch features and `rul_predictions.csv`. The real-model test skips if MLflow isn't running.

The risk thresholds (30/100) and the failure threshold (30) are repeated in several files. Keep them consistent.

## Conventions

- `DECISIONS.md` at the repo root is the project's decision log. After any plan is approved and implemented, add an entry to it in the same commit as the change, before considering the task complete. Keep entries short: it's a decision log, not a design doc.
- When writing a plan in plan mode, always include a short "Why this approach" paragraph near the top, even for small changes, not just a list of what will be done.

## Known state

- `mlflow.db` (SQLite backend) is local and was recreated on 2026-09-23 after the original was deleted (see DECISIONS.md). It holds only that day's seeded retrain: experiments 1-3 are the RF, XGBoost and LightGBM `(unit split, seeded)` searches and 4 is `FD001 RUL Inference`, with artifacts in `mlartifacts/1`-`4` (git-ignored). Run `python src/backup_mlflow.py` (or `make backup-mlflow`) before any operation that touches the training pipeline or MLflow store directly. It snapshots `mlflow.db` into the git-ignored `backups/` folder with a timestamped name, and is safe to run while the server is up. `make` isn't installed on the dev machine, so use the Python command there.
- `mlruns/1`-`4` and `mlartifacts/5`-`8`, `10` are orphaned artifacts from the lost database: model files with no runs, params or metrics. The next new experiment gets ID 5 and will share `mlartifacts/5` with orphans. `mlruns/` folders with 18-digit IDs are the older file-store experiments. Pointing MLflow at `./mlruns` as a file store reports the orphaned integer folders as "Malformed experiment"; that's expected, not corruption. `.env` exists at the root.
- For a throwaway MLflow server (for example, to check reproducibility without touching `mlflow.db`), pass `--default-artifact-root file:///C:/...`. A bare `C:/...` path is parsed as URI scheme `c` and artifact logging fails.
- On Windows, set `PYTHONIOENCODING=utf-8` when piping MLflow script output. Otherwise MLflow's emoji run links crash with cp1252 `UnicodeEncodeError`.
- The `<sensor>_degraded` flags (below 75% of baseline) are 0 on every row of the real data, so they carry no signal.
- `.gitignore`'s `models/` rule also matches `src/models/`. `log_top_model.py` is untracked because of it, `git add src/models/...` warns and exits non-zero even for tracked files, and the Grep tool (ripgrep) skips `src/models/`, so search it with plain `grep`.
- `app_api.py`'s `get_unit_prediction` wraps its 404 in `except Exception`, so an unknown unit returns 500.
- RUL targets are not capped (published FD001 results usually clip at about 125), so metrics aren't directly comparable with the literature.
- The hyperparameter search samples with replacement, so an iteration can repeat an earlier parameter set (reproducibly).
- `src/visualization/visualize.py` is empty.
- MLflow 3 stores logged models under `mlartifacts/<exp>/models/<model_id>/`, not in the run's artifact folder. A `runs:/<run>/<name>` URI first requests the missing run artifact. The server answers 500, and the client's retry backoff adds about 4 minutes before it falls back. `load_best_model` therefore resolves the run's logged model and loads `models:/<model_id>` (about 3 s), keeping `runs:/` only as a fallback.
- Use `127.0.0.1`, not `localhost`, for local services. The MLflow server listens on IPv4 only, and on this machine `localhost` tries IPv6 first and stalls 2 s per request.
- The data CSVs differ from in-memory pipeline output by about 1e-12 (CSV float formatting). Compare streamed features with in-memory batch output for exact checks.
- `app_api.py` imports the gateway relatively (`from .stream_gateway`), so run it as `src.app.app_api`, as documented above.
