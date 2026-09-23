# Decision log

Significant project decisions, oldest first. Add new entries at the bottom, in the same commit as the change they describe. Keep each entry short; details belong in commit messages and CLAUDE.md.

## [2025-09-17] Drop the health_score feature
- Choice: remove the composite `health_score` (sum of `<sensor>_degraded` flags); the code is kept commented out in `engineer_health_indicators.py`.
- Alternatives considered: keep it; the weighted-sum variant the function supported.
- Reasoning: the degradation flags (below 75% of baseline) are 0 on every row of the real data, so the score was a constant zero and carried no signal.
- Commit: b547c3d (models retrained without it in 11f546a)

## [2026-09-23] Label test-set RUL from the RUL_FD001 ground truth
- Choice: offset each test unit's RUL by its true remaining life from `RUL_FD001.txt`, and score predictions at each unit's last observed cycle against it (logged as `test_MAE/RMSE/R2`).
- Alternatives considered: the previous labels, cycles remaining in the observed series (last cycle = 0).
- Reasoning: test engines are truncated before failure, so the observed series doesn't end at failure; the old labels were wrong for every test row. Last-cycle scoring is the standard CMAPSS test evaluation.
- Commit: 0f21b91

## [2026-09-23] Report each unit's latest-cycle prediction
- Choice: the API and every dashboard tab show each unit's latest-cycle prediction, and true RUL comes from the test labels.
- Alternatives considered: the earlier behaviour (first cycle in the API, first/min over history in the dashboard, true RUL read from the train file).
- Reasoning: the latest cycle is the engine's current state; the train file held a different engine that happened to share the unit number.
- Commit: 0f21b91

## [2026-09-23] Validate on an engine-grouped (unit-level) split
- Choice: `GroupShuffleSplit` on `unit` for the 80/20 train/validation split.
- Alternatives considered: the previous random 80/20 row split (`train_test_split`).
- Reasoning: a row split put cycles of the same engine on both sides, and the per-engine baseline features act as an engine fingerprint, which inflated validation scores (XGBoost RMSE 11.6, R2 0.97).
- Commit: 11f546a

## [2026-09-23] New MLflow experiment names when run selection changes
- Choice: log to a new experiment suffix (`(unit split)`, then `(unit split, seeded)`) and have `predict_model.py` select only from the current set.
- Alternatives considered: keep logging to the existing experiments.
- Reasoning: best-run selection picks the lowest RMSE, so leaky or unreproducible older runs would keep winning if mixed in.
- Commit: 11f546a, 83fc440

## [2026-09-23] Seed the hyperparameter search
- Choice: one `SEED = 42` per training script drives the grouped split, a local `random.Random(SEED)` for the search and the model `random_state`; `tests/test_reproducibility.py` fails on any unseeded randomness in `src/`.
- Alternatives considered: the previous global, unseeded `random.choice` search.
- Reasoning: the search was the only unseeded randomness, so each retrain could select a different best model. Two full passes now give identical params and metrics for all 30 runs.
- Commit: 83fc440

## [2026-09-23] MQTT (Mosquitto in Docker) as the streaming broker
- Choice: a Mosquitto broker via docker compose, with QoS 1, retained messages and persistent sessions; paho-mqtt as the client.
- Alternatives considered: Kafka; Redis; a plain in-process Python queue; aiomqtt as the client.
- Reasoning: MQTT is the protocol real device telemetry uses, and Kafka or Redis are overkill for a personal demo. (Inferred, not recorded: a plain queue would keep producer and consumer in one process and survive nothing, so it wouldn't simulate networked telemetry.) aiomqtt needs `add_reader`, which the Windows Proactor loop under uvicorn lacks.
- Commit: 6de7ec9, 627b82d

## [2026-09-23] Per-engine MQTT topics
- Choice: `pm/fd001/engine/{unit}/{telemetry,status,prediction}`, with `unit` also in the payload.
- Alternatives considered: one fleet-wide topic with the engine ID only in the payload.
- Reasoning: subscribers can filter with wildcards, and retained status/prediction messages are kept per topic (one topic would retain one message for the whole fleet). MQTT's per-topic ordering is the per-engine ordering the rolling features need.
- Commit: 6de7ec9

## [2026-09-23] Replay the test set with staggered starts; sessions wipe by default
- Choice: replay `test_FD001.txt` only, engines starting at seeded random offsets (`--stagger`), at a configurable `--interval` (default 1.0s). Each run is a session keyed into every table; a new session wipes the others unless `--keep-history` is passed.
- Alternatives considered: also replaying the run-to-failure train set; lockstep starts; namespacing sessions and never wiping.
- Reasoning: the test set lines up with the batch predictions, so streaming can be checked against them, and staggered starts look like a fleet already in service. Keeping history needs the session in the key, because a second replay reuses the same (unit, cycle) pairs.
- Commit: 6de7ec9, a1228cd

## [2026-09-23] SQLite holds full per-engine history; features recomputed with the batch function
- Choice: store every raw reading in SQLite and, per micro-batch, rerun `engineer_health_indicators()` over each touched engine's full history, keeping only unscored rows.
- Alternatives considered: bounded state (frozen baseline plus last 5 readings); running statistics reimplemented incrementally; in-memory state.
- Reasoning: SQLite lets state survive a restart. Full-history recompute is bitwise identical to batch, while a 5-row buffer drifts by up to 1e-9 because pandas' rolling sums accumulate from the start of the series. It costs about 23 ms per 303-cycle engine, fine at FD001 scale.
- Commit: a1228cd

## [2026-09-23] Hold cycles 1-4 until cycle 5 (warm-up)
- Choice: store readings for cycles 1-4 without predicting, then emit cycles 1-5 together when cycle 5 arrives.
- Alternatives considered: predict right away with a provisional expanding-mean baseline and re-emit corrected predictions at cycle 5.
- Reasoning: the batch baseline averages each engine's first 5 cycles, so earlier rows depend on future readings; holding them makes every emitted value exact, at the cost of a 4-cycle start-up delay.
- Commit: a1228cd

## [2026-09-23] Ack MQTT messages only after the SQLite commit
- Choice: persistent MQTT session (fixed client ID, clean session off, QoS 1) with manual acks sent after each batch's transaction commits; readings inserted idempotently. Mosquitto's in-flight limit raised to 1000.
- Alternatives considered: paho's automatic ack on receipt.
- Reasoning: auto-ack loses messages that were acked but not yet committed when the consumer crashes; ack-after-commit plus idempotent inserts means restarts lose and duplicate nothing. The default in-flight limit of 20 would have capped every batch at 20 readings.
- Commit: 627b82d

## [2026-09-23] Load the model once at consumer startup
- Choice: the consumer loads the best run once via the shared `predict_model.load_best_model()` and tags every prediction with `model_run_id`.
- Alternatives considered: watch MLflow and hot-swap when a better run appears.
- Reasoning: a session's predictions all come from one model and stay comparable with the batch predictions; hot-swapping would mix models mid-stream. Sharing the loader avoids a second copy of the flavour/artifact mapping.
- Commit: 627b82d

## [2026-09-23] WebSocket push via an iframe in Streamlit
- Choice: a gateway bridges retained MQTT predictions to `/ws/live`; the Streamlit dashboard gets a Live tab that iframes the gateway's plain HTML/JS `/live` page. The other tabs stay on the batch CSVs.
- Alternatives considered: Streamlit polling (`st.fragment(run_every=...)`); calling `st.*` from a background thread; a Streamlit custom component (needs a Node build); moving the live tabs to SQLite polling too.
- Reasoning: Streamlit only updates through script reruns, so every Streamlit-side option is polling or an unsupported hack; the iframe gives real push without changing Streamlit's execution model.
- Commit: 13c6120

## [2026-09-23] Mount the streaming gateway on the existing API
- Choice: include the gateway router and its lifespan in `app_api.py`.
- Alternatives considered: a separate gateway process, leaving `app_api.py` untouched.
- Reasoning: one server and one port for the dashboard to reach; the batch `/predictions` routes are unchanged and the API still starts when the broker is down.
- Commit: 13c6120

## [2026-09-23] Test streaming parity against in-memory batch output
- Choice: assert streamed features are bitwise equal to `engineer_health_indicators()` run in memory, compare with the features CSV only within 1e-9, and require streamed predictions to equal `rul_predictions.csv` exactly.
- Alternatives considered: comparing against the features CSV exactly.
- Reasoning: the CSV differs from in-memory output by about 1e-12 because of float formatting, so an exact CSV comparison would fail for reasons unrelated to streaming.
- Commit: 52118aa

## [2026-09-23] Recover the lost MLflow database by retraining, and back it up from now on
- Choice: `mlflow.db` was deleted during troubleshooting by a `del` run against what looked like an empty database but wasn't. No copy existed anywhere on the machine, so a fresh `mlflow.db` was created and the three seeded searches and `predict_model.py` were rerun. From now on the tracking database gets backed up even though it's git-ignored: `python src/backup_mlflow.py` (or `make backup-mlflow`) is a one-command manual snapshot into `backups/`, with no scheduling or retention.
- Alternatives considered: restoring from a backup (none found on C:, including the Recycle Bin and renamed SQLite files); rebuilding `meta.yaml` files for the orphaned `mlruns/1`-`4` (those folders only ever held artifacts, so it would restore no params or metrics); re-scoring the orphaned models to rebuild metrics.
- Reasoning: the seeded pipeline made the retrain an exact recovery. Validation RMSE came out RF 28.23, XGBoost 29.08, LightGBM 30.85, RF was selected with test RMSE 24.96, MAE 18.34, R2 0.64, and `rul_predictions.csv` is byte-identical to the pre-loss inference output. History from before the seeded experiments (params and metrics of the row-split and unseeded runs) is lost; only their artifacts remain. Being git-ignored keeps the database out of the repo but doesn't protect it, so it needs its own backup.
- Commit: none (documentation only; the retrain changed no tracked files)
