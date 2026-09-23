"""
Streaming Consumer
==================

Subscribes to engine telemetry on MQTT, engineers features incrementally
from the per-engine history in SQLite, scores each new row with the best
MLflow model (the same one predict_model.py uses) and publishes the
prediction, retained, to pm/fd001/engine/{unit}/prediction.

Delivery: the MQTT session is persistent (fixed client id, clean_session
off, QoS 1) and messages are acked only after the micro-batch they belong
to is committed to SQLite, so a crash or restart neither loses nor double
counts readings; redeliveries are dropped by the store.

When every engine of the session has finished, last-cycle predictions are
scored against RUL_FD001 and logged to the "FD001 RUL Streaming Inference"
MLflow experiment; test_RMSE should equal the batch inference run's.

Usage (from the repo root; broker and MLflow server running):
    PYTHONPATH=src python src/streaming/consumer.py
"""

import argparse
import json
import logging
import queue
import time
from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

from models.predict_model import (classify_risk, evaluate_against_truth,
                                  feature_columns, load_best_model)
from streaming import config
from streaming.state_store import READING_COLUMNS, StateStore
from streaming.stream_features import StreamFeatureEngine, compute_features

# engineer_health_indicators logs every step at INFO on the root logger;
# keep the root quiet and log the consumer's own progress instead.
logging.getLogger().setLevel(logging.WARNING)
log = logging.getLogger("consumer")
log.setLevel(logging.INFO)

CLIENT_ID = "pm-stream-consumer"
LABELED_TEST_PATH = "data/cleaned/test_FD001_labeled.csv"
STREAM_EXPERIMENT = "FD001 RUL Streaming Inference"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def check_feature_columns(feature_cols):
    """Fails fast if the stream cannot produce every model input column."""
    sample = pd.DataFrame(1.0, index=range(config.WARMUP_CYCLES),
                          columns=READING_COLUMNS)
    sample["unit"] = 1
    sample["time_in_cycles"] = range(1, config.WARMUP_CYCLES + 1)
    missing = set(feature_cols) - set(compute_features(sample).columns)
    if missing:
        raise RuntimeError(f"Stream features lack model inputs: {missing}")


def score(model, feats, feature_cols):
    """Predicts RUL for feature rows, rounded and classified as in batch."""
    ruls = np.round(model.predict(feats[feature_cols]), 2)
    return [
        {"unit": int(u), "time_in_cycles": int(c), "rul": float(r),
         "risk_level": classify_risk(r)}
        for u, c, r in zip(feats["unit"], feats["time_in_cycles"], ruls)
    ]


class StreamProcessor:
    """
    Broker-independent core: applies a batch of (topic, payload) messages
    to the state store and returns the predictions to publish.
    """

    def __init__(self, store, model, model_run_id, feature_cols):
        self.store = store
        self.engine = StreamFeatureEngine(store)
        self.model = model
        self.model_run_id = model_run_id
        self.feature_cols = feature_cols

    def handle(self, messages):
        """Messages are applied in arrival order; telemetry is batched."""
        published, chunk = [], []
        for topic, payload in messages:
            kind = topic.rsplit("/", 1)[-1]
            if kind == "telemetry":
                chunk.append(payload)
                continue
            published += self._flush(chunk)
            chunk = []
            if kind == "session":
                self._on_session(payload)
            elif kind == "status":
                self._on_status(payload)
        return published + self._flush(chunk)

    def _on_session(self, session):
        current = self.store.current_session()
        if current is not None and session["session_id"] < current:
            return  # stale control message from an earlier replay
        with self.store.transaction():
            started = self.store.begin_session(session)
        if started:
            log.info("Session %s started (keep_history=%s)",
                     session["session_id"], session.get("keep_history"))

    def _on_status(self, status):
        if status.get("session_id") != self.store.current_session():
            return
        with self.store.transaction():
            self.store.mark_complete(status["session_id"], status["unit"],
                                     utc_now())

    def _accept_session(self, session_id):
        """True if telemetry of `session_id` should be processed."""
        current = self.store.current_session()
        if session_id == current:
            return True
        if current is not None and session_id < current:
            log.warning("Dropping telemetry from stale session %s",
                        session_id)
            return False
        # Session ids are UTC timestamps, so a larger id is a newer replay
        # whose control message has not been seen; start it with defaults.
        log.warning("Telemetry for unannounced session %s; starting it",
                    session_id)
        with self.store.transaction():
            self.store.begin_session({"session_id": session_id})
        return True

    def _flush(self, records):
        published = []
        by_session = {}
        for rec in records:
            by_session.setdefault(rec["session_id"], []).append(rec)
        for session_id, recs in sorted(by_session.items()):
            if self._accept_session(session_id):
                published += self._process(session_id, recs)
        return published

    def _process(self, session_id, records):
        now = utc_now()
        with self.store.transaction():
            ready = self.engine.ingest(session_id, records, now)
            if ready.empty:
                return []
            preds = score(self.model, ready, self.feature_cols)
            self.store.insert_predictions(session_id, preds,
                                          self.model_run_id, now)
            self.store.insert_features(session_id, ready, self.feature_cols)
        return [{"session_id": session_id, **p,
                 "model_run_id": self.model_run_id, "ts": now}
                for p in preds]

    def evaluate_if_finished(self):
        """
        Once every engine of the current session is complete and scored,
        returns test metrics (last cycle vs RUL_FD001), exactly once per
        session. Otherwise returns None.
        """
        session_id = self.store.current_session()
        key = f"evaluated:{session_id}"
        units = self.store.session_info(session_id).get("units")
        if not units or self.store.get_meta(key):
            return None
        states = self.store.engine_states(session_id)
        for unit in units:
            s = states.get(unit)
            if (s is None or s["status"] != "complete"
                    or s["last_predicted_cycle"] != s["last_cycle"]):
                return None
        preds = self.store.prediction_history(session_id)
        metrics = evaluate_against_truth(
            preds.rename(columns={"rul": "RUL"}),
            pd.read_csv(LABELED_TEST_PATH))
        self.store.set_meta(key, json.dumps(metrics))
        return metrics


def log_streaming_run(session_id, model_run_id, metrics, n_predictions):
    # the tracking URI is set on import of models.predict_model
    try:
        mlflow.set_experiment(STREAM_EXPERIMENT)
        with mlflow.start_run(run_name=f"Stream {session_id}"):
            mlflow.log_param("session_id", session_id)
            mlflow.log_param("model_run_id", model_run_id)
            mlflow.log_param("num_predictions", n_predictions)
            mlflow.log_metrics(metrics)
    except Exception as e:
        log.warning("Could not log streaming run to MLflow: %s", e)


class MqttConsumer:
    """Feeds MQTT messages to a StreamProcessor in micro-batches."""

    def __init__(self, processor, client_id=CLIENT_ID, linger=0.05,
                 max_batch=1000):
        self.processor = processor
        self.inbox = queue.Queue()
        self.linger = linger
        self.max_batch = max_batch
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id=client_id, clean_session=False,
                                  manual_ack=True)
        self.client.on_connect = self._on_connect
        self.client.on_message = lambda c, u, msg: self.inbox.put(msg)

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.error("MQTT connect failed: %s", reason_code)
            return
        client.subscribe([(config.SESSION_TOPIC, 1),
                          (config.TELEMETRY_SUB, 1),
                          (config.STATUS_SUB, 1)])
        log.info("Connected to MQTT broker; subscribed to telemetry")

    def connect(self):
        try:
            self.client.connect(config.BROKER_HOST, config.BROKER_PORT,
                                keepalive=30)
        except OSError as e:
            raise SystemExit(
                f"Cannot reach MQTT broker at {config.BROKER_HOST}:"
                f"{config.BROKER_PORT} ({e}). Try: docker compose up -d")
        self.client.loop_start()

    def next_batch(self, timeout=1.0):
        """Waits for a message, then gathers whatever arrives within linger."""
        try:
            batch = [self.inbox.get(timeout=timeout)]
        except queue.Empty:
            return []
        deadline = time.monotonic() + self.linger
        while len(batch) < self.max_batch:
            try:
                wait = max(0.0, deadline - time.monotonic())
                batch.append(self.inbox.get(timeout=wait))
            except queue.Empty:
                break
        return batch

    def _decode(self, msgs):
        decoded = []
        for msg in msgs:
            try:
                decoded.append((msg.topic, json.loads(msg.payload)))
            except ValueError:
                log.warning("Skipping malformed message on %s", msg.topic)
        return decoded

    def process_batch(self, msgs):
        """
        Applies a batch, then acks and publishes. If handling raises, the
        transaction is rolled back and nothing is acked, so the broker
        redelivers the batch.
        """
        preds = self.processor.handle(self._decode(msgs))
        for msg in msgs:  # committed: safe to acknowledge
            self.client.ack(msg.mid, msg.qos)
        for p in preds:
            self.client.publish(
                config.PREDICTION_TOPIC.format(unit=p["unit"]),
                json.dumps(p), qos=1, retain=True)
        return preds

    def run_forever(self):
        n_scored = 0
        while True:
            msgs = self.next_batch()
            if not msgs:
                continue
            preds = self.process_batch(msgs)
            n_scored += len(preds)
            if preds:
                log.info("Batch of %d messages -> %d predictions "
                         "(%d this run)", len(msgs), len(preds), n_scored)
            self._evaluate()

    def _evaluate(self):
        proc = self.processor
        metrics = proc.evaluate_if_finished()
        if metrics is None:
            return
        session_id = proc.store.current_session()
        n_preds = len(proc.store.prediction_history(session_id))
        log.info("Session %s finished. Test-set evaluation: %s",
                 session_id, metrics)
        log_streaming_run(session_id, proc.model_run_id, metrics, n_preds)

    def stop(self):
        self.client.disconnect()
        self.client.loop_stop()


def main():
    p = argparse.ArgumentParser(description="Streaming RUL consumer")
    p.add_argument("--db", default=str(config.STATE_DB_PATH),
                   help="SQLite state database")
    args = p.parse_args()

    model, best_run = load_best_model()
    feature_cols = feature_columns()
    check_feature_columns(feature_cols)

    store = StateStore.open(args.db)
    store.set_meta("model_run_id", best_run["run_id"])
    processor = StreamProcessor(store, model, best_run["run_id"],
                                feature_cols)
    consumer = MqttConsumer(processor)
    consumer.connect()
    log.info("Consuming with model run %s (%d features); state in %s",
             best_run["run_id"], len(feature_cols), args.db)
    try:
        consumer.run_forever()
    except KeyboardInterrupt:
        log.info("Stopping; unacknowledged messages will be redelivered")
    finally:
        consumer.stop()
        store.close()


if __name__ == "__main__":
    main()
