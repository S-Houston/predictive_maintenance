# tests/test_stream_consumer.py
"""
Tests for the streaming consumer: scoring, session handling, ack-after-commit
and end-of-stream evaluation, plus a full-fleet parity check against the
batch predictions using the real MLflow model (skipped if the MLflow server
or the data is unavailable).
"""

import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from models.predict_model import classify_risk, evaluate_against_truth
from streaming import consumer as consumer_mod
from streaming.consumer import MqttConsumer, StreamProcessor
from streaming.producer import build_start_ticks, iter_ticks, \
    telemetry_payload
from streaming.stream_features import compute_features

from tests.conftest import make_readings

SESSION = "20260101T000000Z"
BATCH_PREDICTIONS = Path("data/processed/rul_predictions.csv")
LABELED_TEST = Path("data/cleaned/test_FD001_labeled.csv")


class FakeModel:
    """Predicts from one feature and checks the input column order."""

    def __init__(self, feature_cols):
        self.feature_cols = feature_cols

    def predict(self, X):
        assert list(X.columns) == self.feature_cols
        return X["sensor_2_rolling_mean"].to_numpy() / 5.0 - 1 / 3


@pytest.fixture
def feature_cols():
    sample = compute_features(pd.DataFrame(make_readings(1, range(1, 6))))
    return [c for c in sample.columns if "sensor" in c]


@pytest.fixture
def processor(store, feature_cols):
    return StreamProcessor(store, FakeModel(feature_cols), "run-1",
                           feature_cols)


def session_msg(session_id=SESSION, units=(1,), keep_history=False):
    return ("pm/fd001/control/session",
            {"session_id": session_id, "units": list(units),
             "keep_history": keep_history})


def telemetry(readings, session_id=SESSION):
    return [(f"pm/fd001/engine/{r['unit']}/telemetry",
             {**r, "session_id": session_id}) for r in readings]


def status(unit, last_cycle, session_id=SESSION):
    return (f"pm/fd001/engine/{unit}/status",
            {"session_id": session_id, "unit": unit, "state": "complete",
             "last_cycle": last_cycle})


def test_processor_scores_rounds_classifies_and_stores(processor, store):
    readings = make_readings(1, range(1, 7))
    preds = processor.handle([session_msg()] + telemetry(readings))

    assert [p["time_in_cycles"] for p in preds] == [1, 2, 3, 4, 5, 6]
    for p in preds:
        assert p["rul"] == round(p["rul"], 2)
        assert p["risk_level"] == classify_risk(p["rul"])
        assert p["session_id"] == SESSION and p["model_run_id"] == "run-1"
    stored = store.prediction_history(SESSION, 1)
    assert list(stored["rul"]) == [p["rul"] for p in preds]


def test_redelivered_messages_are_not_rescored(processor, store):
    readings = make_readings(1, range(1, 7))
    processor.handle([session_msg()] + telemetry(readings))
    assert processor.handle(telemetry(readings)) == []
    assert len(store.prediction_history(SESSION, 1)) == 6


def test_stale_session_telemetry_is_dropped(processor, store):
    processor.handle([session_msg("20260102T000000Z")])
    old = processor.handle(
        telemetry(make_readings(1, range(1, 6)), "20260101T000000Z"))
    assert old == []
    assert store.current_session() == "20260102T000000Z"


def test_unannounced_newer_session_is_started(processor, store):
    processor.handle([session_msg()] +
                     telemetry(make_readings(1, range(1, 6))))
    newer = "20260103T000000Z"
    preds = processor.handle(telemetry(make_readings(1, range(1, 6)), newer))
    assert store.current_session() == newer
    assert len(preds) == 5
    assert store.prediction_history(SESSION).empty  # old session wiped


def test_evaluation_runs_once_when_all_engines_complete(
        processor, tmp_path, monkeypatch):
    labels = pd.DataFrame({"unit": [1] * 6, "time_in_cycles": range(1, 7),
                           "RUL": range(55, 49, -1)})
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)
    monkeypatch.setattr(consumer_mod, "LABELED_TEST_PATH", labels_path)

    readings = make_readings(1, range(1, 7))
    processor.handle([session_msg()] + telemetry(readings[:5]))
    assert processor.evaluate_if_finished() is None
    processor.handle(telemetry(readings[5:]) + [status(1, 6)])

    metrics = processor.evaluate_if_finished()
    last = processor.store.prediction_history(SESSION, 1).iloc[-1]
    assert metrics["test_units"] == 1
    assert metrics["test_MAE"] == pytest.approx(abs(last["rul"] - 50))
    assert processor.evaluate_if_finished() is None  # only once


class StubClient:
    def __init__(self, rc=0):
        self.acks, self.published, self.payloads = [], [], []
        self.rc = rc

    def ack(self, mid, qos):
        self.acks.append(mid)

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, retain))
        self.payloads.append(payload)
        return SimpleNamespace(rc=self.rc)


def _mqtt_messages(pairs):
    return [SimpleNamespace(topic=t, payload=json.dumps(p).encode(), mid=i,
                            qos=1) for i, (t, p) in enumerate(pairs)]


def test_batch_is_acked_and_published_after_commit(processor):
    mc = MqttConsumer(processor)
    mc.client = StubClient()
    msgs = _mqtt_messages([session_msg()] +
                          telemetry(make_readings(1, range(1, 6))))
    preds = mc.process_batch(msgs)

    assert mc.client.acks == [m.mid for m in msgs]
    assert mc.client.published == [("pm/fd001/engine/1/prediction", True)] * 5
    assert len(preds) == 5


def test_failed_batch_is_not_acked(processor, store, monkeypatch):
    mc = MqttConsumer(processor)
    mc.client = StubClient()

    def broken_predict(X):
        raise RuntimeError("model failed")
    monkeypatch.setattr(processor.model, "predict", broken_predict)

    msgs = _mqtt_messages([session_msg()] +
                          telemetry(make_readings(1, range(1, 6))))
    with pytest.raises(RuntimeError):
        mc.process_batch(msgs)
    assert mc.client.acks == [] and mc.client.published == []
    assert store.engine_states(SESSION) == {}  # rolled back; redelivered


def test_failed_publishes_are_logged(processor, caplog):
    mc = MqttConsumer(processor)
    mc.client = StubClient(rc=4)  # MQTT_ERR_NO_CONN
    msgs = _mqtt_messages([session_msg()] +
                          telemetry(make_readings(1, range(1, 6))))
    mc.process_batch(msgs)
    assert "5 of 5 prediction publishes failed" in caplog.text


# --- readiness status and lag ----------------------------------------------

def test_status_payload_reports_state_pid_and_model():
    payload = json.loads(consumer_mod.status_payload("online", "run-1"))
    assert payload["state"] == "online"
    assert payload["pid"] == os.getpid()
    assert payload["model_run_id"] == "run-1"
    assert datetime.fromisoformat(payload["ts"]).tzinfo is not None


def test_last_will_reports_offline(processor):
    mc = MqttConsumer(processor)
    will = mc.client._will_topic.decode(), json.loads(mc.client._will_payload)
    assert will[0] == "pm/fd001/consumer/status"
    assert will[1]["state"] == "offline" and mc.client._will_retain


def test_online_is_published_only_after_subscriptions_succeed(processor):
    mc = MqttConsumer(processor)
    stub = StubClient()
    ok, refused = SimpleNamespace(is_failure=False), \
        SimpleNamespace(is_failure=True)

    mc._on_subscribe(stub, None, 1, [refused, ok], None)
    assert stub.published == []
    mc._on_subscribe(stub, None, 1, [ok, ok, ok], None)
    assert stub.published == [("pm/fd001/consumer/status", True)]
    assert json.loads(stub.payloads[0])["state"] == "online"


def test_max_lag_uses_telemetry_timestamps_only():
    now = datetime(2026, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    msgs = [("pm/fd001/engine/1/telemetry",
             {"ts": "2026-01-01T00:00:07.000+00:00"}),
            ("pm/fd001/engine/2/telemetry",
             {"ts": "2026-01-01T00:00:01.500+00:00"}),
            ("pm/fd001/engine/2/status", {"ts": "2025-01-01T00:00:00+00:00"})]
    assert consumer_mod.max_lag_seconds(msgs, now) == 8.5
    assert consumer_mod.max_lag_seconds(msgs[2:], now) is None


def test_lag_warning_fires_above_threshold_and_is_rate_limited(
        processor, caplog):
    mc = MqttConsumer(processor)
    old = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
    fresh = datetime.now(timezone.utc).isoformat()

    mc._check_lag([("pm/fd001/engine/1/telemetry", {"ts": fresh})])
    assert "behind the producer" not in caplog.text
    mc._check_lag([("pm/fd001/engine/1/telemetry", {"ts": old})])
    mc._check_lag([("pm/fd001/engine/1/telemetry", {"ts": old})])
    assert caplog.text.count("behind the producer") == 1


# --- full-fleet parity with the batch predictions (real model) ------------

def _mlflow_up():
    try:
        urllib.request.urlopen("http://127.0.0.1:5000/health", timeout=2)
        return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def best_model():
    if not _mlflow_up():
        pytest.skip("MLflow server not running at http://127.0.0.1:5000")
    from models.predict_model import feature_columns, load_best_model
    model, best_run = load_best_model()
    return model, best_run["run_id"], feature_columns()


def test_stream_predictions_equal_batch_predictions(store, raw_test_df,
                                                    best_model):
    """
    Replays the whole test set through the consumer core and compares with
    data/processed/rul_predictions.csv. Only meaningful if that file was
    produced by the same model (rerun predict_model.py after retraining).
    """
    if not (BATCH_PREDICTIONS.exists() and LABELED_TEST.exists()):
        pytest.skip("batch predictions or test labels not found")
    model, run_id, feature_cols = best_model
    proc = StreamProcessor(store, model, run_id, feature_cols)
    units = sorted(int(u) for u in raw_test_df["unit"].unique())
    proc.handle([session_msg(units=units)])

    preds, pending = [], []
    ticks = iter_ticks(raw_test_df, build_start_ticks(units, 50, seed=42))
    for tick, readings, completed in ticks:
        pending += telemetry(
            [telemetry_payload(r, SESSION) for r in readings])
        pending += [status(u, c) for u, c in completed]
        if tick % 10 == 9:
            preds += proc.handle(pending)
            pending = []
    preds += proc.handle(pending)

    stream = pd.DataFrame(preds).rename(columns={"rul": "RUL"})
    batch = pd.read_csv(BATCH_PREDICTIONS)
    merged = batch.merge(stream, on=["unit", "time_in_cycles"],
                         suffixes=("_batch", "_stream"))
    assert len(stream) == len(batch) == len(merged)
    mismatched = merged[merged["RUL_batch"] != merged["RUL_stream"]]
    assert mismatched.empty, mismatched.head()
    assert (merged["risk_level_batch"] == merged["risk_level_stream"]).all()

    stream_metrics = proc.evaluate_if_finished()
    batch_metrics = evaluate_against_truth(batch, pd.read_csv(LABELED_TEST))
    assert stream_metrics == pytest.approx(batch_metrics)
    assert np.isfinite(stream_metrics["test_RMSE"])
