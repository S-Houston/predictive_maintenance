# tests/test_stream_features.py
"""
Parity and behaviour tests for the incremental (streaming) feature engine.

The key guarantee: every feature row the stream emits is bitwise identical
to the row the batch pipeline computes from the full file, however the
readings are split into micro-batches.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from streaming.producer import build_start_ticks, iter_ticks, \
    telemetry_payload
from streaming.stream_features import StreamFeatureEngine, compute_features

from tests.conftest import make_readings

SESSION = "20260101T000000Z"
BATCH_FEATURES_CSV = Path("data/features/test_FD001_features.csv")


def feature_cols(df):
    return [c for c in df.columns if "sensor" in c and c != "failure_binary"]


def replay(store, df, units, ticks_per_batch=1, stagger=50):
    """Streams df through a StreamFeatureEngine as the producer would,
    including the JSON round-trip, and returns every emitted row."""
    store.begin_session({"session_id": SESSION})
    engine = StreamFeatureEngine(store)
    df = df[df["unit"].isin(units)]
    emitted, pending = [], []
    ticks = iter_ticks(df, build_start_ticks(units, stagger, seed=42))
    for tick, readings, _ in ticks:
        pending += [json.loads(json.dumps(telemetry_payload(r, SESSION)))
                    for r in readings]
        if (tick + 1) % ticks_per_batch == 0:
            emitted.append(_ingest(store, engine, pending))
            pending = []
    emitted.append(_ingest(store, engine, pending))
    out = pd.concat([e for e in emitted if not e.empty])
    return out.sort_values(["unit", "time_in_cycles"]).reset_index(drop=True)


def _ingest(store, engine, records):
    with store.transaction():
        return engine.ingest(SESSION, records, "now")


def batch_features(df, units):
    batch = compute_features(df[df["unit"].isin(units)].reset_index(drop=True))
    return batch.sort_values(["unit", "time_in_cycles"]).reset_index(drop=True)


# --- parity with the batch pipeline (real FD001 test set) -----------------

@pytest.mark.parametrize("units, ticks_per_batch", [
    (list(range(1, 11)), 1),     # every tick is its own micro-batch
    (list(range(1, 101)), 7),    # whole fleet, batches spanning warm-up
])
def test_stream_features_bitwise_equal_batch(store, raw_test_df, units,
                                             ticks_per_batch):
    stream = replay(store, raw_test_df, units, ticks_per_batch)
    batch = batch_features(raw_test_df, units)
    cols = feature_cols(batch)

    assert len(stream) == len(batch)
    assert (stream[["unit", "time_in_cycles"]].values
            == batch[["unit", "time_in_cycles"]].values).all()
    assert np.array_equal(stream[cols].to_numpy(float),
                          batch[cols].to_numpy(float), equal_nan=True)


def test_stream_features_match_features_csv(store, raw_test_df):
    """The on-disk CSV differs from in-memory batch only by float
    formatting (~1e-12), so compare it with a tolerance."""
    if not BATCH_FEATURES_CSV.exists():
        pytest.skip(f"{BATCH_FEATURES_CSV} not found")
    units = list(range(1, 101))
    stream = replay(store, raw_test_df, units, ticks_per_batch=25)
    csv = pd.read_csv(BATCH_FEATURES_CSV).sort_values(
        ["unit", "time_in_cycles"]).reset_index(drop=True)
    cols = feature_cols(csv)
    np.testing.assert_allclose(stream[cols].to_numpy(float),
                               csv[cols].to_numpy(float),
                               rtol=0, atol=1e-9, equal_nan=True)


# --- warm-up, NaNs and delivery edge cases (synthetic data) ---------------

def test_warmup_holds_rows_until_cycle_5_then_emits_all(store):
    store.begin_session({"session_id": SESSION})
    engine = StreamFeatureEngine(store)
    readings = make_readings(1, range(1, 8))

    for r in readings[:4]:
        assert _ingest(store, engine, [r]).empty
    assert store.engine_states(SESSION)[1]["status"] == "warming_up"

    warm = _ingest(store, engine, [readings[4]])
    assert list(warm["time_in_cycles"]) == [1, 2, 3, 4, 5]
    assert store.engine_states(SESSION)[1]["status"] == "active"

    for cycle, r in zip([6, 7], readings[5:]):
        assert list(_ingest(store, engine, [r])["time_in_cycles"]) == [cycle]


def test_first_cycle_keeps_nan_like_batch(store):
    store.begin_session({"session_id": SESSION})
    engine = StreamFeatureEngine(store)
    out = _ingest(store, engine, make_readings(1, range(1, 6)))
    first = out[out["time_in_cycles"] == 1].iloc[0]
    assert np.isnan(first["sensor_2_rolling_std"])
    assert np.isnan(first["sensor_2_cycle_change"])
    assert not np.isnan(out.iloc[1]["sensor_2_rolling_std"])


def test_duplicate_and_stale_cycles_are_ignored(store):
    store.begin_session({"session_id": SESSION})
    engine = StreamFeatureEngine(store)
    readings = make_readings(1, range(1, 7))
    _ingest(store, engine, readings[:5])

    # A redelivered batch and a stale cycle produce nothing new
    assert _ingest(store, engine, readings[:5]).empty
    assert _ingest(store, engine, [readings[2]]).empty
    out = _ingest(store, engine, [readings[5], readings[5]])
    assert list(out["time_in_cycles"]) == [6]
    assert store.engine_states(SESSION)[1]["n_readings"] == 6


def test_gap_is_processed_like_batch_and_logged(store, caplog):
    store.begin_session({"session_id": SESSION})
    engine = StreamFeatureEngine(store)
    readings = make_readings(1, [1, 2, 3, 4, 5, 7])
    _ingest(store, engine, readings[:5])
    with caplog.at_level(logging.WARNING, logger="stream_features"):
        out = _ingest(store, engine, [readings[5]])
    assert "jumped from cycle 5 to 7" in caplog.text

    expected = compute_features(pd.DataFrame(readings)).iloc[-1]
    cols = feature_cols(out)
    assert np.array_equal(out[cols].iloc[0].to_numpy(float),
                          expected[cols].to_numpy(float), equal_nan=True)


def test_engines_are_independent_within_a_batch(store):
    store.begin_session({"session_id": SESSION})
    engine = StreamFeatureEngine(store)
    batch = make_readings(1, range(1, 6)) + make_readings(2, range(1, 4))
    out = _ingest(store, engine, batch)
    assert set(out["unit"]) == {1}
    states = store.engine_states(SESSION)
    assert states[1]["status"] == "active"
    assert states[2]["status"] == "warming_up"
