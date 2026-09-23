# tests/test_stream_state_store.py
"""
Tests for the streaming SQLite state: sessions, idempotency, atomicity and
resuming after a restart.
"""

import pandas as pd
import pytest

from streaming.state_store import StateStore
from streaming.stream_features import StreamFeatureEngine

from tests.conftest import make_readings


def _ingest(store, session_id, records):
    with store.transaction():
        return StreamFeatureEngine(store).ingest(session_id, records, "now")


def _count(store, table, session_id):
    return store.conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE session_id = ?",
        (session_id,)).fetchone()[0]


def test_new_session_wipes_other_sessions_by_default(store):
    store.begin_session({"session_id": "A"})
    _ingest(store, "A", make_readings(1, range(1, 6)))
    assert store.begin_session({"session_id": "B"}) is True

    assert store.current_session() == "B"
    for table in ["readings", "engine_state", "sessions"]:
        assert _count(store, table, "A") == 0


def test_keep_history_preserves_earlier_sessions(store):
    store.begin_session({"session_id": "A"})
    _ingest(store, "A", make_readings(1, range(1, 6)))
    store.begin_session({"session_id": "B", "keep_history": True})
    _ingest(store, "B", make_readings(1, range(1, 6)))

    assert _count(store, "readings", "A") == 5
    assert _count(store, "readings", "B") == 5
    assert store.session_info("B")["keep_history"] is True


def test_same_session_twice_is_a_no_op(store):
    store.begin_session({"session_id": "A"})
    _ingest(store, "A", make_readings(1, range(1, 3)))
    assert store.begin_session({"session_id": "A"}) is False
    assert _count(store, "readings", "A") == 2


def test_insert_reading_is_idempotent(store):
    store.begin_session({"session_id": "A"})
    reading = make_readings(1, [1])[0]
    assert store.insert_reading("A", reading, "now") is True
    assert store.insert_reading("A", reading, "now") is False
    assert _count(store, "readings", "A") == 1


def test_failed_transaction_rolls_back_everything(store):
    store.begin_session({"session_id": "A"})
    with pytest.raises(RuntimeError):
        with store.transaction():
            StreamFeatureEngine(store).ingest(
                "A", make_readings(1, range(1, 6)), "now")
            raise RuntimeError("scoring failed")
    assert _count(store, "readings", "A") == 0
    assert store.engine_states("A") == {}


def test_restart_resumes_with_identical_output(tmp_path):
    """Stopping mid-stream and reopening the database gives the same rows
    as one uninterrupted run (state lives only in SQLite)."""
    readings = make_readings(1, range(1, 21)) + make_readings(2, range(1, 21))
    readings.sort(key=lambda r: r["time_in_cycles"])

    def run(store, recs):
        return [_ingest(store, "A", [r]) for r in recs]

    uninterrupted = StateStore.open(tmp_path / "one.db")
    uninterrupted.begin_session({"session_id": "A"})
    expected = pd.concat(run(uninterrupted, readings))
    uninterrupted.close()

    path = tmp_path / "two.db"
    first = StateStore.open(path)
    first.begin_session({"session_id": "A"})
    part1 = run(first, readings[:17])
    first.close()
    second = StateStore.open(path)
    part2 = run(second, readings[15:])  # overlap simulates redelivery
    second.close()

    resumed = pd.concat(part1 + part2)
    pd.testing.assert_frame_equal(resumed.reset_index(drop=True),
                                  expected.reset_index(drop=True))


def test_readonly_reader_sees_committed_state(tmp_path):
    path = tmp_path / "state.db"
    writer = StateStore.open(path)
    writer.begin_session({"session_id": "A"})
    _ingest(writer, "A", make_readings(1, range(1, 6)))
    writer.insert_predictions(
        "A", [{"unit": 1, "time_in_cycles": c, "rul": 100.0 - c,
               "risk_level": "Medium"} for c in range(1, 6)], "run", "now")

    reader = StateStore.open_readonly(path)
    assert reader.current_session() == "A"
    assert len(reader.prediction_history("A", 1)) == 5
    with pytest.raises(Exception):
        reader.set_meta("x", "y")
    reader.close()
    writer.close()
