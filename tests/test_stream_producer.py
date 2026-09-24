# tests/test_stream_producer.py
"""
Tests for the telemetry producer's replay schedule (no broker needed).
"""

import json

import pandas as pd
import pytest

from streaming import producer
from streaming.producer import build_start_ticks, iter_ticks, parse_units, \
    telemetry_payload


def test_parse_units():
    available = [1, 2, 3, 4, 5]
    assert parse_units("all", available) == available
    assert parse_units("2-4", available) == [2, 3, 4]
    assert parse_units("5,1", available) == [1, 5]
    with pytest.raises(ValueError):
        parse_units("4-6", available)


def test_start_ticks_are_seeded_bounded_and_start_at_zero():
    units = list(range(1, 101))
    ticks = build_start_ticks(units, stagger=50, seed=42)
    assert ticks == build_start_ticks(units, stagger=50, seed=42)
    assert min(ticks.values()) == 0 and max(ticks.values()) <= 50
    assert len(set(ticks.values())) > 1
    assert set(build_start_ticks(units, 0, 42).values()) == {0}


def test_iter_ticks_publishes_every_cycle_once_in_order():
    df = pd.DataFrame({"unit": [1] * 3 + [2] * 2,
                       "time_in_cycles": [1, 2, 3, 1, 2],
                       "sensor_1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    seen, completed = [], []
    for tick, readings, done in iter_ticks(df, {1: 0, 2: 2}):
        seen += [(tick, r["unit"], r["time_in_cycles"]) for r in readings]
        completed += done
    assert seen == [(0, 1, 1), (1, 1, 2), (2, 1, 3), (2, 2, 1), (3, 2, 2)]
    assert completed == [(1, 3), (2, 2)]


def test_payload_round_trips_values_exactly():
    record = {"unit": 1, "time_in_cycles": 7, "sensor_9": 9046.19,
              "sensor_14": 8138.620000000001, "op_setting_1": -0.0007}
    payload = json.loads(json.dumps(telemetry_payload(record, "S")))
    assert {k: payload[k] for k in record} == record
    assert payload["session_id"] == "S" and payload["schema"] == 1


# --- consumer readiness check ------------------------------------------------

def test_replay_refuses_to_start_without_an_online_consumer(monkeypatch):
    for status in (None, {"state": "offline", "pid": 1}):
        monkeypatch.setattr(producer, "read_consumer_status",
                            lambda timeout=2.0, s=status: s)
        with pytest.raises(SystemExit, match="consumer is not online"):
            producer.check_consumer_online()


def test_online_consumer_passes_the_check(monkeypatch):
    monkeypatch.setattr(producer, "read_consumer_status",
                        lambda timeout=2.0: {"state": "online", "pid": 7})
    producer.check_consumer_online()


def test_no_consumer_check_flag_skips_the_check(monkeypatch):
    class Stop(Exception):
        pass

    def refuse():
        raise AssertionError("consumer check should be skipped")

    def stop_after_check(path):
        raise Stop

    monkeypatch.setattr(producer, "check_consumer_online", refuse)
    monkeypatch.setattr(producer, "load_and_process_txt", stop_after_check)
    with pytest.raises(Stop):
        producer.main(["--no-consumer-check"])
