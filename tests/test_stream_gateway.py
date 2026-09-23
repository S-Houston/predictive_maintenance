# tests/test_stream_gateway.py
"""
Tests for the WebSocket gateway: the fleet snapshot logic of LiveBridge and
the streaming routes (without an MQTT broker).
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app import stream_gateway as gw
from streaming.state_store import StateStore

S1, S2 = "20260101T000000Z", "20260102T000000Z"


def pred(unit, cycle, rul, session=S1):
    return (f"pm/fd001/engine/{unit}/prediction",
            {"session_id": session, "unit": unit, "time_in_cycles": cycle,
             "rul": rul, "risk_level": "Low"})


def apply_all(bridge, messages):
    events = []
    for topic, payload in messages:
        events += bridge.apply(topic, payload)
    return events


def test_bridge_keeps_latest_prediction_per_engine():
    bridge = gw.LiveBridge()
    apply_all(bridge, [("pm/fd001/control/session", {"session_id": S1}),
                       pred(1, 5, 120.0), pred(1, 6, 119.0),
                       pred(2, 5, 80.0)])
    snap = bridge.snapshot()
    assert [(e["unit"], e["time_in_cycles"]) for e in snap["engines"]] == \
        [(1, 6), (2, 5)]


def test_bridge_ignores_redelivered_older_prediction():
    bridge = gw.LiveBridge()
    apply_all(bridge, [pred(1, 6, 119.0)])
    assert bridge.apply(*pred(1, 5, 120.0)) == []
    assert bridge.latest[1]["time_in_cycles"] == 6


def test_new_session_resets_snapshot_and_stale_one_is_ignored():
    bridge = gw.LiveBridge()
    apply_all(bridge, [("pm/fd001/control/session", {"session_id": S1}),
                       pred(1, 5, 120.0)])
    events = bridge.apply("pm/fd001/control/session", {"session_id": S2})
    assert events[0]["type"] == "session" and bridge.latest == {}
    assert bridge.apply(*pred(1, 7, 100.0, session=S1)) == []
    assert bridge.apply("pm/fd001/control/session",
                        {"session_id": S1}) == []


def test_status_before_last_prediction_marks_engine_complete():
    bridge = gw.LiveBridge()
    apply_all(bridge, [pred(1, 30, 40.0)])
    bridge.apply("pm/fd001/engine/1/status",
                 {"session_id": S1, "unit": 1, "state": "complete"})
    apply_all(bridge, [pred(1, 31, 39.0)])
    assert bridge.latest[1]["status"] == "complete"


def test_broker_events_update_connection_state():
    bridge = gw.LiveBridge()
    events = bridge.apply(gw.BROKER_EVENT, {"connected": True})
    assert events == [{"type": "broker", "connected": True}]
    assert bridge.snapshot()["broker_connected"] is True


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(gw, "bridge", gw.LiveBridge())
    monkeypatch.setattr(gw.config, "STATE_DB_PATH", tmp_path / "state.db")
    app = FastAPI()
    app.include_router(gw.router)
    return TestClient(app)


def test_websocket_sends_snapshot_on_connect(client):
    gw.bridge.apply(*pred(3, 8, 55.5))
    with client.websocket_connect("/ws/live") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "snapshot"
    assert msg["engines"][0]["unit"] == 3


def test_live_page_and_plotly_are_served(client):
    assert "/ws/live" in client.get("/live").text
    assert client.get("/live/plotly.min.js").status_code == 200


def test_stream_routes_404_without_state(client):
    assert client.get("/stream/predictions").status_code == 404


def test_stream_routes_read_consumer_state(client):
    store = StateStore.open(gw.config.STATE_DB_PATH)
    with store.transaction():
        store.begin_session({"session_id": S1})
        for cycle in (5, 6):
            store.upsert_engine_state(S1, {
                "unit": 1, "last_cycle": cycle,
                "last_predicted_cycle": cycle, "n_readings": cycle,
                "status": "active"}, "now")
            store.insert_predictions(S1, [{
                "unit": 1, "time_in_cycles": cycle, "rul": 100.0 - cycle,
                "risk_level": "Medium"}], "run", "now")

    latest = client.get("/stream/predictions").json()
    assert latest["session_id"] == S1
    assert [p["time_in_cycles"] for p in latest["predictions"]] == [6]
    history = client.get("/stream/history/1").json()["history"]
    assert [h["rul"] for h in history] == [95.0, 94.0]
    store.close()
