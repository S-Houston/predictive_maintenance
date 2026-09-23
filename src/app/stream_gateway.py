"""
Streaming gateway: pushes live RUL predictions to browsers over WebSocket.

A paho-mqtt thread subscribes to the consumer's retained per-engine
predictions (plus the producer's session and completion topics) and hands
each message to the asyncio loop. A pump task applies them to an in-memory
snapshot of the fleet and broadcasts the changes every 250 ms to every
client on /ws/live. New clients receive the full snapshot first.

The retained prediction topics seed the snapshot as soon as the gateway
connects, so it holds no state of its own. Prediction history for charts
is read from the consumer's SQLite database, opened read-only.

Routes: /live (page), /ws/live (WebSocket), /stream/predictions,
/stream/history/{unit_id}. The batch /predictions routes are unaffected,
and the API starts even if the broker is down (paho keeps retrying).
"""

import asyncio
import json
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

import paho.mqtt.client as mqtt
import plotly
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ..streaming import config
from ..streaming.state_store import StateStore

log = logging.getLogger("stream_gateway")

STATIC_DIR = Path(__file__).parent / "static"
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
PUSH_INTERVAL = 0.25
BROKER_EVENT = "_broker"

router = APIRouter(tags=["streaming"])


class LiveBridge:
    """Fleet snapshot fed by MQTT, fanned out to WebSocket clients."""

    def __init__(self):
        self.session = None
        self.latest = {}
        self.completed = set()
        self.broker_connected = False
        self.clients = set()
        self.queue = None
        self.client = None

    # --- state (pure, no I/O) --------------------------------------------

    def snapshot(self):
        return {
            "type": "snapshot",
            "session": self.session,
            "broker_connected": self.broker_connected,
            "engines": [self.latest[u] for u in sorted(self.latest)],
        }

    def apply(self, topic, payload):
        """Applies one message; returns the events to broadcast."""
        if topic == BROKER_EVENT:
            self.broker_connected = payload["connected"]
            return [{"type": "broker", **payload}]
        kind = topic.rsplit("/", 1)[-1]
        if kind == "session":
            return self._on_session(payload)
        if not self._in_session(payload.get("session_id")):
            return []
        if kind == "prediction":
            return self._on_prediction(payload)
        if kind == "status":
            return self._on_status(payload)
        return []

    def _on_session(self, session):
        current = (self.session or {}).get("session_id")
        if current is not None and session["session_id"] <= current:
            return []
        self.session = session
        self.latest.clear()
        self.completed.clear()
        return [{"type": "session", "session": session}]

    def _in_session(self, session_id):
        """Adopts a newer session seen before its control message."""
        if session_id is None:
            return False
        current = (self.session or {}).get("session_id")
        if current is None or session_id > current:
            self._on_session({"session_id": session_id})
            return True
        return session_id == current

    def _on_prediction(self, pred):
        unit = pred["unit"]
        prev = self.latest.get(unit)
        if prev and prev["time_in_cycles"] >= pred["time_in_cycles"]:
            return []  # retained re-delivery or out-of-order duplicate
        status = "complete" if unit in self.completed else "active"
        self.latest[unit] = {**pred, "status": status}
        return [{"type": "prediction", **self.latest[unit]}]

    def _on_status(self, status):
        unit = status["unit"]
        self.completed.add(unit)
        if unit in self.latest:
            self.latest[unit]["status"] = "complete"
        return [{"type": "status", "unit": unit, "state": "complete"}]

    # --- MQTT thread -> asyncio ------------------------------------------

    def start(self, loop):
        self.queue = asyncio.Queue()

        def push(topic, payload):
            loop.call_soon_threadsafe(self.queue.put_nowait, (topic, payload))

        def on_connect(client, userdata, flags, reason_code, properties):
            if reason_code.is_failure:
                return
            client.subscribe([(config.SESSION_TOPIC, 1),
                              (config.PREDICTION_SUB, 1),
                              (config.STATUS_SUB, 1)])
            push(BROKER_EVENT, {"connected": True})

        def on_disconnect(client, userdata, flags, reason_code, properties):
            push(BROKER_EVENT, {"connected": False})

        def on_message(client, userdata, msg):
            try:
                push(msg.topic, json.loads(msg.payload))
            except ValueError:
                log.warning("Skipping malformed message on %s", msg.topic)

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id=f"pm-gateway-{os.getpid()}")
        self.client.on_connect = on_connect
        self.client.on_disconnect = on_disconnect
        self.client.on_message = on_message
        self.client.connect_async(config.BROKER_HOST, config.BROKER_PORT,
                                  keepalive=30)
        self.client.loop_start()

    def stop(self):
        if self.client is not None:
            self.client.disconnect()
            self.client.loop_stop()

    async def pump(self):
        """Applies queued messages and broadcasts them in batches."""
        while True:
            items = [await self.queue.get()]
            await asyncio.sleep(PUSH_INTERVAL)
            while not self.queue.empty():
                items.append(self.queue.get_nowait())
            events = []
            for topic, payload in items:
                events += self.apply(topic, payload)
            if events:
                await self.broadcast({"type": "events", "events": events})

    async def broadcast(self, message):
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                self.clients.discard(ws)


bridge = LiveBridge()


@asynccontextmanager
async def lifespan(app):
    bridge.start(asyncio.get_running_loop())
    task = asyncio.create_task(bridge.pump())
    try:
        yield
    finally:
        task.cancel()
        bridge.stop()


# --- routes ------------------------------------------------------------


@router.get("/live", include_in_schema=False)
def live_page():
    return FileResponse(STATIC_DIR / "live.html")


@router.get("/live/plotly.min.js", include_in_schema=False)
def plotly_js():
    return FileResponse(PLOTLY_JS, media_type="application/javascript")


@router.websocket("/ws/live")
async def live_socket(ws: WebSocket):
    await ws.accept()
    await ws.send_json(bridge.snapshot())
    bridge.clients.add(ws)
    try:
        while True:
            await ws.receive_text()  # keep the socket open; input ignored
    except WebSocketDisconnect:
        pass
    finally:
        bridge.clients.discard(ws)


def _read_state(fn):
    """Runs fn(store, session_id) against the read-only stream state."""
    if not config.STATE_DB_PATH.exists():
        raise HTTPException(status_code=404,
                            detail="No streaming state yet; start the "
                                   "consumer and producer")
    try:
        store = StateStore.open_readonly(config.STATE_DB_PATH)
        try:
            return fn(store, store.current_session())
        finally:
            store.close()
    except sqlite3.Error as e:
        raise HTTPException(status_code=503,
                            detail=f"Streaming state unavailable: {e}")


@router.get("/stream/predictions")
def stream_predictions():
    """Latest streamed prediction per engine in the current session."""
    return _read_state(lambda store, session: {
        "session_id": session,
        "predictions": store.latest_predictions(session) if session else [],
    })


@router.get("/stream/history/{unit_id}")
def stream_history(unit_id: int):
    """Every streamed prediction for one engine in the current session."""
    def history(store, session):
        if session is None:
            return {"session_id": None, "unit": unit_id, "history": []}
        df = store.prediction_history(session, unit_id)
        return {"session_id": session, "unit": unit_id,
                "history": df.drop(columns="unit").to_dict("records")}
    return _read_state(history)
