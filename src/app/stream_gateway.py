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
/stream/history/{unit_id}, /stream/status (bridge health and counters).
The batch /predictions routes are unaffected, and the API starts even if
the broker is down (paho keeps retrying).

Every hop logs to the "stream_gateway" logger: broker connection and
subscriptions, session changes, WebSocket clients, send failures, and a
stats line every STATS_INTERVAL seconds while messages flow.
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

import paho.mqtt.client as mqtt
import plotly
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ..streaming import config
from ..streaming.state_store import StateStore

log = logging.getLogger("stream_gateway")
# uvicorn only configures its own loggers, so without a handler here INFO
# lines would be dropped.
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)

STATIC_DIR = Path(__file__).parent / "static"
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
PUSH_INTERVAL = 0.25
STATS_INTERVAL = 10.0
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
        self.loop = None
        self.client = None
        self.stopping = False
        # received / malformed / dropped_* / events / broadcasts /
        # send_failures; exposed on /stream/status
        self.counters = Counter()
        self._stats_logged = (time.monotonic(), Counter())

    # --- state (pure, no I/O) --------------------------------------------

    def snapshot(self):
        return {
            "type": "snapshot",
            "session": self.session,
            "broker_connected": self.broker_connected,
            "engines": [self.latest[u] for u in sorted(self.latest)],
        }

    def status(self):
        return {
            "broker_connected": self.broker_connected,
            "session_id": (self.session or {}).get("session_id"),
            "engines": len(self.latest),
            "clients": len(self.clients),
            "counters": dict(self.counters),
        }

    def apply(self, topic, payload):
        """Applies one message; returns the events to broadcast."""
        if topic == BROKER_EVENT:
            self.broker_connected = payload["connected"]
            return [{"type": "broker", **payload}]
        self.counters["received"] += 1
        events = self._apply_message(topic, payload)
        self.counters["events"] += len(events)
        return events

    def _apply_message(self, topic, payload):
        kind = topic.rsplit("/", 1)[-1]
        if kind == "session":
            return self._on_session(payload)
        if not self._in_session(payload.get("session_id")):
            self.counters["dropped_stale_session"] += 1
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
        log.info("Session %s: fleet snapshot reset", session["session_id"])
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
            # retained re-delivery or out-of-order duplicate
            self.counters["dropped_older_cycle"] += 1
            return []
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
        self.loop = loop
        self.queue = asyncio.Queue()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id=f"pm-gateway-{os.getpid()}")
        self.client.on_connect = self._on_connect
        self.client.on_subscribe = self._on_subscribe
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.connect_async(config.BROKER_HOST, config.BROKER_PORT,
                                  keepalive=30)
        self.client.loop_start()

    def _push(self, topic, payload):
        """Hands a message from the paho thread to the asyncio loop."""
        self.loop.call_soon_threadsafe(self.queue.put_nowait,
                                       (topic, payload))

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.error("MQTT connect to %s:%s failed: %s",
                      config.BROKER_HOST, config.BROKER_PORT, reason_code)
            return
        client.subscribe([(config.SESSION_TOPIC, 1),
                          (config.PREDICTION_SUB, 1),
                          (config.STATUS_SUB, 1)])
        log.info("Connected to MQTT broker at %s:%s",
                 config.BROKER_HOST, config.BROKER_PORT)
        self._push(BROKER_EVENT, {"connected": True})

    def _on_subscribe(self, client, userdata, mid, reason_code_list,
                      properties):
        failed = [str(rc) for rc in reason_code_list if rc.is_failure]
        if failed:
            log.error("Subscription refused by broker: %s", failed)
        else:
            log.info("Subscribed to session, prediction and status topics")

    def _on_disconnect(self, client, userdata, flags, reason_code,
                       properties):
        if self.stopping:
            log.info("Disconnected from MQTT broker")
            return
        log.warning("Disconnected from MQTT broker (%s); paho will "
                    "reconnect", reason_code)
        self._push(BROKER_EVENT, {"connected": False})

    def _on_message(self, client, userdata, msg):
        try:
            self._push(msg.topic, json.loads(msg.payload))
        except ValueError:
            self.counters["malformed"] += 1
            log.warning("Skipping malformed message on %s", msg.topic)

    def stop(self):
        self.stopping = True
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
            self.log_stats()

    def log_stats(self, now=None):
        """Logs counter deltas every STATS_INTERVAL s while messages flow."""
        now = time.monotonic() if now is None else now
        last_time, last_counts = self._stats_logged
        if now - last_time < STATS_INTERVAL:
            return False
        delta = self.counters - last_counts
        self._stats_logged = (now, self.counters.copy())
        if not delta:
            return False
        log.info("Last %.0fs: %d MQTT messages, %d events to %d WebSocket "
                 "clients (%d broadcasts); dropped %d stale-session, %d "
                 "older-cycle, %d malformed; %d engines in snapshot",
                 now - last_time, delta["received"], delta["events"],
                 len(self.clients), delta["broadcasts"],
                 delta["dropped_stale_session"],
                 delta["dropped_older_cycle"], delta["malformed"],
                 len(self.latest))
        return True

    async def broadcast(self, message):
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
                self.counters["broadcasts"] += 1
            except Exception as e:
                self.counters["send_failures"] += 1
                self.clients.discard(ws)
                log.warning("Dropped WebSocket client after a failed send "
                            "(%r); %d clients left", e, len(self.clients))


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
    snapshot = bridge.snapshot()
    await ws.send_json(snapshot)
    bridge.clients.add(ws)
    log.info("WebSocket client connected (%d engines in snapshot); %d "
             "clients", len(snapshot["engines"]), len(bridge.clients))
    try:
        while True:
            await ws.receive_text()  # keep the socket open; input ignored
    except WebSocketDisconnect:
        pass
    finally:
        if ws in bridge.clients:
            bridge.clients.discard(ws)
            log.info("WebSocket client disconnected; %d clients",
                     len(bridge.clients))


@router.get("/stream/status")
def stream_status():
    """Bridge health: broker connection, session, clients and counters."""
    return bridge.status()


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
