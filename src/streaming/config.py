"""
Settings shared by the streaming simulation: MQTT broker, topic layout and
the SQLite state database.

Topics are per engine so subscribers can filter with MQTT wildcards and the
broker can retain the latest status/prediction of every engine separately:

    pm/fd001/control/session              retained: current replay session
    pm/fd001/producer/status              retained: online/finished/offline
    pm/fd001/consumer/status              retained: consumer online/offline
    pm/fd001/engine/{unit}/telemetry      QoS 1: one sensor reading per cycle
    pm/fd001/engine/{unit}/status         retained: engine replay complete
    pm/fd001/engine/{unit}/prediction     retained: latest RUL prediction
"""

import os
from pathlib import Path

BROKER_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
BROKER_PORT = int(os.getenv("MQTT_PORT", "1883"))

TOPIC_ROOT = "pm/fd001"
SESSION_TOPIC = f"{TOPIC_ROOT}/control/session"
PRODUCER_STATUS_TOPIC = f"{TOPIC_ROOT}/producer/status"
# JSON {"state": "online"|"offline", "pid", "model_run_id", "ts"}. "online"
# means the model is loaded and telemetry subscriptions are acknowledged.
CONSUMER_STATUS_TOPIC = f"{TOPIC_ROOT}/consumer/status"
TELEMETRY_TOPIC = TOPIC_ROOT + "/engine/{unit}/telemetry"
STATUS_TOPIC = TOPIC_ROOT + "/engine/{unit}/status"
PREDICTION_TOPIC = TOPIC_ROOT + "/engine/{unit}/prediction"
TELEMETRY_SUB = f"{TOPIC_ROOT}/engine/+/telemetry"
STATUS_SUB = f"{TOPIC_ROOT}/engine/+/status"
PREDICTION_SUB = f"{TOPIC_ROOT}/engine/+/prediction"

PAYLOAD_SCHEMA = 1

STATE_DB_PATH = Path("data/stream/stream_state.db")

# Cycles per engine before the first prediction is emitted. The batch
# baseline is the mean of each unit's first 5 cycles (baseline_window in
# engineer_health_indicators), so rows 1-4 can only be scored exactly once
# cycle 5 has arrived.
WARMUP_CYCLES = 5
