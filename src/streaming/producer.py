"""
Telemetry Producer
==================

Replays the CMAPSS FD001 test set as a live stream: one MQTT message per
engine per cycle, published to pm/fd001/engine/{unit}/telemetry at a fixed
cycle interval. Engines join the stream at seeded random offsets
(--stagger) so the fleet looks like engines already in service, and each
engine stops at its last recorded cycle.

Every run is a new replay session, announced on the retained
pm/fd001/control/session topic. The consumer wipes its SQLite state for
earlier sessions unless the producer is run with --keep-history.

Usage (from the repo root, broker running via `docker compose up -d`):
    PYTHONPATH=src python src/streaming/producer.py --interval 0.2
"""

import argparse
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt

from data.make_dataset import load_and_process_txt
from streaming import config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("producer")

DEFAULT_SOURCE = "data/raw/test_FD001.txt"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_units(spec, available):
    """'all', '1-20' or '1,5,7' -> sorted list of units present in the data."""
    available = sorted(int(u) for u in available)
    if spec == "all":
        return available
    units = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            units.update(range(int(lo), int(hi) + 1))
        else:
            units.add(int(part))
    missing = units - set(available)
    if missing:
        raise ValueError(f"Units not in the data: {sorted(missing)}")
    return sorted(units)


def build_start_ticks(units, stagger, seed):
    """
    Tick at which each engine publishes its first cycle, drawn uniformly
    from 0..stagger with a seeded RNG and shifted so the earliest is 0.
    stagger=0 starts every engine together (lockstep).
    """
    rng = random.Random(seed)
    ticks = {u: rng.randint(0, stagger) for u in units}
    first = min(ticks.values())
    return {u: t - first for u, t in ticks.items()}


def iter_ticks(df, start_ticks):
    """
    Yields (tick, readings, completed) for every tick of the replay.
    readings: row dicts published this tick, in unit order.
    completed: (unit, last_cycle) for engines whose series ends this tick.
    """
    df = df.sort_values(["unit", "time_in_cycles"])
    series = {
        int(unit): rows.to_dict("records")
        for unit, rows in df.groupby("unit")
        if int(unit) in start_ticks
    }
    last_tick = max(start_ticks[u] + len(r) for u, r in series.items())
    for tick in range(last_tick):
        readings, completed = [], []
        for unit in sorted(series):
            i = tick - start_ticks[unit]
            rows = series[unit]
            if 0 <= i < len(rows):
                readings.append(rows[i])
                if i == len(rows) - 1:
                    completed.append((unit, rows[i]["time_in_cycles"]))
        yield tick, readings, completed


def telemetry_payload(record, session_id):
    return {
        "schema": config.PAYLOAD_SCHEMA,
        "session_id": session_id,
        **record,
        "ts": utc_now(),
    }


def connect(client_id):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=client_id)
    client.will_set(config.PRODUCER_STATUS_TOPIC, "offline", qos=1,
                    retain=True)
    try:
        client.connect(config.BROKER_HOST, config.BROKER_PORT, keepalive=30)
    except OSError as e:
        raise SystemExit(
            f"Cannot reach MQTT broker at {config.BROKER_HOST}:"
            f"{config.BROKER_PORT} ({e}). Is it running? "
            "Try: docker compose up -d"
        )
    client.loop_start()
    return client


def publish_json(client, topic, payload, retain=False):
    return client.publish(topic, json.dumps(payload), qos=1, retain=retain)


def replay(client, df, start_ticks, session_id, interval):
    """Publishes the replay tick by tick; returns the last publish handle."""
    info = None
    deadline = time.monotonic()
    for tick, readings, completed in iter_ticks(df, start_ticks):
        for record in readings:
            topic = config.TELEMETRY_TOPIC.format(unit=record["unit"])
            info = publish_json(client, topic,
                                telemetry_payload(record, session_id))
        for unit, last_cycle in completed:
            info = publish_json(
                client, config.STATUS_TOPIC.format(unit=unit),
                {"session_id": session_id, "unit": unit,
                 "state": "complete", "last_cycle": last_cycle},
                retain=True,
            )
        if tick % 25 == 0:
            log.info("tick %d: %d readings, %d engines completed",
                     tick, len(readings), len(completed))
        deadline += interval
        time.sleep(max(0.0, deadline - time.monotonic()))
    return info


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("--source", default=DEFAULT_SOURCE,
                   help="raw CMAPSS .txt file to replay")
    p.add_argument("--interval", type=float, default=1.0,
                   help="seconds between cycles (default 1.0)")
    p.add_argument("--units", default="all",
                   help="'all', a range '1-20' or a list '1,5,7'")
    p.add_argument("--stagger", type=int, default=50,
                   help="max start offset per engine, in cycles "
                        "(0 = all engines start together)")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for the staggered start offsets")
    p.add_argument("--keep-history", action="store_true",
                   help="keep earlier sessions in the consumer's SQLite "
                        "state instead of wiping them")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    df = load_and_process_txt(Path(args.source))
    units = parse_units(args.units, df["unit"].unique())
    start_ticks = build_start_ticks(units, args.stagger, args.seed)
    df = df[df["unit"].isin(units)]

    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session = {
        "session_id": session_id,
        "keep_history": args.keep_history,
        "started_at": utc_now(),
        "units": units,
        "interval": args.interval,
        "source": args.source,
    }

    client = connect(f"pm-producer-{session_id}")
    publish_json(client, config.SESSION_TOPIC, session, retain=True)
    client.publish(config.PRODUCER_STATUS_TOPIC, "online", qos=1,
                   retain=True)
    log.info("Session %s: replaying %d engines (%d readings) every %.2fs",
             session_id, len(units), len(df), args.interval)

    status = "finished"
    try:
        info = replay(client, df, start_ticks, session_id, args.interval)
        if info is not None:
            info.wait_for_publish(timeout=30)
    except KeyboardInterrupt:
        status = "stopped"
        log.info("Interrupted; stopping replay")
    client.publish(config.PRODUCER_STATUS_TOPIC, status, qos=1,
                   retain=True).wait_for_publish(timeout=10)
    client.disconnect()
    client.loop_stop()
    log.info("Session %s %s", session_id, status)


if __name__ == "__main__":
    main()
