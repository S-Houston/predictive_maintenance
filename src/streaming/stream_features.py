"""
Incremental feature engineering for the streaming consumer.

Rather than re-implementing the batch features as running statistics, each
micro-batch reruns the batch function engineer_health_indicators() over the
full stored history of the engines it touched and keeps only the rows not
yet scored. Features at cycle t only depend on cycles <= t (except the
baseline, see below), and pandas' rolling aggregations are computed
sequentially from the start of each series, so this reproduces the batch
feature values bit for bit. A bounded buffer of the last 5 readings would
drift by up to ~1e-9 because the rolling algorithm's running sums depend
on the whole series.

Warm-up: the batch baseline is the mean of each engine's first 5 cycles,
so for cycles 1-4 it uses readings that have not arrived yet. Those rows
are held back and emitted together once cycle 5 arrives, which keeps every
emitted row identical to batch at the cost of a 4-cycle start-up delay.
"""

import logging
import warnings

import pandas as pd

from features.driver_health_indicators import SENSOR_COLS
from features.engineer_health_indicators import engineer_health_indicators
from streaming import config

log = logging.getLogger("stream_features")


def compute_features(history, sensor_cols=SENSOR_COLS):
    """Batch feature engineering over raw readings, sorted by unit/cycle."""
    with warnings.catch_warnings():
        # pandas deprecation noise from compute_sensor_baselines' apply()
        warnings.simplefilter("ignore", DeprecationWarning)
        return engineer_health_indicators(history, sensor_cols)


class StreamFeatureEngine:
    def __init__(self, store, sensor_cols=SENSOR_COLS,
                 warmup=config.WARMUP_CYCLES):
        self.store = store
        self.sensor_cols = sensor_cols
        self.warmup = warmup

    def ingest(self, session_id, records, now):
        """
        Stores a micro-batch of readings and returns the feature rows that
        are now ready to score (possibly empty, possibly several cycles of
        one engine when its warm-up completes). Call inside a store
        transaction together with writing the predictions, so engine
        progress and predictions are committed atomically.
        """
        states = self.store.engine_states(session_id)
        touched = {}
        for rec in sorted(records,
                          key=lambda r: (r["unit"], r["time_in_cycles"])):
            unit = int(rec["unit"])
            state = states.get(unit) or self._new_state(unit, now)
            if self._accept(session_id, rec, state, now):
                states[unit] = touched[unit] = state

        if not touched:
            return pd.DataFrame()
        ready = self._ready_rows(session_id, touched)
        for state in touched.values():
            self.store.upsert_engine_state(session_id, state, now)
        return ready

    @staticmethod
    def _new_state(unit, now):
        return {"unit": unit, "last_cycle": 0, "last_predicted_cycle": 0,
                "n_readings": 0, "status": "warming_up", "first_seen": now}

    def _accept(self, session_id, rec, state, now):
        """Stores one reading if it is new; updates the engine state."""
        unit, cycle = int(rec["unit"]), int(rec["time_in_cycles"])
        if cycle <= state["last_cycle"]:
            log.debug("Unit %d cycle %d already processed; ignored",
                      unit, cycle)
            return False
        if cycle != state["last_cycle"] + 1:
            # Batch diff() treats consecutive rows as consecutive cycles
            # too, so this matches batch on the rows that did arrive. A
            # missing cycle that turns up later is dropped as out of order.
            log.warning("Unit %d jumped from cycle %d to %d",
                        unit, state["last_cycle"], cycle)
        if not self.store.insert_reading(session_id, rec, now):
            return False
        state["last_cycle"] = cycle
        state["n_readings"] += 1
        return True

    def _ready_rows(self, session_id, states):
        warm = []
        for unit, state in states.items():
            is_warm = state["n_readings"] >= self.warmup
            if state["status"] != "complete":
                state["status"] = "active" if is_warm else "warming_up"
            if is_warm:
                warm.append(unit)
        if not warm:
            return pd.DataFrame()

        feats = compute_features(
            self.store.load_history(session_id, warm), self.sensor_cols)
        scored_up_to = feats["unit"].map(
            {u: states[u]["last_predicted_cycle"] for u in warm})
        ready = feats[feats["time_in_cycles"] > scored_up_to]
        latest = ready.groupby("unit")["time_in_cycles"].max()
        for unit, cycle in latest.items():
            states[int(unit)]["last_predicted_cycle"] = int(cycle)
        return ready.reset_index(drop=True)
