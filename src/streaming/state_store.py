"""
SQLite state for the streaming simulation.

Holds every raw reading received per engine (the full history is what lets
the stream reproduce the batch features bit for bit), per-engine progress,
emitted predictions and their feature vectors. All rows are keyed by the
replay session so a --keep-history run does not collide with earlier ones.

The consumer is the only writer. Other processes (the API gateway) open the
database read-only; WAL mode lets them read while the consumer writes.
"""

import json
import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = "1"

READING_COLUMNS = (
    ["unit", "time_in_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
_VALUE_COLUMNS = READING_COLUMNS[2:]

SESSION_TABLES = ["readings", "engine_state", "predictions", "features",
                  "sessions"]

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS stream_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    keep_history INTEGER NOT NULL,
    started_at TEXT,
    info_json TEXT
);
CREATE TABLE IF NOT EXISTS readings (
    session_id TEXT NOT NULL,
    unit INTEGER NOT NULL,
    time_in_cycles INTEGER NOT NULL,
    {", ".join(f"{c} REAL" for c in _VALUE_COLUMNS)},
    published_at TEXT,
    received_at TEXT NOT NULL,
    PRIMARY KEY (session_id, unit, time_in_cycles)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS engine_state (
    session_id TEXT NOT NULL,
    unit INTEGER NOT NULL,
    last_cycle INTEGER NOT NULL,
    last_predicted_cycle INTEGER NOT NULL DEFAULT 0,
    n_readings INTEGER NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('warming_up', 'active', 'complete')),
    first_seen TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (session_id, unit)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS predictions (
    session_id TEXT NOT NULL,
    unit INTEGER NOT NULL,
    time_in_cycles INTEGER NOT NULL,
    rul REAL NOT NULL,
    risk_level TEXT NOT NULL,
    model_run_id TEXT NOT NULL,
    predicted_at TEXT NOT NULL,
    PRIMARY KEY (session_id, unit, time_in_cycles)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS features (
    session_id TEXT NOT NULL,
    unit INTEGER NOT NULL,
    time_in_cycles INTEGER NOT NULL,
    feature_json TEXT NOT NULL,
    PRIMARY KEY (session_id, unit, time_in_cycles)
) WITHOUT ROWID;
"""


def _json_value(v):
    """JSON-safe scalar: numpy -> Python, NaN -> None."""
    v = v.item() if hasattr(v, "item") else v
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


class StateStore:
    def __init__(self, conn):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row

    @classmethod
    def open(cls, path):
        """Opens (creating if needed) the database for writing."""
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        store = cls(conn)
        store.init_schema()
        return store

    @classmethod
    def open_readonly(cls, path):
        uri = f"{Path(path).resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, isolation_level=None,
                               check_same_thread=False)
        return cls(conn)

    def close(self):
        self.conn.close()

    def init_schema(self):
        self.conn.executescript(SCHEMA)
        self.set_meta("schema_version", SCHEMA_VERSION)

    @contextmanager
    def transaction(self):
        """One atomic unit of work (BEGIN IMMEDIATE ... COMMIT/ROLLBACK)."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    # --- meta / sessions -------------------------------------------------

    def get_meta(self, key):
        row = self.conn.execute(
            "SELECT value FROM stream_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key, value):
        self.conn.execute(
            "INSERT INTO stream_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value))

    def current_session(self):
        return self.get_meta("current_session")

    def begin_session(self, session):
        """
        Makes `session` (the producer's control message) the current one.
        Unless it asks to keep history, rows of every other session are
        deleted. Returns False if it was already the current session.
        """
        session_id = session["session_id"]
        if session_id == self.current_session():
            return False
        if not session.get("keep_history", False):
            for table in SESSION_TABLES:
                self.conn.execute(
                    f"DELETE FROM {table} WHERE session_id != ?",
                    (session_id,))
        self.conn.execute(
            "INSERT OR IGNORE INTO sessions "
            "(session_id, keep_history, started_at, info_json) "
            "VALUES (?, ?, ?, ?)",
            (session_id, int(bool(session.get("keep_history"))),
             session.get("started_at"), json.dumps(session)))
        self.set_meta("current_session", session_id)
        return True

    def session_info(self, session_id):
        """The producer's control message for a session, or {} if unknown."""
        row = self.conn.execute(
            "SELECT info_json FROM sessions WHERE session_id = ?",
            (session_id,)).fetchone()
        return json.loads(row["info_json"]) if row else {}

    # --- readings / engine state -----------------------------------------

    def insert_reading(self, session_id, record, received_at):
        """Inserts one reading; returns False if it was already stored."""
        cols = READING_COLUMNS + ["published_at"]
        values = [record[c] for c in READING_COLUMNS] + [record.get("ts")]
        cur = self.conn.execute(
            f"INSERT OR IGNORE INTO readings "
            f"(session_id, {', '.join(cols)}, received_at) "
            f"VALUES ({', '.join('?' * (len(cols) + 2))})",
            [session_id, *values, received_at])
        return cur.rowcount == 1

    def engine_states(self, session_id):
        rows = self.conn.execute(
            "SELECT * FROM engine_state WHERE session_id = ?", (session_id,))
        return {r["unit"]: dict(r) for r in rows}

    def upsert_engine_state(self, session_id, state, now):
        self.conn.execute(
            "INSERT INTO engine_state (session_id, unit, last_cycle, "
            "last_predicted_cycle, n_readings, status, first_seen, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(session_id, unit) DO UPDATE SET "
            "last_cycle = excluded.last_cycle, "
            "last_predicted_cycle = excluded.last_predicted_cycle, "
            "n_readings = excluded.n_readings, status = excluded.status, "
            "updated_at = excluded.updated_at",
            (session_id, state["unit"], state["last_cycle"],
             state["last_predicted_cycle"], state["n_readings"],
             state["status"], state.get("first_seen", now), now))

    def mark_complete(self, session_id, unit, now):
        self.conn.execute(
            "UPDATE engine_state SET status = 'complete', updated_at = ? "
            "WHERE session_id = ? AND unit = ?", (now, session_id, unit))

    def load_history(self, session_id, units):
        """Raw readings of `units`, ordered by unit then cycle."""
        units = sorted(int(u) for u in units)
        return pd.read_sql_query(
            f"SELECT {', '.join(READING_COLUMNS)} FROM readings "
            f"WHERE session_id = ? AND unit IN "
            f"({', '.join('?' * len(units))}) "
            f"ORDER BY unit, time_in_cycles",
            self.conn, params=[session_id, *units])

    # --- predictions / features ------------------------------------------

    def insert_predictions(self, session_id, preds, model_run_id, now):
        self.conn.executemany(
            "INSERT OR IGNORE INTO predictions (session_id, unit, "
            "time_in_cycles, rul, risk_level, model_run_id, predicted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(session_id, int(p["unit"]), int(p["time_in_cycles"]),
              float(p["rul"]), p["risk_level"], model_run_id, now)
             for p in preds])

    def insert_features(self, session_id, feats, feature_cols):
        rows = []
        for rec in feats.to_dict("records"):
            vector = {c: _json_value(rec[c]) for c in feature_cols}
            rows.append((session_id, int(rec["unit"]),
                         int(rec["time_in_cycles"]), json.dumps(vector)))
        self.conn.executemany(
            "INSERT OR IGNORE INTO features "
            "(session_id, unit, time_in_cycles, feature_json) "
            "VALUES (?, ?, ?, ?)", rows)

    def latest_predictions(self, session_id):
        """Most recent prediction per engine, joined with engine status."""
        rows = self.conn.execute(
            "SELECT p.unit, p.time_in_cycles, p.rul, p.risk_level, "
            "p.model_run_id, p.predicted_at, e.status "
            "FROM predictions p JOIN engine_state e "
            "ON e.session_id = p.session_id AND e.unit = p.unit "
            "AND e.last_predicted_cycle = p.time_in_cycles "
            "WHERE p.session_id = ? ORDER BY p.unit", (session_id,))
        return [dict(r) for r in rows]

    def prediction_history(self, session_id, unit=None):
        sql = ("SELECT unit, time_in_cycles, rul, risk_level "
               "FROM predictions WHERE session_id = ?")
        params = [session_id]
        if unit is not None:
            sql += " AND unit = ?"
            params.append(int(unit))
        sql += " ORDER BY unit, time_in_cycles"
        return pd.read_sql_query(sql, self.conn, params=params)
