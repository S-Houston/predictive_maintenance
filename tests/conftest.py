# Shared fixtures for the streaming tests

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.make_dataset import load_and_process_txt
from streaming.state_store import READING_COLUMNS, StateStore

RAW_TEST_PATH = Path("data/raw/test_FD001.txt")


@pytest.fixture(scope="session")
def raw_test_df():
    """The real FD001 test set as the producer reads it (skips if absent)."""
    if not RAW_TEST_PATH.exists():
        pytest.skip(f"{RAW_TEST_PATH} not found (data/ is git-ignored)")
    return load_and_process_txt(RAW_TEST_PATH)


@pytest.fixture
def store():
    s = StateStore.open(":memory:")
    yield s
    s.close()


def make_readings(unit, cycles, seed=0):
    """Synthetic raw readings for one engine, as JSON-decoded payload dicts."""
    rng = np.random.default_rng(seed + unit)
    df = pd.DataFrame(rng.normal(500, 25, (len(cycles), len(READING_COLUMNS))),
                      columns=READING_COLUMNS)
    df["unit"] = unit
    df["time_in_cycles"] = list(cycles)
    return [json.loads(json.dumps(r)) for r in df.to_dict("records")]
