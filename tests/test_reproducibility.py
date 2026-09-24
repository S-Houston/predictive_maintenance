# Static checks that all randomness in src/ is explicitly seeded
"""
Guards reproducibility of training runs by scanning src/ for:
- calls to the global `random` module (random.choice, random.shuffle, ...);
  only a seeded `random.Random(<seed>)` instance is allowed
- calls to numpy's global RNG (np.random.rand, np.random.seed, ...);
  only a seeded `np.random.default_rng(<seed>)` / `RandomState(<seed>)` is allowed
- data splitters called without an explicit random_state
"""

import ast
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
SOURCE_FILES = sorted(SRC_DIR.rglob("*.py"))

SEEDED_CONSTRUCTORS = {"Random", "default_rng", "RandomState"}
SPLITTERS = {"train_test_split", "GroupShuffleSplit", "ShuffleSplit", "StratifiedShuffleSplit"}
SHUFFLING_KFOLDS = {"KFold", "GroupKFold", "StratifiedKFold"}


def _dotted_name(node):
    """Returns e.g. 'np.random.choice' for a Call's func node, or None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _has_kwarg(call, name):
    return any(kw.arg == name for kw in call.keywords)


def _is_true_kwarg(call, name):
    return any(
        kw.arg == name and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in call.keywords
    )


def find_unseeded_calls(source):
    """Returns (line, description) for every unseeded random call in source code."""
    problems = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func)
        if name is None:
            continue
        parts = name.split(".")
        func = parts[-1]
        seeded = func in SEEDED_CONSTRUCTORS and (node.args or _has_kwarg(node, "seed"))

        # Global stdlib RNG: random.<fn>(...)
        if len(parts) == 2 and parts[0] == "random" and not seeded:
            problems.append((node.lineno, f"{name}() uses the global/unseeded stdlib RNG"))
        # Global numpy RNG: np.random.<fn>(...) / numpy.random.<fn>(...)
        elif len(parts) == 3 and parts[0] in {"np", "numpy"} and parts[1] == "random" and not seeded:
            problems.append((node.lineno, f"{name}() uses the global/unseeded numpy RNG"))
        # Splitters must pass random_state
        elif func in SPLITTERS and not _has_kwarg(node, "random_state"):
            problems.append((node.lineno, f"{func}() called without random_state"))
        elif func in SHUFFLING_KFOLDS and _is_true_kwarg(node, "shuffle") and not _has_kwarg(node, "random_state"):
            problems.append((node.lineno, f"{func}(shuffle=True) called without random_state"))
    return problems


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: str(p.relative_to(SRC_DIR)))
def test_no_unseeded_randomness(path):
    problems = find_unseeded_calls(path.read_text(encoding="utf-8"))
    assert not problems, "\n".join(f"{path.name}:{line}: {msg}" for line, msg in problems)


# Checks that the detector itself catches what it should

@pytest.mark.parametrize("snippet", [
    "import random\nrandom.choice([1, 2])",
    "import random\nrandom.shuffle(x)",
    "import random\nrandom.seed(1)",
    "import random\nrandom.Random()",
    "import numpy as np\nnp.random.rand(3)",
    "import numpy as np\nnp.random.default_rng()",
    "train_test_split(X, y, test_size=0.2)",
    "GroupShuffleSplit(n_splits=1, test_size=0.2)",
    "KFold(n_splits=5, shuffle=True)",
])
def test_detector_flags_unseeded(snippet):
    assert find_unseeded_calls(snippet)


@pytest.mark.parametrize("snippet", [
    "import random\nrng = random.Random(42)\nrng.choice([1, 2])",
    "import numpy as np\nnp.random.default_rng(42).normal()",
    "import numpy as np\nnp.random.RandomState(seed=0)",
    "train_test_split(X, y, test_size=0.2, random_state=42)",
    "GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)",
    "KFold(n_splits=5)",
])
def test_detector_allows_seeded(snippet):
    assert not find_unseeded_calls(snippet)
