"""Copy mlflow.db to backups/mlflow_<YYYY-MM-DD_HHMM>.db.

Run from the repo root before any operation that touches the training
pipeline or the MLflow store directly:

    python src/backup_mlflow.py

Uses SQLite's backup API rather than a plain file copy, so the snapshot
is consistent even while the MLflow server is running.
"""
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

SOURCE = Path("mlflow.db")
BACKUP_DIR = Path("backups")


def main():
    if not SOURCE.exists():
        sys.exit(f"{SOURCE} not found; run this from the repo root.")
    BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    target = BACKUP_DIR / f"mlflow_{stamp}.db"
    if target.exists():
        sys.exit(f"{target} already exists; not overwriting it.")

    src = sqlite3.connect(f"file:{SOURCE}?mode=ro", uri=True)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    print(f"Backed up {SOURCE} to {target}")


if __name__ == "__main__":
    main()
