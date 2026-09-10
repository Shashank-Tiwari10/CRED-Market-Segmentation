"""
Shared utilities: logging, seeding, timing, and metric persistence.
"""
from __future__ import annotations

import json
import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pandas as pd

import config


def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> logging.Logger:
    """Configure the root logger once and return the package logger."""
    root = logging.getLogger()
    if not root.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        root.addHandler(stream)
        if log_file is not None:
            fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
    root.setLevel(level)
    # third-party chatter
    for noisy in ("matplotlib", "PIL", "optuna", "numba", "shap"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("cred")


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``cred`` namespace."""
    return logging.getLogger(f"cred.{name}")


def set_seed(seed: int = config.SEED) -> np.random.Generator:
    """Seed every RNG we rely on and return a NumPy Generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


@contextmanager
def timer(label: str, logger: Optional[logging.Logger] = None) -> Iterator[Dict[str, float]]:
    """Context manager that logs elapsed wall time for a block."""
    log = logger or logging.getLogger("cred")
    start = time.perf_counter()
    result: Dict[str, float] = {}
    log.info("▶ %s", label)
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["seconds"] = elapsed
        log.info("✔ %s finished in %.1fs", label, elapsed)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Persist a dictionary as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON dictionary."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (as a fraction, e.g. 0.108 = 10.8%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def holder_scale_factor(n_sample_holders: int) -> float:
    """Multiplier that maps sample holder counts to the 38M national market."""
    return config.REAL_MARKET_HOLDERS / max(n_sample_holders, 1)


def fmt_millions(value: float) -> str:
    """Format a raw count as e.g. ``3.8M``."""
    return f"{value / 1e6:.1f}M"


def pass_fail(condition: bool) -> str:
    """Render a boolean as the summary-table status token."""
    return "[PASS]" if condition else "[FAIL]"
