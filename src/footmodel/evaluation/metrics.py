"""Scoring rules and calibration measures.

1X2 functions take P with shape (n, 3) in the order home, draw, away and y with
the class index 0, 1, 2. Binary functions take a probability p and a 0/1 outcome.
Lower is better for log loss, Brier score, RPS and ECE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _onehot(y, k: int) -> np.ndarray:
    return np.eye(k)[np.asarray(y, dtype=int)]


def row_logloss(P, y) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    return -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0))


def logloss(P, y) -> float:
    return float(row_logloss(P, y).mean())


def brier(P, y) -> float:
    P = np.asarray(P, dtype=float)
    return float(((P - _onehot(y, P.shape[1])) ** 2).sum(axis=1).mean())


def rps(P, y) -> float:
    """Ranked probability score for the ordered outcomes home, draw, away."""
    P = np.asarray(P, dtype=float)
    Y = _onehot(y, P.shape[1])
    diff = np.cumsum(P, axis=1) - np.cumsum(Y, axis=1)
    return float((diff[:, :-1] ** 2).sum(axis=1).mean() / (P.shape[1] - 1))


def accuracy(P, y) -> float:
    return float((np.asarray(P).argmax(axis=1) == np.asarray(y, dtype=int)).mean())


def balanced_accuracy(P, y) -> float:
    """Mean recall over the classes that occur in y."""
    pred = np.asarray(P).argmax(axis=1)
    y = np.asarray(y, dtype=int)
    return float(np.mean([(pred[y == k] == k).mean() for k in np.unique(y)]))


def binary_row_logloss(p, o) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    o = np.asarray(o, dtype=float)
    return -(o * np.log(p) + (1 - o) * np.log(1 - p))


def binary_logloss(p, o) -> float:
    return float(binary_row_logloss(p, o).mean())


def binary_brier(p, o) -> float:
    return float(((np.asarray(p, dtype=float) - np.asarray(o, dtype=float)) ** 2).mean())


def calibration_table(p, o, bins: int = 10) -> pd.DataFrame:
    """Equal-width probability bins: mean forecast, actual rate and count."""
    p = np.asarray(p, dtype=float)
    o = np.asarray(o, dtype=float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "mean_forecast": p[m].mean(), "actual_rate": o[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows)


def ece(p, o, bins: int = 10) -> float:
    """Expected calibration error: bin-weighted gap between forecast and actual rate."""
    t = calibration_table(p, o, bins)
    if t.empty:
        return float("nan")
    w = t["n"] / t["n"].sum()
    return float((w * (t["mean_forecast"] - t["actual_rate"]).abs()).sum())


def ece_interval(p, o, bins: int = 10, n_boot: int = 1000, seed: int = 0) -> dict:
    """ECE with a bootstrap 95% interval. With few matches per bin ECE is biased upward,
    so compare intervals, not single values."""
    p = np.asarray(p, dtype=float)
    o = np.asarray(o, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(p)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        vals[b] = ece(p[i], o[i], bins)
    return {"ece": ece(p, o, bins), "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5))}


def paired_bootstrap(diff, n_boot: int = 4000, seed: int = 0, chunk: int = 500) -> dict:
    """Bootstrap the mean of per-match loss differences (model A minus model B).

    A negative mean means A is better. If the 95% interval contains zero, the
    sample cannot tell the two models apart.
    """
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    rng = np.random.default_rng(seed)
    means = []
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(size, n))
        means.append(diff[idx].mean(axis=1))
        done += size
    means = np.concatenate(means)
    return {"mean": float(diff.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": n}
