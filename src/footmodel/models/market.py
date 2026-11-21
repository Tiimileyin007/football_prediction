"""Bookmaker odds as probabilities.

demargin() removes the bookmaker margin from a set of prices. MarketModel wraps
one set of odds columns as a model so it can be scored like any other model.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from ..data.loader import ODDS_1X2_TRIPLES

METHODS = ("proportional", "power", "shin")


def _proportional(pi: np.ndarray) -> np.ndarray:
    return pi / pi.sum()


def _power(pi: np.ndarray) -> np.ndarray:
    """Raise each implied probability to a common power k so they sum to 1."""
    try:
        k = brentq(lambda k: (pi ** k).sum() - 1.0, 0.05, 10.0)
    except ValueError:
        return _proportional(pi)
    return pi ** k


def _shin(pi: np.ndarray) -> np.ndarray:
    """Shin's method: assumes a share z of money comes from insiders."""
    total = pi.sum()

    def q_of(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / total) - z) / (2.0 * (1.0 - z))

    try:
        z = brentq(lambda z: q_of(z).sum() - 1.0, 1e-9, 0.4)
    except ValueError:
        return _proportional(pi)
    q = q_of(z)
    return q / q.sum()


_FUNCS = {"proportional": _proportional, "power": _power, "shin": _shin}


def demargin(odds, method: str = "power") -> np.ndarray:
    """Convert decimal odds (rows of 2 or 3 prices) to probabilities summing to 1.

    Rows with a missing price, or a price of 1.0 or less, come back as NaN.
    """
    if method not in _FUNCS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    o = np.asarray(odds, dtype=float)
    single = o.ndim == 1
    if single:
        o = o[None, :]
    out = np.full(o.shape, np.nan)
    ok = np.isfinite(o).all(axis=1) & (o > 1.0).all(axis=1)
    fn = _FUNCS[method]
    for i in np.flatnonzero(ok):
        out[i] = fn(1.0 / o[i])
    return out[0] if single else out


def overround(odds) -> np.ndarray:
    """Bookmaker margin: sum of implied probabilities minus 1 (NaN if a price is missing)."""
    o = np.asarray(odds, dtype=float)
    return (1.0 / o).sum(axis=-1) - 1.0


class MarketModel:
    """A set of bookmaker odds used as a forecast. It learns nothing from training data."""

    def __init__(self, name: str, triple: str, ou: tuple[str, str] | None = None, method: str = "power"):
        self.name = name
        self.triple = ODDS_1X2_TRIPLES[triple]
        self.ou = ou
        self.method = method

    def fit(self, train):
        return self

    def predict_1x2(self, test) -> np.ndarray:
        return demargin(test[list(self.triple)].to_numpy(dtype=float), self.method)

    def predict_goals(self, test) -> dict:
        if self.ou is None:
            return {}
        both = demargin(test[list(self.ou)].to_numpy(dtype=float), self.method)
        return {"over_2_5": both[:, 0]}
