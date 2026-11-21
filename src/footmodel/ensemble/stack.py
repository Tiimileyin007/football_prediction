"""Combine model forecasts and calibrate them.

Three tools, from simplest to most flexible:

* geometric_mean / logit_mean: equal-weight average of log-odds, nothing to fit.
* OutcomeStacker / BinaryStacker: logistic regression on the models' log-odds.
  This both weights the models and recalibrates the result.
* pool_1x2 / pool_binary with fit_pool_weight: blend a model with the market using
  ONE weight w in [0, 1] on log-odds. w = 0 means "trust the market fully",
  w = 1 means "trust the model fully". A weight near 0 says the model adds
  nothing the market lacks, and it is the shrinkage the value engine needs.

Every fit takes forecasts that were made out of sample, so a stacker trained on
earlier seasons never sees the season it is scored on.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def logodds_1x2(P: np.ndarray) -> np.ndarray:
    """log(home/draw) and log(away/draw): two numbers that carry a 1X2 forecast."""
    P = np.clip(np.asarray(P, dtype=float), EPS, 1.0)
    return np.column_stack([np.log(P[:, 0] / P[:, 1]), np.log(P[:, 2] / P[:, 1])])


def logit(p) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1.0 - p))


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _softmax_rows(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def geometric_mean(forecasts: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Normalised weighted geometric mean of several 1X2 forecasts."""
    w = np.ones(len(forecasts)) / len(forecasts) if weights is None else np.asarray(weights, dtype=float)
    logs = sum(wi * np.log(np.clip(np.asarray(P, dtype=float), EPS, 1.0)) for wi, P in zip(w, forecasts))
    return _softmax_rows(logs)


def logit_mean(ps: list[np.ndarray]) -> np.ndarray:
    return sigmoid(np.mean([logit(p) for p in ps], axis=0))


class OutcomeStacker:
    """Multinomial logistic regression on the log-odds of one or more 1X2 forecasts."""

    def __init__(self, C: float = 1.0):
        self.C = C
        self.lr = None

    def _x(self, forecasts):
        return np.hstack([logodds_1x2(P) for P in forecasts])

    def fit(self, forecasts: list[np.ndarray], y):
        self.lr = LogisticRegression(C=self.C, max_iter=3000).fit(self._x(forecasts), np.asarray(y, dtype=int))
        return self

    def predict(self, forecasts: list[np.ndarray]) -> np.ndarray:
        return self.lr.predict_proba(self._x(forecasts))


class BinaryStacker:
    """Logistic regression on the logits of one or more yes/no forecasts (Platt scaling for one input)."""

    def __init__(self, C: float = 1.0):
        self.C = C
        self.lr = None

    def _x(self, ps):
        return np.column_stack([logit(p) for p in ps])

    def fit(self, ps: list[np.ndarray], o):
        self.lr = LogisticRegression(C=self.C, max_iter=3000).fit(self._x(ps), np.asarray(o, dtype=int))
        return self

    def predict(self, ps: list[np.ndarray]) -> np.ndarray:
        return self.lr.predict_proba(self._x(ps))[:, 1]


def pool_1x2(market: np.ndarray, model: np.ndarray, w: float) -> np.ndarray:
    """Log-linear blend: probabilities proportional to market^(1-w) * model^w."""
    lm = np.log(np.clip(np.asarray(market, dtype=float), EPS, 1.0))
    lb = np.log(np.clip(np.asarray(model, dtype=float), EPS, 1.0))
    return _softmax_rows((1.0 - w) * lm + w * lb)


def pool_binary(market: np.ndarray, model: np.ndarray, w: float) -> np.ndarray:
    return sigmoid((1.0 - w) * logit(market) + w * logit(model))


def fit_pool_weight(market, model, outcome, kind: str, w_max: float = 1.0) -> float:
    """Weight on the model that minimises log loss. kind is "1x2" (outcome = class index) or "binary"."""
    market = np.asarray(market, dtype=float)
    model = np.asarray(model, dtype=float)
    outcome = np.asarray(outcome)

    if kind == "1x2":
        idx = np.arange(len(outcome))
        y = outcome.astype(int)

        def loss(w):
            return float(-np.log(np.clip(pool_1x2(market, model, w)[idx, y], EPS, 1.0)).mean())
    elif kind == "binary":
        o = outcome.astype(float)

        def loss(w):
            p = _clip(pool_binary(market, model, w))
            return float(-(o * np.log(p) + (1 - o) * np.log(1 - p)).mean())
    else:
        raise ValueError("kind must be '1x2' or 'binary'")
    return float(minimize_scalar(loss, bounds=(0.0, w_max), method="bounded").x)
