"""Poisson regression for home and away goals from pre-match features.

Two regularised Poisson regressions give the home and away scoring rates from
the Elo difference, each team's recent shots, shots on target and goals (for and
against), rest days and new-to-league flags. A Dixon–Coles low-score correction,
with rho fitted on the training data, turns the two rates into one scoreline
matrix, from which every market is read exactly as in the Dixon–Coles model.

The features must already be on the match table (add_elo and add_form_features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from ..features.form import FORM_COLUMNS
from .dixon_coles import markets, score_matrices

FEATURES = ["elo_diff"] + FORM_COLUMNS


def fit_rho(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, bound: float = 0.25) -> float:
    """Maximum-likelihood Dixon–Coles rho for given scoring rates."""
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    def nll(rho: float) -> float:
        tau = np.ones_like(lam)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        return float(-np.log(np.maximum(tau, 1e-9)).sum())

    return float(minimize_scalar(nll, bounds=(-bound, bound), method="bounded").x)


class PoissonFeatures:
    def __init__(self, alpha: float = 0.01, max_goals: int = 10, features: list[str] | None = None,
                 name: str = "poisson_features"):
        self.alpha = alpha
        self.max_goals = max_goals
        self.features = features or FEATURES
        self.name = name
        self.scaler = self.home = self.away = None
        self.rho = 0.0

    def fit(self, train: pd.DataFrame):
        X = train[self.features].to_numpy(dtype=float)
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        x = train["home_goals"].to_numpy(dtype=float)
        y = train["away_goals"].to_numpy(dtype=float)
        self.home = PoissonRegressor(alpha=self.alpha, max_iter=2000).fit(Z, x)
        self.away = PoissonRegressor(alpha=self.alpha, max_iter=2000).fit(Z, y)
        self.rho = fit_rho(x, y, self.home.predict(Z), self.away.predict(Z))
        return self

    def _predict(self, test: pd.DataFrame) -> dict:
        Z = self.scaler.transform(test[self.features].to_numpy(dtype=float))
        lam, mu = self.home.predict(Z), self.away.predict(Z)
        M = score_matrices(lam, mu, self.rho, self.max_goals)
        out = markets(M)
        out["score_probs"] = M
        out["lam"], out["mu"] = lam, mu
        return out

    def predict_1x2(self, test: pd.DataFrame) -> np.ndarray:
        o = self._predict(test)
        return np.column_stack([o["p_h"], o["p_d"], o["p_a"]])

    def predict_goals(self, test: pd.DataFrame) -> dict:
        return self._predict(test)
