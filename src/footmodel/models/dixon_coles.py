"""Dixon–Coles scoreline model.

Home and away goals are Poisson with rates
    log(home rate) = c + h + attack[home] - defence[away]
    log(away rate) = c +     attack[away] - defence[home]
and the four lowest scorelines (0-0, 1-0, 0-1, 1-1) are corrected by a factor
that depends on rho, as in Dixon and Coles (1997). Matches are weighted by
exp(-xi * days_ago), so recent form counts more, and attack and defence are
shrunk toward zero (ridge), which keeps teams with little data near average.

One scoreline matrix per match gives every market: 1X2, Over 1.5, Over 2.5,
BTTS, expected goals and correct score, so the markets cannot contradict each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

COLS = ["date", "home_team", "away_team", "home_goals", "away_goals"]


@dataclass
class DCParams:
    teams: dict
    attack: np.ndarray
    defence: np.ndarray
    c: float
    h: float
    rho: float


def _arrays(hist: pd.DataFrame, ref_date, xi: float):
    teams = sorted(set(hist["home_team"]) | set(hist["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    hi = hist["home_team"].map(idx).to_numpy()
    ai = hist["away_team"].map(idx).to_numpy()
    x = hist["home_goals"].to_numpy(dtype=float)
    y = hist["away_goals"].to_numpy(dtype=float)
    days = np.clip((pd.Timestamp(ref_date) - hist["date"]).dt.days.to_numpy(dtype=float), 0, None)
    w = np.exp(-xi * days)
    return teams, idx, hi, ai, x, y, w


def make_objective(hi, ai, x, y, w, n_teams: int, ridge: float):
    """Negative weighted log-likelihood plus ridge, with its analytic gradient."""
    T = n_teams
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    def fun(p):
        att, dfn = p[:T], p[T:2 * T]
        c, h, rho = p[2 * T], p[2 * T + 1], p[2 * T + 2]
        lh = c + h + att[hi] - dfn[ai]
        la = c + att[ai] - dfn[hi]
        lam, mu = np.exp(lh), np.exp(la)

        tau = np.ones_like(lam)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        tau = np.maximum(tau, 1e-9)

        ll = w * (np.log(tau) + x * lh - lam + y * la - mu)

        g_lh = x - lam
        g_la = y - mu
        t00 = lam[m00] * mu[m00] * rho / tau[m00]
        g_lh[m00] -= t00
        g_la[m00] -= t00
        g_lh[m01] += lam[m01] * rho / tau[m01]
        g_la[m10] += mu[m10] * rho / tau[m10]
        g_lh *= w
        g_la *= w

        g_rho = np.zeros_like(lam)
        g_rho[m00] = -lam[m00] * mu[m00] / tau[m00]
        g_rho[m01] = lam[m01] / tau[m01]
        g_rho[m10] = mu[m10] / tau[m10]
        g_rho[m11] = -1.0 / tau[m11]

        g_att = np.bincount(hi, g_lh, T) + np.bincount(ai, g_la, T)
        g_def = -np.bincount(ai, g_lh, T) - np.bincount(hi, g_la, T)
        grad = np.concatenate([g_att, g_def, [g_lh.sum() + g_la.sum(), g_lh.sum(), (w * g_rho).sum()]])
        loss = -ll.sum() + ridge * (att @ att + dfn @ dfn)
        grad = -grad
        grad[:T] += 2.0 * ridge * att
        grad[T:2 * T] += 2.0 * ridge * dfn
        return loss, grad

    return fun


def fit_dc(hist: pd.DataFrame, ref_date, xi: float, ridge: float, rho_bound: float = 0.25) -> DCParams:
    teams, idx, hi, ai, x, y, w = _arrays(hist, ref_date, xi)
    T = len(teams)
    fun = make_objective(hi, ai, x, y, w, T, ridge)
    p0 = np.zeros(2 * T + 3)
    p0[2 * T] = np.log(max(x.mean(), 0.5))
    bounds = [(None, None)] * (2 * T + 2) + [(-rho_bound, rho_bound)]
    res = minimize(fun, p0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    p = res.x
    return DCParams(idx, p[:T], p[T:2 * T], float(p[2 * T]), float(p[2 * T + 1]), float(p[2 * T + 2]))


def rates(params: DCParams, home: pd.Series, away: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Home and away scoring rates. A team never seen in the history gets an average attack and defence."""
    def get(arr, names):
        return np.array([arr[params.teams[t]] if t in params.teams else 0.0 for t in names])

    lam = np.exp(params.c + params.h + get(params.attack, home) - get(params.defence, away))
    mu = np.exp(params.c + get(params.attack, away) - get(params.defence, home))
    return lam, mu


def score_matrices(lam: np.ndarray, mu: np.ndarray, rho: float, max_goals: int = 10) -> np.ndarray:
    """Probability of every scoreline, shape (n, max_goals + 1, max_goals + 1) with home goals first."""
    k = np.arange(max_goals + 1)
    pl = poisson.pmf(k[None, :], lam[:, None])
    pm = poisson.pmf(k[None, :], mu[:, None])
    M = pl[:, :, None] * pm[:, None, :]
    M[:, 0, 0] *= 1.0 - lam * mu * rho
    M[:, 0, 1] *= 1.0 + lam * rho
    M[:, 1, 0] *= 1.0 + mu * rho
    M[:, 1, 1] *= 1.0 - rho
    return M / M.sum(axis=(1, 2), keepdims=True)


def markets(M: np.ndarray) -> dict:
    """Every market from the scoreline matrices."""
    G = M.shape[1]
    i, j = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    total = i + j
    return {
        "p_h": (M * (i > j)).sum(axis=(1, 2)),
        "p_d": (M * (i == j)).sum(axis=(1, 2)),
        "p_a": (M * (i < j)).sum(axis=(1, 2)),
        "over_1_5": (M * (total >= 2)).sum(axis=(1, 2)),
        "over_2_5": (M * (total >= 3)).sum(axis=(1, 2)),
        "btts": (M * ((i >= 1) & (j >= 1))).sum(axis=(1, 2)),
        "exp_home": (M * i).sum(axis=(1, 2)),
        "exp_away": (M * j).sum(axis=(1, 2)),
    }


class DixonColes:
    """Walk-forward Dixon–Coles. Refits every `refit_days` days inside the test season,
    using only results dated before each block, as a live system would."""

    def __init__(self, xi: float = 0.002, ridge: float = 8.0, refit_days: int = 14,
                 max_goals: int = 10, name: str | None = None):
        self.xi, self.ridge, self.refit_days, self.max_goals = xi, ridge, refit_days, max_goals
        self.name = name or "dixon_coles"
        self.train = None
        self._cache_key = None
        self._cache = None
        self.last_rho = None

    def fit(self, train: pd.DataFrame):
        self.train = train[COLS].copy()
        self._cache_key = None
        return self

    def _predict(self, test: pd.DataFrame) -> dict:
        key = (len(test), tuple(test["match_id"].iloc[[0, -1]]), self.train["date"].max())
        if key == self._cache_key:
            return self._cache
        n = len(test)
        block = ((test["date"] - test["date"].min()).dt.days // self.refit_days).to_numpy()
        lam, mu, rho = np.zeros(n), np.zeros(n), np.zeros(n)
        test_cols = test[COLS]
        for b in np.unique(block):
            m = block == b
            start = test["date"][m].min()
            hist = pd.concat([self.train, test_cols[test["date"] < start]], ignore_index=True)
            params = fit_dc(hist, start, self.xi, self.ridge)
            lam[m], mu[m] = rates(params, test["home_team"][m], test["away_team"][m])
            rho[m] = params.rho
            self.last_rho = params.rho
        M = np.concatenate([
            score_matrices(lam[block == b], mu[block == b], float(rho[block == b][0]), self.max_goals)
            for b in np.unique(block)
        ])
        order = np.concatenate([np.flatnonzero(block == b) for b in np.unique(block)])
        M_sorted = np.empty_like(M)
        M_sorted[order] = M
        out = markets(M_sorted)
        out["score_probs"] = M_sorted
        out["lam"], out["mu"] = lam, mu
        self._cache_key, self._cache = key, out
        return out

    def predict_1x2(self, test: pd.DataFrame) -> np.ndarray:
        o = self._predict(test)
        return np.column_stack([o["p_h"], o["p_d"], o["p_a"]])

    def predict_goals(self, test: pd.DataFrame) -> dict:
        return self._predict(test)
