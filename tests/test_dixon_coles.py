import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from scipy.stats import poisson

from footmodel.evaluation.walkforward import walk_forward
from footmodel.models import BaseRate
from footmodel.models.dixon_coles import (
    DixonColes, _arrays, fit_dc, make_objective, markets, rates, score_matrices)


def _simulate(n_seasons=4, per_season=190, seed=3):
    """Matches from known strengths so the fit can be checked against the truth."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(10)]
    att = np.linspace(0.35, -0.35, 10)
    dfn = np.linspace(0.25, -0.25, 10)
    rows = []
    day = pd.Timestamp("2014-08-01")
    for s in range(n_seasons):
        for i in range(per_season):
            h, a = rng.choice(10, 2, replace=False)
            lam = np.exp(0.1 + 0.3 + att[h] - dfn[a])
            mu = np.exp(0.1 + att[a] - dfn[h])
            hg, ag = rng.poisson(lam), rng.poisson(mu)
            day += pd.Timedelta(days=2)
            rows.append({"season_start": 2014 + s, "season": f"{2014 + s}/{15 + s}", "date": day,
                         "home_team": teams[h], "away_team": teams[a], "home_goals": hg, "away_goals": ag,
                         "match_id": f"{s}-{i}"})
    d = pd.DataFrame(rows)
    d["total_goals"] = d.home_goals + d.away_goals
    d["result_code"] = np.where(d.home_goals > d.away_goals, 0, np.where(d.home_goals == d.away_goals, 1, 2)).astype(float)
    d["over_1_5"] = (d.total_goals >= 2).astype("Int64")
    d["over_2_5"] = (d.total_goals >= 3).astype("Int64")
    d["btts"] = ((d.home_goals > 0) & (d.away_goals > 0)).astype("Int64")
    return d


def test_gradient_matches_numerical_gradient():
    d = _simulate(1, 120)
    teams, idx, hi, ai, x, y, w = _arrays(d, d["date"].max(), 0.002)
    f = make_objective(hi, ai, x, y, w, len(teams), 4.0)
    p0 = np.random.default_rng(0).normal(0, 0.1, 2 * len(teams) + 3)
    p0[-1] = 0.05
    err = check_grad(lambda q: f(q)[0], lambda q: f(q)[1], p0)
    assert err < 1e-4


def test_fit_recovers_home_advantage_and_team_order():
    d = _simulate(4, 190)
    p = fit_dc(d, d["date"].max() + pd.Timedelta(days=1), xi=0.0, ridge=1.0)
    assert 0.15 < p.h < 0.45                      # true value 0.3
    strength = {t: p.attack[i] + p.defence[i] for t, i in p.teams.items()}
    order = sorted(strength, key=strength.get, reverse=True)
    assert order[0] in ("T0", "T1", "T2") and order[-1] in ("T7", "T8", "T9")


def test_score_matrix_is_a_probability_distribution():
    lam, mu = np.array([1.6, 0.9, 2.4]), np.array([1.1, 1.3, 0.7])
    M = score_matrices(lam, mu, rho=-0.08)
    assert M.shape == (3, 11, 11)
    assert M.sum(axis=(1, 2)) == pytest.approx(np.ones(3))
    assert (M >= 0).all()
    m = markets(M)
    assert m["p_h"] + m["p_d"] + m["p_a"] == pytest.approx(np.ones(3))
    assert (m["over_2_5"] <= m["over_1_5"]).all()
    assert m["exp_home"] == pytest.approx(lam, rel=0.02)


def test_rho_zero_is_independent_poisson_and_negative_rho_lifts_low_draws():
    lam, mu = np.array([1.5]), np.array([1.2])
    M0 = score_matrices(lam, mu, rho=0.0)
    assert M0[0, 2, 1] == pytest.approx(poisson.pmf(2, 1.5) * poisson.pmf(1, 1.2), rel=1e-4)
    M1 = score_matrices(lam, mu, rho=-0.1)
    assert M1[0, 0, 0] > M0[0, 0, 0] and M1[0, 1, 1] > M0[0, 1, 1]
    assert M1[0, 1, 0] < M0[0, 1, 0]


def test_equal_teams_give_symmetric_1x2():
    m = markets(score_matrices(np.array([1.3]), np.array([1.3]), rho=-0.05))
    assert m["p_h"] == pytest.approx(m["p_a"])


def test_unseen_team_gets_average_strength():
    d = _simulate(2, 100)
    p = fit_dc(d, d["date"].max(), 0.001, 8.0)
    lam, mu = rates(p, pd.Series(["NEW"]), pd.Series(["T3"]))
    assert 0.3 < lam[0] < 5 and 0.3 < mu[0] < 5


def test_forecasts_never_use_results_from_the_future():
    d = _simulate(3, 120)
    model = DixonColes(xi=0.001, ridge=8.0, refit_days=14)
    base = walk_forward(d, [model], first_test_start=2016)
    d2 = d.copy()
    last = d2.index[-1]
    d2.loc[last, ["home_goals", "away_goals"]] = [9, 0]   # change the very last result
    changed = walk_forward(d2, [DixonColes(xi=0.001, ridge=8.0, refit_days=14)], first_test_start=2016)
    cols = ["p_h", "p_d", "p_a", "p_o25", "p_btts"]
    # every forecast except the changed match itself must be identical
    keep = base["match_id"] != d.loc[last, "match_id"]
    assert np.allclose(base.loc[keep, cols].to_numpy(), changed.loc[keep, cols].to_numpy())


def test_walk_forward_outputs_all_markets_and_valid_probabilities():
    d = _simulate(3, 120)
    preds = walk_forward(d, [DixonColes(refit_days=28), BaseRate()], first_test_start=2016)
    dc = preds[preds["model"] == "dixon_coles"]
    assert np.allclose(dc[["p_h", "p_d", "p_a"]].sum(axis=1), 1)
    for c in ["p_o15", "p_o25", "p_btts", "exp_h", "exp_a", "p_score"]:
        assert dc[c].notna().all()
    assert dc["p_score"].between(0, 1).all()
    assert (dc["p_o25"] <= dc["p_o15"]).all()
