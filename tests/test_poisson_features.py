import numpy as np
import pandas as pd
import pytest

from footmodel.evaluation.walkforward import walk_forward
from footmodel.features import FORM_COLUMNS, add_form_features
from footmodel.models import add_elo
from footmodel.models.poisson_features import PoissonFeatures, fit_rho
from test_dixon_coles import _simulate


def _with_shots(n_seasons=3, per_season=120, seed=5):
    d = _simulate(n_seasons, per_season, seed)
    rng = np.random.default_rng(seed)
    d["home_shots"] = d["home_goals"] * 3 + rng.integers(4, 12, len(d))
    d["away_shots"] = d["away_goals"] * 3 + rng.integers(3, 10, len(d))
    d["home_shots_on_target"] = d["home_goals"] + rng.integers(1, 5, len(d))
    d["away_shots_on_target"] = d["away_goals"] + rng.integers(1, 4, len(d))
    return add_elo(d)


def test_features_use_only_the_past():
    d = _with_shots()
    f1 = add_form_features(d)
    d2 = d.copy()
    last = d2.index[-1]
    d2.loc[last, ["home_goals", "away_goals", "home_shots", "away_shots"]] = [9, 0, 40, 1]
    f2 = add_form_features(d2)
    # every feature, including the last match's own, is recorded before its result
    assert np.array_equal(f1[FORM_COLUMNS].to_numpy(), f2[FORM_COLUMNS].to_numpy())


def test_first_match_starts_from_generic_values_and_rest_is_capped():
    f = add_form_features(_with_shots())
    first = f.iloc[0]
    assert first["h_sf"] == pytest.approx(12.0) and first["a_gf"] == pytest.approx(1.4)
    assert first["h_rest"] == 14.0
    assert (f["h_rest"] <= 14).all() and (f["a_rest"] >= 0).all()


def test_features_move_toward_recent_results():
    d = _with_shots(1, 60)
    f = add_form_features(d, halflife=4)
    team = d.iloc[0]["home_team"]
    rows = f[(f["home_team"] == team)]
    # after its first match the team's goals-for estimate has moved off the starting value
    assert rows["h_gf"].iloc[1] != rows["h_gf"].iloc[0]


def test_new_team_flag():
    d = _with_shots(3, 100)
    f = add_form_features(d)
    assert (f.loc[f["season_start"] == f["season_start"].min(), ["new_home", "new_away"]] == 0).all().all()
    # in later seasons the flag is 1 only for a team absent from the previous season
    prev = set(d.loc[d["season_start"] == 2014, "home_team"]) | set(d.loc[d["season_start"] == 2014, "away_team"])
    s15 = f[f["season_start"] == 2015]
    expected = ~s15["home_team"].isin(prev)
    assert (s15["new_home"] == expected.astype(float)).all()


def test_unsorted_input_is_refused():
    with pytest.raises(ValueError):
        add_form_features(_with_shots().iloc[::-1])


def test_fit_rho_finds_negative_correlation_when_low_scores_are_overrepresented():
    rng = np.random.default_rng(0)
    lam = np.full(5000, 1.4)
    mu = np.full(5000, 1.1)
    x, y = rng.poisson(lam).astype(float), rng.poisson(mu).astype(float)
    # add extra 0-0 and 1-1 draws
    x[:600], y[:600] = 0.0, 0.0
    x[600:1000], y[600:1000] = 1.0, 1.0
    assert fit_rho(x, y, lam, mu) < 0


def test_model_gives_valid_forecasts_and_no_future_leakage():
    d = add_form_features(_with_shots(3, 120))
    preds = walk_forward(d, [PoissonFeatures()], first_test_start=2016)
    p = preds[["p_h", "p_d", "p_a"]].to_numpy()
    assert np.allclose(p.sum(axis=1), 1) and ((p >= 0) & (p <= 1)).all()
    for c in ["p_o15", "p_o25", "p_btts", "exp_h", "exp_a", "p_score"]:
        assert preds[c].notna().all()
    assert (preds["p_o25"] <= preds["p_o15"]).all()

    # changing the last result must not change any other forecast
    d2 = _with_shots(3, 120)
    last = d2.index[-1]
    d2.loc[last, ["home_goals", "away_goals", "home_shots", "away_shots"]] = [9, 0, 40, 1]
    d2["total_goals"] = d2.home_goals + d2.away_goals
    d2 = add_form_features(add_elo(d2.drop(columns=["elo_home", "elo_away", "elo_diff"])))
    preds2 = walk_forward(d2, [PoissonFeatures()], first_test_start=2016)
    keep = preds["match_id"] != d.loc[last, "match_id"]
    assert np.allclose(preds.loc[keep, ["p_h", "p_o25"]].to_numpy(), preds2.loc[keep, ["p_h", "p_o25"]].to_numpy())
