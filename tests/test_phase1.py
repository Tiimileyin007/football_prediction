import numpy as np
import pandas as pd
import pytest

from footmodel.evaluation import metrics as M
from footmodel.evaluation.walkforward import common_matches, walk_forward
from footmodel.models import BaseRate, EloLogit, MarketModel, add_elo, demargin, overround


# ---------- metrics ----------

def test_uniform_forecast_scores():
    P = np.full((4, 3), 1 / 3)
    y = np.array([0, 1, 2, 0])
    assert M.logloss(P, y) == pytest.approx(np.log(3))
    assert M.brier(P, y) == pytest.approx(6 / 9)
    assert M.rps(np.full((1, 3), 1 / 3), np.array([0])) == pytest.approx(5 / 18)


def test_perfect_forecast():
    P = np.eye(3)[[0, 1, 2]]
    y = np.array([0, 1, 2])
    assert M.logloss(P, y) < 1e-9
    assert M.brier(P, y) == 0
    assert M.rps(P, y) == 0
    assert M.accuracy(P, y) == 1
    assert M.balanced_accuracy(P, y) == 1


def test_rps_penalises_distance():
    # Forecasting home when the result is away is worse than forecasting draw.
    home = np.array([[1.0, 0.0, 0.0]])
    draw = np.array([[0.0, 1.0, 0.0]])
    assert M.rps(home, np.array([2])) > M.rps(draw, np.array([2]))


def test_ece_zero_when_calibrated_and_positive_when_not():
    p = np.full(1000, 0.7)
    o = np.r_[np.ones(700), np.zeros(300)]
    assert M.ece(p, o) == pytest.approx(0, abs=1e-9)
    assert M.ece(np.full(1000, 0.9), o) == pytest.approx(0.2)


def test_paired_bootstrap_interval_contains_mean_and_excludes_zero_for_clear_gap():
    d = np.full(200, -0.05) + np.random.default_rng(1).normal(0, 0.01, 200)
    r = M.paired_bootstrap(d, n_boot=500)
    assert r["lo"] <= r["mean"] <= r["hi"]
    assert r["hi"] < 0


# ---------- margin removal ----------

@pytest.mark.parametrize("method", ["proportional", "power", "shin"])
def test_demargin_sums_to_one(method):
    odds = np.array([[1.80, 3.60, 4.50], [2.10, 3.40, 3.60], [1.30, 5.50, 11.0]])
    q = demargin(odds, method)
    assert q.sum(axis=1) == pytest.approx(np.ones(3))
    assert (q > 0).all()
    # lower price means higher probability
    assert (np.argsort(odds, axis=1) == np.argsort(-q, axis=1)).all()


def test_demargin_fair_odds_unchanged_and_two_way_works():
    fair = np.array([2.0, 4.0, 4.0])
    assert demargin(fair, "power") == pytest.approx([0.5, 0.25, 0.25])
    q = demargin(np.array([[1.90, 1.90]]), "shin")
    assert q[0] == pytest.approx([0.5, 0.5])


def test_demargin_missing_prices_give_nan():
    q = demargin(np.array([[2.0, np.nan, 3.0], [0.9, 3.0, 3.0]]))
    assert np.isnan(q).all()


def test_overround():
    assert overround(np.array([[2.0, 3.0, 6.0]])) == pytest.approx([0.0])
    assert overround(np.array([1.9, 3.5, 4.2]))[()] > 0


# ---------- Elo ----------

def _toy_matches(n_seasons=3, per_season=40, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    day = pd.Timestamp("2014-08-01")
    for s in range(n_seasons):
        for i in range(per_season):
            h, a = rng.choice(teams, 2, replace=False)
            hg, ag = rng.poisson(1.5), rng.poisson(1.1)
            day += pd.Timedelta(days=3)
            rows.append({"season_start": 2014 + s, "season": f"{2014 + s}/{15 + s}", "date": day,
                         "home_team": h, "away_team": a, "home_goals": hg, "away_goals": ag,
                         "match_id": f"{s}-{i}"})
    d = pd.DataFrame(rows)
    d["total_goals"] = d.home_goals + d.away_goals
    d["result_code"] = np.where(d.home_goals > d.away_goals, 0, np.where(d.home_goals == d.away_goals, 1, 2)).astype(float)
    d["over_1_5"] = (d.total_goals >= 2).astype("Int64")
    d["over_2_5"] = (d.total_goals >= 3).astype("Int64")
    d["btts"] = ((d.home_goals > 0) & (d.away_goals > 0)).astype("Int64")
    return d


def test_elo_first_matches_start_equal_and_use_only_the_past():
    d = _toy_matches()
    e = add_elo(d)
    assert e["elo_diff"].iloc[0] == 0
    # Change the result of the last match: no earlier pre-match rating may move.
    d2 = d.copy()
    d2.loc[d2.index[-1], "home_goals"] += 5
    e2 = add_elo(d2)
    assert (e["elo_diff"].iloc[:-1].to_numpy() == e2["elo_diff"].iloc[:-1].to_numpy()).all()


def test_elo_rewards_winner():
    d = _toy_matches(1, 10)
    d.loc[d.index[0], ["home_goals", "away_goals"]] = [3, 0]
    h, a = d.loc[d.index[0], ["home_team", "away_team"]]
    e = add_elo(d)
    later = e[(e["home_team"] == h) & (e.index > e.index[0])]
    if len(later):
        assert later["elo_home"].iloc[0] > 1500


def test_elo_requires_sorted_dates():
    d = _toy_matches()
    with pytest.raises(ValueError):
        add_elo(d.iloc[::-1])


# ---------- walk-forward ----------

class Spy:
    name = "spy"

    def __init__(self):
        self.seen = []

    def fit(self, train):
        self.seen.append(int(train["season_start"].max()))
        return self

    def predict_1x2(self, test):
        self.seen.append(-int(test["season_start"].min()))
        return np.full((len(test), 3), 1 / 3)

    def predict_goals(self, test):
        return {}


def test_walk_forward_never_trains_on_the_test_season_or_later():
    d = _toy_matches(4, 20)
    spy = Spy()
    preds = walk_forward(d, [spy], first_test_start=2015)
    # pairs of (max training season, -test season): training is always earlier
    pairs = list(zip(spy.seen[0::2], spy.seen[1::2]))
    assert pairs and all(tr < -te for tr, te in pairs)
    assert sorted(preds["season_start"].unique()) == [2015, 2016, 2017]
    assert preds["model"].eq("spy").all()


def test_walk_forward_with_real_models_gives_valid_probabilities():
    d = add_elo(_toy_matches(4, 60))
    preds = walk_forward(d, [BaseRate(), BaseRate(2), EloLogit()], first_test_start=2015)
    p = preds[["p_h", "p_d", "p_a"]].to_numpy()
    assert np.allclose(p.sum(axis=1), 1)
    assert ((p >= 0) & (p <= 1)).all()
    base = preds[preds["model"] == "base_rate_all"]
    assert base["p_o25"].between(0, 1).all()


def test_common_matches_drops_rows_a_model_cannot_forecast():
    d = _toy_matches(3, 20)
    d["ps_close_h"], d["ps_close_d"], d["ps_close_a"] = 2.0, 3.5, 4.0
    d.loc[d.index[-5:], "ps_close_h"] = np.nan
    m = MarketModel("m", "ps_close")
    preds = walk_forward(d, [BaseRate(), m], first_test_start=2015)
    common = common_matches(preds, ["base_rate_all", "m"], "p_h")
    assert common["match_id"].nunique() == preds["match_id"].nunique() - 5
