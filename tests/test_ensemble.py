import numpy as np
import pandas as pd
import pytest

from footmodel.ensemble import (BinaryStacker, OutcomeStacker, fit_pool_weight, geometric_mean, logit_mean,
                                pool_1x2, pool_binary)
from footmodel.ensemble.walkforward import GOAL_MODELS, MARKET, ONE_X_TWO_MODELS, run_ensemble, to_long
from footmodel.evaluation import metrics as M


def _truth(n, seed):
    rng = np.random.default_rng(seed)
    logits = np.column_stack([rng.normal(0.2, 0.7, n), np.zeros(n), rng.normal(-0.2, 0.7, n)])
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=pi) for pi in p])
    return p, y, rng


def _sharpen(p, k):
    q = p ** k
    return q / q.sum(axis=1, keepdims=True)


def test_geometric_mean_of_identical_forecasts_is_unchanged():
    p = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    assert geometric_mean([p, p, p]) == pytest.approx(p)
    assert geometric_mean([p, _sharpen(p, 2)]).sum(axis=1) == pytest.approx([1, 1])
    q = np.array([0.7, 0.2])
    assert logit_mean([q, q]) == pytest.approx(q)


def test_stacker_repairs_an_overconfident_model():
    p, y, _ = _truth(6000, 1)
    over = _sharpen(p, 2.5)
    tr, te = slice(0, 3000), slice(3000, 6000)
    st = OutcomeStacker(C=10).fit([over[tr]], y[tr])
    assert M.logloss(st.predict([over[te]]), y[te]) < M.logloss(over[te], y[te]) - 0.02


def test_binary_stacker_removes_a_bias():
    rng = np.random.default_rng(2)
    true = rng.uniform(0.3, 0.7, 6000)
    o = (rng.uniform(size=6000) < true).astype(int)
    biased = np.clip(true - 0.08, 0.01, 0.99)
    st = BinaryStacker(C=10).fit([biased[:3000]], o[:3000])
    pred = st.predict([biased[3000:]])
    assert abs(pred.mean() - o[3000:].mean()) < abs(biased[3000:].mean() - o[3000:].mean())


def test_pool_endpoints():
    m = np.array([[0.5, 0.3, 0.2]])
    b = np.array([[0.2, 0.3, 0.5]])
    assert pool_1x2(m, b, 0.0) == pytest.approx(m)
    assert pool_1x2(m, b, 1.0) == pytest.approx(b)
    assert pool_binary(np.array([0.6]), np.array([0.4]), 0.0) == pytest.approx([0.6])
    assert pool_binary(np.array([0.6]), np.array([0.4]), 1.0) == pytest.approx([0.4])


def test_pool_weight_is_near_zero_for_a_noise_model_and_high_for_a_better_one():
    p, y, rng = _truth(8000, 3)
    market = p                                    # the market knows the truth
    noise = _sharpen(np.clip(p + rng.normal(0, 0.12, p.shape), 0.02, 0.98), 1.0)
    noise = noise / noise.sum(axis=1, keepdims=True)
    assert fit_pool_weight(market, noise, y, "1x2") < 0.2
    weak_market = _sharpen(p, 0.3)                # market is washed out; the model has the truth
    assert fit_pool_weight(weak_market, p, y, "1x2") > 0.6


def test_ece_interval_contains_point_estimate():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.2, 0.8, 2000)
    o = (rng.uniform(size=2000) < p).astype(int)
    r = M.ece_interval(p, o, n_boot=200)
    assert r["lo"] <= r["ece"] <= r["hi"] + 1e-9 or r["lo"] <= r["hi"]


# ---------- walk-forward ensemble ----------

def _toy(n_seasons=5, per_season=300, seed=7):
    rng = np.random.default_rng(seed)
    n = n_seasons * per_season
    p, y, _ = _truth(n, seed)
    o25 = (rng.uniform(size=n) < 0.55).astype(float)
    o15 = np.maximum(o25, (rng.uniform(size=n) < 0.8).astype(float))
    btts = (rng.uniform(size=n) < 0.52).astype(float)
    hg, ag = rng.poisson(1.5, n), rng.poisson(1.2, n)
    meta = pd.DataFrame({
        "match_id": [f"m{i}" for i in range(n)], "season": "x", "season_start": np.repeat(np.arange(2014, 2014 + n_seasons), per_season),
        "date": pd.date_range("2014-08-01", periods=n, freq="D"), "result_code": y.astype(float),
        "home_goals": hg, "away_goals": ag, "total_goals": hg + ag,
        "over_1_5": o15, "over_2_5": o25, "btts": btts})
    A = {}
    for m, k in zip(ONE_X_TWO_MODELS, (0.8, 0.9, 1.1)):
        A[m] = {"p1x2": _sharpen(p, k)}
    A[MARKET] = {"p1x2": p, "o25": np.clip(0.55 + rng.normal(0, 0.05, n), 0.05, 0.95)}
    for m in GOAL_MODELS:
        A[m].update({"o15": np.clip(0.78 + rng.normal(0, 0.04, n), 0.05, 0.99), "o25": np.clip(0.52 + rng.normal(0, 0.05, n), 0.05, 0.95),
                     "btts": np.clip(0.5 + rng.normal(0, 0.04, n), 0.05, 0.95), "exp_h": 1.4 + rng.normal(0, 0.1, n), "exp_a": 1.1 + rng.normal(0, 0.1, n)})
    return meta, A


def test_ensemble_outputs_are_valid_and_only_in_test_seasons():
    meta, A = _toy()
    out, w = run_ensemble(meta, A, first_test_start=2017)
    te = meta["season_start"].to_numpy() >= 2017
    for k in ("avg3", "avg3_cal", "stack3", "pool_avg3"):
        assert np.allclose(out[k][te].sum(axis=1), 1)
        assert np.isnan(out[k][~te]).all()
    for k in ("o25_cal", "o25_pool", "o15_cal", "btts_cal"):
        assert ((out[k][te] > 0) & (out[k][te] < 1)).all()
    assert sorted(w["1x2"]) == [2017, 2018]
    assert all(0 <= v <= 1 for v in w["1x2"].values())


def test_ensemble_forecasts_never_depend_on_the_season_they_predict():
    meta, A = _toy(n_seasons=6)
    base, _ = run_ensemble(meta, A, first_test_start=2017)
    changed = meta.copy()
    mask = changed["season_start"] == 2018
    changed.loc[mask, "result_code"] = (changed.loc[mask, "result_code"] + 1) % 3   # rewrite 2018 outcomes
    changed.loc[mask, "over_2_5"] = 1 - changed.loc[mask, "over_2_5"]
    other, _ = run_ensemble(changed, A, first_test_start=2017)
    s18 = mask.to_numpy()
    for k in ("avg3_cal", "stack3", "pool_avg3", "o25_cal", "o25_pool", "exp_h_cal"):
        assert np.allclose(base[k][s18], other[k][s18])            # 2018 forecasts ignore 2018 results
    s17 = (meta["season_start"] == 2017).to_numpy()
    assert np.allclose(base["stack3"][s17], other["stack3"][s17])
    s19 = (meta["season_start"] == 2019).to_numpy()
    assert not np.allclose(base["stack3"][s19], other["stack3"][s19])  # but later seasons may learn from them


def test_to_long_layout():
    meta, A = _toy()
    out, _ = run_ensemble(meta, A, first_test_start=2017)
    long = to_long(meta, out)
    assert {"model", "p_h", "p_o25", "exp_h", "p_score", "result_code"} <= set(long.columns)
    assert set(long["model"]) >= {"ens_stack3", "ens_goals_cal", "ens_o25_pool"}
    assert long["season_start"].min() == 2017
