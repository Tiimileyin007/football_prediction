from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from footmodel.data.checks import missing_seasons, run_checks
from footmodel.data.loader import ODDS_COLUMNS, load_all, parse_dates, season_from_path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
needs_raw = pytest.mark.skipif(not list(RAW.glob("E0_*.csv")), reason="no raw season files")


def test_season_from_path():
    assert season_from_path("data/raw/E0_2012_13.csv") == (2012, "2012/13")
    with pytest.raises(ValueError):
        season_from_path("data/raw/2012-13.csv")


def test_parse_dates_handles_both_formats():
    s = pd.Series(["18/08/12", "08/08/2015", "bad"])
    d = parse_dates(s)
    assert d.iloc[0] == pd.Timestamp(2012, 8, 18)
    assert d.iloc[1] == pd.Timestamp(2015, 8, 8)
    assert pd.isna(d.iloc[2])


def test_missing_seasons_finds_gap():
    assert missing_seasons([2012, 2013, 2015, 2016]) == [2014]
    assert missing_seasons([2012, 2013]) == []
    assert missing_seasons([]) == []


@pytest.fixture(scope="module")
def matches():
    return load_all(RAW)


@needs_raw
def test_no_error_level_checks_fail(matches):
    failed = [str(c) for c in run_checks(matches) if not c.passed and c.severity == "error"]
    assert not failed, failed


@needs_raw
def test_all_clean_columns_exist(matches):
    for col in ODDS_COLUMNS:
        assert col in matches.columns
    for col in ["match_id", "season", "date", "home_team", "away_team", "total_goals", "btts", "over_2_5"]:
        assert col in matches.columns


@needs_raw
def test_sorted_by_date(matches):
    assert matches["date"].is_monotonic_increasing


@needs_raw
def test_targets_are_consistent(matches):
    assert (matches["total_goals"] == matches["home_goals"] + matches["away_goals"]).all()
    assert set(matches["result_code"].unique()) <= {0, 1, 2}
    assert ((matches["btts"] == 1) == ((matches["home_goals"] > 0) & (matches["away_goals"] > 0))).all()


@needs_raw
def test_pinnacle_closing_present_for_old_seasons(matches):
    old = matches[matches["season_start"] <= 2018]
    assert old[["ps_close_h", "ps_close_d", "ps_close_a"]].notna().all().all()
