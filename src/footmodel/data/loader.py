"""Load football-data.co.uk Premier League season files into one clean table.

The raw files change shape over the years (different bookmakers, different
column names for the average and maximum odds, two date formats, a Time column
that only appears from 2020/21). This module maps every season onto one set of
clean column names so later stages never touch raw column names.

Odds naming used in the clean table
-----------------------------------
<book>_open_*   price in the main (pre-match) odds columns of the file
<book>_close_*  price in the closing-odds columns (not in every season)

ps    = Pinnacle
b365  = Bet365
avg   = market average. 2012/13-2018/19 files carry the Betbrain average
        (BbAv*), later files carry the football-data average (Avg*). The two
        are close but not identical, so `odds_source` records which era a row
        comes from.
max   = best available price (BbMx* in the old era, Max* later)
ou    = Over/Under 2.5 goals
ah    = Asian handicap
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# clean name -> raw column in the files
MATCH_COLUMNS = {
    "date": "Date",
    "time": "Time",
    "home_team": "HomeTeam",
    "away_team": "AwayTeam",
    "home_goals": "FTHG",
    "away_goals": "FTAG",
    "result": "FTR",
    "ht_home_goals": "HTHG",
    "ht_away_goals": "HTAG",
    "referee": "Referee",
    "home_shots": "HS",
    "away_shots": "AS",
    "home_shots_on_target": "HST",
    "away_shots_on_target": "AST",
    "home_fouls": "HF",
    "away_fouls": "AF",
    "home_corners": "HC",
    "away_corners": "AC",
    "home_yellows": "HY",
    "away_yellows": "AY",
    "home_reds": "HR",
    "away_reds": "AR",
}

# clean name -> candidate raw columns, first one present in a file wins
ODDS_COLUMNS = {
    # 1X2, Pinnacle
    "ps_open_h": ["PSH"], "ps_open_d": ["PSD"], "ps_open_a": ["PSA"],
    "ps_close_h": ["PSCH"], "ps_close_d": ["PSCD"], "ps_close_a": ["PSCA"],
    # 1X2, Bet365
    "b365_open_h": ["B365H"], "b365_open_d": ["B365D"], "b365_open_a": ["B365A"],
    "b365_close_h": ["B365CH"], "b365_close_d": ["B365CD"], "b365_close_a": ["B365CA"],
    # 1X2, market average and best price
    "avg_open_h": ["AvgH", "BbAvH"], "avg_open_d": ["AvgD", "BbAvD"], "avg_open_a": ["AvgA", "BbAvA"],
    "avg_close_h": ["AvgCH"], "avg_close_d": ["AvgCD"], "avg_close_a": ["AvgCA"],
    "max_open_h": ["MaxH", "BbMxH"], "max_open_d": ["MaxD", "BbMxD"], "max_open_a": ["MaxA", "BbMxA"],
    "max_close_h": ["MaxCH"], "max_close_d": ["MaxCD"], "max_close_a": ["MaxCA"],
    # Over/Under 2.5
    "ps_ou_open_over": ["P>2.5"], "ps_ou_open_under": ["P<2.5"],
    "ps_ou_close_over": ["PC>2.5"], "ps_ou_close_under": ["PC<2.5"],
    "avg_ou_open_over": ["Avg>2.5", "BbAv>2.5"], "avg_ou_open_under": ["Avg<2.5", "BbAv<2.5"],
    "avg_ou_close_over": ["AvgC>2.5"], "avg_ou_close_under": ["AvgC<2.5"],
    "max_ou_open_over": ["Max>2.5", "BbMx>2.5"], "max_ou_open_under": ["Max<2.5", "BbMx<2.5"],
    "max_ou_close_over": ["MaxC>2.5"], "max_ou_close_under": ["MaxC<2.5"],
    # Asian handicap (line is for the home team)
    "ah_open_line": ["AHh", "BbAHh"], "ah_close_line": ["AHCh"],
    "ah_avg_open_home": ["AvgAHH", "BbAvAHH"], "ah_avg_open_away": ["AvgAHA", "BbAvAHA"],
    "ah_avg_close_home": ["AvgCAHH"], "ah_avg_close_away": ["AvgCAHA"],
}

ODDS_1X2_TRIPLES = {
    "ps_open": ("ps_open_h", "ps_open_d", "ps_open_a"),
    "ps_close": ("ps_close_h", "ps_close_d", "ps_close_a"),
    "b365_open": ("b365_open_h", "b365_open_d", "b365_open_a"),
    "b365_close": ("b365_close_h", "b365_close_d", "b365_close_a"),
    "avg_open": ("avg_open_h", "avg_open_d", "avg_open_a"),
    "avg_close": ("avg_close_h", "avg_close_d", "avg_close_a"),
    "max_open": ("max_open_h", "max_open_d", "max_open_a"),
    "max_close": ("max_close_h", "max_close_d", "max_close_a"),
}

_FILE_RE = re.compile(r"E0_(\d{4})_(\d{2})\.csv$")


def season_from_path(path: str | Path) -> tuple[int, str]:
    """E0_2012_13.csv -> (2012, '2012/13')."""
    m = _FILE_RE.search(str(path))
    if not m:
        raise ValueError(f"unexpected file name (want E0_YYYY_YY.csv): {path}")
    start = int(m.group(1))
    return start, f"{start}/{m.group(2)}"


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse dd/mm/yyyy and dd/mm/yy, which both occur in the raw files."""
    s = values.astype(str).str.strip()
    full = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return full.fillna(short)


def _pick(raw: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in raw.columns:
            return pd.to_numeric(raw[col], errors="coerce")
    return pd.Series(np.nan, index=raw.index, dtype="float64")


def load_season(path: str | Path) -> pd.DataFrame:
    """Load one season file and map it onto the clean column names."""
    start, label = season_from_path(path)
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.dropna(how="all")
    raw = raw[raw["HomeTeam"].notna() & raw["AwayTeam"].notna()].copy()

    out = pd.DataFrame(index=raw.index)
    out["season"] = label
    out["season_start"] = start

    for clean, col in MATCH_COLUMNS.items():
        if col not in raw.columns:
            out[clean] = np.nan
        elif clean == "date":
            out[clean] = parse_dates(raw[col])
        elif clean in ("home_team", "away_team", "referee", "time", "result"):
            out[clean] = raw[col].astype("string").str.strip()
        else:
            out[clean] = pd.to_numeric(raw[col], errors="coerce")

    for clean, candidates in ODDS_COLUMNS.items():
        out[clean] = _pick(raw, candidates)

    out["odds_source"] = "football-data" if "AvgH" in raw.columns else "betbrain"

    out["match_id"] = (
        label + "|" + out["home_team"].astype(str) + "|" + out["away_team"].astype(str)
    )
    return out.reset_index(drop=True)


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["result_code"] = df["result"].map({"H": 0, "D": 1, "A": 2})
    df["over_1_5"] = (df["total_goals"] >= 2).astype("Int64")
    df["over_2_5"] = (df["total_goals"] >= 3).astype("Int64")
    df["btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype("Int64")
    return df


def load_all(raw_dir: str | Path) -> pd.DataFrame:
    """Load every E0_YYYY_YY.csv in a folder, sorted by date."""
    raw_dir = Path(raw_dir)
    files = sorted(p for p in raw_dir.glob("E0_*.csv") if _FILE_RE.search(p.name))
    if not files:
        raise FileNotFoundError(f"no E0_YYYY_YY.csv files in {raw_dir}")
    frames = [load_season(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df = _add_targets(df)
    df = df.sort_values(["date", "time", "home_team"], kind="stable").reset_index(drop=True)
    return df
