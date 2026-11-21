"""Validation checks for the clean match table.

Each check returns a Check with severity "error" (the data cannot be trusted
until it is fixed) or "warning" (worth knowing, does not block modelling).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loader import ODDS_1X2_TRIPLES


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    severity: str = "error"  # "error" or "warning"

    def __str__(self) -> str:
        status = "PASS" if self.passed else ("FAIL" if self.severity == "error" else "WARN")
        return f"[{status}] {self.name}" + (f": {self.detail}" if self.detail else "")


def missing_seasons(season_starts: list[int]) -> list[int]:
    """Season start years missing between the first and last one present."""
    present = sorted(set(int(s) for s in season_starts))
    if not present:
        return []
    return [y for y in range(present[0], present[-1] + 1) if y not in present]


def _season_label(start: int) -> str:
    return f"{start}/{str(start + 1)[-2:]}"


def run_checks(df: pd.DataFrame, matches_per_season: int = 380) -> list[Check]:
    checks: list[Check] = []

    # 1. no duplicated matches
    dup_id = int(df["match_id"].duplicated().sum())
    dup_date = int(df.duplicated(["date", "home_team", "away_team"]).sum())
    checks.append(Check("no duplicate matches", dup_id == 0 and dup_date == 0,
                        f"{dup_id} repeated ids, {dup_date} repeated date+teams"))

    # 2. results present
    cols = ["date", "home_team", "away_team", "home_goals", "away_goals", "result"]
    nulls = {c: int(df[c].isna().sum()) for c in cols if df[c].isna().any()}
    checks.append(Check("every match has a date, teams, goals and result", not nulls,
                        str(nulls) if nulls else ""))

    # 3. result agrees with goals
    implied = np.where(df["home_goals"] > df["away_goals"], "H",
                       np.where(df["home_goals"] < df["away_goals"], "A", "D"))
    bad = int((implied != df["result"].astype(str)).sum())
    checks.append(Check("result agrees with the goals scored", bad == 0, f"{bad} mismatches"))

    # 4. full seasons
    sizes = df.groupby("season").size()
    short = {s: int(n) for s, n in sizes.items() if n != matches_per_season}
    checks.append(Check(f"every season has {matches_per_season} matches", not short, str(short) if short else ""))

    # 5. each team plays 19 home and 19 away
    unbalanced = []
    for season, g in df.groupby("season"):
        home = g["home_team"].value_counts()
        away = g["away_team"].value_counts()
        for team in sorted(set(home.index) | set(away.index)):
            if home.get(team, 0) != matches_per_season // 20 or away.get(team, 0) != matches_per_season // 20:
                unbalanced.append(f"{season} {team}")
    checks.append(Check("each team plays equal home and away games per season", not unbalanced,
                        ", ".join(unbalanced[:5])))

    # 6. dates inside the season window (1 Aug start year to 31 Jul next year)
    out_of_window = 0
    for season, g in df.groupby("season"):
        start = int(g["season_start"].iloc[0])
        lo, hi = pd.Timestamp(start, 8, 1), pd.Timestamp(start + 1, 7, 31)
        out_of_window += int(((g["date"] < lo) | (g["date"] > hi)).sum())
    checks.append(Check("dates fall inside their season", out_of_window == 0, f"{out_of_window} outside"))

    # 7. odds are valid prices
    # Best-price (max) odds take the best price per outcome across many books, so
    # their overround is often negative; they get their own warning below.
    bad_price = 0
    bad_overround = 0
    arbitrage_rows = 0
    for name, cols3 in ODDS_1X2_TRIPLES.items():
        sub = df[list(cols3)].dropna()
        if sub.empty:
            continue
        bad_price += int(((sub <= 1.0) | (sub > 1000)).any(axis=1).sum())
        overround = (1.0 / sub).sum(axis=1) - 1.0
        if name.startswith("max_"):
            arbitrage_rows += int((overround < 0).sum())
        else:
            bad_overround += int(((overround < -0.01) | (overround > 0.25)).sum())
    checks.append(Check("1X2 odds are valid prices (above 1.0)", bad_price == 0, f"{bad_price} bad rows"))
    checks.append(Check("single-book and average 1X2 overround is plausible (-1% to 25%)", bad_overround == 0,
                        f"{bad_overround} implausible rows"))
    checks.append(Check("best-price (max) odds never imply an arbitrage", arbitrage_rows == 0,
                        f"{arbitrage_rows} rows do; treat max odds with caution", severity="warning"))

    # 8. gaps in the run of seasons (warning only)
    gaps = missing_seasons(df["season_start"].unique().tolist())
    checks.append(Check("no gaps in the run of seasons", not gaps,
                        "missing: " + ", ".join(_season_label(y) for y in gaps) if gaps else "",
                        severity="warning"))

    return checks


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Share of matches per season that have each key odds set (1.0 = all)."""
    rows = {}
    for season, g in df.groupby("season"):
        row = {"matches": len(g)}
        for name, cols3 in ODDS_1X2_TRIPLES.items():
            row[f"1x2_{name}"] = round(float(g[list(cols3)].notna().all(axis=1).mean()), 3)
        row["ou_ps_close"] = round(float(g[["ps_ou_close_over", "ps_ou_close_under"]].notna().all(axis=1).mean()), 3)
        row["ou_avg_open"] = round(float(g[["avg_ou_open_over", "avg_ou_open_under"]].notna().all(axis=1).mean()), 3)
        row["ah_open_line"] = round(float(g["ah_open_line"].notna().mean()), 3)
        row["shots"] = round(float(g[["home_shots", "away_shots"]].notna().all(axis=1).mean()), 3)
        rows[season] = row
    return pd.DataFrame(rows).T
