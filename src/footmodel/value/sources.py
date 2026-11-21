"""Odds sources. Each returns Quote objects; swap the source without touching the rest.

FootballDataFixtures reads the upcoming-fixtures file from football-data.co.uk (a URL or
a local path). TheOddsApi reads The Odds API (needs a key; check which bookmakers and
how many requests your plan includes). Manual quotes (for example SportyBet prices you
type in) go through `cli quote`, not through a source.
"""

from __future__ import annotations

import csv
import io
import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

FD_BOOKS = {"PS": "pinnacle", "B365": "bet365", "BW": "betandwin", "IW": "interwetten",
            "WH": "williamhill", "VC": "vcbet", "Avg": "market_avg", "Max": "market_max"}

# Odds API names that differ from football-data / training-data names.
TEAM_ALIASES = {
    "Manchester United": "Man United", "Manchester City": "Man City", "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest", "Tottenham Hotspur": "Tottenham", "Wolverhampton Wanderers": "Wolves",
    "Brighton and Hove Albion": "Brighton", "West Ham United": "West Ham", "Leeds United": "Leeds",
    "Leicester City": "Leicester", "Sheffield United": "Sheffield United", "AFC Bournemouth": "Bournemouth",
}
ODDS_API_BOOKS = {"pinnacle": "pinnacle", "bet365": "bet365", "williamhill": "williamhill", "betfair_ex_eu": "betfair"}


@dataclass
class Quote:
    kickoff: str          # naive UTC ISO
    home: str
    away: str
    book: str
    market: str           # "1x2" or "ou25"
    selection: str
    odds: float


def norm_team(name: str) -> str:
    return TEAM_ALIASES.get(name.strip(), name.strip())


def _num(x):
    try:
        v = float(x)
        return v if v > 1.0 else None
    except (TypeError, ValueError):
        return None


def parse_fixtures_csv(text: str) -> list[Quote]:
    """football-data.co.uk fixtures.csv: Date dd/mm/yyyy and Time (UK local), then <book>H/D/A and <book>>2.5 / <2.5."""
    out: list[Quote] = []
    for row in csv.DictReader(io.StringIO(text.lstrip("﻿"))):
        if not row.get("HomeTeam") or not row.get("Date"):
            continue
        fmt = "%d/%m/%Y" if len(row["Date"]) == 10 else "%d/%m/%y"
        local = datetime.strptime(row["Date"] + " " + (row.get("Time") or "15:00"), fmt + " %H:%M").replace(tzinfo=ZoneInfo("Europe/London"))
        kickoff = local.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
        h, a = norm_team(row["HomeTeam"]), norm_team(row["AwayTeam"])
        for code, book in FD_BOOKS.items():
            for sel, col in (("H", code + "H"), ("D", code + "D"), ("A", code + "A")):
                if (v := _num(row.get(col))):
                    out.append(Quote(kickoff, h, a, book, "1x2", sel, v))
            for sel, col in (("O", code + ">2.5"), ("U", code + "<2.5")):
                if (v := _num(row.get(col))):
                    out.append(Quote(kickoff, h, a, book, "ou25", sel, v))
    return out


class FootballDataFixtures:
    def __init__(self, location: str = "https://www.football-data.co.uk/fixtures.csv"):
        self.location = location

    def fetch(self) -> list[Quote]:
        if self.location.startswith("http"):
            with urllib.request.urlopen(self.location, timeout=30) as r:
                text = r.read().decode("utf-8-sig", errors="replace")
        else:
            text = open(self.location, encoding="utf-8-sig").read()
        rows = parse_fixtures_csv(text)
        # keep the English Premier League only: the file mixes divisions
        return rows if "Div" not in text.splitlines()[0] else _only_e0(text, rows)


def _only_e0(text: str, rows: list[Quote]) -> list[Quote]:
    keep = {(norm_team(r["HomeTeam"]), norm_team(r["AwayTeam"])) for r in csv.DictReader(io.StringIO(text.lstrip("﻿"))) if r.get("Div") == "E0"}
    return [q for q in rows if (q.home, q.away) in keep]


def parse_odds_api(events: list[dict]) -> list[Quote]:
    out: list[Quote] = []
    for ev in events:
        kickoff = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
        home, away = norm_team(ev["home_team"]), norm_team(ev["away_team"])
        for bk in ev.get("bookmakers", []):
            book = ODDS_API_BOOKS.get(bk["key"], bk["key"])
            for mk in bk.get("markets", []):
                for o in mk.get("outcomes", []):
                    if mk["key"] == "h2h":
                        sel = "H" if o["name"] == ev["home_team"] else "A" if o["name"] == ev["away_team"] else "D"
                        out.append(Quote(kickoff, home, away, book, "1x2", sel, float(o["price"])))
                    elif mk["key"] == "totals" and abs(float(o.get("point", 0)) - 2.5) < 1e-9:
                        out.append(Quote(kickoff, home, away, book, "ou25", "O" if o["name"] == "Over" else "U", float(o["price"])))
    return out


class TheOddsApi:
    def __init__(self, api_key: str | None = None, sport: str = "soccer_epl", regions: str = "uk,eu"):
        self.key, self.sport, self.regions = api_key or os.environ.get("ODDS_API_KEY", ""), sport, regions

    def fetch(self) -> list[Quote]:
        if not self.key:
            raise RuntimeError("Set ODDS_API_KEY (or pass api_key) to use The Odds API")
        qs = urllib.parse.urlencode({"apiKey": self.key, "regions": self.regions, "markets": "h2h,totals", "oddsFormat": "decimal"})
        with urllib.request.urlopen(f"https://api.the-odds-api.com/v4/sports/{self.sport}/odds?{qs}", timeout=30) as r:
            return parse_odds_api(json.loads(r.read()))
