from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from footmodel.value import collector, edge, forecast, scan, sources, store, tracker
from footmodel.value.sources import Quote, parse_fixtures_csv, parse_odds_api


def test_pool_endpoints_and_edge_math():
    m, d = np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.3, 0.5])
    assert edge.pool(m, d, 0.0) == pytest.approx(m)
    assert edge.pool(m, d, 1.0) == pytest.approx(d)
    assert edge.edge(0.5, 2.2) == pytest.approx(0.10)
    assert edge.kelly_fraction(0.4, 2.0) == 0.0          # no edge, no stake
    assert edge.stake(0.55, 2.2, 1000, 0.25, 0.02) == 20.0  # capped at 2% of bankroll
    assert edge.candidates({"H": 0.5}, {"H": 2.2, "D": 3.0}, 0.03)[0]["edge"] == pytest.approx(0.10)
    assert edge.candidates({"H": 0.5}, {"H": 20.0}, 0.03, max_odds=12) == []


def _db(tmp_path):
    return store.connect(str(tmp_path / "t.db"))


def _seed(con, kickoff="2099-01-01T15:00:00"):
    for t, odds in (("2098-12-30T10:00:00", (2.0, 3.5, 3.9)), ("2099-01-01T14:30:00", (1.9, 3.6, 4.2))):
        collector.snapshot(con, [Quote(kickoff, "Arsenal", "Chelsea", "pinnacle", "1x2", s, o) for s, o in zip("HDA", odds)], t)
    return store.match_key(kickoff, "Arsenal", "Chelsea")


def test_open_and_close_come_from_timestamps(tmp_path):
    con = _db(tmp_path)
    key = _seed(con)
    op, _ = store.quote_set(con, key, "pinnacle", "1x2", "open")
    cl, t = store.quote_set(con, key, "pinnacle", "1x2", "latest", before="2099-01-01T15:00:00")
    assert op["H"] == 2.0 and cl["H"] == 1.9 and t == "2099-01-01T14:30:00"
    # a snapshot taken after kick-off must never become the close
    collector.snapshot(con, [Quote("2099-01-01T15:00:00", "Arsenal", "Chelsea", "pinnacle", "1x2", s, 9.0) for s in "HDA"], "2099-01-01T16:00:00")
    assert store.quote_set(con, key, "pinnacle", "1x2", "latest", before="2099-01-01T15:00:00")[0]["H"] == 1.9


def test_scan_bet_settle_and_clv(tmp_path):
    con = _db(tmp_path)
    key = _seed(con)
    t = "2098-12-31T09:00:00"
    for s, o in zip("HDA", (2.3, 3.4, 3.9)):
        store.add_quote(con, key, "sportybet", "1x2", s, o, t)
    con.commit()
    rows = scan.scan(con, "sportybet", w=0.0, min_edge=0.03, now="2098-12-31T10:00:00")
    assert [r["selection"] for r in rows][0] == "H" and rows[0]["edge"] > 0.03
    scan.record_bet(con, rows[0], 10.0)
    tracker.set_result(con, key, 2, 1)
    assert tracker.settle(con) == 1
    b = con.execute("SELECT * FROM bets").fetchone()
    assert b["status"] == "won" and b["profit"] == pytest.approx(10 * (rows[0]["odds"] - 1))
    clv = tracker.clv_rows(con, now="2099-01-02T00:00:00")
    assert len(clv) == 1 and clv[0]["stale_min"] == pytest.approx(30)
    assert "ROI" in tracker.report(con)


def test_outcome_rules():
    assert tracker.outcome("1x2", "D", 1, 1) and tracker.outcome("ou25", "O", 2, 1) and not tracker.outcome("ou25", "O", 1, 1)


def test_parse_fixtures_csv_converts_uk_time_to_utc():
    text = "Div,Date,Time,HomeTeam,AwayTeam,PSH,PSD,PSA,B365H,B365D,B365A\nE0,14/03/2026,15:00,Arsenal,Chelsea,2.0,3.5,3.9,2.05,3.4,3.8\n"
    q = parse_fixtures_csv(text)
    assert {x.book for x in q} == {"pinnacle", "bet365"} and q[0].kickoff == "2026-03-14T15:00:00"
    summer = parse_fixtures_csv(text.replace("14/03/2026", "14/08/2026"))
    assert summer[0].kickoff == "2026-08-14T14:00:00"    # BST is UTC+1


def test_parse_odds_api_payload():
    ev = [{"commence_time": "2026-12-01T15:00:00Z", "home_team": "Manchester United", "away_team": "Arsenal",
           "bookmakers": [{"key": "pinnacle", "markets": [
               {"key": "h2h", "outcomes": [{"name": "Manchester United", "price": 3.0}, {"name": "Draw", "price": 3.4}, {"name": "Arsenal", "price": 2.3}]},
               {"key": "totals", "outcomes": [{"name": "Over", "point": 2.5, "price": 1.9}, {"name": "Under", "point": 2.5, "price": 1.95}]}]}]}]
    q = parse_odds_api(ev)
    assert q[0].home == "Man United" and {x.selection for x in q if x.market == "1x2"} == set("HDA") and len(q) == 5


def test_forecast_runs_and_blends(tmp_path):
    con = _db(tmp_path)
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C", "D"]
    rows = [(pd.Timestamp("2025-01-01") + pd.Timedelta(days=i), teams[i % 4], teams[(i + 1 + i // 4) % 4 if (i + 1 + i // 4) % 4 != i % 4 else (i + 2) % 4], rng.poisson(1.5), rng.poisson(1.1)) for i in range(120)]
    matches = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_goals", "away_goals"])
    collector.snapshot(con, [Quote("2099-01-01T15:00:00", "A", "B", "pinnacle", "1x2", s, o) for s, o in zip("HDA", (2.0, 3.5, 3.9))], "2098-12-01T00:00:00")
    assert forecast.run_forecasts(con, matches, w=0.2, now="2098-12-02T00:00:00") == 1
    final = forecast.latest_forecast(con, "2099-01-01|A|B", "1x2", "final_w0.2")
    assert sum(final.values()) == pytest.approx(1)


def test_dashboard_draws_from_result_tables_and_skips_when_empty(tmp_path):
    from footmodel.evaluation.dashboard import build
    assert build(tmp_path) is None                        # nothing to draw yet: no crash
    pd.DataFrame({"model": ["elo_logit", "market_avg_open"], "n": [10, 10], "log_loss": [1.0, 0.95], "brier": [0.6, 0.57],
                  "accuracy": [0.5, 0.55], "balanced_acc": [0.45, 0.47]}).to_csv(tmp_path / "ensemble_1x2.csv", index=False)
    assert build(tmp_path).exists()


def test_backfill_closes_fills_only_stale_or_missing_closes(tmp_path):
    con = _db(tmp_path)
    key = _seed(con)                       # real snapshots: close at 14:30, 30 min before kick-off, so fresh
    csv_text = "Date,HomeTeam,AwayTeam,PSCH,PSCD,PSCA\n01/01/2099,Arsenal,Chelsea,1.8,3.7,4.4\n"
    assert tracker.backfill_closes(con, csv_text) == 0                         # fresh close is never overwritten
    assert store.quote_set(con, key, "pinnacle", "1x2", "latest", "2099-01-01T15:00:00")[0]["H"] == 1.9
    # a fixture whose only snapshot is days old gets the file's close
    collector.snapshot(con, [Quote("2099-02-01T15:00:00", "Everton", "Fulham", "pinnacle", "1x2", s, o) for s, o in zip("HDA", (2.5, 3.2, 3.0))], "2099-01-20T10:00:00")
    text2 = "Date,HomeTeam,AwayTeam,PSCH,PSCD,PSCA\n01/02/2099,Everton,Fulham,2.3,3.3,3.2\n"
    assert tracker.backfill_closes(con, text2) == 1
    k2 = store.match_key("2099-02-01T15:00:00", "Everton", "Fulham")
    prices, t = store.quote_set(con, k2, "pinnacle", "1x2", "latest", "2099-02-01T15:00:00")
    assert prices["H"] == 2.3 and t == "2099-02-01T14:59:00"


def test_cloud_log_round_trip_and_snapshot_rules(tmp_path):
    from footmodel.value import cloud
    log = tmp_path / "log.csv"
    ko = "2026-12-01T15:00:00"
    q = [Quote(ko, "Arsenal", "Chelsea", b, "1x2", s, o) for b in ("pinnacle", "vcbet") for s, o in zip("HDA", (2.0, 3.5, 3.9))]
    r = cloud.snapshot_to_log(q, log, now="2026-11-30T10:00:00")           # first ever: written; vcbet not kept
    assert r["written"] == 3 and len(cloud.read_log(log)) == 3
    assert cloud.snapshot_to_log(q, log, now="2026-11-30T11:00:00")["written"] == 0      # far from kick-off, recent: skipped
    assert cloud.snapshot_to_log(q, log, now="2026-12-01T13:00:00")["written"] == 3      # 2 hours before kick-off: written
    assert cloud.snapshot_to_log(q, log, now="2026-12-05T10:00:00")["written"] == 0      # nothing upcoming
    con = _db(tmp_path)
    assert cloud.import_log(con, log) == {"quotes_added": 6}
    assert cloud.import_log(con, log) == {"quotes_added": 0}                              # repeating adds nothing
    key = store.match_key(ko, "Arsenal", "Chelsea")
    prices, t = store.quote_set(con, key, "pinnacle", "1x2", "latest", ko)
    assert t == "2026-12-01T13:00:00" and prices["H"] == 2.0


def test_snapshot_workflow_file_is_valid_yaml():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load((Path(__file__).resolve().parents[1] / ".github" / "workflows" / "snapshot.yml").read_text())
    assert wf["jobs"]["snapshot"]["steps"] and (wf.get(True) or wf.get("on"))["schedule"]
