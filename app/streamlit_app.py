"""Personal football prediction app.   Run from the project folder:   streamlit run app/streamlit_app.py"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from footmodel.value.cli import _read as _read_text
from footmodel.value import collector, forecast, scan, sources, store, tracker, views

st.set_page_config(page_title="Football predictions", page_icon="⚽", layout="wide")

with st.sidebar:
    st.header("Settings")
    db_path = st.text_input("Database file", os.environ.get("FOOTMODEL_DB", str(ROOT / "football.db")))
    book = st.text_input("Your bookmaker", "sportybet")
    w = st.slider("Weight on our model (0 = trust the market fully)", 0.0, 1.0, 0.2, 0.05,
                  help="Phase 3 found 0.0 in every season: our models have not yet beaten the market.")
    min_edge = st.slider("Minimum edge to show", 0.0, 0.15, 0.03, 0.005, format="%.3f")
    bankroll = st.number_input("Bankroll", 1.0, 1e9, 1000.0, 50.0)
    kelly = st.slider("Kelly fraction", 0.05, 1.0, 0.25, 0.05)
    cap = st.slider("Max stake (share of bankroll)", 0.005, 0.1, 0.02, 0.005, format="%.3f")
    results_dir = st.text_input("Results folder", str(ROOT / "data" / "results"))
    matches_csv = st.text_input("Clean matches file", str(ROOT / "data" / "clean" / "matches.csv"))

con = store.connect(db_path)
st.title("Football predictions")
c = views.counts(con)
a, b_, c_, d_ = st.columns(4)
a.metric("Upcoming fixtures", c["fixtures"]); b_.metric("Odds quotes stored", c["quotes"])
c_.metric("Bets logged", c["bets"]); d_.metric("Settled", c["settled"])
st.caption("No profitability claim is made until prospective results (see Monitoring) show it.")

tab_fx, tab_value, tab_bets, tab_mon, tab_data = st.tabs(["Fixtures", "Value bets", "My bets", "Monitoring", "Data"])

with tab_fx:
    fx = views.fixtures_table(con, book, w)
    if fx.empty:
        st.info("No upcoming fixtures yet. Go to the Data tab and take an odds snapshot.")
    else:
        market = st.radio("Market", ["1x2", "ou25"], horizontal=True, format_func=lambda m: {"1x2": "Match result (1X2)", "ou25": "Over/Under 2.5"}[m])
        show = fx[fx["market"] == market].drop(columns=["match_key", "market"])
        pct = {k: "{:.1%}" for k in ("model_p", "market_fair_p", "our_p", "edge")}
        st.dataframe(show.style.format({**pct, "book_odds": "{:.2f}"}, na_rep="–"), width="stretch", hide_index=True)
        st.caption("our_p blends the sharp market's fair probability with our model. edge = our_p × your odds − 1.")

    st.subheader(f"Enter {book} prices")
    keys = [r["match_key"] for r in con.execute("SELECT match_key FROM fixtures WHERE kickoff>? ORDER BY kickoff", (store.now_utc(),))]
    if keys:
        with st.form("quote"):
            key = st.selectbox("Match", keys)
            mk = st.selectbox("Market", list(store.SELECTIONS), format_func=lambda m: {"1x2": "1X2 (home / draw / away)", "ou25": "Over/Under 2.5 (over / under)"}[m])
            cols = st.columns(len(store.SELECTIONS[mk]))
            vals = [col.number_input(sel, 1.01, 1000.0, 2.0, 0.05, key=f"q_{mk}_{sel}") for col, sel in zip(cols, store.SELECTIONS[mk])]
            if st.form_submit_button("Save prices"):
                t = store.now_utc()
                for sel, v in zip(store.SELECTIONS[mk], vals):
                    store.add_quote(con, key, book, mk, sel, v, t)
                con.commit()
                st.success("Saved. Refresh the Value bets tab.")

with tab_value:
    rows = scan.scan(con, book, w, min_edge, bankroll, kelly, cap)
    if not rows:
        st.info(f"No selections beat the fair price by {min_edge:.1%}. Enter {book} prices on the Fixtures tab, and take a snapshot and forecast on the Data tab.")
    for i, r in enumerate(rows):
        with st.container(border=True):
            l, m, rr = st.columns([4, 3, 2])
            l.markdown(f"**{r['match_key'].split('|', 1)[1].replace('|', ' v ')}**  \n{r['kickoff']} UTC · {r['market']} · **{r['selection']}** @ {r['odds']:.2f}")
            m.markdown(f"Fair {r['p_fair']:.1%} · edge **{r['edge']:+.1%}**  \n_{r['source']}_")
            stake_in = rr.number_input("Stake", 0.0, 1e9, float(r["stake"]), 1.0, key=f"stake{i}")
            if rr.button("Log this bet", key=f"log{i}"):
                scan.record_bet(con, r, stake_in)
                st.success("Bet logged.")
                st.rerun()

with tab_bets:
    bt = views.bets_table(con)
    if bt.empty:
        st.info("No bets logged yet.")
    else:
        st.dataframe(bt.style.format({"edge": "{:+.1%}", "p_final": "{:.1%}", "odds": "{:.2f}", "stake": "{:.2f}", "profit": "{:+.2f}"}, na_rep="–"), width="stretch", hide_index=True)
    st.subheader("Results")
    url = st.text_input("Results CSV (path or URL)", "https://www.football-data.co.uk/mmz4281/2627/E0.csv")
    if st.button("Load results and settle bets"):
        try:
            text = _read_text(url)
            n = tracker.results_from_csv(con, text)
            filled = tracker.backfill_closes(con, text)
            st.success(f"{n} results loaded, {tracker.settle(con)} bets settled, {filled} missing closing prices filled in.")
        except Exception as e:
            st.error(f"Could not load results: {e}")

with tab_mon:
    st.code(tracker.report(con, book), language=None)
    clv = tracker.clv_rows(con)
    if clv:
        st.subheader("Closing-line value per bet")
        st.bar_chart(pd.DataFrame(clv).set_index("id")["clv"])
    rd = Path(results_dir)
    dash = rd / "performance_dashboard.png"
    st.subheader("Model performance")
    if dash.exists():
        st.image(str(dash), width="stretch")
        st.caption("Redrawn every time a run script finishes (python -m footmodel.evaluation.run_ensemble ...).")
    else:
        st.info("No dashboard yet: run the evaluation scripts first.")
    if (rd / "reliability.png").exists():
        with st.expander("Calibration (reliability diagrams)"):
            st.image(str(rd / "reliability.png"), width="stretch")

with tab_data:
    st.write("Take an odds snapshot, then forecasts. Snapshot often (every 30 minutes if you can): the closing price is the last snapshot before kick-off.")
    src = st.selectbox("Odds source", ["footballdata", "oddsapi"], help="oddsapi needs the ODDS_API_KEY environment variable")
    loc = st.text_input("football-data location (URL or file)", "https://www.football-data.co.uk/fixtures.csv")
    l, r_ = st.columns(2)
    if l.button("Take odds snapshot"):
        try:
            s = sources.TheOddsApi() if src == "oddsapi" else sources.FootballDataFixtures(loc)
            st.success(str(collector.snapshot(con, s.fetch())))
        except Exception as e:
            st.error(f"Snapshot failed: {e}")
    log_path = st.text_input("Always-on snapshot log", str(ROOT / "data" / "live" / "quotes_log.csv"))
    if st.button("Import cloud snapshots"):
        from footmodel.value import cloud
        st.success(str(cloud.import_log(con, log_path)))
    if r_.button("Make forecasts"):
        try:
            n = forecast.run_forecasts(con, pd.read_csv(matches_csv), w)
            st.success(f"Forecasts made for {n} fixtures.")
        except Exception as e:
            st.error(f"Forecast failed: {e}")
