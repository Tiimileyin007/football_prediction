# football_prediction
a football prediction model, work in progress

# footmodel

Football match prediction model. Phase 0 (data foundation) is done: one loader
for all Premier League season files, one set of clean column names, and checks
that must pass before any model is trained.

## Set up

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

This installs pandas, numpy, scipy, scikit-learn, matplotlib and (on Windows) tzdata, plus pytest with `[dev]`.
Python 3.11 or newer.

## Running scripts

Run everything from the project folder (`football-model/`). Both styles work: `python -m footmodel.evaluation.run_baselines`
and VS Code's "Run Python File" button (or `python src/footmodel/evaluation/run_baselines.py`).

## Data

Put the season files from football-data.co.uk in `data/raw/`, named
`E0_YYYY_YY.csv` (for example `E0_2012_13.csv`). Check the dates inside each
file: two of the files supplied were labelled one season too late.

## Build the clean dataset and run the checks

```bash
python -m footmodel.data.build --raw data/raw --out data/clean
pytest
```

This writes `matches.csv`, `coverage.csv` and `checks.txt` to `data/clean/` and
exits with an error if any error-level check fails.

## Phase 1: baselines and the evaluation harness

```bash
python -m footmodel.evaluation.run_baselines --raw data/raw --out data/results
```

Trains on earlier seasons only, forecasts each season from 2014/15 on, and
prints score tables (log loss, Brier, RPS, accuracy, balanced accuracy, draw
calibration), per-season results and paired bootstrap intervals. Models compared:
base rates (all history, last 3 seasons), Elo, and three market benchmarks
(Pinnacle closing, Pinnacle opening, market-average opening) with the margin
removed by the power method.

To add a model, give it a `name`, `fit(train)`, `predict_1x2(test)` and
`predict_goals(test)` (see `evaluation/walkforward.py`) and add it to
`build_models()` in `run_baselines.py`.

The Elo constants (K 20, home advantage 65, 25% regression between seasons,
promoted teams start at 1450) are fixed defaults and were not tuned on the test seasons.

## Phase 2: Dixon–Coles scoreline model

```bash
python -m footmodel.evaluation.tune_dc --raw data/raw --out data/results     # picks xi and ridge on 2014/15-2018/19 only
python -m footmodel.evaluation.run_phase2 --raw data/raw --out data/results  # final test on 2020/21 onward
```

`models/dixon_coles.py` fits attack and defence per team with time decay and
shrinkage, refits every 14 days inside the test season (using only earlier
results), and builds one scoreline matrix per match. Every market is read from
that matrix: 1X2, Over 1.5, Over 2.5, BTTS, expected goals, correct score.
Teams new to the league start at average strength (a known weakness).

## Phase 2b: Poisson regression with features, and the market residual test

```bash
python -m footmodel.evaluation.run_poisson --raw data/raw --out data/results      # tunes on 2014/15-2018/19, tests on 2020/21 on
python -m footmodel.evaluation.residual_test --forecasts data/results/poisson_forecasts.csv
```

`features/form.py` builds pre-match form (rolling shots, shots on target, goals
for and against, rest days, new-to-league flags) using only earlier matches.
`models/poisson_features.py` fits home and away goal-rate regressions on those
features plus the Elo difference. `residual_test.py` stacks the models with the
market's probabilities to see whether they add anything the market lacks.

## Phase 3: ensembles, calibration and market pooling

```bash
python -m footmodel.evaluation.run_ensemble --forecasts data/results/poisson_forecasts.csv --out data/results
```

`ensemble/stack.py` has equal-weight averages, logistic-regression stackers
(which also recalibrate), and a one-weight log-odds pool between a model and the
market. `ensemble/walkforward.py` fits every ensemble on earlier seasons' forecasts
only. The script prints score tables, the pooling weight with a bootstrap
interval, calibration error with intervals, and writes `reliability.png` and
`reliability_points.csv` (the data behind the diagram).

Per-market calibration can make markets disagree slightly; the app should show
the calibrated probabilities and use the scoreline matrix only for likely scores.

## Project layout

Folders end with `/`. Everything else is a file.

```
football-model/
├── README.md                      how to run everything
├── pyproject.toml                 package settings and dependencies
│
├── data/
│   ├── raw/                       season files as downloaded (E0_2012_13.csv ... E0_2025_26.csv)
│   ├── clean/
│   │   ├── matches.csv            one clean table, all seasons
│   │   ├── coverage.csv           what data exists per season
│   │   └── checks.txt             data check results
│   └── results/                   outputs of the evaluation runs (tables, forecasts, plots)
│
├── src/
│   └── footmodel/                 the Python package
│       ├── data/
│       │   ├── loader.py          reads and cleans the season files
│       │   ├── checks.py          data checks
│       │   └── build.py           builds data/clean
│       ├── features/
│       │   └── form.py            pre-match form features
│       ├── models/
│       │   ├── market.py          bookmaker odds to probabilities
│       │   ├── baselines.py       base rates and Elo
│       │   ├── dixon_coles.py     scoreline model
│       │   └── poisson_features.py  goal-rate regression
│       ├── ensemble/
│       │   ├── stack.py           averaging, stacking, market pooling
│       │   └── walkforward.py     ensembles fitted on earlier seasons only
│       ├── evaluation/
│       │   ├── metrics.py         log loss, Brier, calibration
│       │   ├── walkforward.py     time-ordered testing
│       │   ├── report.py          score tables
│       │   ├── reliability_plot.py  calibration diagram
│       │   ├── dashboard.py         performance dashboard picture (drawn after every run)
│       │   └── run_*.py, tune_dc.py, residual_test.py   runnable scripts
│       └── value/                 phase 5
│           ├── edge.py            edge, blending, stake size
│           ├── store.py           SQLite log (fixtures, odds, bets, results)
│           ├── sources.py         odds sources
│           ├── collector.py       stores one odds snapshot
│           ├── forecast.py        live forecasts
│           ├── scan.py            finds value bets, logs bets
│           ├── tracker.py         settlement, ROI, CLV, margin
│           ├── views.py           tables for the app
│           └── cli.py             the command line tool
│
└── tests/                         automated checks (test_*.py)
```

## Phase 5: value engine

```bash
python -m footmodel.value.cli --db football.db snapshot            # schedule every 30 min
python -m footmodel.value.cli --db football.db forecast --matches data/clean/matches.csv
python -m footmodel.value.cli --db football.db fixtures            # shows match keys
python -m footmodel.value.cli --db football.db quote --key KEY --market 1x2 --odds 2.1 3.4 3.6   # SportyBet prices you type in
python -m footmodel.value.cli --db football.db scan
python -m footmodel.value.cli --db football.db bet --key KEY --market 1x2 --selection H --odds 2.1 --stake 10
python -m footmodel.value.cli --db football.db results --csv https://www.football-data.co.uk/mmz4281/2627/E0.csv
# loading results also fills in missing Pinnacle closing prices from the same file
python -m footmodel.value.cli --db football.db settle
python -m footmodel.value.cli --db football.db report
```

The fair probability blends Pinnacle (margin removed) with Dixon-Coles, weight `--w` on the model (default 0.2; phase 3 found 0.0). Loading results also backfills any missing closing price (a fixture whose last snapshot before kick-off is missing or older than 90 minutes) from the file's Pinnacle closing columns; real snapshots are never overwritten. The snapshot source is untested against the live sites from this workspace. Check it on your machine first.

## Performance dashboard

Every run script (`run_baselines`, `run_phase2`, `run_poisson`, `run_ensemble`) redraws
`data/results/performance_dashboard.png` when it finishes: 1X2 log loss and Brier score per model,
log loss by season, Over 2.5 log loss, accuracy against balanced accuracy, and calibration error with
intervals. To redraw it without re-running the models, or to add the betting record:

```bash
python -m footmodel.evaluation.dashboard --out data/results --db football.db
```

### In VS Code

1. Open the folder in VS Code and select your `.venv` interpreter.
2. Open `data/results/performance_dashboard.png` once in an editor tab (click it in the Explorer). Keep the tab open
   and run any script from the terminal: the picture redraws at the end of the run.
3. To make it pop up on its own, set `FOOTMODEL_SHOW=1` (PowerShell: `$env:FOOTMODEL_SHOW=1`; macOS/Linux: `export FOOTMODEL_SHOW=1`),
   or add `--show` to the dashboard command. It opens in your default image viewer.
4. `Terminal > Run Task` offers "Run ensembles + dashboard" and "Redraw dashboard only" (see `.vscode/tasks.json`).

With `--db` it also writes `betting_record.png` (cumulative profit from settled bets).

## Phase 6: personal app (Streamlit)

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

Tabs: **Fixtures** (our probabilities, the sharp market's, your bookmaker's price and the edge; form to type in
SportyBet prices), **Value bets** (selections that beat the fair price, with suggested stakes and a button to log
the bet), **My bets** (log, results, settlement), **Monitoring** (ROI interval, CLV, margin, bets needed, and the
performance dashboard) and **Data** (take an odds snapshot, make forecasts). Settings in the sidebar.
`src/footmodel/value/views.py` holds the tables, `tests/test_app.py` starts the app headlessly.

## Clean table at a glance

- Match columns: `season`, `date`, `home_team`, `away_team`, `home_goals`,
  `away_goals`, `result` (H/D/A), half-time goals, shots, shots on target,
  corners, fouls, cards, `referee`.
- Targets: `total_goals`, `result_code` (0 home, 1 draw, 2 away), `over_1_5`,
  `over_2_5`, `btts`.
- Odds: `<book>_open_*` and `<book>_close_*` for Pinnacle (`ps`), Bet365
  (`b365`), market average (`avg`) and best price (`max`); Over/Under 2.5
  (`*_ou_*`); Asian handicap (`ah_*`).

## Known data limits

- 2019/20 is missing from the files supplied.
- Pinnacle odds stop on 8 January 2026, so the last 170 matches of 2025/26 have none.
- Closing odds for 2012/13 to 2018/19 exist for Pinnacle only; Bet365 and
  market-average closing odds start in 2020/21.
- The market average in 2012/13 to 2018/19 is the Betbrain average, and later it
  is the football-data average (`odds_source` records which).
- Over/Under 2.5 closing odds from Pinnacle only exist from 2020/21.
- Best-price (`max`) odds sometimes imply an arbitrage; use them with caution.
- No Over 1.5 or BTTS odds in the files.
