"""Choose the Dixon–Coles time decay (xi) and shrinkage (ridge) on early seasons only.

    python -m footmodel.evaluation.tune_dc --raw data/raw --out data/results

Settings are scored on the walk-forward seasons 2014/15 to 2018/19. Later seasons
(2020/21 on) are never used here, so they stay a clean final test. The score is
1X2 log loss plus Over 2.5 log loss (the two markets that have bookmaker odds).
"""

from __future__ import annotations

# Lets this file also be run directly (for example with VS Code's "Run Python File").
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "footmodel.evaluation"

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from ..data.loader import load_all
from ..models.dixon_coles import DixonColes
from . import metrics as M
from .walkforward import walk_forward

SELECTION_LAST_SEASON = 2018
XIS = [0.0, 0.001, 0.002, 0.003, 0.005]
RIDGES = [2.0, 8.0, 32.0]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/results")
    args = ap.parse_args(argv)

    df = load_all(args.raw)
    models = [DixonColes(xi=xi, ridge=r, name=f"dc_xi{xi}_r{r}") for xi, r in itertools.product(XIS, RIDGES)]
    preds = walk_forward(df, models, first_test_start=2014, last_test_start=SELECTION_LAST_SEASON)

    rows = []
    for m in models:
        g = preds[preds["model"] == m.name]
        P = g[["p_h", "p_d", "p_a"]].to_numpy()
        y = g["result_code"].to_numpy(dtype=int)
        ll_1x2 = M.logloss(P, y)
        ll_o25 = M.binary_logloss(g["p_o25"], g["over_2_5"])
        ll_btts = M.binary_logloss(g["p_btts"], g["btts"])
        rows.append({"xi": m.xi, "ridge": m.ridge, "ll_1x2": ll_1x2, "ll_over25": ll_o25, "ll_btts": ll_btts,
                     "selection_score": ll_1x2 + ll_o25, "n": len(g)})
    grid = pd.DataFrame(rows).sort_values("selection_score").reset_index(drop=True)

    pd.set_option("display.width", 160)
    print(f"Selection seasons 2014/15 to {SELECTION_LAST_SEASON}/{str(SELECTION_LAST_SEASON + 1)[-2:]}, "
          f"{int(grid['n'].iloc[0])} matches. Lower is better.\n")
    print(grid.round(4).to_string(index=False))

    best = grid.iloc[0]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "dc_tuning_grid.csv", index=False)
    (out / "dc_params.json").write_text(json.dumps({"xi": float(best["xi"]), "ridge": float(best["ridge"])}) + "\n")
    print(f"\nChosen: xi={best['xi']}, ridge={best['ridge']}  (written to {out / 'dc_params.json'})")


if __name__ == "__main__":
    main()
