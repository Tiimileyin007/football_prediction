"""Build the clean dataset from the raw season files.

    python -m footmodel.data.build --raw data/raw --out data/clean

Writes matches.csv (and matches.parquet when pyarrow is installed),
coverage.csv and checks.txt, prints the check results, and exits with a
non-zero code if any error-level check fails.
"""

from __future__ import annotations

# Lets this file also be run directly (for example with VS Code's "Run Python File").
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "footmodel.data"

import argparse
import sys
from pathlib import Path

from .checks import coverage_report, run_checks
from .loader import load_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default="data/raw", help="folder with E0_YYYY_YY.csv files")
    parser.add_argument("--out", default="data/clean", help="folder for the clean files")
    args = parser.parse_args(argv)

    df = load_all(args.raw)
    checks = run_checks(df)
    coverage = coverage_report(df)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "matches.csv", index=False)
    try:
        df.to_parquet(out / "matches.parquet", index=False)
    except ImportError:
        pass  # pyarrow not installed; the CSV is enough for now
    coverage.to_csv(out / "coverage.csv")
    (out / "checks.txt").write_text("\n".join(str(c) for c in checks) + "\n", encoding="utf-8")

    print(f"{len(df)} matches, {df['season'].nunique()} seasons, "
          f"{df['home_team'].nunique()} teams, {df['date'].min().date()} to {df['date'].max().date()}")
    for c in checks:
        print(c)
    print("\nOdds coverage by season (1.0 = every match):")
    print(coverage.drop(columns=["1x2_b365_close", "1x2_avg_close", "1x2_max_close"]).to_string())

    return 1 if any((not c.passed) and c.severity == "error" for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
