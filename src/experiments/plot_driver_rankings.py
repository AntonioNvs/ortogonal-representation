"""CLI: plot cumulative driver rankings for a season."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath("src"))

from visualization.driver_rankings import plot_driver_rankings, resolve_driver_labels
from visualization.style import resolve_plot_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot driver season rankings")
    parser.add_argument("--rankings", type=str, default="output/skill_rankings/latest/rankings.csv")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--driver", action="append", required=True, dest="drivers")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mode", type=str, default="rank", choices=["rank", "skill"])
    args = parser.parse_args()

    rankings = pd.read_csv(args.rankings)
    resolved = resolve_driver_labels(rankings, args.drivers, args.season)
    for r in resolved:
        if r["driverId"] is None:
            print(f"WARNING: could not resolve driver '{r['query']}'")
        else:
            print(
                f"  {r['query']} -> {r['driverRef']} (id={r['driverId']}, "
                f"team={r.get('constructorRef', '?')})"
            )

    out = args.output or f"output/skill_rankings/plots/season_{args.season}_rankings"
    plot_driver_rankings(rankings, args.season, args.drivers, output_path=out, mode=args.mode)
    out_dir, _ = resolve_plot_output_dir(out)
    print(f"saved plots to {out_dir}/")


if __name__ == "__main__":
    main()
