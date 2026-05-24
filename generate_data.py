"""
F1 Orthogonal Representation — Data Extraction Pipeline

Entry point CLI for extracting and pre-processing Formula 1 data from
RelBench rel-f1 (relational/team data) and tabular driver features.

Usage
-----
    python main.py --year 2023 --modules all
    python main.py --year 2023 --modules driver
    python main.py --year 2023 2024 --modules team track
"""
import argparse
import sys
import os

sys.path.append(os.path.abspath("src"))

from data_extractor import run_extraction, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="F1 Orthogonal Representation — Data Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py --year 2023 --modules all
  python main.py --year 2023 --modules driver
  python main.py --year 2023 2024 --modules team track
        """,
    )

    parser.add_argument(
        "--year", "-y",
        type=int,
        nargs="+",
        required=True,
        help="Season year(s) to extract (e.g. 2023 or 2023 2024)",
    )

    parser.add_argument(
        "--modules", "-m",
        nargs="+",
        default=["all"],
        choices=["all", "driver", "team", "track", "dataset"],
        help="Extraction modules to run (default: all)",
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: ./output)",
    )

    parser.add_argument(
        "--cache-dir", "-c",
        type=str,
        default=None,
        help="FastF1 cache directory (default: ./cache)",
    )

    args = parser.parse_args()

    setup_logging()

    for year in args.year:
        run_extraction(
            year=year,
            modules=args.modules,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
        )


if __name__ == "__main__":
    main()
