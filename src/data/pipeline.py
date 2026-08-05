"""Single CLI entry point for the rel-f1 enrichment pipeline.

Subcommands:
    extract   Pre-warm the Jolpica HTTP cache + raw JSON snapshots for every
              round not yet covered by the pristine snapshot, without writing
              the enriched database. Mostly useful to separate network I/O
              from the (fast, offline) merge step.
    build     Run extraction (if needed) + schema normalization + ID
              reconciliation, and write data/enriched/rel-f1/db/*.parquet +
              manifest.json. Idempotent: re-running only fetches rounds that
              were not already cached, unless --refresh-last-n-rounds is set.
    validate  Run the 6 validation checks against the current build.
    all       build, then validate.

Examples:
    python -m src.data.pipeline build --max-year 2026
    python -m src.data.pipeline build --refresh-last-n-rounds 2
    python -m src.data.pipeline validate
    python -m src.data.pipeline all
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def _cmd_extract(args: argparse.Namespace) -> int:
    from .build_enriched_db import _determine_new_rounds, load_pristine_db
    from .sources.jolpica import JolpicaClient

    client = JolpicaClient(raw_dir=args.raw_dir)
    pristine = load_pristine_db()
    rounds, _ = _determine_new_rounds(
        client, pristine.table_dict["races"].df, args.max_year, args.refresh_last_n_rounds
    )
    logger.info("Pre-fetched %d round(s) of raw Jolpica snapshots into %s", len(rounds), args.raw_dir)
    for year, round_ in rounds:
        client.get_qualifying(year, round_)
        client.get_sprint(year, round_)
        client.get_driver_standings(year, round_)
        client.get_constructor_standings(year, round_)
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from .build_enriched_db import build_enriched_db
    from .sources.jolpica import JolpicaClient

    client = JolpicaClient(raw_dir=args.raw_dir)
    manifest = build_enriched_db(
        output_dir=args.output_dir,
        max_year=args.max_year,
        refresh_last_n_rounds=args.refresh_last_n_rounds,
        client=client,
    )
    logger.info(
        "Build complete: %d new round(s), rows_added=%s",
        manifest["new_rounds_added"], manifest["rows_added_per_table"],
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from .validate_enriched import run_full_validation

    report = run_full_validation(
        enriched_dir=args.output_dir,
        max_year=args.max_year,
        skip_f1db=args.skip_f1db,
        skip_graph=args.skip_graph,
    )
    print(report.summary())
    return 0 if report.all_passed else 1


def _cmd_all(args: argparse.Namespace) -> int:
    rc = _cmd_build(args)
    if rc != 0:
        return rc
    return _cmd_validate(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="rel-f1 enrichment pipeline")
    parser.add_argument("--raw-dir", default="data/raw/jolpica", help="Raw Jolpica JSON snapshot directory")
    parser.add_argument("--output-dir", default="data/enriched/rel-f1", help="Enriched database output directory")
    parser.add_argument("--max-year", type=int, default=2026, help="Last season to include")
    parser.add_argument(
        "--refresh-last-n-rounds", type=int, default=0,
        help="Bypass the HTTP cache for the N most recently held rounds (post-race data corrections)",
    )
    parser.add_argument("--skip-f1db", action="store_true", help="Skip the f1db cross-source validation check")
    parser.add_argument("--skip-graph", action="store_true", help="Skip the graph/task smoke test check")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("extract", help="Pre-warm raw Jolpica snapshots only")
    sub.add_parser("build", help="Extract + normalize + merge -> enriched db/*.parquet")
    sub.add_parser("validate", help="Run the 6 validation checks")
    sub.add_parser("all", help="build, then validate")
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    handlers = {
        "extract": _cmd_extract,
        "build": _cmd_build,
        "validate": _cmd_validate,
        "all": _cmd_all,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
