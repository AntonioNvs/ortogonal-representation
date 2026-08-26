"""Hyperparameter sweep for the SAGE qualifying-position regressor.

Runs the single-seed runner (``sage_qualifying_run.py``) over a grid of
``(num_layers, hidden_dim)`` and a set of seeds, parses the ``RESULT_JSON``
line each run emits, and reports **per-config medians** (test MAE/RMSE and the
driver-vs-car paired ΔMAE / p). The headline number is a robust median over
seeds rather than one lucky draw, so the per-constructor significance is not
flattered (or sunk) by a single random init.

Usage (from the repo root, on the A100 box):
    python src/experiments/sage_sweep.py --num-layers 2 3 4 --hidden-dim 64 128 256 --seeds 42 7 123
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict

RUNNER = "src/experiments/sage_qualifying_run.py"


def _parse_result(line: str):
    return json.loads(line[len("RESULT_JSON="):])


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE qualifying sweep")
    parser.add_argument("--num-layers", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--hidden-dim", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    configs = [(L, H) for L in args.num_layers for H in args.hidden_dim]
    per_config = defaultdict(list)  # (L, H) -> list of result dicts

    for L, H in configs:
        for seed in args.seeds:
            cmd = [
                sys.executable, RUNNER,
                "--num-layers", str(L),
                "--hidden-dim", str(H),
                "--epochs", str(args.epochs),
                "--lr", str(args.lr),
                "--weight-decay", str(args.weight_decay),
                "--patience", str(args.patience),
                "--seed", str(seed),
            ]
            print(f"\n=== {L} layers / {H} hidden / seed {seed} ===", flush=True)
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            result_line = None
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                if line.startswith("RESULT_JSON="):
                    result_line = line.strip()
            proc.wait()
            if proc.returncode != 0:
                print(f"[sweep] run failed rc={proc.returncode}; skipping")
                continue
            if result_line is None:
                print("[sweep] no RESULT_JSON line found; skipping")
                continue
            per_config[(L, H)].append(_parse_result(result_line))

    # --- summary over seeds ---------------------------------------------------
    print("\n\n=== SWEEP SUMMARY (median over seeds) ===")
    header = f"{'layers':>6} {'hidden':>6} {'seeds':>5} {'MAE':>8} {'RMSE':>8} {'Δvs-ctor':>10} {'p<.05':>6}"
    print(header)
    best = None
    for (L, H), results in sorted(per_config.items()):
        if not results:
            continue
        maes = [r["test_mae"] for r in results]
        rmses = [r["test_rmse"] for r in results]
        dmaes = [r["paired"]["per-constructor"]["dmae"] for r in results]
        ps = [r["paired"]["per-constructor"]["p"] for r in results]
        med_mae = statistics.median(maes)
        med_rmse = statistics.median(rmses)
        med_dmae = statistics.median(dmaes)
        frac_sig = sum(1 for p in ps if p < 0.05) / len(ps)
        print(
            f"{L:>6} {H:>6} {len(results):>5} {med_mae:>8.4f} {med_rmse:>8.4f} "
            f"{med_dmae:>10.4f} {frac_sig:>6.2f}"
        )
        if best is None or med_mae < best[1]:
            best = ((L, H), med_mae)

    if best:
        print(f"\nBest config: {best[0][0]} layers, {best[0][1]} hidden -> median MAE {best[1]:.4f}")


if __name__ == "__main__":
    main()
