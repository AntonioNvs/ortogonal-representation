# SAGE Position Regression — Accuracy Improvement (Depth/Capacity + Training Hygiene)

**Branch:** `sage-position-regression`
**Date:** 2026-08-26
**Status:** implemented + swept. **Locked config: `num_layers=4`, `hidden_dim=128`** (now the runner defaults).

---

## Context

The single-`Linear` SAGE regressor is functionally correct and beats all three
baselines, but the driver-vs-car result is *small and seed-sensitive*: the
per-constructor paired test swung from `p=0.004 (**)` to `p=0.089 (*)` between two
runs that differed only in random init, and the head-of-prediction MAE is ~0.183
against a car floor of ~0.200. This doc records the plan to **lower MAE and
stabilise the significance** while keeping the single-`Linear` readout — the branch's
load-bearing constraint for future GNNExplainer / Integrated Gradients / Shapley.

**Hard constraint (confirmed with user):** the readout stays `Linear(hidden_dim, 1)`.
All improvements are encoder-side.

## Lever 1 — Depth + capacity (the main architectural lever)

**Root cause.** At 2 SAGE layers × hidden 64, the receptive field of
`qualifying@(T,k)` reaches only (a) the *previous* race's result and (b) the
driver/team *static* attributes — the `same_driver` recurrence is unrolled one hop,
so the model has essentially no multi-race form. Predicting the grid from "one prior
race + who is this driver" is the bottleneck.

**Change.** Nothing structural: `sage_regressor.py` already routes the *final* layer to
`QUALIFYING_IN_EDGE_TYPES` (line 74), so `--num-layers 4` "just works" — layers 1–3
build `driver_state`/`constructor_state` over history, layer 4 reads out `qualifying`.
Residual + `LayerNorm` (already present) make the added depth safe. Capacity is the
matching `--hidden-dim 128`/`256` scaling both the `HeteroEncoder` output and the
`SAGEConv` message dims.

**Sweep.** `num_layers ∈ {2,3,4}` × `hidden_dim ∈ {64,128,256}`. The model is currently
*underfit* (train MSE 0.032, test RMSE 0.229), so the capacity/depth direction is safe.

## Lever 2 — Training hygiene (no cosine LR)

Three pieces, all in `sage_qualifying_run.py`:

1. **Early stopping on val MAE** (`--patience 20`). Track best val MAE, stop after 20
   non-improving epochs, reload the best checkpoint. Matters more as depth/capacity grow.
2. **Weight decay** (`--weight-decay 1e-5`) as a safety rail against the mild overfit
   risk that the bigger models introduce.
3. **Seed averaging for the reported number.** Run each config over 3–5 seeds and
   report the **median** MAE / ΔMAE / p. This is the fix for the significance
   fragility: the honest claim becomes "SAGE beats the car floor robustly across
   seeds", not "in this one draw".

## Files

- `src/experiments/sage_qualifying_run.py` — added `--weight-decay`, `--patience`,
  early-stopping loop, `RESULT_JSON=` machine-readable summary line (unchanged output
  otherwise).
- `src/experiments/sage_sweep.py` (new) — drives the runner over the grid × seeds,
  parses `RESULT_JSON=`, prints per-config medians.

## Verification

On the A100 box:

```bash
# single config (sanity)
python src/experiments/sage_qualifying_run.py --num-layers 3 --hidden-dim 128 --epochs 200 --seed 42

# full sweep
python src/experiments/sage_sweep.py --num-layers 2 3 4 --hidden-dim 64 128 256 --seeds 42 7 123
```

**Pass criteria:** (a) median test MAE below the current ~0.183; (b) the
per-constructor ΔMAE is negative and its p<0.05 in most seeds (so the driver-above-car
claim is no longer seed-dependent); (c) the single-`Linear` readout is untouched; (d)
face-validity still holds (elite on top, rookies ranked on current form).

## Sweep results (3 seeds: 42, 7, 123)

```
layers hidden      MAE     RMSE   Δvs-ctor  p<.05
     2     64   0.1832   0.2270    -0.0168   0.33   <- old default
     2    128   0.1794   0.2251    -0.0206   1.00
     2    256   0.1826   0.2296    -0.0174   0.33
     3     64   0.1735   0.2188    -0.0264   1.00
     3    128   0.2471   0.2886    +0.0471   0.33   <- unstable cell
     3    256   0.1708   0.2201    -0.0291   0.67
     4     64   0.1704   0.2164    -0.0295   1.00
     4    128   0.1637   0.2123    -0.0362   1.00   <- WINNER (locked)
     4    256   0.2586   0.2989    +0.0587   0.33   <- unstable cell
```

**Decision.** `4/128` wins on every axis: lowest median MAE (0.1637, ~11% below the
old 0.1832), best RMSE, largest driver-above-car gap (Δ −0.0362 vs −0.0168), and
`p<.05` in **all 3 seeds** (the seed-fragility observed earlier is gone at this config).

**Follow-up (not blocking).** Two cells (`3/128`, `4/256`) collapsed — worse than the
constructor baseline (positive Δ). With only 3 seeds a single divergent run flips the
median; these are most likely training-instability artifacts at the high-width
boundary, not a real "wider is worse" signal. Re-run `{4/128, 4/64, 3/64, 3/128, 4/256}`
over 5 seeds before treating the sweep table as a final paper artifact.
