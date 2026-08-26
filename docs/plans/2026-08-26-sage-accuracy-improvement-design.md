# SAGE Position Regression — Accuracy Improvement (Depth/Capacity + Training Hygiene)

**Branch:** `sage-position-regression`
**Date:** 2026-08-26
**Status:** implemented (runner edits + sweep script; awaiting A100 sweep results)

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
