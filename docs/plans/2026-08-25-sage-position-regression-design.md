# SAGE Position Regression — Design

**Branch:** `sage-position-regression` (from `master`)
**Date:** 2026-08-25
**Motivating question:** A pure-GNN baseline for predicting a driver's **qualifying grid position** in a specific race, keeping the model maximally interpretable so that future graph-importance / attribution methods (GNNExplainer, Integrated Gradients, Shapley-over-neighbours) can be applied cleanly on top.

---

## Positioning

The two causal branches (`counterfactual-swap`, `marginal-attribution`) rest on a base GNN whose causal quantity is *extracted* by intervening on the readout. Before investing further in that machinery, we want a **minimal, honest SAGE regression model** with a single, transparent readout — the cleanest possible substrate for later attribution.

This branch inherits only the model-agnostic plumbing from `master` (`build_graph`, `get_active_task`, the `graph_meta.pt` schema snapshot, the RelBench task registry). It deliberately does **not** carry the `F1OrthogonalPipeline` skeleton (MLP classifier + aux heads + fusion + orthogonality loss), which is exactly the "external skeleton" the user wants to avoid because it scatters attribution across opaque layers.

## Design principle

> **Interpretability lives in the readout, not in the message passing.**

The GNN *encoder* (input encoder + SAGE stack) may be as deep/expressive as needed; attribution methods degrade when the *readout* is a deep nonlinear MLP with feature fusion and auxiliary heads. So the rule is:

- **Encoder**: keep expressive (input `HeteroEncoder` + bidirectional SAGE + residual/LayerNorm).
- **Readout**: reduce to a **single `Linear(hidden_dim, 1)`** — a linear, decomposable sum `pred = w·h + b`.

The single linear readout *preserves* interpretability instead of destroying it, giving a clean two-level decomposition: (a) which *input features* produced `h` (gradient through SAGE), and (b) which *dimensions of `h`* drove the prediction (the weights `w`, readable directly).

## Target

Predict `qualifying.position` — the driver's **qualifying grid position** for a specific race. Rationale:

- Continuous, dense, pre-race (no DNF ambiguity), no ~22% NaN problem that `results.position` has.
- The `qualifying` node already exists in the graph with `number`, `position`, `date` features.
- Lower is better; report **MAE / RMSE** as the primary regression metrics.

Requires a **new RelBench task** `qualifying-position` (entity_table=`qualifying`, target_col=`position`, `remove_columns=[("qualifying", "position")]`), so the target is stripped from the node's input features and cannot leak into the prediction (mirrors how `results-position` removes `position` from `results`).

## Graph schema — temporal meta-nodes

Adopts the **meta-node** design (the same idea already specified in `counterfactual-swap`): instead of one static node per driver/constructor spanning the whole career, each entity gets **one node per season**. This is what makes temporal evolution of driver skill representable at all — a single static `drivers` node cannot distinguish "Verstappen in 2017" from "Verstappen in 2023".

**Node types**

- `driver_season` — one node per `(driver, season)` that exists in the data. `verstappen@2017` and `verstappen@2023` are distinct nodes.
- `constructor_season` — one node per `(constructor, season)`.
- `race` — one per race (features: round, year).
- `circuit` — static (one per circuit).
- `qualifying` — one per `(driver, race)`; the **target node** (features: number, date; label: `position`).
- `results`, `standings`, `constructor_results`, `constructor_standings` — raw season-evidence nodes (unchanged from the base graph).

**Edge types**

- `results → driver_season@T`, `standings → driver_season@T` — a season node aggregates its own-season race/championship evidence.
- `results → constructor_season@T`, `constructor_standings → constructor_season@T`.
- `same_driver: driver_season@T → driver_season@T+1` — **directional**, carries a driver's identity/skill forward in time (message flows @T → @T+1).
- `same_constructor: constructor_season@T → constructor_season@T+1` — same for teams.
- `race → circuit`.
- `qualifying → race` — the target node sees the race's circuit/era context.
- **Reverse context edges**: `driver_season@T → qualifying`, `constructor_season@T → qualifying` — the target node aggregates its driver's and team's season embeddings.

**Why temporal evolution is now representable.** `driver_season@2023` aggregates its own 2023 results and, via the `same_driver` chain, every prior season's embedding. `driver_season@2017` has a much shorter chain (only its first seasons) and far less evidence. The SAGE can learn different embeddings for `verstappen@2017` (young, sparse, uncertain) vs `verstappen@2023` (mature, strong), and the single-Linear readout over the target node reflects that difference.

Message flow:

```
results/standings (season T)
   └─> driver_season@T ──(same_driver)──> driver_season@T+1 ──> ... ──> driver_season@2023
   └─> constructor_season@T ─(same_constructor)─> ...

qualifying (target)  ── aggregates ──>  driver_season@T + constructor_season@T
                                          + race ──> circuit
        └─> Linear(hidden, 1) ──> predicted grid position
```

Reverse edges are not a new model family — they are the standard way to make a heterogeneous GNN bidirectional so a *leaf* node (the row we predict on) can attend to its parents. 100% message passing; no readout complexity added.

## Model

1. **`HeteroEncoder`** (input side, "por baixo") — encodes categorical/numerical/timestamp columns to embeddings. Not part of the skeleton; does not harm attribution. Loaded via the `graph_meta.pt` snapshot (read *before* `build_graph` overwrites it, as fixed in the prior branch).

2. **SAGE stack** (`HeteroConv` of `SAGEConv`):
   - conv1 `mean` aggregation, `hidden_dim` channels.
   - conv2 `max` aggregation, `hidden_dim` channels (keep a latent representation, not a scalar).
   - Residual connection + `LayerNorm` on `driver_season`, `constructor_season` and `qualifying` (combats oversmoothing, stabilises training).
   - Edge-type dict covers the raw `results`/`standings` → season-node edges, the directional `same_driver`/`same_constructor` edges, and the reverse context edges into `qualifying`.

3. **Readout**: `Linear(hidden_dim, 1)` applied to the `qualifying` node embedding → scalar grid position.

**Removed** vs. `F1OrthogonalPipeline`: `classifier` MLP, `aux_piloto`/`aux_equipe` heads, `[driver||constructor]` fusion, pre-race feature concatenation, `OrthogonalSeparationLoss`.

## Temporal isolation (no leakage)

Cross-season ordering is enforced by **graph structure**, not just masks: the `same_driver`/`same_constructor` edges are directional (@T → @T+1), so a season node can only aggregate the past. `verstappen@2024` reaches `@2023`, `@2022`, … and can never reach `@2025`. The rest of the leakage control reuses the existing year-mask machinery (`add_edge_year_masks`) keyed on the source table's `raceId`:

- **Split** (fixed, year-based): train = **1950–2021** (the whole dataset up to the val window), val = **2022–2023**, test = **2024–2026** (the most recent seasons). Implemented by setting `MIN_YEAR = 1950` in `src/config.py` while keeping `EXTENDED_VAL_TIMESTAMP = 2022-01-01` / `EXTENDED_TEST_TIMESTAMP = 2024-01-01`, so `_years_from_timestamps` yields exactly these ranges.
- **Split masks**: train edges = `TRAIN_YEARS`, val edges = `TRAIN_YEARS ∪ VAL_YEARS`, test edges = all, applied to the raw leaf edges (`results`/`standings` → season nodes) keyed on the source table's year/`raceId`.
- **Reverse edges** (`season node → qualifying`) inherit their forward counterpart's mask — a reverse `driver_season@T → qualifying` edge is the same underlying `(driver, race)` row as the forward `qualifying → driver_season@T` edge, so it inherits the same temporal mask. Masking must be defined on the *underlying row*, not the edge direction.
- **Within-season round leakage** remains for the fixed split: predicting the qualifying of round k from results of round > k in the same season. For the initial test this is left at year-level granularity; the round-mask machinery (`add_edge_round_masks`) already exists in `train.py` and is the documented follow-up if the fixed split shows signal.

## Evaluation

- **Metrics**: MAE and RMSE (primary). Lower is better; MAE is the headline number (it is what a mean-predictor minimises, making the baseline comparison direct).
- **Report** val/test on the fixed split, with the standard seed handling.

## Baselines

Two leak-free baselines, both computed **only on training-years data** (1950–2021), so neither can peek at the future:

1. **Trivial floor — global mean.** Predict the mean grid position over all training qualifying sessions — a single constant for every prediction. This is the sanity floor, not the interesting comparison; any model that does not beat it is broken.

2. **Driver-naive — per-driver trailing mean.** For each driver, predict the mean of *their own* qualifying positions over the training window (`qualifying.position` grouped by `driverId`, restricted to train years). This captures "how well does this driver usually qualify" with no car, team, or race context — the crudest possible driver-skill signal. **This is the primary baseline:** the SAGE model must beat it to show it extracts more than "who is this driver".

A third baseline is noted but deferred: **constructor-naive** (per-team trailing mean grid position). It is the natural counterpart once the driver-naive comparison passes — it isolates the car signal and is the first step toward the project's driver-vs-car question. Not required for this initial test.

## Files to create (in this branch)

```
src/data/temporal_graph.py           # meta-node graph builder (driver_season/constructor_season + same_* edges)
src/models/sage_regressor.py        # HeteroEncoder + bidirectional SAGE stack + Linear readout
src/training/sage_train.py          # training loop (or extend train.py minimally)
src/experiments/sage_qualifying_run.py   # end-to-end runner
```

Reuse (from `master`): `src/data/tasks.py`, `src/data/enriched_dataset.py`, `build_graph`/`get_active_task` in `train.py`, and the `graph_meta.pt` snapshot convention.

## Exit criteria

- The model **trains** and beats **both** the global-mean floor and the driver-naive (per-driver mean) baseline on MAE (and does not blow up on RMSE).
- The readout is a single `Linear` (no MLP/aux heads/fusion) — a clean substrate for a future `GNNExplainer` / gradient-attribution pass.
- (Informal) a face-validity check: top drivers (Verstappen, Hamilton, Leclerc…) tend to get low (good) predicted grid positions.

If the model does not beat the driver-naive baseline, record why before adding complexity — the point of this branch is a minimal, honest substrate, not peak accuracy.

## Timeline (indicative)

- **Day 1**: `sage_regressor.py` + graph reverse-edge plumbing + temporal masks.
- **Day 2**: training loop + baseline comparison + metrics report.
- **Day 3 (buffer)**: fix leakage/masking edge cases, produce the interpretability-ready checkpoint.
