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

## Graph schema — causal round-state graph

The graph is **temporally causal by construction**: every edge points from an event at an earlier time to a node at a later time. There are no bidirectional edges across time, so no node can ever aggregate information from its own future. This is the property that makes leakage *impossible* rather than merely masked.

**Design principle — round-granularity states.** A season-level meta-node (`driver_season@T`) would aggregate the *entire* season and therefore leak a mid-season qualifying's own later rounds. To close that, the state node is indexed by **round**, not season: one state node per `(entity, race)`, chained directionally.

**Node types**

- `driver_state@(T,k)` — the driver's latent skill state *just before* race `(T,k)` (after all races with `year < T`, or `year == T and round < k`). One node per `(driver, race)` in which the driver appears (~27k). Input features are the driver's **static** attributes (`dob`, `nationality`, `code`, …); the temporal evidence is injected via message passing.
- `constructor_state@(T,k)` — the team's latent state just before race `(T,k)`, one per `(constructor, race)` (~13k). Input features are the constructor's static attributes.
- `results@(T,k)` — the raw race-result evidence (position, points, grid, …). A leaf node.
- `qualifying@(T,k)` — one per `(driver, race)`; the **target node** (features: `number`, `date`; label: `position`).
- `race@(T,k)` — one per race (features: round, year).
- `circuit` — static (one per circuit).

**Edge types (all directional, causal)**

- `same_driver`: `driver_state@(T,k-1) → driver_state@(T,k)` — the within-season recurrence carrying a driver's skill forward one round.
- `same_driver_cross`: `driver_state@(T-1, last) → driver_state@(T, 1)` — carries skill across the season boundary. Only exists if the driver raced in season T-1; a rookie's `driver_state@(T,1)` has no predecessor and relies on static features.
- `same_constructor` / `same_constructor_cross`: the team analogue.
- `result_of_driver`: `results@(T,k-1) → driver_state@(T,k)` — the previous race's result feeds the state.
- `result_of_constructor`: `constructor_results@(T,k-1) → constructor_state@(T,k)` (team evidence comes from the `constructor_results` table — one row per (constructor, race) — not `results`, which is per-driver).
- `circuit → race` — the race node aggregates its circuit.
- `race → qualifying` — the target aggregates the race's circuit/era context.
- **Context edges**: `driver_state@(T,k) → qualifying@(T,k)` and `constructor_state@(T,k) → qualifying@(T,k)` — the target aggregates its driver's and team's pre-race state.

(Message passing convention: in every edge `src → dst`, the *destination* aggregates the source. So `qualifying` aggregates `driver_state`, `constructor_state`, and `race`; `race` aggregates `circuit`.)

**Why temporal evolution is representable, and why it is leak-free.** The recurrence

```
results@(T,k-1) ─(result_of_driver)→ driver_state@(T,k) ─(same_driver)→ driver_state@(T,k+1) → …
```

means `driver_state@(T,k)` encodes exactly the driver's history up to round `k-1`, and nothing after. `verstappen@(2023, 5)` sees 2023 rounds 1–4 plus all prior seasons; `verstappen@(2017, 5)` sees only 2015–2017 rounds 1–4. The SAGE can therefore learn different embeddings for a young, sparse-history Verstappen vs a mature one, and the single-Linear readout reflects that. Because the chain is acyclic in time, the receptive field of every node is exactly its causal past — **no edge masking is required anywhere**.

Message flow:

```
results@(T,k-1) ──> driver_state@(T,k) ──(same_driver)──> driver_state@(T,k+1) ──> …
results@(T,k-1) ──> constructor_state@(T,k) ─(same_constructor)─> …

qualifying@(T,k)  ── aggregates ──>  driver_state@(T,k) + constructor_state@(T,k)
                                     + race@(T,k) ──> circuit
        └─> Linear(hidden, 1) ──> predicted grid position
```

**Depth note (expressiveness, not correctness).** The temporal chain spans up to ~72 seasons × ~20 rounds ≈ 1400 hops, far beyond a 2–3-layer SAGE's reach. This is acceptable for the target: predicting *current* qualifying depends overwhelmingly on recent form and the driver/team static identity, both of which are within a few hops (the static identity is baked into the state node's own features). Long-range career-arc effects are a secondary signal; if they matter, depth is the tuning knob (start at 2–3 layers). This is a modelling choice, not a leakage risk.

## Model

1. **`HeteroEncoder`** (input side, "por baixo") — encodes categorical/numerical/timestamp columns to embeddings. Not part of the skeleton; does not harm attribution. Loaded via the `graph_meta.pt` snapshot (read *before* `build_graph` overwrites it, as fixed in the prior branch).
2. **SAGE stack** (`HeteroConv` of `SAGEConv`):

   - conv1 `mean` aggregation, `hidden_dim` channels.
   - conv2 `max` aggregation, `hidden_dim` channels (keep a latent representation, not a scalar).
   - Residual connection + `LayerNorm` on `driver_state`, `constructor_state` and `qualifying` (combats oversmoothing, stabilises training).
   - Edge-type dict covers the `result_of_*` evidence edges, the directional `same_*` temporal edges, and the context edges into `qualifying`. Depth is a hyperparameter (start at 2).
3. **Readout**: `Linear(hidden_dim, 1)` applied to the `qualifying` node embedding → scalar grid position.

**Removed** vs. `F1OrthogonalPipeline`: `classifier` MLP, `aux_piloto`/`aux_equipe` heads, `[driver||constructor]` fusion, pre-race feature concatenation, `OrthogonalSeparationLoss`.

## Leakage analysis (exhaustive)

The requirement is a model that is theoretically sound **from the start**, so every temporal leakage source is enumerated below and closed by *construction* (not by an ad-hoc mask that could be forgotten or misapplied).

**Split.** Fixed, year-based: train = **1950–2021**, val = **2022–2023**, test = **2024–2026** (most recent seasons). Implemented by setting `MIN_YEAR = 1950` in `src/config.py` while keeping `EXTENDED_VAL_TIMESTAMP = 2022-01-01` / `EXTENDED_TEST_TIMESTAMP = 2024-01-01`, so `_years_from_timestamps` yields exactly these ranges.

**Because the graph is acyclic in time, the split is purely a *label* mask over target nodes** — no edge masking is needed. Every `qualifying@(T,k)` node's receptive field is exactly `{events with (year, round) strictly before (T,k)}`. Concretely, each leakage source and its closure:

1. **Cross-season leakage** (`driver_state@2024` seeing 2025 data). *Closed by structure:* `same_driver_cross` points `@T-1 → @T` only. A 2024 node can reach 2023, 2022, … and never 2025.
2. **Within-season leakage** (`qualifying@(T,k)` seeing results of round `≥ k`). *Closed by structure:* `qualifying@(T,k)` → `driver_state@(T,k)`, which reaches at most `results@(T,k-1)` via `result_of_driver`. `results@(T,k)` (same race, *after* qualifying) is not in the ancestry.
3. **Target's own label** (`qualifying.position`). *Closed by the task:* the `qualifying-position` task removes `position` from the node's input features (mirrors `results-position`). The label is never a feature.
4. **`results.grid` as a proxy for the target.** `results.grid` is the starting grid ≈ the qualifying result of that race. *Closed by structure:* the only `results` in a target's ancestry is `results@(T,k-1)` (previous race), whose `grid` is *that* race's — legitimately past. `results@(T,k)` (whose `grid` is the current qualifying being predicted) is unreachable.
5. **Championship standings leakage.** *Closed by omission:* `standings`/`constructor_standings` are **not** wired as evidence in v1 (only `results` is). If added later, they must feed `driver_state@(T,k)` only for `round < k` — the same round-indexing rule.
6. **Non-leaking but noteworthy features.** `qualifying.number` (car number) and `qualifying.date` (the session timestamp) are known at prediction time and therefore not leakage. However `number` is a potential *shortcut* (the reigning champion often carries `#1`), and `date` lets the model learn era/grid-size effects. Both are legitimate, but the `number` shortcut should be checked (an ablation dropping it) so the skill signal isn't conflated with a number.
7. **Team continuity across renames.** `same_constructor_cross` follows the constructor id, so a rebrand (Toro Rosso → AlphaTauri → RB) is treated as a discontinuity. This is a *data-semantics* choice (an under-counting of team continuity), not a leakage: it can only make the model *less* informed, never more. Documented for completeness.

**Consequence for implementation.** No `add_edge_year_masks` / `add_edge_round_masks` calls are needed for correctness. The training loop can build one static graph over 1950–2026 and run a single forward pass, selecting target nodes by year for train/val/test labels. This is both simpler and less error-prone than the mask-based approach it replaces.

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
src/data/temporal_graph.py           # causal round-state graph builder (driver_state/constructor_state + same_*/result_of_* edges)
src/models/sage_regressor.py        # HeteroEncoder + causal SAGE stack + Linear readout
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

- **Day 1**: `temporal_graph.py` (causal round-state graph) + graph sanity checks (per-round node/edge counts, acyclicity check).
- **Day 2**: `sage_regressor.py` + training loop + baseline comparison + metrics report.
- **Day 3 (buffer)**: acyclicity/leakage unit checks (assert every node's ancestors are strictly earlier), produce the interpretability-ready checkpoint.
