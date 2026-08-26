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
- Lower is better; report MAE / RMSE / Spearman ρ (and optionally a "top-3 in qualifying" AUROC, mirroring the existing `auroc_top3` idiom).

## Graph schema

Reuse the existing `make_pkey_fkey_graph` heterogeneous graph (node types `drivers`, `constructors`, `circuits`, `races`, `results`, `qualifying`, `standings`, `constructor_results`, `constructor_standings`).

**Critical change — bidirectional edges.** In the current graph, `qualifying → drivers` and `qualifying → constructors` run *only* source→dest; the `qualifying` node never receives a message, so a prediction on the `qualifying` node has no access to who the driver/team is. We add reverse edges:

- `drivers → qualifying` (via `f2p_driverId`)
- `constructors → qualifying` (via `f2p_constructorId`)

This lets the target node aggregate the driver/constructor embeddings, which in turn aggregate each entity's full historical neighbourhood. Message flow:

```
history (past results / qualifying / standings)
   └─> drivers, constructors      (aggregate their own history)
          └─[reverse edge]─> qualifying   (aggregates driver + constructor)
                 └─> SAGE → single Linear → grid position
```

Reverse edges are not a new model family — they are the standard way to make a heterogeneous GNN bidirectional so a *leaf* node (a row we want to predict on) can attend to its parents. 100% message passing; no readout complexity added.

## Model

1. **`HeteroEncoder`** (input side, "por baixo") — encodes categorical/numerical/timestamp columns to embeddings. Not part of the skeleton; does not harm attribution. Loaded via the `graph_meta.pt` snapshot (read *before* `build_graph` overwrites it, as fixed in the prior branch).

2. **SAGE stack** (`HeteroConv` of `SAGEConv`):
   - conv1 `mean` aggregation, `hidden_dim` channels.
   - conv2 `max` aggregation, `hidden_dim` channels (keep a latent representation, not a scalar).
   - Residual connection + `LayerNorm` on the target node types (combats oversmoothing, stabilises training).
   - Reverse edges included in the `HeteroConv` edge-type dict.

3. **Readout**: `Linear(hidden_dim, 1)` applied to the `qualifying` node embedding → scalar grid position.

**Removed** vs. `F1OrthogonalPipeline`: `classifier` MLP, `aux_piloto`/`aux_equipe` heads, `[driver||constructor]` fusion, pre-race feature concatenation, `OrthogonalSeparationLoss`.

## Temporal isolation (no leakage)

The graph spans 2000–2026. A qualifying prediction for race R must only see information available *before* R. Reuse the existing year-mask / round-mask machinery (`add_edge_year_masks`, `add_edge_round_masks`) keyed on the source table's `raceId`:

- **Split masks**: train edges = `TRAIN_YEARS`, val edges = `TRAIN_YEARS ∪ VAL_YEARS`, test edges = all (same as the current fixed-split scheme in `prepare_data`).
- **Reverse edges are masked consistently with their forward counterparts** — a reverse `drivers → qualifying` edge is the same `(driver, qualifying)` pair as the forward `qualifying → drivers` edge, so it inherits the same temporal mask. This is the one subtle correctness point: masking must be defined on the *underlying row* (keyed by the source table's `raceId`), not on the edge direction.

For the initial test, the fixed year-based split is sufficient; walk-forward (round-by-round) masking is a follow-up only if the fixed split shows signal.

## Evaluation

- **Metrics**: MAE, RMSE, Spearman ρ (primary); optional top-3-in-qualifying AUROC (secondary, ranking-flavoured).
- **Baseline**: constant predictor (predict the mean/median grid position) and, if cheap, a `Linear`-on-raw-`qualifying`-features model — to show the SAGE over the graph adds signal over the node's own features.
- Report val/test on the fixed split, with the standard seed handling.

## Files to create (in this branch)

```
src/models/sage_regressor.py        # HeteroEncoder + bidirectional SAGE stack + Linear readout
src/training/sage_train.py          # training loop (or extend train.py minimally)
src/experiments/sage_qualifying_run.py   # end-to-end runner
```

Reuse (from `master`): `src/data/tasks.py`, `src/data/enriched_dataset.py`, `build_graph`/`get_active_task` in `train.py`, and the `graph_meta.pt` snapshot convention.

## Exit criteria

- The model **trains** and beats the constant-predictor baseline on the qualifying-position target (MAE / Spearman).
- The readout is a single `Linear` (no MLP/aux heads/fusion) — a clean substrate for a future `GNNExplainer` / gradient-attribution pass.
- (Informal) a face-validity check: top drivers (Verstappen, Hamilton, Leclerc…) tend to get low (good) predicted grid positions.

If the model does not beat baseline, record why before adding complexity — the point of this branch is a minimal, honest substrate, not peak accuracy.

## Timeline (indicative)

- **Day 1**: `sage_regressor.py` + graph reverse-edge plumbing + temporal masks.
- **Day 2**: training loop + baseline comparison + metrics report.
- **Day 3 (buffer)**: fix leakage/masking edge cases, produce the interpretability-ready checkpoint.
