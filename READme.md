# F1 Driver Skill — Validation-First Pipeline

Research project on **car-adjusted driver performance** in Formula 1 using a GNN-first architecture with model-agnostic validation benchmarks.

## Validation-first workflow

Define and run benchmarks **before** promoting any new model:

```bash
# 1. Export baselines
python src/experiments/run_bradley_terry.py --max-year 2025
python src/experiments/run_bayesian_ssm.py --start-year 2014 --end-year 2021 --smoke-test  # dev only

# 2. Unified benchmark (career gates + locked 2024–2025 PL + Shapley decomposition)
python src/experiments/run_validation_benchmark.py --sources bradley_terry

# 3. Publication plots
python src/experiments/plots/plot_team_tier_heatmap.py --start-year 2014 --end-year 2025
python src/experiments/plots/plot_driver_season_skill.py --source bradley_terry --season 2024 --driver verstappen
python src/experiments/plots/plot_driver_rank_evolution.py --source bradley_terry --driver verstappen --driver hamilton --driver leclerc --driver norris --start-year 2018 --end-year 2024
python src/experiments/plots/plot_entity_attribution.py --source bradley_terry --season 2024
```

## Contract

Every model exports `SkillExport` (`src/skill/contract.py`):

- Per-race: `raw_skill`, calibrated `skill_0_10` ∈ [0,10], uncertainty, driver/constructor/context contributions
- Per-season aggregates for career validation
- Inference modes: `filtered` (causal) vs `smoothed` (descriptive only)

See [docs/model_contract.md](docs/model_contract.md) for gates, baselines, and claim levels.

## Primary model (when trained)

```bash
python src/experiments/train_skill_gnn.py --seed 42
python src/experiments/run_validation_benchmark.py --sources skill_gnn bradley_terry bayesian_ssm
```

## OrthogonalShapleyGNN (candidate, arch v2)

Port of the historical SAGE+MLP orthogonal pipeline onto the causal temporal graph ([`src/models/orthogonal_shapley_gnn.py`](src/models/orthogonal_shapley_gnn.py)). Pre-race **context** combines normalized grid/round scalars with a **race-node embedding** (circuit/era signal from `circuit→race` message passing). Training target is **classified race finish order**.

**Data:** enriched RelBench F1 relational DB → `build_temporal_graph` ([`src/data/temporal_graph.py`](src/data/temporal_graph.py)): per-round `driver_state` / `constructor_state` / `race` nodes; race rows supply `position`, `grid`, `round`, `race_idx`, and state indices.

**Architecture (defaults):** RelBench `HeteroEncoder` → **3-layer** heterogeneous SAGE (`mean` then `max`, hidden **64**) with residual `LayerNorm` on `driver_state`, `constructor_state`, and `race` → `context_mlp([grid_norm, round_norm, race_emb])` → concatenate `[driver_emb ‖ constructor_emb ‖ ctx]` → **3-layer** fusion MLP (`mlp_hidden=64`); auxiliary linear heads on driver, constructor, and context channels.

**Loss:**

```
total = PL_fused
      + 0.5 × PL_driver_aux
      + 0.75 × PL_constructor_aux
      + 0.25 × PL_context_aux
      + 0.25 × pairwise_ranking(fused)
      + 0.1 × attribution_balance(Shapley subsample)
      + λ_orth(epoch) × orth_loss
```

- `orth_loss` = squared cosine similarity of driver↔constructor embeddings **and** driver↔context (default `λ_orth=2.0`, 10-epoch warmup).
- `attribution_balance` penalizes driver Shapley share above 38% on a 20% race subsample.
- Model selection: composite `val_pl + 0.5 × (1 − val_pairwise_acc)`.

**Skill readout:** at export, driver skill is the exact 3-coalition Shapley value (driver / constructor / projected context) of the fused MLP utility, with train-only baselines ([`src/explain/coalition_shapley.py`](src/explain/coalition_shapley.py)).

**Train & validate:**

```bash
# Default v2 config (3L/64)
python src/experiments/train_orthogonal_shapley_gnn.py --seed 42

# SkillGNN-scale ablation preset
python src/experiments/train_orthogonal_shapley_gnn.py --seed 42 --num-layers 4 --hidden-dim 128 --mlp-hidden 128

python src/experiments/run_orthogonal_shapley_pipeline.py --stages all
python src/experiments/run_validation_benchmark.py --sources orthogonal_shapley bradley_terry
```

**Target gates (locked 2024–2025):** pairwise acc ≥ BT − 0.01 (~0.685), PL NLL ≤ BT + 0.01, driver Shapley share 35–45%, context share 15–25%, partial ρ CI low > 0.

## Dependencies

Core: `torch`, `torch-geometric`, `relbench`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pyarrow`

Bayesian baseline: `cmdstanpy`, `arviz` (+ [CmdStan](https://mc-stan.org/users/interfaces/cmdstan) installed separately)
