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

## Dependencies

Core: `torch`, `torch-geometric`, `relbench`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pyarrow`

Bayesian baseline: `cmdstanpy`, `arviz` (+ [CmdStan](https://mc-stan.org/users/interfaces/cmdstan) installed separately)
