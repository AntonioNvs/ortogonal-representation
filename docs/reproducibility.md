# Reproducibility

Exact commands, seeds, and locked splits for the SSAC27 abstract results.

## Environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

- Python 3.11+ recommended.
- GPU (CUDA) recommended for OrthogonalShapleyGNN / SkillGNN training; CPU is
  enough for Bradley–Terry and for loading the frozen checkpoint.
- Bayesian SSM baseline additionally needs [CmdStan](https://mc-stan.org/users/interfaces/cmdstan)
  (`cmdstanpy`, `arviz` are in `requirements.txt`).

## Data

The enriched RelBench F1 database ships in-repo:

```text
data/enriched/rel-f1/
  manifest.json
  sprint_results.parquet
  db/*.parquet          # circuits, constructors, drivers, races, results, …
```

Rebuild (optional):

```bash
python -m src.data.pipeline build
python -m src.data.validate_enriched
```

Raw Jolpica / f1db dumps under `data/raw/` are gitignored.

## Locked protocol

| Item | Value |
|------|-------|
| Seed | `42` (all stochastic entrypoints) |
| Career era window | seasons ≥ **2014** |
| Locked ranking test | races in **2024–2025** |
| Inference mode for gates | `filtered` (causal; races 1…R only) |
| Underrated cohort | fixed once from model-free `teammate_residual` |
| Frozen checkpoint | `output/orthogonal_shapley_model/` (Model A, arch v3, 4×128) |

Do **not** use `smoothed` inference for headline gates.

## Reproduce abstract numbers (no retraining)

Frozen Model A weights are committed under `output/orthogonal_shapley_model/`.

```bash
# Unified benchmark (career gates + locked 2024–2025 PL + Shapley shares)
python src/experiments/run_validation_benchmark.py \
  --sources orthogonal_shapley bradley_terry bayesian_ssm

# Application scripts used in the abstract figure
python src/experiments/run_goat_equal_car.py
python src/experiments/run_undervalued_backtest.py
python src/experiments/plots/plot_ssac27_abstract_figure.py
```

Expected artifacts:

- `output/validation_benchmark/benchmark.json` / `benchmark.md`
- `output/applications/goat_equal_car/`
- `output/applications/undervalued/`
- `output/applications/ssac27_abstract_figure/`

## Retrain from scratch

```bash
# Primary model
python src/experiments/train_orthogonal_shapley_gnn.py --seed 42
python src/experiments/run_orthogonal_shapley_pipeline.py --stages all

# Walk-forward Bradley–Terry baseline
python src/experiments/run_bradley_terry.py --max-year 2025

# Bayesian state-space (optional; needs CmdStan)
python src/experiments/run_bayesian_ssm.py --start-year 2014 --end-year 2025

# SkillGNN baseline (optional ablation)
python src/experiments/train_skill_gnn.py --seed 42

# Re-run the unified benchmark against freshly exported skills
python src/experiments/run_validation_benchmark.py \
  --sources orthogonal_shapley bradley_terry bayesian_ssm
```

Default training output directory for OrthogonalShapleyGNN is
`output/orthogonal_shapley_model/` (same path as the frozen abstract checkpoint —
back it up before overwriting).

## Publication plots

```bash
python src/experiments/plots/plot_validation_figures.py \
  --benchmark-json output/validation_benchmark/benchmark.json
python src/experiments/plots/plot_entity_attribution.py \
  --source orthogonal_shapley --season 2024
python src/experiments/plots/plot_team_tier_heatmap.py \
  --start-year 2014 --end-year 2025
python src/experiments/plots/plot_driver_season_skill.py \
  --source orthogonal_shapley --season 2024 --driver verstappen
python src/experiments/plots/plot_driver_rank_evolution.py \
  --source orthogonal_shapley \
  --driver verstappen --driver hamilton --driver leclerc --driver norris \
  --start-year 2018 --end-year 2024
```

## Tests

```bash
python -m pytest tests -q
```

## Claim language

Export and discuss results as **car-adjusted performance**. Reserve "pure skill"
wording only if disentanglement gates in [`model_contract.md`](model_contract.md)
pass.
