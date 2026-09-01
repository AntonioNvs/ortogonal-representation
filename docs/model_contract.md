# Model contract — validation-first driver skill (2026-08-31)

Single source of truth for **what every skill model must export**, **how we validate before promotion**, and **how we evaluate**.

---

## Estimand

For driver **D**, team **T**, race **R** (round k of season year):

```text
systematic(D,T,R) = driver(D,R) + constructor(T,R) + context(R)
f(D,T,R)          = driver(D,R)   # exported skill readout (higher = better)
skill_0_10        = 10 * sigmoid(alpha * (f - mu_train) / sigma_train)
```

- **Cumulative season skill** at round r: mean of `f(D,T,R)` over rounds 1…r only (**filtered / causal** mode).
- Do **not** call outputs "pure skill" unless disentanglement gates pass.
- **Context** = modeled race-level non-driver/non-constructor effects (grid, circuit/event terms). **Residual/chance** is separate and never folded into context.

## Inference modes

| Mode | Definition | Allowed uses |
|------|------------|--------------|
| `filtered` | Only races 1…R when scoring round R | Career gates, locked test, ranking export |
| `smoothed` | Full-interval posterior / descriptive smoothing | Paper-style plots only; **never** headline gates |

## Mandatory model capabilities

1. **Per-race skill on [0, 10]** — anchored logistic calibration fit on **training years only**; uncertainty propagated through the same monotone map.
2. **Temporal evolution** — race-level `f(D,T,R)` and cumulative as-of-round summaries.
3. **Entity decomposition** — normalized **Shapley variance shares** for driver / constructor / context (sum to 100%); residual reported separately.

## Architecture (non-negotiable for primary GNN)

| Role | Model | File |
|------|-------|------|
| Graph substrate | Causal round-state `HeteroData` | `src/data/temporal_graph.py` |
| **Primary skill GNN** | SkillGNN — PL ranking on race results | `src/models/skill_gnn.py` |
| **GNN baseline (predictive)** | SAGE qualifying regressor (4/128) | `src/models/sage_regressor.py` |
| **Benchmark BT** | Walk-forward race-level Bradley–Terry | `src/baselines/bradley_terry_skill.py` |
| **Benchmark Bayesian** | Lindner et al. state-space (Stan/NUTS) | `src/baselines/bayesian_ssm.py` |
| Simple baseline | Teammate-residual | `src/baselines/teammate_residual.py` |

## Common export contract

All skill sources implement `SkillExport` (`src/skill/contract.py`):

**Race columns:** `driverId`, `season`, `round`, `raceId`, `constructorId`, `lineage_id`, `driver_name`, `constructor_name`, `raw_skill`, `skill_0_10`, `skill_lo`, `skill_hi`, `contrib_driver`, `contrib_constructor`, `contrib_context`, `contrib_residual`, `skill_source`, `inference_mode`, `as_of_round`

**Season columns:** `driverId`, `season`, `skill_score`, `skill_0_10`, `skill_lo`, `skill_hi`, `skill_source`, `inference_mode`, `as_of_round`, `n_obs`, `support_bucket`

Artifacts written under `output/skill_exports/{source}/` as `race_skill.parquet`, `season_skill.csv`, `metadata.json`.

## Validation (headline — run **before** trusting any new model)

### 1. Contract & data integrity
- Temporal cutoff respected (`filtered` mode)
- Lineage continuity (rebrands: Sauber → Audi, etc.)
- Mobility / support flags

### 2. Score behavior
- [0,10] bounds, calibration anchors (~1/5/9 at −2/0/+2 train SD)
- IQR, central mass, saturation diagnostics
- Uncertainty width / coverage

### 3. Locked test (2024–2025)
- Race Plackett–Luce NLL and pairwise accuracy (true per-race, not global proxy)
- Qualifying log-score / RMSE where supported
- Bayesian: posterior predictive checks + MCMC diagnostics (R-hat, ESS, divergences)

### 4. Disentanglement
- Constructor leakage |ρ| < 0.3 (SkillGNN XAI)
- Swap invariance on driver readout
- Shapley driver/constructor/context shares

### 5. Career validity (primary gate)
- **Partial Spearman ρ ≥ 0.15** with cluster CI low > 0 (skill vs forward tier, controlling tier-at-T)
- Within-season permutation p-value
- **Eligible promotion AUROC** (below S-tier at T only)

### 6. Robustness
- DNF policies (classified / finished / all entries)
- ≥5 seeds where stochastic
- Era windows (e.g. hybrid 2014–2021 for Bayesian replication)

**Do not gate on:** raw Spearman alone, all-driver moved-up AUROC.

## Publication plots (CLI)

| Plot | Command |
|------|---------|
| Team tier heatmap | `python src/experiments/plots/plot_team_tier_heatmap.py` |
| Season skill trajectory | `python src/experiments/plots/plot_driver_season_skill.py` |
| Multi-season rank panels | `python src/experiments/plots/plot_driver_rank_evolution.py` |
| Shapley attribution bars | `python src/experiments/plots/plot_entity_attribution.py` |

All plots: seaborn styling, English labels, DB-backed proper names, PNG+SVG+PDF, sidecar metadata JSON.

## Canonical commands

```bash
# Build enriched DB (if missing)
python -m src.data.pipeline build

# Benchmark baselines (validation-first)
python src/experiments/run_bradley_terry.py --max-year 2025 --output-dir output/skill_exports/bradley_terry
python src/experiments/run_bayesian_ssm.py --start-year 2014 --end-year 2021 --output-dir output/skill_exports/bayesian_ssm

# Unified validation benchmark
python src/experiments/run_validation_benchmark.py --sources bradley_terry bayesian_ssm

# Primary GNN (when ready)
python src/experiments/train_skill_gnn.py --seed 42
python src/experiments/run_validation_benchmark.py --sources skill_gnn bradley_terry bayesian_ssm

# Plots
python src/experiments/plots/plot_team_tier_heatmap.py --start-year 2014 --end-year 2025
python src/experiments/plots/plot_driver_season_skill.py --source bradley_terry --season 2024 --driver verstappen
python src/experiments/plots/plot_driver_rank_evolution.py --source bradley_terry --driver verstappen --driver hamilton --driver leclerc --driver norris --start-year 2018 --end-year 2024
python src/experiments/plots/plot_entity_attribution.py --source bradley_terry --season 2024
```
