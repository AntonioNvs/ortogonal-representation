# F1 Car-Adjusted Driver Performance

Open-source research code for ranking Formula 1 drivers by **car-adjusted
performance** — isolating the driver signal from car, team, and race context
on a causal temporal graph.

Prepared for the **MIT Sloan Sports Analytics Conference (SSAC27)** Research
Paper Competition (*Other Sports* track).

**Repository:** https://github.com/AntonioNvs/ortogonal-representation

> Claims are framed as **car-adjusted performance**, not "pure skill", unless
> disentanglement gates pass. See [`docs/model_contract.md`](docs/model_contract.md).

## Estimand

For driver **D**, team **T**, race **R**:

```text
systematic(D,T,R) = driver(D,R) + constructor(T,R) + context(R)
f(D,T,R)          = driver(D,R)   # exported skill readout (higher = better)
```

Cumulative season skill at round *r* uses only races 1…*r* (`filtered` / causal
mode). Context covers modeled race-level non-driver/non-constructor effects
(grid, circuit/event). Residual chance is reported separately.

## Method (headline)

**OrthogonalShapleyGNN** ([`src/models/orthogonal_shapley_gnn.py`](src/models/orthogonal_shapley_gnn.py)):

- Causal round-state hetero graph ([`src/data/temporal_graph.py`](src/data/temporal_graph.py))
- Heterogeneous SAGE encoder + MLP fusion over `[driver ‖ constructor ‖ context]`
- Trained with Plackett–Luce NLL on classified race finishes (+ aux heads, orthogonal loss)
- Skill export = exact 3-coalition Shapley value of the driver channel

**Baselines:** walk-forward Bradley–Terry, Bayesian state-space (Lindner et al.),
SkillGNN, teammate-residual. See [`docs/orthogonal-shapley-approach.md`](docs/orthogonal-shapley-approach.md).

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Enriched RelBench F1 DB ships in-repo (data/enriched/rel-f1/).
# Rebuild from RelBench + Jolpica/Ergast if needed:
python -m src.data.pipeline build
```

### Reproduce the abstract results

```bash
# Frozen Model A checkpoint is under output/orthogonal_shapley_model/
python src/experiments/run_validation_benchmark.py \
  --sources orthogonal_shapley bradley_terry bayesian_ssm

# Application figures (GOAT equal-car, undervalued screen)
python src/experiments/run_goat_equal_car.py
python src/experiments/run_undervalued_backtest.py
python src/experiments/plots/plot_ssac27_abstract_figure.py
```

Published benchmark numbers live in
[`output/validation_benchmark/benchmark.md`](output/validation_benchmark/benchmark.md).

Full command list, seeds, and locked splits:
[`docs/reproducibility.md`](docs/reproducibility.md).

### Retrain OrthogonalShapleyGNN

```bash
python src/experiments/train_orthogonal_shapley_gnn.py --seed 42
python src/experiments/run_orthogonal_shapley_pipeline.py --stages all
```

## Validation gates

Primary career gate (era ≥ 2014, fixed model-free cohort):

| Gate | Criterion |
|------|-----------|
| Partial Spearman ρ (skill vs rest-of-career tier, residualised on constructor tier) | ρ > 0; cluster CI low > 0 |
| Locked test 2024–2025 (race PL NLL / pairwise acc) | beat or match Bradley–Terry (±0.01) |
| Shapley driver / constructor / context shares | diagnostic; report, do not over-claim |

Details: [`docs/career_validation_framework.md`](docs/career_validation_framework.md).

## Repository layout

```text
data/enriched/rel-f1/   # Shared RelBench F1 tables (parquet) + manifest
docs/                   # Contract, validation, reproducibility, SSAC rules
output/                 # Frozen Model A, benchmarks, abstract figures
src/
  applications/         # GOAT equal-car helpers
  baselines/            # BT, Bayesian SSM, SkillGNN, OrthogonalShapley loaders
  data/                 # Pipeline, temporal graph, RelBench tasks
  experiments/          # Train / validate / plot CLIs
  explain/              # Coalition Shapley + XAI probes
  models/               # OrthogonalShapleyGNN, SkillGNN, SAGE regressor
  skill/                # SkillExport contract + calibration
  validation/           # Career metrics, benchmark runner, tiers
  visualization/        # Publication plot builders
tests/                  # Unit / smoke tests
```

## Data

- **In-repo:** `data/enriched/rel-f1/` (≈1.4 MB parquet tables through ~2025/26).
- **Sources:** [RelBench](https://relbench.stanford.edu/) `rel-f1`, enriched with
  [Ergast](http://ergast.com/mrd/) / [Jolpica](https://github.com/jolpica/jolpica-f1)
  for recent seasons. Raw snapshots under `data/raw/` are gitignored; rebuild with
  `python -m src.data.pipeline build`.
- Shared per SSAC open-source requirements. Private fields are not present.

## License

MIT — see [`LICENSE`](LICENSE). Upstream RelBench / Ergast / Jolpica retain their
own licenses; cite them when redistributing derived data.
