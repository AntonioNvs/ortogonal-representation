# Career Validation Report

- Generated at: `2026-08-18T02:36:54.629324+00:00`
- Skill source: `bradley_terry`
- Tier window: `3` seasons (trailing)
- Forward horizon: `3` seasons
- Tier proportions: S=`30%`, A=`35%`, B=remainder
- Joined (driver, season) rows: `287` across `46` drivers

## Headline: skill vs. forward tier outcome

| Statistic | Value | 95% CI | p |
|---|---:|:---:|---:|
| **Spearman rho** (cluster-bootstrap by driverId) | **+0.4056** | [+0.2341, +0.5571] | — |
| Fisher-z pooled rho (across 24 seasons) | +0.4676 | [+0.3569, +0.5653] | — |
| Within-season permutation | rho=+0.4056 | — | **5e-05** |
| Kendall tau | +0.3004 | — | 2.5e-12 |

## Effect size — does skill add signal *above* the driver's current team?

| Metric | Value | 95% CI |
|---|---:|:---:|
| **Partial Spearman rho** (controlling for tier(T)) | **+0.2175** | [+0.0701, +0.3652] |
| **AUROC 'moved up a tier'** | **0.4312** | [0.3183, 0.5519] |

AUROC computed over 287 rows, 75 positive (driver ended up at a strictly higher-tier team on average).

**Interpretation.** The cluster-bootstrap CI is the honest interval — it treats each driver's sequence as one unit of information rather than one per (driver, season) row. The within-season permutation p-value tests whether skill orders drivers *within* a season, on top of what the grid already dictates. Partial rho is the sharpest test of the paper's thesis: if it stays positive with a CI not crossing zero, the skill score carries signal that 'the driver is at a Tier-S team right now' cannot explain.

### Diagnostic (do not quote — assumes iid rows)

- Naive row-bootstrap 95% CI: [+0.2992, +0.5015]
- Naive iid p-value from scipy.stats.spearmanr: `8.59e-13`

## Per-Season Spearman

| Season | n | Spearman rho |
|---|---:|---:|
| 2000 | 10 | +0.8899 |
| 2001 | 12 | +0.3091 |
| 2002 | 13 | +0.5915 |
| 2003 | 14 | +0.3839 |
| 2004 | 12 | -0.0108 |
| 2005 | 11 | -0.0753 |
| 2006 | 11 | -0.1376 |
| 2007 | 13 | +0.3026 |
| 2008 | 13 | +0.2814 |
| 2009 | 10 | +0.0137 |
| 2010 | 8 | -0.4910 |
| 2011 | 9 | +0.5533 |
| 2012 | 12 | +0.7430 |
| 2013 | 12 | +0.5215 |
| 2014 | 14 | +0.4311 |
| 2015 | 13 | +0.7031 |
| 2016 | 11 | +0.7433 |
| 2017 | 13 | +0.6774 |
| 2018 | 11 | +0.2872 |
| 2019 | 12 | +0.3025 |
| 2020 | 12 | +0.6881 |
| 2021 | 14 | +0.6792 |
| 2022 | 13 | +0.7504 |
| 2023 | 14 | +0.3808 |
