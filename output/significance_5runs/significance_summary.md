# Ortho Model Statistical Comparison

- Generated at: `2026-06-06T14:16:52.940139+00:00`
- Metric: `auroc`

## AUROC Per Model

| Model | Mean | Std | 95% CI | n |
|---|---:|---:|---:|---:|
| high | 0.883301 | 0.004367 | [0.880362, 0.887097] | 5 |
| low | 0.875826 | 0.006497 | [0.870853, 0.881578] | 5 |
| zero | 0.864589 | 0.010255 | [0.856147, 0.872647] | 5 |

## Pairwise Significance

| Pair | Delta Mean (A-B) | 95% CI | p-value | p-adjusted (Holm) | Reject H0 | n |
|---|---:|---:|---:|---:|---:|---:|
| low_vs_high | -0.007475 | [-0.012125, -0.004742] | 0.062997 | 0.062997 | False | 5 |
| zero_vs_high | -0.018711 | [-0.024572, -0.013420] | 0.061497 | 0.184491 | False | 5 |
| zero_vs_low | -0.011237 | [-0.017879, -0.005296] | 0.062697 | 0.125394 | False | 5 |
