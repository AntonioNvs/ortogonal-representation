# Validation benchmark report

## bradley_terry
- Partial ρ: 0.1425771967182368 (CI low: 0.054730752085572563)
- Underrated resolution: 0.673469387755102 (CI low: 0.5, n=49)
- Underrated promotion AUROC: 0.5662878787878788
- Locked PL NLL: 1.8892964124679565
- Locked pairwise acc: 0.6951018161805174

## bayesian_ssm
- Partial ρ: 0.43417778004866686 (CI low: 0.1442892918085608)
- Underrated resolution: 1.0 (CI low: 1.0, n=6)
- Underrated promotion AUROC: nan
- Locked PL NLL: nan
- Locked pairwise acc: nan

## orthogonal_shapley
- Partial ρ: 0.269663660234158 (CI low: 0.17756569155900787)
- Underrated resolution: 0.8 (CI low: 0.5263157894736842, n=15)
- Underrated promotion AUROC: 0.6666666666666666
- Locked PL NLL: 1.8732308149337769
- Locked pairwise acc: 0.6834067143643369

## Gates
- **bradley_terry**: partial=False, resolution=False, underrated_auroc=False, pl=True, pairwise=True
- **bayesian_ssm**: partial=True, resolution=True, underrated_auroc=False, pl=False, pairwise=False
- **orthogonal_shapley**: partial=True, resolution=True, underrated_auroc=False, pl=True, pairwise=False