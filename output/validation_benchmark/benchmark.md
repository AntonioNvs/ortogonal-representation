# Validation benchmark report

## bradley_terry
- Partial ρ: 0.10978738918820966 (CI low: -0.07250624184557873)
- Underrated resolution: 0.46153846153846156 (CI low: 0.18518518518518517, n=26)
- Underrated promotion AUROC: 0.4107142857142857
- Locked PL NLL: 1.88713538646698
- Locked pairwise acc: 0.6938635112823335

## bayesian_ssm
- Partial ρ: 0.32373370760456127 (CI low: 0.12497723282085876)
- Underrated resolution: 0.46153846153846156 (CI low: 0.18518518518518517, n=26)
- Underrated promotion AUROC: 0.7380952380952381
- Locked PL NLL: 1.915357232093811
- Locked pairwise acc: 0.6927627958172813

## orthogonal_shapley
- Partial ρ: 0.38273632382457257 (CI low: 0.15758369439996373)
- Underrated resolution: 0.46153846153846156 (CI low: 0.18518518518518517, n=26)
- Underrated promotion AUROC: 0.6964285714285714
- Locked PL NLL: 1.8093644380569458
- Locked pairwise acc: 0.7429829389102917

## Gates
- **bradley_terry**: partial=False, resolution=False, underrated_auroc=True, pl=True, pairwise=True
- **bayesian_ssm**: partial=True, resolution=False, underrated_auroc=True, pl=False, pairwise=True
- **orthogonal_shapley**: partial=True, resolution=False, underrated_auroc=False, pl=True, pairwise=True