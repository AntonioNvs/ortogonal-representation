# Validation benchmark report

## bradley_terry
- Partial ρ: 0.1066283886947603 (CI low: -0.07794884501343767)
- Underrated resolution: 0.46153846153846156 (CI low: 0.18518518518518517, n=26)
- Underrated promotion AUROC: 0.40476190476190477
- Locked PL NLL: 1.8892964124679565
- Locked pairwise acc: 0.6951018161805174

## bayesian_ssm
- Partial ρ: 0.32373370760456127 (CI low: 0.12497723282085876)
- Underrated resolution: 0.46153846153846156 (CI low: 0.18518518518518517, n=26)
- Underrated promotion AUROC: 0.7380952380952381
- Locked PL NLL: 1.915357232093811
- Locked pairwise acc: 0.6927627958172813

## orthogonal_shapley
- Partial ρ: 0.3999437620155319 (CI low: 0.19421245305179574)
- Underrated resolution: 0.46153846153846156 (CI low: 0.18518518518518517, n=26)
- Underrated promotion AUROC: 0.6785714285714286
- Locked PL NLL: 1.8035427331924438
- Locked pairwise acc: 0.7491744634012107

## Gates
- **bradley_terry**: partial=False, resolution=False, underrated_auroc=True, pl=True, pairwise=True
- **bayesian_ssm**: partial=True, resolution=False, underrated_auroc=True, pl=False, pairwise=True
- **orthogonal_shapley**: partial=True, resolution=False, underrated_auroc=False, pl=True, pairwise=True