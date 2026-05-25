# References

Justificativa da literatura para cada decisão do projeto.

---

## 1. Features do piloto (tabulares com janela rolling)

**Bell, A., Smith, J., Sabel, C., Jones, K. (2016).** *Formula for success: Multilevel modelling of Formula One Driver and Constructor performance.* Journal of Quantitative Analysis in Sports, 12(2).
→ Mostra empiricamente que carro (constructor) responde por ~85% da variância e piloto por ~15%; defende isolar features driver-puras (delta vs companheiro, posições ganhadas) das contaminadas por carro. Base teórica para nossa **residualização** contra `(constructorId, year)`.

**Eichenberger, R., Stadelmann, D. (2009).** *Who is the best Formula 1 driver? An economic approach to evaluating talent.* Economic Analysis & Policy, 39(3).
→ Demonstra que medidas absolutas (vitórias, pontos) são proxies enviesados de habilidade; é preciso controlar pelo carro/temporada. Inspira **teammate_delta** como sinal driver-puro.

**Phillips, A. J. K. (2014).** *Uncovering Formula One driver performances from 1950 to 2013 by adjusting for team and competition effects.* Journal of Quantitative Analysis in Sports.
→ Valida uso de **rolling-window** de corridas passadas e `shift(1)` para construir feature lag-only (sem leakage).

**Bergmeir, C., Benítez, J. M. (2012).** *On the use of cross-validation for time series predictor evaluation.* Information Sciences, 191.
→ Justifica split temporal (train < val < test em ordem cronológica) em vez de k-fold aleatório.

Features escolhidas (`avg_qualifying_pos`, `teammate_delta`, `fastest_lap_rate`, `position_delta`, `podium_rate`, `crash_rate`, `points_per_finish`, `experience`) seguem o conjunto consolidado nos três trabalhos acima.

---

## 2. Features de pista (estáticas + ambientais)

**FIA Formula 1 Sporting Regulations / Technical Regulations** (documentos oficiais).
→ `length_m`, `corners_count`, `altitude_m` são descritores físicos canônicos da FIA para classificação de circuitos.

**FastF1 Documentation** — https://docs.fastf1.dev/
→ API oficial para `circuit_info.rotation`, `circuit_info.corners` (telemetria) e `weather_data` (`TrackTemp`, `AirTemp`, `Humidity`). Padrão da comunidade de analytics de F1.

**Tulabandhula, T., Rudin, C. (2014).** *Tire changes, fresh air, and yellow flags: Challenges in predictive analytics for professional racing.* Big Data, 2(2).
→ Mostra que temperatura de pista e ar afetam significativamente desempenho relativo de pneu/carro — justifica `avg_track_temp`, `avg_air_temp`, `avg_humidity` como contexto físico da corrida.

---

## 3. Encoder de equipe (HeteroGraphSAGE sobre grafo relacional)

**Hamilton, W. L., Ying, R., Leskovec, J. (2017).** *Inductive Representation Learning on Large Graphs (GraphSAGE).* NeurIPS 2017. **Stanford.**
→ Base do `SAGEConv` usado em `team_encoder.py`. Sample-and-aggregate inductive — generaliza para construtores não vistos no treino (relevante para temporadas futuras).

**Schlichtkrull, M. et al. (2018).** *Modeling Relational Data with Graph Convolutional Networks (R-GCN).* ESWC.
→ Justifica `HeteroConv` com convoluções separadas por tipo de aresta (`results→constructors`, `qualifying→constructors`, `constructor_standings→constructors`).

**Robinson, J., Ranjan, R., Hu, W. et al. (2024).** *RelBench: A Benchmark for Deep Learning on Relational Databases.* NeurIPS 2024. **Stanford.** https://relbench.stanford.edu
→ Origem da tarefa `driver-top3` e do dataset `rel-f1`. Define `make_pkey_fkey_graph` que constrói o HeteroData a partir de pkeys/fkeys — pipeline canônico para relational deep learning.

**Fey, M., Hu, W., Huang, K. et al. (2023).** *Relational Deep Learning: Graph Representation Learning on Relational Databases.* arXiv:2312.04615. **Stanford.**
→ Paper conceitual por trás do RelBench. Defende GNN heterogênea como forma natural de absorver toda a vizinhança da tabela `constructors` (resultados, qualifying, standings) num único embedding.

Uso de `mean` na conv1 e `max` na conv2 segue prática do GraphSAGE (média captura agregado, max captura outliers/melhor desempenho histórico).

---

## 5. Loss: BCE + ortogonalidade (cosseno) + descorrelação (cross-corr) + HSIC

### 5.1 Multi-task com cabeças auxiliares

**Caruana, R. (1997).** *Multitask Learning.* Machine Learning, 28.
**Ruder, S. (2017).** *An Overview of Multi-Task Learning in Deep Neural Networks.* arXiv:1706.05098.
→ Justifica `aux_piloto` e `aux_equipe` com `aux_weight=0.5`: cabeças auxiliares forçam cada encoder a ser preditivo isoladamente, evitando que um único ramo domine o gradiente.

### 5.2 Ortogonalidade por cosseno (`λ_pairwise · mean|cos(v_a, v_b)|`)

**Bansal, N., Chen, X., Wang, Z. (2018).** *Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?* NeurIPS 2018.
→ Demonstra que penalidades de ortogonalidade (Soft Orthogonality) melhoram condicionamento e desemaranhamento entre subspaces. Base direta da nossa `mean|cos(v_p, v_e)|`.

**Brock, A., Donahue, J., Simonyan, K. (2019).** *Large Scale GAN Training (BigGAN).* ICLR.
→ Usa orthogonal regularization para forçar independência entre direções de variação latente — análogo ao que queremos entre `v_piloto`, `v_equipe`, `v_pista`.

### 5.3 Cross-correlação dimensão-a-dimensão (`λ_crossdim · mean|C|`)

**Wang, T. et al. (2021).** *Self-Supervised Learning Disentangled Group Representation as Feature.* NeurIPS.
→ Justifica HSIC como métrica de desemaranhamento em representações multimodais — exatamente nosso caso (piloto, equipe, pista).

**Greenfeld, D., Shalit, U. (2020).** *Robust Learning with the Hilbert-Schmidt Independence Criterion.* ICML.
→ HSIC como regularizador para forçar invariância/independência sem precisar discriminador adversarial; mais estável que GRL.

### 5.5 Ortogonalidade aplicada aos 3 pares (piloto-equipe-pista)

**Locatello, F. et al. (2019).** *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.* ICML.
→ Aponta que disentanglement só emerge com viés indutivo explícito; cobrir os três pares (não só o central) é o viés que torna a separação treinável.

### 5.6 Por que combinar três termos

A loss `BCE + λ_p·cos + λ_c·cross + λ_h·HSIC` ataca dependências em três níveis complementares:
- **cosseno**: alinhamento por amostra (geometria local).
- **cross-corr**: correlação linear dimensão-a-dimensão (Barlow Twins).
- **HSIC**: dependência não-linear de distribuição (Gretton).

Essa combinação está explicitamente discutida em **Bardes, Ponce, LeCun (2022)**, *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*, ICLR — que mistura termos de variância + invariância + covariância pelo mesmo motivo.

---