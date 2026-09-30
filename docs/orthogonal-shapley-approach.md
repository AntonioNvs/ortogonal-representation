# OrthogonalShapleyGNN — como a abordagem funciona (detalhado)

**Branch:** `sage-position-regression`
**Data:** 2026-09-04
**Escopo:** documento de contexto para análise e propostas de mudança. Não é o
contrato de export (ver `model_contract.md`) nem o plano de identificação dura
(ver `plans/2026-09-03-hard-identification-design.md`). É o "porquê + como"
completo do modelo candidato atual.

---

## 0. A pergunta e o estimand

> **Conseguimos ranquear pilotos pela habilidade pura, isolada do carro/equipe/era?**

O alvo de inferência é o *driver effect* — a parcela do resultado de uma corrida
que pertence ao piloto, e só a ele. Formalmente, para piloto **D**, construtor
**C** e corrida **R**:

```text
systematic(D,C,R) = driver(D,R) + constructor(C,R) + context(R)
skill(D,T)        = career_skill(D) + season_offset(D,T)
```

Duas camadas separam isso:

1. **Decomposição em jogadores** — o valor sistemático de uma corrida se parte em
   três jogadores aditivos: **driver**, **constructor**, **context** (circuito/era/
   grid/round). O *driver effect* é o valor Shapley do jogador "driver".
2. **Identificação dentro do driver** — o driver effect, por sua vez, se parte em
   **career_skill** (um vetor por piloto, constante na carreira, livre de carro)
   + **season_offset** (um desvio por temporada). Trocar de equipe é a alavanca
   causal: o mesmo piloto indo bem na equipe A e mal na B força a parte constante
   a absorver "o piloto" e o construtor a absorver "o nível da equipe".

O score exportado é o **valor Shapley do driver, centrado dentro da corrida**
(intra-race deviation), depois agregado por temporada como média das corridas.

---

## 1. Substrato: grafo temporal causal round-state

Arquivo: `src/data/temporal_graph.py`. Não é um grafo de entidades estático — é um
grafo **temporalmente causal por construção**: toda aresta aponta de um evento no
passado para um nó no futuro, então nenhum nó agrega o próprio futuro (sem leakage
temporal).

### Tipos de nó

| Nó | Cardinalidade | Significado |
|----|---------------|-------------|
| `driver_state` | um por `(driverId, raceId)` | estado do piloto **antes** da corrida |
| `constructor_state` | um por `(constructorId, raceId)` | estado da equipe antes da corrida |
| `race` | um por corrida | features `year`, `round`, `name`, `date` |
| `circuit` | estático | local, país, lat/lng, alt |
| `results` | folha (por driver) | evidência de resultado: `grid`, `position`, `points`, etc. |
| `constructor_results` | folha (por equipe) | pontos agregados da equipe |
| `qualifying` | alvo do regressor auxiliar | features `number`, `date`; label `position` |

### Arestas (todas direcionais, causais)

| Aresta | Fluxo |
|--------|-------|
| `same_driver` / `same_driver_cross` | cadeia temporal do piloto (dentro da temporada / atravessando a virada de ano) |
| `same_constructor` / `same_constructor_cross` | análogo da equipe |
| `result_of_driver` | `results@(T,k−1)` → `driver_state@(T,k)`: o resultado passado alimenta o estado atual |
| `result_of_constructor` | `constructor_results@(T,k−1)` → `constructor_state@(T,k)` |
| `circuit_to_race` | o `race` agrega o circuito |
| `race_to_qualifying` / `driver_state_to_qualifying` / `constructor_state_to_qualifying` | o alvo de quali agrega contexto + estados (usado pelo regressor SAGE auxiliar) |

O nó `results` carrega metadados críticos para o modelo de ranking: `driver_id`,
`constructor_id`, `driver_state_idx`, `constructor_state_idx`, `race_idx`, `grid`,
`round`, `year`, `in_ranking` (tem posição de chegada) e `driver_career_idx`.

### `driver_career_idx` (identificação dura)

Um índice contíguo **por piloto** (`0..n_drivers−1`), estável ao longo da carreira.
É ele que indexa o embedding de carreira (Seção 3.3). Mapeamento feito em
`temporal_graph.py:396-405` a partir de `drivers["driverId"].unique()` ordenado.

---

## 2. Encoder: SAGE heterogêneo

Arquivo: `src/models/orthogonal_shapley_gnn.py`.

```
features → HeteroEncoder (→ hidden_dim=128)
        → 4 × [ HeteroConv(SAGE) + ReLU + residual + LayerNorm ]
        → x_dict["driver_state"], ["constructor_state"], ["race"], ...
```

- **HeteroEncoder** converte as colunas tabulares (cat/num) para embeddings.
- **4 camadas** de `HeteroConv` sobre todas as arestas de `EDGE_TYPES`.
  Aggregator = `mean` na camada 0, `max` nas demais.
- **Residual + LayerNorm** em `driver_state`, `constructor_state`, `race`.

Saída: `x_dict` com os estados latentes. O `driver_state` aqui é o **offset por
temporada**; **não** é a habilidade de carreira.

---

## 3. Readout: valor de utility e os três jogadores

### 3.1 Contexto (terceiro jogador)

```python
context_vector = context_mlp([grid_norm, round_norm] ‖ race_emb)   # → 32 dims
```

`grid_norm = (grid−1)/19`, `round_norm = (round−1)/(max_round−1)`. O contexto
mistura **escalares pré-corrida** (posição de grid, rodada) com o embedding do
circuito/era (`race`). É o jogador "tudo que não é piloto nem equipe".

### 3.2 Readouts

Dois modos, controlados por `use_additive_readout`:

- **`fused` (legado):** `classifier([d ‖ c ‖ ctx])` — MLP não-linear sobre a
  concatenação. O Shapley sobre isso **não** é aditivo (termos de interação
  dominam a atribuição). Por isso foi abandonado como padrão.
- **`additive` (padrão):** `u = u_d + u_c + u_x`, com três cabeças lineares:
  ```python
  u_c = aux_constructor(c_emb)     # valor da equipe
  u_x = aux_context(ctx)           # valor do contexto
  u_d = driver_skill(d_emb, career_emb)   # valor do piloto (ver 3.3)
  ```
  Sobre três jogadores **lineares**, o Shapley é exato por construção:
  `phi_i = u_i(real) − u_i(baseline)`, resíduo de eficiência = 0.

### 3.3 Driver effect: carreira + temporada

```python
driver_skill(d_emb, career_emb) = aux_driver_career(career_emb) + aux_driver_season(d_emb)
```

- `career_emb = driver_career(driver_career_idx)` — `nn.Embedding(n_drivers, 128)`,
  **fora** do `HeteroConv` (não recebe mensagens de construtor ⇒ car-free por
  construção). Um vetor por piloto, constante na carreira.
- `d_emb` — o nó `driver_state` por `(driver, race)`, é o **offset de temporada**.

Sem `career_emb` (modo legado), o driver effect volta a ser só `aux_driver_season`.

---

## 4. Shapley de coalizão (exato, 3 jogadores)

Arquivo: `src/explain/coalition_shapley.py`.

- **Jogadores:** `driver=1`, `constructor=2`, `context=4` (bitmask).
- **Baselines:** embeddings/context médios do **treino** para jogadores ausentes
  (`CoalitionBaselines`). O driver ausente usa `driver_emb` **e** `driver_career_emb`
  (dois baselines: offset e carreira).
- **Valor de coalizão:** `v(S) = utility_additive` substituindo cada jogador ausente
  pelo baseline correspondente. Com readout aditivo, `v` é a soma dos jogadores
  presentes (mais constantes).
- **Shapley exato:** para n=3, `phi_i = Σ_{S⊆N\{i}} w(|S|) · (v(S∪{i}) − v(S))`,
  com pesos `w(0)=1/3, w(1)=1/6, w(2)=1/3`. O driver é o jogador composto
  carreira+offset via `utility_additive(..., career_emb)`.

Resultado: `phi_d` (driver), `phi_c` (constructor), `phi_x` (context) e o resíduo
de eficiência `v(N) − v(∅) − Σ phi_i` — que no modo aditivo é numericamente ~1e−9.

**Importante:** o jogador "driver" conta como **um** jogador (não dois), mesmo
sendo a soma de carreira + temporada. O count continua 3, e a eficiência aditiva
é preservada.

---

## 5. Função de perda

Arquivo: `src/experiments/train_orthogonal_shapley_gnn.py`, `race_loss_for_mask`.

```text
L = PL(fused)                       # ranking Plackett-Luce sobre o readout principal
  + 0.5·PL(driver)                  # auxiliares: força cada cabeça a também
  + 0.75·PL(constructor)            #   ranquear a corrida sozinha
  + 0.25·PL(context)
  + 0.25·pairwise                   # ranking pairwise sobre o fused
  + λ_orth · orth_loss              # penalidade de ortogonalidade (rampa 0.2→2.0)
  + 0.10·attribution_balance        # equilíbrio das shares Shapley
```

Componentes:

1. **Plackett-Luce NLL** sobre o fused **e** sobre as três cabeças auxiliares.
   Isso garante que cada jogador, sozinho, contém sinal de ranking — senão o
   Shapley devolveria um jogador "vazio".
2. **Pairwise ranking loss** no fused (estabiliza o PL).
3. **`orth_loss`** (`paired_orthogonal_loss`): penaliza a similaridade (cos²)
   entre `driver_state`↔`constructor_state` e entre driver↔contexto, projetado via
   `driver_ctx_orth`. λ sobe de 0.2 até 2.0 durante warmup — separação *soft*.
4. **`attribution_balance_loss`**: `share_i = |phi_i| / Σ|phi|`. Penaliza a share
   do driver acima de `0.38` e a do construtor acima de `0.30` (`relu²`). É o
   "teto" que impede um canal de monopolizar a utility. Calculada só numa
   subamostra de ~20% das corridas (custo do Shapley exato).

> A separação *soft* (ortogonalidade + teto de share) é exatamente o que a
> identificação dura (Seção 3.3) veio **endurecer**: em vez de penalizar a
> correlação, ela muda a estrutura para que a habilidade de carreira seja
> estruturalmente livre do carro.

---

## 6. Export do score

Arquivo: `src/baselines/orthogonal_shapley_skill.py`, `export_race_skills`.

1. `phi_d` (Shapley do driver) é calculado por corrida.
2. **Centering dentro da corrida:** `raw_skill = phi_d − mean(phi_d | race)`.
   PL é invariante a translação por corrida, então é um transform pós-hoc que não
   conflita com o treino. Remove o *nível* de carro/era/circuito por construção.
3. **Canais de atribuição** (`contrib_driver`, `contrib_constructor`,
   `contrib_context`) ficam **sem centering** — a eficiência
   (`Σ contrib = v(N) − v(∅)`) é preservada.
4. Score por temporada = média dos `raw_skill` nas corridas 1…R (modo `filtered`).

A skill exportada passa por calibração logística para `[0,10]` (âncora no treino),
conforme `model_contract.md`.

---

## 7. Validação (o que "bom" significa)

Arquivo: `docs/career_validation_framework.md` (v4).

A hipótese central é a **fair-market**: o mercado de pilotos é eficiente, então
equipes melhores contratam pilotos melhores. Um score de *skill* (e não de carro)
deve prever o futuro da carreira **acima** do que o carro atual já explica.

Métricas primárias (na janela comum `≥2014`):

| Métrica | O que mede | Gate |
|---------|-----------|------|
| `partial_rho_continuous` | Spearman parcial do score vs. promoção de tier, residualizando no tier do construtor | Orth ≥ Bayesian |
| Cox HR (censurado) | tempo até promoção de tier, hazard por unidade de skill | HR > 1, IC exclui 1 |
| Recoverability probe | o canal de carreira vaza a identidade do construtor? | AUC ≤ null p95 |

E também: `underrated_resolution` (rate de promoção entre os subestimados),
`eligible_promotion_auroc`, e a comparação de ranking (`PL NLL`, `pairwise_acc`).

**Resultados da última rodada (2026-09-04, janela comum ≥2014; `plackett_luce` pendente):**

| Modelo | partial ρ (contínuo) | Cox HR (eligible) | PL NLL | pairwise |
|---|---|---|---|---|
| Bradley-Terry | 0.087 [−0.094, +0.232] | 1.134 [0.773, 1.791] ns | 1.889 | 0.695 |
| Bayesian SSM | 0.309 [+0.086, +0.488] | 4.902 [1.047, 24.20] | 1.915* | 0.693* |
| **OrthogonalShapley** | **0.364 [+0.152, +0.536]** | **3.907 [1.413, 11.49]** p=2.6e-4 | **1.804** | **0.749** |

\* Bayesian é nível-temporada e `smoothed`/in-sample — seu locked-test não é
comparação held-out justa.

Leitura: Orth lidera nos **três**. Ganha o ρ contínuo por margem (0.364 > 0.309 do
Bayesian; o BT nem exclui 0), tem o Cox com o IC mais apertado que exclui 1 (o
Bayesian tem ponto maior, 4.90, mas IC inferior encosta em 1.05 — near-separated), e
é o melhor ranking held-out.

---

## 8. Onde falta contexto (pontos abertos para você propor mudanças)

Ordenados por impacto potencial no objetivo (bater o Bayesian no readout de skill):

1. **Leakage residual do canal de carreira — RESOLVIDO (2026-09-04).** O probe
   de carreira (team-switchers, GroupKFold) agora passa: `macro_auc ≈ 0.504` vs
   null p95 ≈ 0.521 (`leakage = false`, n=364 pilotos / 1210 pares). A Seção 3.3
   removeu o carro de fato; o que resta no offset por temporada é nível de equipe
   **por design**. A claim segue "car-adjusted performance".

2. **Só resultado de corrida, sem sinal de quali.** O Bayesian usa quali (pace
   limpo). O Orth ranqueia só a ordem de chegada. Adicionar quali como segundo
   sinal de pace é o candidato mais óbvio a fechar o gap — **agora desenhado** em
   `plans/2026-09-04-qualifying-pace-signal-design.md` (cabeça auxiliar de quali).

3. **Prior de suavidade temporal — DESENHADO (2026-09-04).** Especificado em
   `plans/2026-09-04-temporal-smoothness-prior-design.md`: RW + shrinkage sobre o
   escalar do offset (análogo ao GP random-walk do Bayesian). Ainda não rodado.

4. **Offset pode dominar a carreira.** Risco listado no design: se
   `aux_driver_season` dominar, o readout volta a ser skill estática por temporada
   (o que tínhamos antes). O diagnóstico (`offset_frac`) e a mitigação (shrinkage)
   estão especificados no prior temporal (ver ponto 3).

5. **Context player — já não é o gargalo.** Shares D/C/X ~0.35/0.42/0.23
   (`shapley_season_mean`, rodada 2026-09-04); a menção a ~6% era de uma rodada
   antiga. O grid/largada é colinear com carro+piloto, então a maior parte dele
   aparece na share do constructor/driver, não no context — o context carrega o
   *resíduo marginal*. É o comportamento esperado do Shapley, não
   sub-parametrização.

6. **Comparação justa com o Bayesian.** O Bayesian é nível-temporada com prior;
   o Orth é nível-corrida. "Bater" no ρ pontual de 0.438 (n=147) pode ser a
   métrica errada — a defesa mais forte do Orth é o Cox crível + cobertura por
   corrida, não o ρ pontual. Vale decidir a narrativa antes de perseguir o número.

7. **Shapley re-narração.** Com o driver = carreira + offset, a figura de
   atribuição precisa narrar "skill de carreira + desvio de temporada", não um nó
   estático por temporada. Deferido no design doc, Seção 6.

---

## 9. Arquivos-chave (mapa)

| Arquivo | Responsabilidade |
|---------|------------------|
| `src/models/orthogonal_shapley_gnn.py` | encoder SAGE + readout aditivo + carreira/offset |
| `src/explain/coalition_shapley.py` | Shapley exato 3-jogadores + baselines + balance loss |
| `src/data/temporal_graph.py` | grafo causal round-state + `driver_career_idx` |
| `src/experiments/train_orthogonal_shapley_gnn.py` | loss composta + loop de treino |
| `src/baselines/orthogonal_shapley_skill.py` | export + centering intra-corrida |
| `src/explain/orthogonal_shapley_probes.py` | probes de leakage / swap / eficiência |
| `src/validation/benchmark.py` | orquestra benchmark + gates por era window |
| `docs/career_validation_framework.md` | metodologia de validação (v3) |
| `docs/model_contract.md` | contrato de export comum a todos os modelos |
| `docs/plans/2026-09-03-hard-identification-design.md` | plano da identificação dura (Seções 1–2) |
