# Sinal de Pace do Qualifying (2º sinal) para o OrthogonalShapleyGNN — Design

**Branch:** `sage-position-regression`
**Data:** 2026-09-04
**Status:** desenhado; ainda não rodado no A100
**Antecessor:** `plans/2026-09-04-temporal-smoothness-prior-design.md` (prior de
suavidade temporal, segunda alavanca). Este documento especifica a **terceira
alavanca** — a cabeça auxiliar de qualifying, o "2º sinal de pace" — o maior gap
estrutural restante vs. o Bayesian SSM (`orthogonal-shapley-approach.md` ponto
aberto nº 2).

---

## Motivação

O Bayesian SSM (Lindner et al.) usa **qualifying** como pace limpo. O
OrthogonalShapleyGNN ranqueia **só a ordem de chegada** — que mistura safety car,
DNF e estratégia. O grafo já contém o nó `qualifying` com arestas `*_to_qualifying`
(herdadas de `sage_regressor.py`), mas o `OrthogonalShapleyGNN` **computa o
embedding do nó e o descarta** (peso morto). O nó agrega `driver_state +
constructor_state + race`, e o alvo `qualifying.y` já está normalizado
(`(position−1)/(n_cars−1)`, pole→0, fundo→1).

O target é vencer em **partial ρ / Cox HR**, não em PL NLL / pairwise (fora de
escopo, como nos designs anteriores).

### Escopo (decisão travada)

**Apenas a cabeça de quali.** Não tocamos no encoder, no readout, no context
player, no export, nem na decomposição Shapley. Uma mudança estrutural limpa →
o efeito em ρ/Cox é atribuível a esta alavanca.

---

## 1. O mecanismo: um alvo supervisionado, não um jogador novo

A confusão comum é "adicionar quali = adicionar feature". **Não.** O quali entra
como **alvo de uma cabeça auxiliar** (multi-task), não como feature nem como
jogador. A diferença é decisiva:

- A posição de largada (`grid`) **já** entra no context player (`grid_norm`), mas
  é quase uma função do carro e do piloto — no Shapley, esse "nível" da largada é
  **colinear com constructor e driver**, então é absorvido por esses dois jogadores,
  não pelo context. Adicionar quali como feature teria o mesmo destino: dissolver-se
  na colinearidade carro↔largada, sem ensinar nada novo.
- Uma cabeça `Linear` sobre o nó `qualifying` prevendo a posição normalizada
  **força os embeddings a codificar pace limpo** (o sinal que o Bayesian usa e o
  Orth não tem), em vez de deixá-lo dissolver.

```python
self.quali_readout = nn.Linear(hidden_dim, 1)   # lê o nó "qualifying" (128 dims)
```

No loop de treino, **um termo de loss novo** (não um jogador novo):

```python
qpred = model.quali_readout(x_dict["qualifying"]).squeeze(-1)   # (N_q,)
quali_loss = F.mse_loss(qpred[quali_train_mask], quali_y[quali_train_mask])
```

### Por que NÃO contamina a decomposição Shapley

`utility_additive` continua exatamente `u_d + u_c + u_x` — três cabeças lineares,
Shapley exato por construção. O `quali_readout` **não entra no utility**; é um
leitor separado (um "termômetro") que mede o pace do embedding, sem participar da
soma que gera o score.

O MSE de quali injeta **gradiente** de volta no encoder compartilhado
(`HeteroConv`), dizendo "deixa `driver_state`, `constructor_state` e `race`
codificarem pace". Mas **quem decide a divisão driver-vs-carro continua sendo a
maquinaria de separação existente** (orth_loss + identificação dura + attribution
balance). Quaili adiciona **sinal**, não jogador.

### Por que cabeça **fused**, e não separada

Uma cabeça `Linear_d(driver_state)` prevendo posição **absoluta** forçaria o
embedding do piloto a codificar o carro (determinante dominante da posição de
quali) → vazamento → recoverability acusaria. A cabeça fused (Linear sobre o nó
`qualifying`, que já agrega `driver + constructor + race`) deixa o canal do
construtor absorver a parte do carro e o canal do driver a parte do piloto —
**a divisão a cargo do orth/hard-id**, que é exatamente o que eles existem para
fazer.

---

## 2. Causalidade (sem leakage temporal)

O nó `qualifying@k` é destino de três arestas, todas apontando para ele:

```
race → qualifying                  (contexto circuito/era)
driver_state@k → qualifying        (estado do piloto antes da corrida k)
constructor_state@k → qualifying   (estado da equipe antes da corrida k)
```

O estado `@k` é construído a partir de `results@(T,k−1)` via `result_of_driver` /
`result_of_constructor`. Logo o nó `qualifying@k` agrega **apenas o passado**,
nunca o resultado da própria corrida k. Quali é **pré-corrida** no mesmo fim de
semana — prever `qualifying.y@k` a partir de estados `@k` é a ordem temporal
correta, igual ao readout de ranking. Nenhum nó agrega o próprio futuro.

A loss de quali roda **só na máscara de treino** (`year ≤ train_max_year`),
idêntico ao `orth_loss` e ao RW/shrinkage — o A/B continua honesto.

### O mecanismo honesto de por que move ρ/Cox

Quali absoluto é dominado pelo carro, então a cabeça fused aprende a prever
posição de quali quase inteiramente a partir do `constructor_state` — e isso é o
**ponto** da proposta, não uma falha:

O `partial_rho_continuous` residualiza o skill sobre `constructor_score_at_T` (o
pace contínuo do construtor). Se o embedding do construtor codifica pace **melhor**
(agora supervisionado por quali, não só por ordem de chegada ruidosa), então (a) o
residual do skill fica mais limpo, (b) o Shapley atribui a parte do carro ao
construtor com mais precisão, (c) o `partial_rho` sobe por **remover mais carro**,
não por inflar o skill.

Consequência: o sinal de quali entra **predominantemente pelo construtor**. Efeito
esperado: `partial_rho_continuous` ↑, recoverability do *constructor* ↑ (bom),
recoverability do *driver/career* **não piora** — e se piorar, é o alarme de
leakage (subir para a abordagem B).

---

## 3. Implementação

### 3.1 Modelo — 1 atributo novo

`src/models/orthogonal_shapley_gnn.py`, no `__init__` (após os aux heads):

```python
self.quali_readout = nn.Linear(hidden_dim, 1)
```

`encode()` já produz `x_dict["qualifying"]` (128 dims), pois as arestas
`*_to_qualifying` estão em `EDGE_TYPES` (`sage_regressor.py`). **Zero mudança no
encoder.**

### 3.2 Loop de treino

`src/experiments/train_orthogonal_shapley_gnn.py`:

- **Antes do loop** (onde o `chain` já é montado), capturar alvo e máscara uma vez:
  ```python
  quali_y = graph_data["qualifying"].y.to(device)          # (N_q,)
  quali_train_mask = graph_data["qualifying"].year.to(device) <= train_max_year
  ```
- **Dentro do loop**, após `x_dict = model.encode(...)` e o cômputo de
  `train_total`/`orth_loss`/`rw`/`shrink`, somar:
  ```python
  qpred = model.quali_readout(x_dict["qualifying"]).squeeze(-1)
  quali_loss = F.mse_loss(qpred[quali_train_mask], quali_y[quali_train_mask])
  total_loss = (train_total + lam*orth_loss + lambda_rw*rw_loss
                + lambda_shrink*shrink_loss + lambda_quali*quali_loss)
  ```
- **Log:** acrescentar `quali {quali_loss:.4f}` à linha de época.
- **Meta:** gravar `lambda_quali` em `config` e `quali_loss` (treino/val) em `metrics`.

### 3.3 Argumento novo

```python
parser.add_argument("--lambda-quali", type=float, default=0.0)
```

Propagar por `train_one_config(...)` (assinatura + `common` dict), como os demais
`lambda_*`.

### 3.4 O que **não** muda

- **Export** (`orthogonal_shapley_skill.py`): intocado. O Shapley soma
  carreira+offset; o quali não entra no score exportado.
- **Shapley / baselines / balance loss:** intocados (mesmos 3 jogadores).
- **Context player, encoder, readout:** intocados.
- **Todo o pipeline de validação:** intocado.

Arquivos tocados: só `orthogonal_shapley_gnn.py` (1 atributo) e
`train_orthogonal_shapley_gnn.py` (arg + 1 termo + log).

---

## 4. Escala do λ e sweep

`quali.y` é `[0,1]`; o MSE é O(1). O `λ_quali` proposto o mantém pequeno ao lado
do PL (~1.8):

| termo | default | efeito esperado no total |
|-------|---------|--------------------------|
| `λ_quali` | 0.3 | ~0.03–0.1 |

Sweep mínimo e honesto (mesma seed, um eixo por vez): `λ_quali ∈ {0.1, 0.3, 1.0}`.
O baseline do A/B é `λ_quali=0` (o modelo com prior temporal, sem quali).

---

## 5. Protocolo A/B honesto

- Mesma seed (42), mesmos hiperparâmetros, **uma mudança por vez**.
- **Baseline A** = prior temporal ativo, quali desligado:
  `--lambda-rw 0.5 --lambda-shrink 0.05 --lambda-quali 0`.
- **Variante B** = idem + quali: `--lambda-quali 0.3`, mantendo os mesmos
  `--lambda-rw/--lambda-shrink`.
- Janela **comum ≥2014**, reportando lado a lado.

**Sequenciamento crítico:** o prior temporal (design 2026-09-04) e o quali são
levers independentes, mas o quali entra "por cima" do prior. O A/B honesto do quali
deve ter o prior **fixo nos dois braços** — senão não se sabe qual lever moveu o
número. Por isso o baseline A aqui já é "prior ligado", não o modelo cru.

---

## 6. Critérios de aceitação

Medidos na janela **comum ≥2014**, A (baseline, prior ligado, quali desligado) vs.
B (quali ligado):

| # | Gate | Passa se | Prioridade |
|---|------|----------|------------|
| 1 | `partial_rho_continuous` | B ≥ A **e** CI low > 0 | **Must** |
| 2 | Recoverability do driver (career, team-switchers) | AUC ≤ null p95, **não pior que A** | **Must** (alarme de leakage) |
| 3 | Cox HR (eligible) | HR > 1, CI exclui 1, não mais largo que A | Should |
| 4 | PL NLL / pairwise (locked 2024–25) | não degrada materialmente vs. A | Guard |
| 5 | `quali_loss` (val) | cai de A→B e estabiliza | sanity |
| 6 | Share do constructor (Shapley) | ↑ modesto (pace indo para o construtor) | diagnóstico |

Gate 1 é o objetivo. Gate 2 é o guardrail central — a prova de que o quali não
vazou carro para o driver; se falhar, subir para a abordagem B. Gate 5 garante que
a cabeça realmente aprendeu.

---

## 7. Riscos

| Risco | Sintoma | Mitigação |
|-------|---------|-----------|
| Quali vaza carro para o driver | recoverability do driver ↑ (Gate 2) | subir para abordagem B (cabeças separadas + residual) |
| `λ_quali` forte demais | pairwise cai, quali_loss→0 mas ranking degrada | baixar `λ_quali`; quali é auxiliar |
| Cabeça não aprende | quali_loss não cai (Gate 5) | conferir `quali_train_mask`; sinal não chega |
| Construtor domina demais | share do construtor dispara, skill espremido | teto via `attribution_balance` (target 0.30) |
| Interação com o prior | não dá para atribuir o ganho | A/B com prior fixo nos dois braços |

---

## 8. Comandos canônicos (A100)

```bash
git checkout sage-position-regression && git pull

# Baseline A (prior temporal ligado, quali desligado)
python src/experiments/train_orthogonal_shapley_gnn.py \
  --seed 42 --use-additive-readout \
  --lambda-rw 0.5 --lambda-shrink 0.05 --lambda-quali 0 \
  --output-dir output/orthogonal_shapley_model_A

# Variante B (quali ligado)
python src/experiments/train_orthogonal_shapley_gnn.py \
  --seed 42 --use-additive-readout \
  --lambda-rw 0.5 --lambda-shrink 0.05 --lambda-quali 0.3 \
  --output-dir output/orthogonal_shapley_model_B

# Benchmark na janela comum (mesmo comando do design do prior)
python src/experiments/run_validation_benchmark.py \
  --sources bradley_terry bayesian_ssm orthogonal_shapley \
  --horizon inf --min-year 2014 --fixed-cohort --era-windows
```

---

## 9. Diferido (não neste round)

- **Abordagem B** (cabeças separadas driver/carro + residual por companheiro) —
  só se o Gate 2 falhar.
- **Pace contínuo (tempo de volta, Q3/milliseconds)** — hoje o grafo só tem posição
  normalizada; pace de verdade exige enriquecer o `qualifying` e rebuild do DB no
  A100.
- **Re-narração da figura Shapley** como "skill de carreira + desvio de temporada"
  (`orthogonal-shapley-approach.md` ponto aberto nº 7).
