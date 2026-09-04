# Prior de Suavidade Temporal para o Offset do Driver-Skill — Design

**Branch:** `sage-position-regression`
**Data:** 2026-09-04
**Status:** desenhado; ainda não rodado no A100
**Antecessor:** `plans/2026-09-03-hard-identification-design.md` (identificação dura,
Seções 1–2, já implementada). Este documento especifica a **segunda alavanca**
diferida naquele design (Seção 6: "Temporal smoothness prior … GP random-walk
analogue").

---

## Motivação

O `career_validation_framework` (v3) julga um score de skill por prever a promoção
de tier de um piloto **acima** do que o carro atual explica (`partial_rho_continuous`,
Cox HR). Nesse critério o Bayesian SSM ainda lidera. O documento de identificação
dura (2026-09-03) listou três lacunas estruturais vs. o Bayesian; a nº 3 é:

> O Bayesian modela a habilidade como um **GP random-walk** (trajetória de carreira);
> o Orth tem `career_skill` estático + um `season_offset` por corrida **sem prior de
> trajetória**.

O `orthogonal-shapley-approach.md` repete isso no ponto aberto nº 3 ("Sem prior de
suavidade temporal") e no nº 4 ("Offset pode dominar a carreira"). Este design ataca
os dois com **um único mecanismo**: um prior de suavidade + encolhimento sobre o
**escalar do offset**, o análogo direto ao random-walk do Bayesian.

O alvo é vencer em **partial ρ / Cox HR**, não em PL NLL / pairwise (fora de escopo,
como no design anterior).

### Escopo (decisão travada)

**Apenas o prior temporal.** Não tocamos no context player, no encoder, no readout,
nem no export. Uma mudança estrutural limpa → o efeito em ρ/Cox é atribuível a esta
alavanca, mantendo o protocolo A/B honesto ("uma mudança por vez").

---

## 1. A ideia: nível de carreira + trajetória suave

A identificação dura já parte o driver effect em dois termos aditivos:

```text
u_d = u_career(D) + u_season(D, T, k)
    = aux_driver_career(career_emb[D]) + aux_driver_season(driver_state_emb[D,T,k])
```

O nó `driver_state` é **por corrida** (`(driverId, raceId)`), e as arestas
`same_driver` / `same_driver_cross` (`temporal_graph.py:214-223`) já ligam corridas
consecutivas do mesmo piloto em ordem cronológica (`dsrc → ddst`, `ddst = dsrc+1`).
O esqueleto do random-walk **já existe no grafo**; falta só a penalidade.

Hoje `u_career` (constante) e `u_season` (por corrida) sofrem de um **empate aditivo
não-identificável**: somar `c` à carreira e subtrair `c` do offset não muda `u_d`.
É por isso que o offset pode "dominar a carreira" (risco nº 4). Dois termos resolvem,
ambos sobre o **escalar** `u_season`:

1. **Random-walk (RW):** penaliza `(u_season[k] − u_season[k−1])²` ao longo da cadeia
   `same_driver ∪ same_driver_cross`. A trajetória do piloto só muda suavemente de
   corrida para corrida — o análogo do GP random-walk do Bayesian.
2. **Shrinkage leve:** penaliza `u_season²` (nível do offset → 0). Quebra o empate
   aditivo e empurra a **identidade** para o `career_skill` car-free. Deliberadamente
   **leve**: não é para matar o offset (isso mataria a trajetória que o RW modela),
   só para ancorar seu nível.

Resultado: `u_d = nível(car-free, estável) + desvio(suave, centrado em ~0)` — uma
decomposição do GP em média + inovações suaves.

---

## 2. Os dois termos de perda

Ambos operam sobre `u_season = aux_driver_season(x_dict["driver_state"]).squeeze(-1)`,
um escalar por nó `driver_state`, calculado uma vez por época (Linear sobre
`N_ds × 128` → `N_ds`, trivial). Seja `E = [same_driver ‖ same_driver_cross]` a cadeia
(concat das duas arestas) e `M` uma máscara booleana sobre nós `driver_state`
restringindo à janela de treino:

```text
# random-walk: suavidade ao longo de arestas com AMBOS os extremos no treino
keep    = M[src] & M[dst]                     # src, dst = E
L_rw    = mean_{e ∈ E, keep}  (u_season[dst] − u_season[src])²

# shrinkage: encolhe o nível do offset nos nós de treino
L_shrink = mean_{i ∈ driver_state, M[i]}  u_season[i]²
```

**Por que restringir ao treino (`M`).** O `orth_loss` já é calculado só sobre
`train_mask` (`train_...py:161`). Restringir o RW/shrinkage às arestas/nós da janela
de treino mantém o A/B honesto: só o sinal das temporadas de treino molda o modelo,
sem usar estrutura futura. `M` é derivada de `train_mask`: os índices `driver_state`
referenciados pelas linhas de `results` no treino. Arestas cujo `src` **ou** `dst`
caia fora de `M` são descartadas do RW (evita "vazar" um passo de suavização para uma
corrida de val/test).

**Simplicidade (YAGNI).** Um único `λ_rw` para `same_driver` e `same_driver_cross`
(o passo de virada de ano recebe o mesmo peso do passo intra-temporada). Sem warmup:
são priores leves e estáveis desde a época 0 (ao contrário do orth, que precisa de
rampa para não dominar o gradiente cedo). Se aparecer instabilidade, warmup fica como
refinamento futuro — não neste round.

---

## 3. Implementação

### 3.1 Modelo — novo método (espelha `paired_orthogonal_loss`)

`src/models/orthogonal_shapley_gnn.py`:

```python
def temporal_smoothness_loss(
    self,
    x_dict: Dict[str, torch.Tensor],
    chain_edge_index: torch.Tensor,        # (2, E) = [same_driver ‖ same_driver_cross]
    node_mask: torch.Tensor | None = None, # (N_ds,) bool: nós driver_state no treino
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(L_rw, L_shrink) sobre o escalar do offset por driver_state.

    Análogo GP random-walk: penaliza saltos do offset entre corridas consecutivas
    do mesmo piloto (RW) e o nível do offset (shrinkage), empurrando a identidade
    para o career_skill car-free.
    """
    u = self.aux_driver_season(x_dict["driver_state"]).squeeze(-1)  # (N_ds,)
    src, dst = chain_edge_index[0], chain_edge_index[1]
    if node_mask is not None:
        keep = node_mask[src] & node_mask[dst]
        src, dst = src[keep], dst[keep]
    rw = torch.mean((u[dst] - u[src]) ** 2) if src.numel() > 0 else u.new_zeros(())
    u_shrink = u[node_mask] if node_mask is not None else u
    shrink = torch.mean(u_shrink ** 2) if u_shrink.numel() > 0 else u.new_zeros(())
    return rw, shrink
```

Sem `driver_career` (modo legado, `num_drivers==0`) o método ainda roda — mas o prior
só faz sentido acoplado à identificação dura, então será chamado apenas com
`--use-additive-readout` + carreira ativa.

### 3.2 Diagnóstico do risco nº 4 (offset domina a carreira)

Sobre as linhas de treino, reportar por época e gravar no `meta`:

```text
offset_frac = std(u_season) / (std(u_career) + std(u_season) + 1e-9)
```

`u_career = aux_driver_career(career_emb)`, `u_season = aux_driver_season(d_emb)`.
Se o shrinkage funciona, `offset_frac` cai (a carreira carrega a variância da
identidade). Um `offset_frac` alto (≳0.5) mesmo com shrinkage sinaliza que o offset
ainda domina — o gatilho para subir `λ_shrink`.

### 3.3 Loop de treino

`src/experiments/train_orthogonal_shapley_gnn.py`:

- **Antes do loop:** montar a cadeia e a máscara uma vez.
  ```python
  chain = torch.cat([
      edge_index_dict[("driver_state","same_driver","driver_state")],
      edge_index_dict[("driver_state","same_driver_cross","driver_state")],
  ], dim=1).to(device)
  n_ds = graph_data["driver_state"].num_nodes
  train_ds_mask = torch.zeros(n_ds, dtype=torch.bool, device=device)
  train_ds_mask[res.driver_state_idx[train_mask].to(device)] = True
  ```
- **Dentro do loop**, após `x_dict = model.encode(...)` e o cômputo de
  `train_total`/`orth_loss` (linha ~481-497), somar os termos:
  ```python
  rw_loss, shrink_loss = model.temporal_smoothness_loss(x_dict, chain, train_ds_mask)
  total_loss = (train_total + lam * orth_loss
                + lambda_rw * rw_loss + lambda_shrink * shrink_loss)
  ```
  Espelha exatamente o padrão da linha 497 (`total_loss = train_total + lam*orth_loss`).
- **Log:** acrescentar `rw {rw_loss:.4f} | shrink {shrink_loss:.4f} | off_frac {offset_frac:.2f}`
  à linha de época.
- **Meta:** gravar `lambda_rw`, `lambda_shrink` em `config` e `offset_frac` (treino/val)
  em `metrics`.

### 3.4 Argumentos novos

```python
parser.add_argument("--lambda-rw", type=float, default=0.5)     # suavidade RW
parser.add_argument("--lambda-shrink", type=float, default=0.05) # encolhe o offset
```

Propagar por `train_one_config(...)` (assinatura + `common` dict em `main`), como os
demais `lambda_*`. Defaults como ponto de partida — ver §4 para o sweep.

### 3.5 O que **não** muda

- **Export** (`orthogonal_shapley_skill.py`): intocado. O Shapley já soma
  carreira+offset; o offset agora é mais suave e menor, a carreira carrega a
  identidade. O `raw_skill` centrado herda a trajetória suave de graça.
- **Shapley / baselines / balance loss:** intocados (mesmos 3 jogadores).
- **Context player, encoder, readout:** intocados.

Arquivos tocados: só `orthogonal_shapley_gnn.py` (novo método) e
`train_orthogonal_shapley_gnn.py` (args + 2 termos + diagnóstico).

---

## 4. Escala dos λ e sweep

`u_season` vive na escala do logit PL (O(1)). Logo `L_rw` e `L_shrink` são O(1), e os
λ propostos os mantêm pequenos ao lado do PL (~1.8):

| termo | default | efeito esperado no total |
|-------|---------|--------------------------|
| `λ_rw` | 0.5 | ~0.05–0.3 |
| `λ_shrink` | 0.05 | ~0.02–0.1 (leve, por design) |

Sweep mínimo e honesto (mesma seed, um eixo por vez):
`λ_rw ∈ {0.1, 0.5, 1.0}` mantendo `λ_shrink=0.05`; se `offset_frac` continuar alto,
subir `λ_shrink` para `0.1`. O baseline do A/B é `λ_rw=λ_shrink=0` (o modelo atual).

---

## 5. Por que isto move ρ/Cox sem mexer no ranking

- **PL/pairwise ~estáveis.** RW e shrinkage agem **só** em `u_season`, um componente
  aditivo. O RW restringe a trajetória **entre** corridas, não a ordem **dentro** de
  uma corrida; a carreira + construtor + contexto seguem carregando o sinal de ranking
  intra-corrida. Espera-se pairwise/PL praticamente inalterados (fora de escopo, como
  no design anterior — se moverem pouco, tudo bem).
- **ρ/Cox melhoram.** O score exportado é `phi_d` centrado por corrida ≈ carreira +
  offset suave. Com o offset ancorado em ~0 e suavizado, (a) a **identidade** migra
  para o canal car-free `career_skill` — o que o probe de recuperabilidade precisa que
  aconteça (Gate 1 da identificação dura) — e (b) a **trajetória** por temporada fica
  coerente com o histórico ("olhar para o passado"), reduzindo ruído corrida-a-corrida
  que hoje enfraquece a correlação com promoção de tier.

Não prometemos bater o ρ pontual de 0.438 do Bayesian (n=147, near-separated); a
defesa forte do Orth segue sendo o Cox crível + cobertura por corrida (ponto aberto
nº 6). O prior temporal fecha a lacuna estrutural nº 3 e sustenta essa narrativa.

---

## 6. Critérios de aceitação

Medidos na janela **comum ≥2014**, A (baseline, λ=0) vs. B (variante):

| # | Gate | Passa se | Prioridade |
|---|------|----------|------------|
| 1 | `offset_frac` (treino) | cai de A→B (carreira carrega a identidade) | **Must** |
| 2 | Recoverability probe (career, team-switchers) | AUC ≤ null p95, não pior que A | **Must** |
| 3 | `partial_rho_continuous` | B ≥ A **e** B ≥ Bayesian | **Must** |
| 4 | Cox HR (eligible) | HR > 1, IC exclui 1, não mais largo que A | Should |
| 5 | PL NLL / pairwise | não degrada materialmente vs. A | Guard |

Gate 1 é o teste direto de que o mecanismo fez o que promete (ataca o risco nº 4).
Gate 5 é o guardrail: se o ranking desabar, o prior está forte demais (baixar `λ_rw`).

### Riscos

| Risco | Sintoma | Mitigação |
|-------|---------|-----------|
| RW forte demais | pairwise cai, `offset_frac`→0, ρ cai | baixar `λ_rw`; offset colapsado = sem trajetória |
| Shrinkage forte demais | offset ~0, vira skill estática por carreira | manter `λ_shrink` leve (≤0.1) |
| Prior não move nada | ρ/Cox iguais a A | offset já era pequeno; o gargalo é outro (quali, ponto nº 2) |
| Máscara de treino mal montada | RW/shrink ≈ 0 sempre | conferir `train_ds_mask` cobre os nós esperados |

---

## 7. Protocolo A/B honesto

- Mesma seed, mesmos hiperparâmetros; muda **só** os dois termos novos.
- Baseline A = `--lambda-rw 0 --lambda-shrink 0` (idêntico ao modelo atual).
- Variante B = `--lambda-rw 0.5 --lambda-shrink 0.05`.
- Reportar lado a lado: `offset_frac`, recoverability AUC, `partial_rho_continuous`,
  Cox HR, PL NLL, pairwise — em `common_2014`. Uma mudança por vez.

---

## 8. Comandos canônicos (A100)

```bash
git checkout sage-position-regression && git pull
python -m src.data.pipeline build            # só se data/enriched/rel-f1/db faltar

# Baseline A (modelo atual: prior desligado)
python src/experiments/train_orthogonal_shapley_gnn.py \
  --seed 42 --use-additive-readout \
  --lambda-rw 0 --lambda-shrink 0 \
  --output-dir output/orthogonal_shapley_model_baseline

# Variante B (prior temporal ligado)
python src/experiments/train_orthogonal_shapley_gnn.py \
  --seed 42 --use-additive-readout \
  --lambda-rw 0.5 --lambda-shrink 0.05 \
  --output-dir output/orthogonal_shapley_model

# Re-fit do Bayesian (baseline da janela comum >=2014)
python -m src.experiments.run_bayesian_ssm --start-year 2014 --end-year 2025

# Benchmark dos três na janela comum
python src/experiments/run_validation_benchmark.py \
  --sources bradley_terry bayesian_ssm orthogonal_shapley \
  --horizon inf --era-windows
```

---

## 9. Diferido (não neste round)

- Quali como segundo sinal de pace (ponto aberto nº 2 — a maior lacuna restante).
- Warmup / `λ_rw` separado para `same_driver_cross` (só se houver instabilidade).
- Re-narração da figura Shapley como "skill de carreira + desvio de temporada"
  (ponto aberto nº 7).
