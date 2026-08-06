# TODO — Fortalecimento do Paper

---

## 1. Revisão da Literatura

- [ ] Buscar métodos de decomposição de desempenho (agente vs. contexto) em **outros esportes e domínios**
- [ ] Levantar os **principais trabalhos de análise de dados em F1** e explicitar como o seu método se diferencia
- [ ] Montar **tabela comparativa** (método, domínio, tipo de decomposição, limitações)

## 2. Validação Experimental

- [ ] Implementar **baselines** para comparação (média bruta, regressão linear, ranking por pontos)
- [ ] Rodar o modelo em **temporadas adicionais** e verificar consistência
- [ ] Construir **validação externa**: cruzar ranking do modelo com promoções/rebaixamentos reais de pilotos entre temporadas
- [ ] Calcular **correlação estatística** (Spearman/Kendall) entre ranking do modelo e o validador externo

## 3. Seção de Aplicação

- [ ] Detalhar **casos de uso concretos** (scouting, decisão de line-up, avaliação isolada do carro)
- [ ] Criar **visualizações** que demonstrem insights acionáveis

## 4. Próximos Passos

- [X] Marcar conversa com Pedro para alinhar direção
- [ ] Definir conferência foco + deadline

---

## Related-Work

### Direções do Pedro (reunião)

- Related-Work: Padrão é ter 30 referências - trabalhar nisso
- Onde mais foi aplicado a orthogonality constrained?
- Isso já é meu POC II, estender para um artigo científico (então fica muito forte para o poc já)
- Não precisa colocar std ou IC nas tabelas - espaço é algo bom
- Como fica a eficácia da predição conforme a temporada percorre: eixo X = número de corridas transcorridas na temporada, eixo Y = AUROC
  - Desempenho médio do modelo (pega o range de temporadas de teste)
- Como fica o modelo que só olha para equipe/piloto
- Como validar isso? Olha a carreira deles - mudou para uma equipe melhor/pior? desempenho de pontos nas próximas temporadas (montar os rótulos)
- Como rankear as equipes: ferrari é um outlier de todas, definir categorias de equipe (tier 1, 2 e 3), fica mais estável, começar simples e depois mudar
- Como definir que o sinal é um bom ou ruim? SHAP já dá isso - analisar sobre
- Framework ser quantitativo, análise de múltiplas temporadas, related work para comparar caracterização de piloto

### Referências externas

- [Sloan Sports Conference - Research Paper Competition](https://www.sloansportsconference.com/research-paper-competition)
- Paper para estudar: *Deep Reinforcement Learning for NBA Player Valuation: A Temporal Difference Approach with Shapley Attribution*
- [Google Docs - rascunho do paper](https://docs.google.com/document/d/12zoFMnFujCkQcYCD-c14W5C9tJjXa2T8cqWHk9sNkM4/edit?tab=t.0)
- Abstract em outubro

---

## 5. Sprint — Até 04 de Agosto (12 dias)

**Objetivo:** Entregar 5 artefatos simples e de alto impacto que respondem diretamente às perguntas do Pedro.

### 5.1 Curva de Eficácia Temporal (AUROC vs. Corridas Transcorridas) 🔴

- [ ] Criar script `eval_temporal_curve.py` que avalia o modelo em janelas progressivas da temporada
- [ ] Para cada temporada de teste e cada k de 1 até max_races: filtrar instâncias até a k-ésima corrida, calcular AUROC
- [ ] Plotar curva com eixo X = número de corridas, eixo Y = AUROC, uma linha por temporada + banda de variação
- [ ] Média entre temporadas com desvio padrão

**Por que é simples:** Modelo já treinado. Só truncar os dados de teste e avaliar. ~100 linhas.
**Por que é eficaz:** Figura central do paper — mostra que o modelo é útil com poucas corridas (scouting) e melhora conforme a temporada avança. Nenhum paper de F1 tem isso.
**Artefato:** Script `eval_temporal_curve.py` + figura PNG.

### 5.2 Modelos Single-Modality (Só Piloto / Só Equipe) 🔴

- [ ] Criar script `eval_single_modality.py` que carrega modelo treinado e avalia `aux_piloto` e `aux_equipe` isoladamente
- [ ] Calcular AUROC de cada sinal isolado vs. modelo completo (full)
- [ ] Gerar tabela comparativa: full vs. só piloto vs. só equipe
- [ ] Opcional: plotar curva ROC sobreposta (full + piloto + equipe)

**Por que é simples:** As aux heads já existem no `F1OrthogonalPipeline`. É só extrair e avaliar. ~50 linhas.
**Por que é eficaz:** Ablação fundamental que mostra que o modelo combinado supera cada sinal isolado. Fortalece a seção experimental.
**Artefato:** Script `eval_single_modality.py` + tabela CSV.

### 5.3 Categorização de Equipes em Tiers 🔴

- [ ] Criar script `team_tiers.py` que classifica equipes em Tier 1, 2, 3 com base em desempenho histórico
- [ ] Heurística: pontos médios por temporada ou posição no campeonato de construtores
- [ ] Definir tiers por era (ex.: 2000-2009, 2010-2013, 2014-2021, 2022+ para capturar mudanças de dominância)
- [ ] Gerar tabela de classificação e salvar em CSV
- [ ] Opcional: plotar heatmap (equipe vs. ano, cor = tier)

**Por que é simples:** Heurística baseada em regras sobre dados já disponíveis. ~80 linhas.
**Por que é eficaz:** Estabiliza a análise de transferências; pré-requisito para validação externa. Resolve o problema "Ferrari é outlier".
**Artefato:** Script `team_tiers.py` + `team_tiers.csv`.

### 5.4 SHAP: Análise de Qualidade do Sinal 🟡

- [ ] Estender notebook `shap_encoder_importance_2021.ipynb` com análise de acerto/erro
- [ ] Comparar `driver_importance` com o resultado real da corrida
- [ ] Identificar se erros vêm de over-reliance no sinal do piloto ou da equipe
- [ ] Classificar previsões corretas vs. incorretas e analisar distribuição de importância
- [ ] Gerar figura: boxplot de driver_importance para acertos vs. erros

**Por que é simples:** Infraestrutura SHAP já existe no notebook. ~60 linhas adicionais.
**Por que é eficaz:** Transforma SHAP descritivo em prescritivo — mostra quando o modelo confia no sinal errado.
**Artefato:** Notebook atualizado com seção "Signal Quality Analysis".

### 5.5 Coleta de Referências para Related Work ⚪

- [ ] Buscar no Google Scholar / Semantic Scholar os termos abaixo
- [ ] Manter arquivo `related_work.md` com tabela: autor, título, ano, resumo de 1-2 frases, relevância
- [ ] Meta: 15-20 referências coletadas até 04/08 (as 30 finais vêm depois)

**Termos de busca por área:**
1. **Orthogonality-constrained representation learning:** disentanglement, Barlow Twins (Zbontar et al., 2021), VicReg (Bardes et al., 2022), redundancy reduction, decorrelation losses
2. **Player valuation em sports analytics:** decomposição jogador vs. time na NBA, futebol, baseball — como separam contribuição individual do contexto
3. **Formula 1 data analysis:** driver ranking, constructor contribution, métodos quantitativos de avaliação de piloto
4. **Shapley value attribution:** NBA player valuation paper (Sloan), SHAP em esportes, feature attribution
5. **Graph neural networks for sports:** grafos temporais, GNN heterogêneo para previsão esportiva
6. **Representation learning for multi-entity systems:** driver+team, player+team, decomposição de fatores latentes

**Referências-chave a estudar primeiro:**
- *Deep Reinforcement Learning for NBA Player Valuation* (Sloan) — referência principal de comparação
- *Barlow Twins* (Zbontar et al., 2021) — fundamento teórico de orthogonality constraint
- *VicReg* (Bardes et al., 2022) — alternativa ao Barlow Twins
- Artigos do Sloan Sports Conference sobre F1 ou motorsport (se existirem)

**Por que é simples:** Pesquisa bibliográfica, sem código. 3-4 horas.
**Por que é eficaz:** 30 referências não se fazem em uma semana; começar agora evita desespero. Abstract é em outubro.
**Artefato:** `related_work.md` com tabela de referências.

### 5.6 Cronograma

| Data | Ação | Esforço |
|------|------|---------|
| 23-24 Jul | 5.5 — Coleta inicial de referências (paralelo) | 3-4h |
| 25-26 Jul | 5.2 — Modelos single-modality | 2-3h |
| 27-28 Jul | 5.1 — Curva AUROC vs. corridas | 3-4h |
| 29-30 Jul | 5.3 — Tiers de equipes | 2-3h |
| 31 Jul-02 Ago | 5.4 — SHAP: qualidade do sinal | 3-4h |
| 03-04 Ago | Buffer / refinamento / integração | - |

**Total estimado:** ~15-20 horas em 12 dias.
Framework ser quantitativo, análise de múltiplas temporadas, related work para comparar caracterização de piloto

[ieeexplore.ieee.org/abstract/document/10932140](https://ieeexplore.ieee.org/abstract/document/10932140)

[ieeexplore.ieee.org/abstract/document/11134599](https://ieeexplore.ieee.org/abstract/document/11134599)

[dl.acm.org/doi/abs/10.1145/3672608.3707766](https://dl.acm.org/doi/abs/10.1145/3672608.3707766)
