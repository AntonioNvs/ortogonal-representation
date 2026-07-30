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

## Related-Work

- Related-Work: Padrão é ter 30 referências - trabalhar nisso
- Onde mais foi aplicado a orthogonality constrained?
- Isso já é meu POC II, estender para um artigo científico (então fica muito forte para o poc já)
- Não precisa colocar std ou IC nas tabelas - espaço é algo bom
- Como fica a eficácia da predição conforme a temporada percorre (o quanto útil é os dados da temporada passada) - eixo X: número de corridas transcorridas na temporada, eixo Y: aucroc
  - Desempenho médio do modelo (pega o range de temporadas de teste)
- Como fica o modelo que só olha para equipe/piloto
- Como validar isso? Olha a carreira deles - mudou para uma equipe melhor/pior? desempenho de pontos nas próximas temporadas (montar os rótulos)
- Como rankear as equipes: ferrari é um outlier de todas, definir categorias de equipe (tier 1, 2 e 3), fica mais estável, começar simples e depois mudar, mas o importante é criar o framework de ranqueamento
- Como definir que o sinal é um bom ou ruim? Uma heurística direta no espaço latente? SHAP já dá isso - analisar sobre

[www.sloansportsconference.com/research-paper-competition](https://www.sloansportsconference.com/research-paper-competition)

Olhar esse artigo
Deep Reinforcement Learning for NBA Player Valuation: A Temporal Difference Approach with Shapley Attribution

Focar legal nele

Abstract em outubro

[docs.google.com/document/d/12zoFMnFujCkQcYCD-c14W5C9tJjXa2T8cqWHk9sNkM4/edit?tab=t.0](https://docs.google.com/document/d/12zoFMnFujCkQcYCD-c14W5C9tJjXa2T8cqWHk9sNkM4/edit?tab=t.0)

Framework ser quantitativo, análise de múltiplas temporadas, related work para comparar caracterização de piloto

[ieeexplore.ieee.org/abstract/document/10932140](https://ieeexplore.ieee.org/abstract/document/10932140)

[ieeexplore.ieee.org/abstract/document/11134599](https://ieeexplore.ieee.org/abstract/document/11134599)

[dl.acm.org/doi/abs/10.1145/3672608.3707766](https://dl.acm.org/doi/abs/10.1145/3672608.3707766)
