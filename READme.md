### To-Do

**04/06**

- [X] Reformulação completa do sistema, simplificando a base de dados e o modelo
  - [X] Diminuição de loss
  - [X] Remoção do *track encoder*
  - [X] Somente uma base de dados, a relbench
  - [X] Ter uma GNN que gera dois espaços latentes, ao invés de dois modelos
- [X] Fazer um treinamento completo e analisá-lo no analysis.ipynb
- [X] Verificar o temporal leakage no treinamento, adicionando uma máscara nas arestas
- [ ] Estudar sobre espaços latentes ortogonais

**05/06**

- [ ] Agrupamento de referências literárias sobre o tema
- [ ] Estruturação do relatório final, nas definições que são propostas

### Perguntas que preciso responder e saber bem

#### 1. Porque usar o relbench dataset?

- Padronização e Relevância: O Relbench é um benchmark moderno e padronizado especificamente desenhado para aprendizado de máquina em bancos de dados relacionais (Deep Learning on Relational Databases).
- Complexidade Realista: Diferente de datasets tabulares simples, ele reflete a complexidade do mundo real com múltiplas tabelas, chaves primárias/estrangeiras e, crucialmente, dados temporais.
- Comparabilidade: Usar um benchmark estabelecido permite que seus resultados sejam comparados de forma justa com o estado da arte (state-of-the-art), validando a eficácia da sua abordagem de representação ortogonal contra baselines conhecidos.

#### 2. Porque usar um modelo de rede neural para grafos?

- Mapeamento Natural: Bancos de dados relacionais são, em sua essência, grafos. As linhas das tabelas são os nós (entidades) e as chaves estrangeiras são as arestas (relações).
- Captura de Contexto Multi-hop: Modelos tradicionais (como XGBoost ou MLPs) exigem engenharia de features manual (feature engineering) para juntar (JOIN) tabelas. As GNNs (como o seu graph_encoder.py) conseguem agregar informações de vizinhanças complexas e distantes de forma automática, capturando dependências estruturais que modelos tabulares ignoram.

#### 3. Porque o conjunto de treino/validação/teste foi esse?

- Divisão Temporal (Time-based Split): Em dados relacionais e temporais (como os do Relbench), divisões aleatórias (random splits) causam vazamento de dados (data leakage). A justificativa ideal é que o split foi feito de forma cronológica (ex: treinar em dados até 2020, validar em 2021, testar em 2022).
- Simulação do Mundo Real: Isso garante que o modelo seja avaliado exatamente como seria usado na vida real: prevendo o futuro com base apenas no passado.

#### 4. Há temporal leakage no treinamento?

- O pipeline de amostragem, gerenciado pelo Relbench e tratado no train.py garante que, para prever um alvo no tempo t, o modelo só tem acesso a features e arestas do grafo onde o carimbo de tempo (timestamp) é estritamente menor que t.

#### 5. Porque aplicar loss de ortogonalidade?

- Redução de Redundância: Em redes neurais profundas, é comum que diferentes neurônios ou cabeças de atenção aprendam a mesma coisa (colapso de representação). A loss de ortogonalidade força os vetores de representação (embeddings) a serem independentes/descorrelacionados.
- Desemaranhamento (Disentanglement): Ajuda o modelo a aprender características distintas e complementares dos dados. Como você tem um pipeline_fusion.py, a ortogonalidade pode estar sendo usada para garantir que as diferentes fontes de informação (ou diferentes sub-redes) não sobreponham suas representações antes da fusão.

#### 6. A loss de ortogonalidade realmente da certo?

- ESCREVER APÓS OS TESTES FINAIS

#### 7. Como comprovar que a loss de ortogonalidade ajuda?

- Estudo de Ablação: lambda = 0, lambda > 0
- Testes de Significância: aucroc melhor, metrica de ortogonalidade melhor
- Análise qualitativa: matriz de covariância das representações geradas

#### 8. Como saber que as aplicações que eu quero podem ser de valia?

- Comparação com Baselines Fortes
- Impacto no negócio: exemplo de uso dos encoders
