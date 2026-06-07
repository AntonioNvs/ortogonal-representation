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

- Porque usar o relbench dataset?
- Porque usar um modelo de rede neural para grafos?
- Porque o conjunto de treino/validação/teste foi esse?
- Há temporal leakage no treinamento?
- Porque aplicar loss de ortogonalidade?
- A loss de ortogonalidade realmente da certo?
- Como comprovar que a loss de ortogonalidade ajuda?
- Como saber que as aplicações que eu quero podem ser de valia?
