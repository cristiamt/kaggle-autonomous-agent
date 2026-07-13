# 🤖 Autonomous Machine Learning Agent - Zero-Shot Architecture

Este repositório contém a arquitetura de um Cientista de Dados Autônomo construído para resolver problemas de classificação tabular do zero. O agente é coordenado pelo modelo **Gemini Flash (Free API)** e foi desenhado para escrever, executar e submeter pipelines complexos de Machine Learning em uma única chamada (*Zero-Shot*), respeitando limites rígidos de *Rate Limit* (RPM/RPD) de APIs gratuitas.

## 🧠 Arquitetura do Agente e Estratégia Matemática

Para evitar o *overfitting* de prompt (onde o agente fica viciado em um único dataset) e contornar a explosão de dimensionalidade ao criar variáveis cegamente, implementamos um pipeline dinâmico de 4 fases que se adapta à matriz de dados de forma autônoma:

1. **O Scout (Seleção de Variáveis):** O LLM treina um modelo raso preliminar (`LightGBM`) apenas para mapear os pesos e a importância real (Sinal vs. Ruído) das colunas originais.
2. **Engenharia Direcionada:** Criação de variáveis cruzadas e interações matemáticas focadas estritamente nas **Top 10 features** mais importantes, injetando não-linearidade sem estourar o limite de memória.
3. **Poda Rígida:** O agente identifica e descarta sumariamente colunas com importância igual a zero no modelo Scout.
4. **Ensemble Final:** Combinação de peso pesado usando **XGBoost, LightGBM e CatBoost**, validados em um esquema `Stratified 5-Fold K-Fold` para garantir estabilidade e maximizar a métrica ROC AUC.

## 🏆 Resultados e Generalização (Benchmark)

A robustez da arquitetura foi testada em múltiplos datasets independentes (cenários *Kaggle-in-Kaggle*), provando a capacidade do agente de recalcular rotas e se adaptar a diferentes níveis de esparsidade de dados sem nenhuma alteração manual no prompt:

* **Dataset 01 (`kaggle_in_kaggle_train_01`):** ROC AUC ~0.71 (Teto de extração informacional alcançado).
* **Dataset 02 (`kaggle_in_kaggle_train_02`):** ROC AUC **0.962** (Generalização perfeita demonstrando a eficácia do filtro Scout e do Ensemble).
* **Tempo médio de autonomia:** ~55 segundos do código em branco até a submissão final.

## 📁 Estrutura do Repositório

* `prompts/system.md`: O "cérebro" matemático do projeto. Contém as instruções do sistema e as restrições arquiteturais para o LLM.
* `agent.yaml`: Configurações de conexão do agente e definição das ferramentas permitidas.
* `test_all_folders.py`: Script de automação *built-in* para orquestrar e testar o agente iterativamente através de 16 cenários de dados diferentes, com freios de segurança contra bloqueios de API.
* `.gitignore`: Blindagem de dados sensíveis e pesados.