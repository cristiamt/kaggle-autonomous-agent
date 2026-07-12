# 🤖 Autonomous ML Agent - Kaggle Competition

Este repositório contém um agente autônomo de Machine Learning construído para competir em desafios de dados tabulares no Kaggle. O agente é coordenado por um LLM e desenhado para escrever, executar e submeter pipelines completos de ML sem intervenção humana.

## 🧠 Arquitetura de IA
* **Modelo Base:** Gemini 3.1 Flash Lite.
* **Estratégia de Prompting:** *Single-Shot Execution*. O agente opera sob restrições rígidas para contornar análises exploratórias longas, entregando um script `master_pipeline.py` completo e robusto em uma única chamada de ferramenta. Isso maximiza a velocidade e minimiza os custos de API.

## ⚙️ Estratégia de Machine Learning
O cérebro do agente foi instruído com um prompt focado em estabilidade e prevenção de *overfitting*, utilizando táticas clássicas de competições:
* **Pré-processamento:** *Ordinal Encoding* para variáveis categóricas, evitando a dispersão (sparsity) gerada por One-Hot Encoding em algoritmos de árvore.
* **Validação:** Validação Cruzada Estratificada com 5 dobras (*5-Fold Stratified CV*).
* **Modelagem:** Ensemble de **XGBoost** e **LightGBM**, extraindo a média das probabilidades.
* **Métrica Alvo:** ROC AUC.

## 🏆 Resultados (Benchmark)
* **Public Leaderboard:** 0.707
* **Private Leaderboard:** 0.704
* **Tempo de Execução:** ~28 segundos (Totalmente autônomo).

## 📁 Estrutura do Repositório
* `prompts/system.md`: O núcleo do projeto. Contém as instruções do sistema e o template matemático rígido para o LLM.
* `agent.yaml`: Configurações do agente, definição do modelo e mapeamento de ferramentas permitidas (write_file, run_command, etc).
* `.gitignore`: Blindagem de dados de treino (`data/`), chaves de API (`.env`) e lixo de compilação.