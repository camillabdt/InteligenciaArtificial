# 🧠 Inteligência Artificial para Detecção de Fake News com XAI

Este projeto é um estudo aplicado no contexto da disciplina de Inteligência Artificial (UFMS - 2025/1), com foco na aplicação de classificadores supervisionados e técnicas de explicabilidade (XAI) para identificar desinformação em textos jornalísticos.

## 🔍 Objetivo

Avaliar e comparar algoritmos de aprendizado supervisionado utilizando um conjunto de **red flags linguísticas** extraídas de notícias verdadeiras e falsas. O projeto também busca interpretar os resultados por meio de explicações geradas com **SHAP (SHapley Additive Explanations)**.

---

## 🗂 Estrutura do Projeto

```bash
📦IA - MESTRADO/
├── load-data.py                # Leitura e preparação dos dados
├── train-models.py             # Treinamento dos classificadores
├── plot-results.py            # Visualização comparativa dos resultados
├── salva-final-table.py       # Geração da tabela final
├── model_results.csv          # Resultados salvos em CSV
├── comparacao_modelos.png     # Gráfico comparando acurácia final
├── tabela_resultados_bonita.png  # Tabela final formatada
├── Fakenews-dataset-final.csv # Dataset final com red flags
├── requirements.txt           # Bibliotecas utilizadas
├── .gitignore                 # Arquivos ignorados pelo Git (ex: venv/)
└── README.md                  # Este arquivo
```

---

## 📊 Dataset

O dataset **`Fakenews-dataset-final.csv`** foi construído a partir de:
- Um subconjunto balanceado (500 reais, 500 falsas) baseado no [ISOT Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets).
- Atribuição de **16 red flags linguísticas** com apoio de **LLMs** como Qwen, DeepSeek, LLaMA e Gemma.
- Adição de uma coluna chamada **Temperatura** (soma total das red flags por amostra).

---

## 🧪 Classificadores Avaliados

- 🌲 **Random Forest**
- ⚙️ **XGBoost**
- 💡 **MLP (Multi-Layer Perceptron)**
- 📐 **SVM (Support Vector Machine)**
- 📊 **KNN (K-Nearest Neighbors)**

---

## 📈 Métricas de Avaliação

- Accuracy
- Precision
- Recall
- F1-Score

Os resultados estão salvos no arquivo `model_results.csv` e visualmente comparados em `comparacao_modelos.png`.

---

## 🧠 XAI com SHAP

O projeto utiliza o pacote **SHAP** para analisar os fatores linguísticos (red flags) que mais influenciam as decisões dos classificadores, especialmente no Random Forest e XGBoost.

---

## ⚙️ Requisitos

Para rodar o projeto, instale os pacotes abaixo em um ambiente virtual Python 3.10+:

```bash
pip install -r requirements.txt
```

---

## 🚀 Execução

1. Carregue os dados:
   ```bash
   python load-data.py
   ```

2. Treine os modelos:
   ```bash
   python train-models.py
   ```

3. Gere os gráficos:
   ```bash
   python plot-results.py
   ```

4. Exporte a tabela final:
   ```bash
   python salva-final-table.py
   ```

---


## 👩‍💻 Autora

**Camilla Borchhardt Quincozes**

- 📚 Mestranda em Ciência da Computação
- 🏫 Universidade Federal de Mato Grosso do Sul (UFMS)
- 💡 Projeto desenvolvido para fins educacionais e científicos

---

## 📜 Licença

Este projeto é de uso acadêmico e livre para fins de estudo, respeitando as fontes originais do dataset e ferramentas utilizadas.
