# Previsão de Preços de Casas com Machine Learning em Python

## 📊 Visão Geral do Projeto

Este projeto tem como objetivo prever o preço de casas utilizando técnicas de Machine Learning.

Foi utilizado o dataset **California Housing**, bastante conhecido em estudos de ciência de dados, para treinar e avaliar modelos capazes de prever o valor médio de casas com base em características socioeconômicas e geográficas.

O projeto demonstra o fluxo completo de um projeto de Ciência de Dados, desde a exploração dos dados até a avaliação e interpretação do modelo.

---

## 📚 Dataset

O projeto utiliza o **California Housing Dataset**, um conjunto de dados amplamente utilizado em estudos de Ciência de Dados e Machine Learning.

O dataset contém informações sobre distritos da Califórnia, incluindo características como:

- renda média da região
- número médio de quartos
- localização geográfica
- densidade populacional

Essas variáveis são utilizadas para prever o valor médio das casas em cada região.

---

## 🧠 Modelos de Machine Learning Utilizados

Foram testados dois algoritmos:

- Regressão Linear
- Random Forest Regressor

A comparação entre os modelos permite avaliar qual algoritmo consegue capturar melhor os padrões presentes nos dados.

---

## 📈 Métricas de Avaliação

Os modelos foram avaliados utilizando as seguintes métricas:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

Essas métricas permitem medir a precisão das previsões realizadas pelos modelos.

---

## 🔍 Interpretação do Modelo

Além da previsão de preços, o projeto também analisa **quais variáveis têm maior impacto na previsão**, utilizando a importância das variáveis do modelo Random Forest.

Essa etapa é importante para entender quais fatores mais influenciam o valor das casas.

---

## 🛠 Tecnologias Utilizadas

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 📂 Estrutura do Projeto

ml-house-price-project
│
├── house_prediction.py
├── README.md
├── requirements.txt
├── data/
└── images/

---

## ▶️ Como executar o projeto

1. Clonar o repositório
git clone https://github.com/SimoesBia/house-price-ml-project.git

2. Instalar as dependências
pip install -r requirements.txt

3. Executar o script
python house_prediction.py


---

## 🚀 Objetivo

O objetivo deste projeto é praticar técnicas de Machine Learning e desenvolver um projeto de portfólio para oportunidades de estágio em **Ciência de Dados, Machine Learning ou Análise de Dados**.

---

## 👩‍💻 Autora

Projeto desenvolvido por **Bia** como parte dos estudos em Ciência da Computação e Machine Learning.