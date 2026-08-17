# 📊 Modelos USP — Laboratório de Forecasting & MMM

# 🦇 Squad B.A.T.M.A.N.
> **B**ayesian **A**nalytics, **T**ime-Series & **M**arketing **A**ttribution **N**etwork

*Grupo de estudos e laboratório investigativo de modelagem econométrica e previsão de séries temporais estruturado por pesquisadores e alunos de Ciência de Dados — ICMC / USP.*

---

[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg?style=flat-flat&logo=python)](https://www.python.org/)
[![R Version](https://img.shields.io/badge/R-4.x-blue.svg?style=flat-flat&logo=r)](https://www.r-project.org/)
[![Google Meridian](https://img.shields.io/badge/Google-Meridian%20(v1.8.0)-brightgreen.svg?style=flat-flat&logo=google)](https://github.com/google/meridian)
[![Meta Robyn](https://img.shields.io/badge/Meta-Robyn-blue.svg?style=flat-flat&logo=meta)](https://github.com/facebookexperimental/Robyn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### 🛠️ Estrutura Metodológica e Tecnologias

A análise de mix de marketing e previsão de demanda requer a combinação de diferentes abordagens estatísticas. A metodologia do time é dividida nos seguintes pilares complementares:

| Componente | Abordagem / Objetivo Metodológico | Tecnologias Utilizadas |
| :--- | :--- | :--- |
| **Forecasting Univariado** | Modelagem estatística de tendência, sazonalidade anual e impacto de feriados | `Facebook Prophet` + `cmdstanpy` |
| **MMM Bayesiano** | Atribuição probabilística de investimentos com incorporação de distribuições a priori (priors) | `Google Meridian` + `JAX` |
| **MMM Otimizado** | Modelagem de retornos decrescentes e adstock temporal via algoritmos de otimização evolutiva | `Meta Robyn` (R) |
| **Modelagem Customizada** | Implementação manual de regressões (OLS/Ridge) com parametrização direta das curvas de Hill e Adstock geométrico | `Python` (`scikit-learn`, `NumPy`) |

---

## 🧠 Cobertura Metodológica e Teórica

Para fundamentar as análises estatísticas e de modelagem preditiva, este repositório cobre os seguintes tópicos teóricos divididos por vertentes de Ciência de Dados:

### 1. Modelagem Preditiva & Métodos Dinâmicos
* **Séries Temporais & Modelos de Espaço de Estado:** Decomposição estrutural estocástica de sinais, identificação de sazonalidade, tendência e impacto de feriados locais (implementado via *Facebook Prophet*).
* **Modelos Autoregressivos Clássicos (Box-Jenkins):** Suavização exponencial, diagnóstico de estacionariedade e formulação de modelos autoregressivos (`AR`, `MA`, `ARMA`, `ARIMA` e `SARIMA`) e expansões (*Método Theta*).
* **Modelagem Temporal de Adstock:** Métodos de atenuação e carregamento temporal de mídia por meio de decaimentos geométricos e funções de suavização.

### 2. Atribuição Causal & Inferência Bayesiana
* **Paradigma Bayesiano e Teorema de Bayes:** Formulação probabilística para cálculo do efeito incremental de investimentos, mapeamento de distribuições *a priori* (priors) informativas e não-informativas, e estimação de parâmetros por cadeias de Markov (MCMC).
* **Computação Bayesiana:** Implementações probabilísticas robustas com otimização em JAX (*Google Meridian*) e flexibilidade para simulações condicionais.

### 3. Modelos de Regressão e Aprendizado Supervisionado
* **Regressão Linear Múltipla & Diagnóstico:** Estimadores clássicos OLS e análise de qualidade de ajuste (cálculo de $R^2$, MAPE, RMSE) e resíduos.
* **Modelos Lineares Generalizados (GLM):** Extensões para a família exponencial, com ênfase na Regressão Logística para classificação probabilística e diagnóstico frequentista/bayesiano.
* **Seleção e Regularização:** Penalizações matriciais L2 (*Ridge*) para lidar com multicolinearidade severa em investimentos e algoritmos de seleção de variáveis.

### 4. Aprendizado Não-Supervisionado & Dimensionalidade
* **Redução de Dimensionalidade & Segmentação:** Técnicas de Análise de Componentes Principais (*PCA*) para extração de variância explicada e agrupamento de dados (*Clustering* via *K-Means*) para identificação de perfis latentes.

### 5. Tratamento de Sinais & Dados Não-Estruturados
* **Saneamento e Limpeza de Bases:** Engenharia de atributos, tratamento estatístico de nulos, imputação e outliers.
* **Processamento de Linguagem Natural (NLP):** Captura automatizada (Web Scraping) e vetorização de texto por meio de matrizes termo-documento (Bag of Words) e TF-IDF.

---

## 📁 Estrutura do Projeto

Abaixo está o mapeamento dos diretórios e arquivos do repositório:

```text
modelos_USP/
├── data/
│   ├── data_raw/            # Bases de dados brutas baixadas (.gitkeep)
│   └── data_processed/      # Relatórios e matrizes geradas pelos modelos (.gitkeep)
├── EDA/                     # Pasta com scripts de Análise Exploratória de Dados
│   └── eda_*.py             # Scripts de visualização e perfil de cada dataset
├── models/                  # Pasta com scripts de modelagem estatística e ML
│   ├── 02_model_prophet.py  # Modelo de forecasting univariado (Prophet)
│   ├── 03_model_meridian.py # Modelo MMM Bayesiano do Google (Meridian)
│   ├── 04_model_robyn.R     # Modelo MMM da Meta (Robyn em R)
│   ├── 05_model_custom_manual.py # MMM customizado (Ridge/OLS + Adstock/Hill manuais)
│   ├── 06_compare_results.py # Consolidação e comparação de ROI e métricas
│   ├── 07_model_classification.py # Classificação comparativa (Regressão Logística/SVM/RF)
│   ├── 08_customer_segmentation_pca.py # Agrupamento K-Means com redução por PCA
│   ├── 09_benchmark_arima_theta.py # Benchmark de previsões temporais (ARIMA vs Theta)
│   ├── 10_survival_analysis.py # Modelagem de sobrevivência de clientes (Kaplan-Meier/Cox)
│   └── 11_competitor_scraping_nlp.py # Captura via Web Scraping e matrizes BoW/TF-IDF
├── 00_download_data.py      # Script para baixar automaticamente bases de dados do Kaggle
├── 00_test_env.py           # Script para validação de imports do ambiente conda/JAX/Prophet
├── 01_data_prep.py          # Script de carregamento, limpeza e simulação de dados
├── .gitignore               # Configurações de arquivos ignorados no repositório Git
├── environment.yml          # Especificação completa do ambiente virtual Conda
└── README.md                # Guia de introdução e documentação do projeto
```

---

## 🔍 Onde Obter as Bases de Dados (Fontes)

Para executar os scripts e testar a eficácia dos modelos, você pode alimentar a pasta `data/raw/` com dados das seguintes origens:

| Fonte | Tipo de Dados | Descrição / Como Obter |
| :--- | :--- | :--- |
| **GA4 E-commerce** | Dados de Vendas/Canais | Disponível publicamente no BigQuery em `bigquery-public-data.ga4_obfuscated_sample_ecommerce`. Consulte as queries SQL abaixo para extração. |
| **Google Trends** | Controle / Demanda | Acesse a tabela pública `bigquery-public-data.google_trends.top_terms` para capturar a sazonalidade e volume de busca. |
| **Datasets do Kaggle** | MMM e Series Temporais | Baixe os datasets oficiais (Rossmann, Store Sales, etc.) rodando o script local `python 00_download_data.py`. |
| **Gerador Sintético** | Dados Simulados | Execute `python 01_data_prep.py` para gerar uma base sintética de 156 semanas (3 anos) simulando investimentos em Google, Meta e TikTok Ads. |

---

## 🛢️ Consultas BigQuery (GA4)

Caso opte por utilizar os dados públicos do **GA4** no Google BigQuery Store, utilize as consultas estruturadas a seguir para extrair as métricas formatadas:

### 1. Agregação Diária (Source/Medium)
Esta consulta agrega os dados por dia e mantém todas as combinações de `source` e `medium` da base do GA4:

```sql
SELECT
  PARSE_DATE('%Y%m%d', event_date) AS event_date,
  COALESCE(traffic_source.source, '(direct)') AS source,
  COALESCE(traffic_source.medium, '(none)') AS medium,
  COUNT(DISTINCT CONCAT(user_pseudo_id, CAST(event_timestamp AS STRING))) AS total_events,
  COUNT(DISTINCT user_pseudo_id) AS active_users,
  COUNTIF(event_name = 'purchase') AS total_orders,
  ROUND(SUM(COALESCE(event_value_in_usd, 0)), 2) AS total_revenue_usd
FROM
  `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
GROUP BY
  1, 2, 3
ORDER BY
  event_date DESC, total_revenue_usd DESC;
```

### 2. Agregação Semanal com Agrupamento de Canais (Recomendada para MMM)
Esta consulta processa toda a base bruta do GA4 desde o início, padroniza e agrupa os canais em categorias agregadas por semana, retornando o formato ideal para modelos de séries temporais de médio prazo e MMM:

```sql
WITH base_events AS (
  SELECT
    -- Trunca a data para a segunda-feira da respectiva semana
    DATE_TRUNC(PARSE_DATE('%Y%m%d', event_date), WEEK(MONDAY)) AS week_start_date,
    user_pseudo_id,
    event_name,
    COALESCE(event_value_in_usd, 0) AS revenue,
    -- Padronização simples dos agrupamentos de canais
    CASE
      WHEN traffic_source.medium = 'cpc' OR traffic_source.source = 'google' AND traffic_source.medium LIKE '%paid%' THEN 'paid_search_google'
      WHEN traffic_source.medium = 'organic' THEN 'organic_search'
      WHEN traffic_source.medium IN ('referral', 'app') THEN 'referral'
      WHEN traffic_source.source = '(direct)' OR traffic_source.medium = '(none)' THEN 'direct'
      ELSE 'other_channels'
    END AS channel_group
  FROM
    `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
)

SELECT
  week_start_date AS date,
  
  -- Métricas de Conversão do Negócio
  COUNTIF(event_name = 'purchase') AS total_orders,
  ROUND(SUM(CASE WHEN event_name = 'purchase' THEN revenue ELSE 0 END), 2) AS total_revenue_usd,
  
  -- Volume de Usuários Ativos Totais na Semana
  COUNT(DISTINCT user_pseudo_id) AS total_active_users,
  
  -- Distribuição de Usuários por Canal (Variáveis de Volume/Exposição)
  COUNT(DISTINCT CASE WHEN channel_group = 'paid_search_google' THEN user_pseudo_id END) AS users_paid_search,
  COUNT(DISTINCT CASE WHEN channel_group = 'organic_search' THEN user_pseudo_id END) AS users_organic_search,
  COUNT(DISTINCT CASE WHEN channel_group = 'direct' THEN user_pseudo_id END) AS users_direct,
  COUNT(DISTINCT CASE WHEN channel_group = 'referral' THEN user_pseudo_id END) AS users_referral
FROM
  base_events
GROUP BY
  week_start_date
ORDER BY
  week_start_date ASC;
```

---

## 🚀 Como Configurar o Ambiente Local

Recomendamos o uso de **Conda** ou **Miniforge** (otimizado para processadores macOS Apple Silicon / Intel):

### 1. Clonar o Repositório e Navegar
```bash
git clone <URL_DO_REPOSITORIO>
cd modelos_USP
```

### 2. Criar e Ativar o Ambiente Conda
```bash
conda create -n env_modelos_usp python=3.11 -c conda-forge -y
conda activate env_modelos_usp
```

### 3. Instalar Dependências Principais
```bash
# Dependências do Prophet e compiladores Stan
conda install -c conda-forge cmdstanpy prophet jax pandas numpy -y

# Instalar bibliotecas de Machine Learning e visualização do Python
pip install google-meridian scikit-learn matplotlib seaborn
```

### 4. Testar a Instalação
Execute o script de sanidade técnica para garantir que todas as dependências críticas (Prophet, JAX, Meridian) estão funcionando adequadamente:
```bash
python 00_test_env.py
```

---

## 💻 Fluxo de Execução

Siga a ordem dos scripts abaixo para executar a modelagem completa:

```bash
# Passo 1: Fazer download das bases de dados opcionais do Kaggle
python 00_download_data.py

# Passo 2: Preparar os dados (Gera data/processed_data.csv)
python 01_data_prep.py

# Passo 3: Executar previsões de séries temporais com Prophet
python models/02_model_prophet.py

# Passo 4: Rodar o MMM Bayesiano da Google (Meridian)
python models/03_model_meridian.py

# Passo 5: Rodar o MMM da Meta (Robyn - requer ambiente R instalado)
Rscript models/04_model_robyn.R

# Passo 6: Rodar a regressão manual de Mix de Marketing
python models/05_model_custom_manual.py

# Passo 7: Comparar os coeficientes estimados e curvas de resposta de cada modelo
python models/06_compare_results.py
```

---

## 🔗 Referências Externas de Competição

* **Kaggle Competition:** [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

---

## 🤝 Como Contribuir (Fluxo de Git / PRs)

Para colaborar neste projeto, siga o fluxo abaixo para criar branches e abrir Pull Requests:

### 1. Clonar o Repositório
```bash
git clone https://github.com/seu-usuario/modelos_USP.git
cd modelos_USP
```

### 2. Criar uma Feature Branch
Crie uma branch específica para o modelo ou alteração em que você está trabalhando:
```bash
git checkout -b feature/nome-da-sua-feature
# Exemplo: git checkout -b feature/modelo-prophet
```

### 3. Fazer as Alterações e Commitar
Faça suas alterações e crie commits claros seguindo o padrão de commits semânticos (Conventional Commits):
```bash
git add .
git commit -m "feat: implementa modelo X com adstock geométrico"
```

### 4. Enviar para o GitHub e Abrir um Pull Request
Envie sua branch para o repositório remoto:
```bash
git push origin feature/nome-da-sua-feature
```
Depois disso, acesse a página do repositório no GitHub e abra um **Pull Request (PR)** para revisão e mesclagem.

