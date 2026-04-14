# 🏦 Credit Risk ML Pipeline

> Pipeline end-to-end Análise financeira e Risco de Crédito com MLOps
> **Stack:** Airflow 2.9 · dbt 1.8 · MLflow 2.14 · FastAPI 0.115 · XGBoost 2.1 · DuckDB · MinIO · Docker · GenAI Copilot (API HuggingFace)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![Airflow](https://img.shields.io/badge/Apache_Airflow-2.9.3-017CEE?logo=apacheairflow)](https://airflow.apache.org)
[![dbt](https://img.shields.io/badge/dbt-1.8-FF694B?logo=dbt)](https://getdbt.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.14-0194E2?logo=mlflow)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-FF6600)](https://xgboost.ai)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docker.com)

---

## Visão Geral

Pipeline completo de MLOps para modelagem de **Probabilidade de Default (PD)** inspirado nas stacks de risco de crédito de fintechs brasileiras (Nubank, XP, Itaú). Cobre todo o ciclo: ingestão → transformação → treinamento → serving → explicabilidade com GenAI.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ARQUITETURA DO PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────────────────────────────────────┐  │
│  │ Airflow  │───▶│           DATA LAKE (MinIO/DuckDB)           │  │
│  │  DAGs    │    │  Bronze (raw) → Silver (clean) → Gold (ML)   │  │
│  └──────────┘    └────────────────────┬─────────────────────────┘  │
│       │                               │ dbt models                  │
│       ▼                               ▼                             │
│  ┌──────────┐    ┌──────────────────────────────────────────────┐  │
│  │  Python  │───▶│              MLflow                          │  │
│  │ generate │    │  LR Baseline → XGBoost Champion              │  │
│  │  _data   │    │  AUC >0.85 | KS >0.40 | Gini >0.70         │  │
│  └──────────┘    └────────────────────┬─────────────────────────┘  │
│                                       │ Model Registry              │
│                                       ▼                             │
│                  ┌──────────────────────────────────────────────┐  │
│                  │         FastAPI (Docker)                      │  │
│                  │  /predict  /explain  /predict/batch           │  │
│                  │  + GenAI Copilot (Claude API)                 │  │
│                  └────────────────────┬─────────────────────────┘  │
│                                       │ JSON                        │
│                                       ▼                             │
│                  ┌──────────────────────────────────────────────┐  │
│                  │          Power BI Dashboard                   │  │
│                  │  Carteira · Score · SHAP · Vintage · AUC     │  │
│                  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Diagrama Mermaid (Lineage Completo)

```mermaid
flowchart TD
    A[("📂 Raw CSV/Parquet\n(Bronze)")] -->|Airflow DAG| B[stg_contratos\ndbt Silver]
    B --> C[fato_contrato\ndbt Gold]
    B --> D[atrasos\ndbt Gold]
    C --> E[features_risco\ndbt Gold]
    D --> E
    E -->|train_model.py| F{{"🔬 MLflow\nExperimentos"}}
    F -->|AUC 0.72| G[LR Baseline]
    F -->|AUC 0.88| H[XGBoost Champion]
    H -->|Staging → Production| I[("📦 MLflow\nModel Registry")]
    I -->|pyfunc.load_model| J["⚡ FastAPI\n/predict"]
    J -->|SHAP + Claude API| K["🤖 GenAI Copilot\nExplicação PT-BR"]
    E -->|SQL Export| L["📊 Power BI\nDashboard"]

    style H fill:#FF6600,color:#fff
    style I fill:#0194E2,color:#fff
    style K fill:#7C3AED,color:#fff
```

---

## Estrutura do Projeto

```
credit-risk-pipeline/
├── 🐳 docker-compose.yml          # Stack completo: Airflow, MLflow, MinIO, FastAPI
├── ⚙️  Makefile                    # Comandos de conveniência
├── 📦 requirements.txt             # Dependências Python
│
├── 📁 dags/
│   └── ingestion_dag.py           # DAG: Bronze → Silver → Gold → ML trigger
│
├── 📁 dbt/
│   ├── dbt_project.yml
│   ├── profiles.yml                # DuckDB adapter
│   └── models/
│       ├── silver/
│       │   ├── stg_contratos.sql  # Limpeza, cast, dedup
│       │   └── schema.yml         # Testes dbt
│       └── gold/
│           ├── fato_contrato.sql  # Fato principal + segmentação de risco
│           ├── atrasos.sql        # Roll-rates, buckets, vintage curves
│           ├── features_risco.sql # Feature store para ML
│           └── schema.yml
│
├── 📁 scripts/
│   ├── generate_data.py           # 10k linhas sintéticas realistas BR
│   ├── train_model.py             # LR + XGBoost + MLflow completo
│   └── init_db.sql                # Init Postgres
│
├── 📁 app/
│   ├── main.py                    # FastAPI: /predict, /explain, /batch
│   ├── copilot.py                 # GenAI Copilot (Claude API + fallback)
│   ├── Dockerfile                 # Multi-stage build
│   └── requirements.txt
│
└── 📁 tests/
    ├── conftest.py
    ├── test_data_quality.py       # 14 testes de qualidade de dados
    └── test_api.py                # 15 testes de API
```

---

## Início Rápido (< 2 horas)

### Pré-requisitos
- Docker Desktop instalado e rodando
- Python 3.11+ (para desenvolvimento local)
- 8GB RAM recomendado

### 1. Clone e configure

```bash
git clone https://github.com/seu-usuario/credit-risk-pipeline.git
cd credit-risk-pipeline
cp .env.example .env
# Edite .env para adicionar ANTHROPIC_API_KEY (opcional)
```

### 2. Sobe o stack completo

```bash
make up
# ou: docker-compose up -d
```

Aguarde ~2 minutos. Acesse:

| Serviço | URL | Credenciais |
|---------|-----|-------------|
| **Airflow UI** | http://localhost:8080 | admin / admin |
| **MLflow UI** | http://localhost:5000 | - |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin |
| **FastAPI Docs** | http://localhost:8000/docs | Bearer: credito-demo-token-2024 |

### 3. Executa o pipeline completo

```bash
# Opção A: via Makefile (local, sem Docker)
make pipeline   # gera dados + dbt + treina modelos

# Opção B: via Airflow (com Docker)
make dag-trigger
# Acompanhe em: http://localhost:8080
```

### 4. Testa a API

```bash
# Predição single
make predict-test

# Ou via curl direto:
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer credito-demo-token-2024" \
  -H "Content-Type: application/json" \
  -d '{
    "cliente": {
      "idade": 32,
      "renda_mensal": 7500,
      "score_interno": 580,
      "score_serasa": 550,
      "limite_credito": 20000,
      "saldo_devedor": 14000,
      "num_parcelas": 36,
      "valor_parcela": 390,
      "idade_contrato": 18,
      "historico_atrasos_30d": 2,
      "tem_cpf_negativado": 0,
      "num_dependentes": 2,
      "tempo_relacionamento": 24
    }
  }'
```

---

## Métricas de Performance (Holdout Temporal)

| Modelo | AUC-ROC | KS Stat | Gini | Avg Precision |
|--------|---------|---------|------|---------------|
| LR Baseline | 0.77 | 0.42 | 0.54 | 0.28 |
| **XGBoost Champion** | **0.88** | **0.58** | **0.76** | **0.45** |

> Split temporal: treino = safras até Jun/2024 | teste = Jul-Dez/2024

---

## Roadmap de Implementação (6 Sprints)

| Sprint | Entrega | Tempo Est. | Dificuldade |
|--------|---------|-----------|-------------|
| **Sprint 1** | Docker Compose + Airflow + Bronze | 6-8h | ⭐⭐ |
| **Sprint 2** | dbt Silver/Gold + Feature Store | 8-10h | ⭐⭐⭐ |
| **Sprint 3** | MLflow: treino + experimentos + registry | 6-8h | ⭐⭐⭐ |
| **Sprint 4** | FastAPI + Docker deploy + testes | 6-8h | ⭐⭐ |
| **Sprint 5** | Power BI dashboard (conecta SQL Gold) | 8-10h | ⭐⭐⭐ |
| **Sprint 6** | GenAI Copilot + README + LinkedIn post | 4-6h | ⭐⭐ |

---

## Power BI — Guia de Conexão

1. Abra Power BI Desktop
2. **Obter Dados** → **DuckDB** (ou use **Arquivo CSV** das exports Gold)
3. Conecte em: `data/warehouse.duckdb`
4. Importe as tabelas: `gold.fato_contrato`, `gold.atrasos`, `gold.features_risco`

**Visuais recomendados:**
- Cartão: Taxa de inadimplência atual, Perda esperada total
- Gráfico de barras: PD por segmento_risco, por produto, por UF
- Linha temporal: AUC ao longo do tempo (por safra)
- Scatter: DTI vs Score colorido por inadimplente
- Matriz: Confusão do modelo (calculada em Python, exportada como CSV)
- Treemap: Concentração de carteira por faixa_score × produto

---

## Testes

```bash
make test          # todos os testes
make test-data     # qualidade de dados
make test-api      # testes de API
make test-cov      # com cobertura de código
```

---

## Troubleshooting Comum

| Problema | Causa | Solução |
|---------|-------|---------|
| Airflow "db not initialized" | Init não completou | `make down && make up` |
| MLflow connection refused | Serviço ainda iniciando | Aguarde 2-3 min |
| dbt "profile not found" | profiles.yml fora do lugar | `cd dbt && dbt run --profiles-dir .` |
| FastAPI 503 "modelo não disponível" | MLflow sem modelo Production | Rode `make train` primeiro |
| MinIO bucket não existe | Criação automática falhou | Crie `mlflow-artifacts` manual no MinIO Console |

---

## Sobre o Projeto

Construído como projeto portfólio para demonstrar domínio de **MLOps end-to-end** no contexto de risco de crédito brasileiro. Combina:
- Engenharia de dados com padrões Medallion (Bronze/Silver/Gold)
- Feature engineering focado em variáveis de bureau (DTI, LTV, Serasa)
- MLOps com versionamento e promoção automática de modelos
- API production-ready com explicabilidade (SHAP + LLM)
- GenAI para tradução de outputs técnicos em linguagem de negócio

---

*Desenvolvido com Python 3.11 · Airflow 2.9.3 · dbt 1.8.7 · MLflow 2.14.3 · FastAPI 0.115.6 · XGBoost 2.1.3 · DuckDB 1.1.3*
