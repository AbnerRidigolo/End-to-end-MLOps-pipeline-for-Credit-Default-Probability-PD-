# ============================================================
# Makefile — Comandos de conveniência para o pipeline
# Uso: make <comando>
# ============================================================

.PHONY: help setup data train test api up down logs clean

PYTHON := python
DATA_PATH := data/raw/clientes_contratos.parquet

help: ## Mostra este menu
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────
setup: ## Instala dependências Python locais
	pip install -r requirements.txt

setup-dev: ## Instala deps + pre-commit hooks
	pip install -r requirements.txt
	pre-commit install

# ── Dados ──────────────────────────────────────────────────────
data: ## Gera dataset sintético (10k linhas)
	$(PYTHON) scripts/generate_data.py --n 10000

data-large: ## Gera dataset grande (50k linhas)
	$(PYTHON) scripts/generate_data.py --n 50000 \
		--output data/raw/clientes_contratos_50k.parquet

# ── dbt ────────────────────────────────────────────────────────
dbt-deps: ## Instala pacotes dbt
	cd dbt && dbt deps --profiles-dir .

dbt-run: ## Roda todos os modelos dbt
	cd dbt && dbt run --profiles-dir . --no-version-check

dbt-test: ## Roda testes de qualidade dbt
	cd dbt && dbt test --profiles-dir . --no-version-check

dbt-docs: ## Gera e serve documentação dbt (abre browser)
	cd dbt && dbt docs generate --profiles-dir . && dbt docs serve

dbt-silver: ## Roda só modelos Silver
	cd dbt && dbt run --select silver --profiles-dir .

dbt-gold: ## Roda só modelos Gold
	cd dbt && dbt run --select gold --profiles-dir .

# ── ML ─────────────────────────────────────────────────────────
train: ## Treina modelos (LR baseline + XGBoost champion)
	$(PYTHON) scripts/train_model.py --data $(DATA_PATH)

train-custom: ## Treina com nome customizado
	$(PYTHON) scripts/train_model.py \
		--data $(DATA_PATH) \
		--run-name "xgb_$(shell date +%Y%m%d_%H%M%S)"

# ── Testes ─────────────────────────────────────────────────────
test: ## Roda todos os testes pytest
	pytest tests/ -v --tb=short

test-data: ## Testa somente qualidade de dados
	pytest tests/test_data_quality.py -v

test-api: ## Testa somente a API
	pytest tests/test_api.py -v

test-cov: ## Roda testes com cobertura
	pytest tests/ --cov=scripts --cov=app --cov-report=html

# ── API ────────────────────────────────────────────────────────
api: ## Sobe FastAPI local (sem Docker)
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

api-test: ## Testa endpoint de health
	curl -s http://localhost:8000/health | python -m json.tool

predict-test: ## Testa endpoint de predição
	curl -s -X POST http://localhost:8000/predict \
		-H "Authorization: Bearer credito-api-token-2024" \
		-H "Content-Type: application/json" \
		-d '{"cliente": {"idade": 35, "renda_mensal": 5000, "score_interno": 650, \
			"score_serasa": 620, "limite_credito": 15000, "saldo_devedor": 6000, \
			"num_parcelas": 24, "valor_parcela": 250, "idade_contrato": 12, \
			"tem_cpf_negativado": 0, "num_dependentes": 1, \
			"tempo_relacionamento": 36}}' | python -m json.tool

# ── Docker ─────────────────────────────────────────────────────
up: ## Sobe todo o stack (Airflow + MLflow + MinIO + FastAPI)
	docker-compose up -d
	@echo "\nStack subindo... aguarde ~2 min para inicializacao completa"
	@echo "  Airflow UI:  http://localhost:8080  (admin / senha do .env)"
	@echo "  MLflow UI:   http://localhost:5000"
	@echo "  MinIO:       http://localhost:9001  (minioadmin / senha do .env)"
	@echo "  FastAPI:     http://localhost:8000/docs"

down: ## Para e remove containers
	docker-compose down

down-v: ## Para containers e apaga volumes (CUIDADO: apaga dados!)
	docker-compose down -v

logs: ## Mostra logs do Airflow scheduler
	docker-compose logs -f airflow-scheduler

logs-api: ## Mostra logs da FastAPI
	docker-compose logs -f fastapi

restart: ## Reinicia um serviço específico (ex: make restart SERVICE=fastapi)
	docker-compose restart $(SERVICE)

build: ## Reconstrói imagem da FastAPI
	docker-compose build fastapi

# ── Pipeline Completo ──────────────────────────────────────────
pipeline: data dbt-run train ## Roda pipeline completo local (sem Docker)
	@echo "\n[OK] Pipeline completo executado!"

# ── Limpeza ────────────────────────────────────────────────────
clean: ## Remove arquivos temporários e cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
	rm -rf dbt/target dbt/dbt_packages

clean-data: ## Remove dados gerados (CUIDADO!)
	rm -f data/raw/*.parquet data/raw/*.csv data/raw/*.json

# ── Airflow ────────────────────────────────────────────────────
dag-trigger: ## Dispara a DAG de ingestão manualmente
	docker exec credit_airflow_scheduler \
		airflow dags trigger pipeline_ingestion_credito

dag-status: ## Mostra status da última execução da DAG
	docker exec credit_airflow_scheduler \
		airflow dags list-runs -d pipeline_ingestion_credito --limit 5
