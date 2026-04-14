#!/usr/bin/env bash
# ============================================================
# setup.sh — Inicialização completa do pipeline de crédito
# Uso: bash setup.sh
# ============================================================
set -e

# Cores para output
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${GREEN}[OK]${NC} $1"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WAIT]${NC} $1"; }
step() { echo -e "\n${BOLD}${CYAN}━━━ $1 ━━━${NC}"; }
err()  { echo -e "${RED}[ERRO]${NC} $1"; exit 1; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Pipeline de Risco de Crédito — Setup Completo     ║"
echo "║   Airflow + dbt + MLflow + MinIO + FastAPI           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Pré-requisitos ────────────────────────────────────────────────────────────
step "VERIFICANDO PRÉ-REQUISITOS"

command -v docker >/dev/null 2>&1 || err "Docker não encontrado. Instale o Docker Desktop."
command -v python >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1 || err "Python não encontrado."

PYTHON=$(command -v python3 2>/dev/null || command -v python)
log "Docker: $(docker --version | cut -d' ' -f3 | tr -d ',')"
log "Python: $($PYTHON --version)"

# Cria diretórios necessários
mkdir -p data/raw data/silver data/gold
log "Diretórios data/ criados"

# ── Passo 1: Build e subida dos containers ────────────────────────────────────
step "PASSO 1/5 — BUILD E SUBIDA DOS CONTAINERS"
info "Building imagens customizadas (Airflow com ML deps)..."
info "Primeira vez leva ~5-10 min (download de pacotes). Aguarde..."

docker compose build --no-cache 2>&1 | grep -E "(Building|Built|Error|error)" || true
docker compose up -d

log "Containers iniciados"

# ── Passo 2: Aguarda serviços ficarem prontos ──────────────────────────────────
step "PASSO 2/5 — AGUARDANDO SERVIÇOS FICAREM PRONTOS"

wait_for_service() {
    local name=$1
    local url=$2
    local max_wait=${3:-180}
    local elapsed=0

    warn "Aguardando $name ficar pronto ($url)..."
    until curl -sf "$url" >/dev/null 2>&1; do
        if [ $elapsed -ge $max_wait ]; then
            err "$name não ficou pronto em ${max_wait}s. Verifique: docker compose logs $name"
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo -n "."
    done
    echo ""
    log "$name pronto! ($elapsed s)"
}

# PostgreSQL (via docker exec)
info "Aguardando PostgreSQL..."
until docker exec credit_postgres pg_isready -U airflow >/dev/null 2>&1; do
    sleep 3; echo -n "."
done
echo ""; log "PostgreSQL pronto!"

# MLflow (demora pois instala pacotes na primeira vez)
wait_for_service "MLflow" "http://localhost:5000/health" 240

# Airflow
wait_for_service "Airflow" "http://localhost:8080/health" 180

log "Todos os serviços estão prontos!"

# ── Passo 3: Gera dados sintéticos ────────────────────────────────────────────
step "PASSO 3/5 — GERANDO DADOS SINTÉTICOS (10k registros)"

info "Instalando dependências Python localmente..."
$PYTHON -m pip install pandas numpy pyarrow --quiet

info "Rodando generate_data.py..."
$PYTHON scripts/generate_data.py --n 10000 --seed 42 \
    --output data/raw/clientes_contratos.parquet

log "Dados gerados em data/raw/clientes_contratos.parquet"

# ── Passo 4: Treina modelos com MLflow ────────────────────────────────────────
step "PASSO 4/5 — TREINANDO MODELOS (LR + XGBoost)"
info "Isso pode levar 2-3 minutos..."

info "Instalando dependências de ML..."
$PYTHON -m pip install scikit-learn xgboost shap mlflow boto3 scipy matplotlib --quiet

info "Rodando train_model.py..."
MLFLOW_TRACKING_URI=http://localhost:5000 \
AWS_ACCESS_KEY_ID=minioadmin \
AWS_SECRET_ACCESS_KEY=minioadmin \
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
$PYTHON scripts/train_model.py \
    --mlflow-uri http://localhost:5000 \
    --data data/raw/clientes_contratos.parquet \
    --run-name "xgb_champion_v1"

log "Modelos treinados e registrados no MLflow!"

# ── Passo 5: Aguarda FastAPI carregar o modelo ────────────────────────────────
step "PASSO 5/5 — VALIDANDO API"

info "Reiniciando FastAPI para carregar modelo de Production..."
docker compose restart fastapi

wait_for_service "FastAPI" "http://localhost:8000/health" 120

# Testa predição
info "Testando predição de exemplo..."
RESPONSE=$(curl -s -X POST http://localhost:8000/predict \
    -H "Authorization: Bearer credito-api-token-2024" \
    -H "Content-Type: application/json" \
    -d '{
        "cliente": {
            "idade": 35,
            "renda_mensal": 5000,
            "score_interno": 520,
            "score_serasa": 490,
            "limite_credito": 15000,
            "saldo_devedor": 10000,
            "num_parcelas": 24,
            "valor_parcela": 420,
            "idade_contrato": 12,
            "historico_atrasos_30d": 2,
            "tem_cpf_negativado": 0,
            "num_dependentes": 2,
            "tempo_relacionamento": 18
        }
    }' 2>/dev/null)

if echo "$RESPONSE" | grep -q "probabilidade_default"; then
    PD=$(echo "$RESPONSE" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['probabilidade_default']*100:.1f}%\")" 2>/dev/null || echo "N/A")
    SEG=$(echo "$RESPONSE" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(d['segmento_risco'])" 2>/dev/null || echo "N/A")
    log "API funcionando! PD=$PD | Segmento=$SEG"
else
    warn "API respondeu mas resposta inesperada: $RESPONSE"
fi

# ── Resumo Final ──────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║            SETUP COMPLETO COM SUCESSO!               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}${BOLD}"
echo "  SERVIÇOS DISPONÍVEIS:"
echo ""
echo "  🌐 Airflow UI    → http://localhost:8080"
echo "     Login: admin / admin"
echo ""
echo "  🔬 MLflow UI     → http://localhost:5000"
echo "     Veja experimentos e modelos registrados"
echo ""
echo "  🪣 MinIO Console → http://localhost:9001"
echo "     Login: minioadmin / minioadmin"
echo ""
echo "  ⚡ FastAPI Docs  → http://localhost:8000/docs"
echo "     Token: credito-demo-token-2024"
echo ""
echo "  PRÓXIMOS PASSOS:"
echo ""
echo "  1. Abra http://localhost:8080 e ative a DAG:"
echo "     'pipeline_ingestion_credito'"
echo ""
echo "  2. Teste a API em http://localhost:8000/docs"
echo ""
echo "  3. Conecte o Power BI em:"
echo "     data/warehouse.duckdb  (tabelas gold.*)"
echo -e "${NC}"
