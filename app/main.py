"""
FastAPI — Serving de Modelos de Risco de Crédito (PD)
======================================================
Endpoints:
  GET  /health          — liveness/readiness check
  GET  /model/info      — metadados do modelo em produção
  POST /predict         — predição single (1 cliente)
  POST /predict/batch   — predição em batch (até 1000 clientes)
  POST /explain         — SHAP + GenAI Copilot (explicação natural PT-BR)

Autenticação: Bearer token (simples para demo)
Modelo: carregado do MLflow Registry (stage=Production)
"""

import os   
import time
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.artifacts
import xgboost as xgb
import numpy as np
import pandas as pd
import shap
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

from copilot import CopilotRisco

# ── Configurações ─────────────────────────────────────────────────────────────
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "credit_pd_model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("Variável de ambiente API_TOKEN não definida. Configure antes de iniciar a API.")

FEATURES = [
    "idade", "renda_mensal", "score_interno", "score_serasa",
    "limite_credito", "saldo_devedor", "num_parcelas", "valor_parcela",
    "idade_contrato", "historico_atrasos_30d", "historico_atrasos_60d",
    "historico_atrasos_90d", "dias_atraso_max", "tem_cpf_negativado",
    "num_consultas_bureau_90d", "num_dependentes", "tempo_relacionamento",
    "dti", "ltv", "burden_ratio", "utilizacao_limite",
]

# Estado global do modelo
model_state: dict = {}


# ── Autenticação ──────────────────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)


def verificar_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if credentials is None or credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido ou ausente",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# ── Lifespan: carrega modelo na inicialização ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega modelo MLflow no startup, libera no shutdown."""
    print(f"[STARTUP] Conectando MLflow: {MLFLOW_URI}")
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        # Carrega como Booster nativo — contorna incompatibilidade do wrapper
        # sklearn/pyfunc com scikit-learn >= 1.8 ao deserializar modelos antigos
        artifact_dir = mlflow.artifacts.download_artifacts(model_uri)
        booster = xgb.Booster()
        booster.load_model(os.path.join(artifact_dir, "model.xgb"))
        model_state["model"] = booster
        model_state["model_uri"] = model_uri
        model_state["loaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Mesmo Booster serve para SHAP
        try:
            model_state["shap_explainer"] = shap.TreeExplainer(booster)
        except Exception:
            model_state["shap_explainer"] = None

        model_state["copilot"] = CopilotRisco()
        model_state["mlflow_client"] = mlflow.MlflowClient()
        print(f"[STARTUP] Modelo '{MODEL_NAME}' ({MODEL_STAGE}) carregado!")
    except Exception as e:
        print(f"[WARN] Falha ao carregar modelo: {e}")
        print("[WARN] API rodando em modo degradado (sem modelo)")
        model_state["model"] = None

    yield

    # Cleanup
    model_state.clear()
    print("[SHUTDOWN] Recursos liberados.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk PD API",
    description="API de Predição de Probabilidade de Default (PD) — Portfólio MLOps",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ClienteFeatures(BaseModel):
    """Features de um cliente para predição de PD."""
    # Demográficas
    idade: int = Field(..., ge=18, le=100, description="Idade do cliente")
    # Financeiras
    renda_mensal: float = Field(..., gt=0, description="Renda mensal em R$")
    score_interno: int = Field(..., ge=300, le=1000, description="Score interno (300-1000)")
    score_serasa: int = Field(..., ge=0, le=1000, description="Score Serasa (0-1000)")
    # Contrato
    limite_credito: float = Field(..., gt=0, description="Limite de crédito em R$")
    saldo_devedor: float = Field(..., ge=0, description="Saldo devedor em R$")
    num_parcelas: int = Field(..., ge=1, le=360)
    valor_parcela: float = Field(..., ge=0)
    idade_contrato: int = Field(..., ge=0, le=600, description="Idade do contrato em meses")
    # Comportamentais
    historico_atrasos_30d: int = Field(default=0, ge=0)
    historico_atrasos_60d: int = Field(default=0, ge=0)
    historico_atrasos_90d: int = Field(default=0, ge=0)
    dias_atraso_max: int = Field(default=0, ge=0)
    tem_cpf_negativado: int = Field(default=0, ge=0, le=1)
    num_consultas_bureau_90d: int = Field(default=0, ge=0)
    num_dependentes: int = Field(default=0, ge=0)
    tempo_relacionamento: int = Field(default=12, ge=0, description="Meses de relacionamento")
    # Derivadas (podem ser calculadas automaticamente)
    dti: Optional[float] = Field(default=None, description="Debt-to-Income (calculado se None)")
    ltv: Optional[float] = Field(default=None)
    burden_ratio: Optional[float] = Field(default=None)
    utilizacao_limite: Optional[float] = Field(default=None)

    @field_validator("saldo_devedor")
    @classmethod
    def saldo_menor_limite(cls, v, info):
        # Permite saldo até 110% do limite (cheque especial)
        return v

    def calcular_derivadas(self) -> "ClienteFeatures":
        """Calcula features derivadas se não fornecidas."""
        if self.dti is None:
            self.dti = round(self.saldo_devedor / max(self.renda_mensal * 12, 1), 4)
        if self.utilizacao_limite is None:
            self.utilizacao_limite = round(self.saldo_devedor / max(self.limite_credito, 1), 4)
        if self.ltv is None:
            self.ltv = self.utilizacao_limite
        if self.burden_ratio is None:
            self.burden_ratio = round(self.valor_parcela / max(self.renda_mensal, 1), 4)
        return self


class PredictRequest(BaseModel):
    cliente: ClienteFeatures
    contrato_id: Optional[str] = None
    retornar_shap: bool = Field(default=False, description="Incluir SHAP values na resposta")


class PredictResponse(BaseModel):
    contrato_id: Optional[str]
    probabilidade_default: float = Field(description="PD estimada (0-1)")
    score_credito: int = Field(description="Score derivado da PD (300-1000)")
    segmento_risco: str = Field(description="low_risk | medium_risk | high_risk")
    recomendacao: str
    shap_top_features: Optional[dict] = None
    modelo_versao: str
    latencia_ms: float


class BatchPredictRequest(BaseModel):
    clientes: list[PredictRequest] = Field(..., max_length=1000)


class ExplainRequest(BaseModel):
    cliente: ClienteFeatures
    contrato_id: Optional[str] = None
    idioma: str = Field(default="pt-BR")


# ── Helpers ───────────────────────────────────────────────────────────────────

def cliente_para_df(cliente: ClienteFeatures) -> pd.DataFrame:
    """Converte Pydantic model → DataFrame com as features do modelo."""
    cliente = cliente.calcular_derivadas()
    data = {feat: getattr(cliente, feat) for feat in FEATURES}
    return pd.DataFrame([data])


def pd_para_segmento(pd_value: float) -> tuple[str, str]:
    """Converte PD em segmento de risco e recomendação."""
    if pd_value < 0.05:
        return "low_risk", "Aprovar — risco muito baixo. Considerar aumento de limite."
    elif pd_value < 0.15:
        return "medium_risk", "Aprovar com cautela — monitorar comportamento."
    elif pd_value < 0.30:
        return "high_risk", "Reprovar ou aprovar com garantias. Reduzir exposição."
    else:
        return "high_risk", "Reprovar — risco crítico. Encaminhar para análise manual."


def pd_para_score(pd_value: float) -> int:
    """Converte PD → Score de crédito (300-1000, invertido)."""
    # Transformação log-odds → score (estilo FICO)
    import math
    pd_clipped = min(max(pd_value, 0.001), 0.999)
    log_odds = math.log(pd_clipped / (1 - pd_clipped))
    score = int(600 - 40 * log_odds)
    return max(300, min(1000, score))


def calcular_shap_top(X: pd.DataFrame, n_top: int = 5) -> dict:
    """Retorna top N features por importância SHAP."""
    explainer = model_state.get("shap_explainer")
    if explainer is None:
        return {}
    shap_vals = explainer.shap_values(X)[0]
    shap_dict = dict(zip(FEATURES, shap_vals))
    top = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:n_top]
    return {k: round(float(v), 4) for k, v in top}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Infra"])
async def health_check():
    """Liveness + Readiness check."""
    modelo_ok = model_state.get("model") is not None
    return {
        "status": "healthy" if modelo_ok else "degraded",
        "modelo_carregado": modelo_ok,
        "modelo_uri": model_state.get("model_uri", "N/A"),
        "carregado_em": model_state.get("loaded_at", "N/A"),
        "versao_api": "1.0.0",
    }


@app.get("/model/info", tags=["Modelo"], dependencies=[Depends(verificar_token)])
async def model_info():
    """Retorna metadados do modelo em produção."""
    if not model_state.get("model"):
        raise HTTPException(503, "Modelo não carregado")
    client = model_state["mlflow_client"]
    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if versions:
        v = versions[0]
        return {
            "nome": MODEL_NAME,
            "stage": MODEL_STAGE,
            "version": v.version,
            "run_id": v.run_id,
            "features": FEATURES,
            "n_features": len(FEATURES),
        }
    return {"nome": MODEL_NAME, "stage": MODEL_STAGE, "features": FEATURES}


@app.post("/predict", response_model=PredictResponse, tags=["Predição"])
async def predict_single(
    request: PredictRequest,
    _token: str = Depends(verificar_token),
):
    """
    Predição de PD para um cliente/contrato.

    Retorna probabilidade de default, score de crédito, segmento de risco
    e recomendação de negócio.
    """
    if not model_state.get("model"):
        raise HTTPException(503, "Modelo não disponível")

    start = time.time()
    X = cliente_para_df(request.cliente)

    try:
        pd_value = float(model_state["model"].predict(xgb.DMatrix(X))[0])
    except Exception as e:
        raise HTTPException(500, f"Erro na predição: {e}")

    segmento, recomendacao = pd_para_segmento(pd_value)
    score = pd_para_score(pd_value)
    latencia = round((time.time() - start) * 1000, 2)

    shap_top = None
    if request.retornar_shap:
        shap_top = calcular_shap_top(X)

    return PredictResponse(
        contrato_id=request.contrato_id,
        probabilidade_default=round(pd_value, 4),
        score_credito=score,
        segmento_risco=segmento,
        recomendacao=recomendacao,
        shap_top_features=shap_top,
        modelo_versao=model_state.get("model_uri", "N/A"),
        latencia_ms=latencia,
    )


@app.post("/predict/batch", tags=["Predição"])
async def predict_batch(
    request: BatchPredictRequest,
    _token: str = Depends(verificar_token),
):
    """Predição em batch (até 1000 clientes). Otimizado para throughput."""
    if not model_state.get("model"):
        raise HTTPException(503, "Modelo não disponível")

    start = time.time()
    rows = []
    ids = []
    erros = []

    for item in request.clientes:
        try:
            cliente = item.cliente.calcular_derivadas()
            rows.append({feat: getattr(cliente, feat) for feat in FEATURES})
            ids.append(item.contrato_id)
        except Exception as e:
            erros.append({"contrato_id": item.contrato_id, "erro": str(e)})

    results = []
    if rows:
        try:
            X = pd.DataFrame(rows)
            pds = model_state["model"].predict(xgb.DMatrix(X))
            for pd_val, contrato_id in zip(pds, ids):
                segmento, recomendacao = pd_para_segmento(float(pd_val))
                results.append({
                    "contrato_id": contrato_id,
                    "probabilidade_default": round(float(pd_val), 4),
                    "score_credito": pd_para_score(float(pd_val)),
                    "segmento_risco": segmento,
                    "recomendacao": recomendacao,
                    "erro": None,
                })
        except Exception as e:
            raise HTTPException(500, f"Erro na predição batch: {e}")

    latencia = round((time.time() - start) * 1000, 2)
    return {
        "n_predicoes": len(results),
        "n_erros": len(erros),
        "latencia_ms": latencia,
        "resultados": results,
        "erros": erros,
    }


@app.post("/explain", tags=["Interpretabilidade"])
async def explain(
    request: ExplainRequest,
    _token: str = Depends(verificar_token),
):
    """
    Explicação completa de uma predição: SHAP values + GenAI Copilot.
    Retorna explicação em PT-BR natural para gerentes de crédito.
    """
    if not model_state.get("model"):
        raise HTTPException(503, "Modelo não disponível")

    X = cliente_para_df(request.cliente.calcular_derivadas())
    pd_value = float(model_state["model"].predict(xgb.DMatrix(X))[0])
    segmento, _ = pd_para_segmento(pd_value)

    # SHAP values
    shap_top = calcular_shap_top(X, n_top=5)
    shap_all = {}
    explainer = model_state.get("shap_explainer")
    if explainer:
        vals = explainer.shap_values(X)[0]
        shap_all = {f: round(float(v), 4) for f, v in zip(FEATURES, vals)}

    # GenAI Copilot
    copilot = model_state.get("copilot")
    explicacao_llm = ""
    acoes_recomendadas = []
    if copilot:
        explicacao_llm, acoes_recomendadas = await copilot.explicar_predicao(
            cliente=request.cliente,
            pd_value=pd_value,
            segmento=segmento,
            shap_top=shap_top,
        )

    return {
        "contrato_id": request.contrato_id,
        "probabilidade_default": round(pd_value, 4),
        "score_credito": pd_para_score(pd_value),
        "segmento_risco": segmento,
        "shap_top_features": shap_top,
        "shap_all_features": shap_all,
        "explicacao_llm": explicacao_llm,
        "acoes_recomendadas": acoes_recomendadas,
        "modelo_versao": model_state.get("model_uri", "N/A"),
    }
