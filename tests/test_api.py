"""
Testes da FastAPI — sem dependência de MLflow (mock do modelo).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Mock do modelo antes de importar o app ────────────────────────────────────
# Evita conexão real ao MLflow durante testes
@pytest.fixture(scope="module")
def client():
    """TestClient com modelo mockado."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.08])

    mock_state = {
        "model": mock_model,
        "model_uri": "models:/credit_pd_model/Production",
        "loaded_at": "2024-01-01 00:00:00",
        "shap_explainer": None,
        "copilot": None,
    }

    with patch.dict("app.main.model_state", mock_state):
        from app.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


API_TOKEN = "credito-api-token-2024"
AUTH_HEADER = {"Authorization": f"Bearer {API_TOKEN}"}


# ── Testes de Health ──────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "modelo_carregado" in data

    def test_health_sem_auth(self, client):
        """Health deve ser público (sem auth)."""
        r = client.get("/health")
        assert r.status_code == 200


# ── Testes de Autenticação ────────────────────────────────────────────────────

class TestAutenticacao:
    def test_predict_sem_token_retorna_401(self, client):
        r = client.post("/predict", json={"cliente": {}})
        assert r.status_code == 401

    def test_predict_token_invalido_retorna_401(self, client):
        r = client.post(
            "/predict",
            json={"cliente": {}},
            headers={"Authorization": "Bearer token-errado"},
        )
        assert r.status_code == 401


# ── Testes de Predição ────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_cliente_valido(self, client, sample_cliente_dict):
        r = client.post(
            "/predict",
            json={"cliente": sample_cliente_dict},
            headers=AUTH_HEADER,
        )
        assert r.status_code == 200
        data = r.json()
        assert "probabilidade_default" in data
        assert 0 <= data["probabilidade_default"] <= 1
        assert "segmento_risco" in data
        assert data["segmento_risco"] in ["low_risk", "medium_risk", "high_risk"]
        assert "score_credito" in data
        assert 300 <= data["score_credito"] <= 1000
        assert "recomendacao" in data
        assert "latencia_ms" in data

    def test_predict_com_contrato_id(self, client, sample_cliente_dict):
        r = client.post(
            "/predict",
            json={"cliente": sample_cliente_dict, "contrato_id": "CTR0000001"},
            headers=AUTH_HEADER,
        )
        assert r.status_code == 200
        assert r.json()["contrato_id"] == "CTR0000001"

    def test_predict_idade_invalida(self, client, sample_cliente_dict):
        cliente = sample_cliente_dict.copy()
        cliente["idade"] = 15  # menor que 18
        r = client.post("/predict", json={"cliente": cliente}, headers=AUTH_HEADER)
        assert r.status_code == 422  # Validation Error

    def test_predict_renda_zero(self, client, sample_cliente_dict):
        cliente = sample_cliente_dict.copy()
        cliente["renda_mensal"] = 0
        r = client.post("/predict", json={"cliente": cliente}, headers=AUTH_HEADER)
        assert r.status_code == 422

    def test_predict_score_fora_range(self, client, sample_cliente_dict):
        cliente = sample_cliente_dict.copy()
        cliente["score_interno"] = 1500  # acima de 1000
        r = client.post("/predict", json={"cliente": cliente}, headers=AUTH_HEADER)
        assert r.status_code == 422

    def test_predict_sem_derivadas_calcula_automatico(self, client, sample_cliente_dict):
        """API deve calcular DTI/LTV se não fornecidos."""
        cliente = sample_cliente_dict.copy()
        del cliente["dti"]
        del cliente["ltv"]
        del cliente["burden_ratio"]
        del cliente["utilizacao_limite"]
        r = client.post("/predict", json={"cliente": cliente}, headers=AUTH_HEADER)
        assert r.status_code == 200


# ── Testes de Batch ───────────────────────────────────────────────────────────

class TestBatch:
    def test_batch_dois_clientes(self, client, sample_cliente_dict):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.05, 0.25])

        with patch.dict("app.main.model_state", {"model": mock_model,
                                                  "shap_explainer": None}):
            r = client.post(
                "/predict/batch",
                json={"clientes": [
                    {"cliente": sample_cliente_dict, "contrato_id": "CTR001"},
                    {"cliente": sample_cliente_dict, "contrato_id": "CTR002"},
                ]},
                headers=AUTH_HEADER,
            )
        assert r.status_code == 200
        data = r.json()
        assert data["n_predições"] == 2
        assert len(data["resultados"]) == 2

    def test_batch_limite_excedido(self, client, sample_cliente_dict):
        clientes = [{"cliente": sample_cliente_dict}] * 1001  # acima do limite
        r = client.post("/predict/batch", json={"clientes": clientes}, headers=AUTH_HEADER)
        assert r.status_code == 422


# ── Testes de Lógica de Negócio ───────────────────────────────────────────────

class TestLogicaNegocio:
    def test_pd_baixo_resulta_low_risk(self):
        from app.main import pd_para_segmento
        segmento, recomendacao = pd_para_segmento(0.03)
        assert segmento == "low_risk"
        assert "Aprovar" in recomendacao

    def test_pd_alto_resulta_high_risk(self):
        from app.main import pd_para_segmento
        segmento, recomendacao = pd_para_segmento(0.35)
        assert segmento == "high_risk"
        assert "Reprovar" in recomendacao

    def test_pd_para_score_invertido(self):
        from app.main import pd_para_score
        score_baixo_risco = pd_para_score(0.02)
        score_alto_risco = pd_para_score(0.40)
        assert score_baixo_risco > score_alto_risco, \
            "Score deve ser maior para clientes de menor risco"

    def test_score_range_valido(self):
        from app.main import pd_para_score
        for pd_val in [0.001, 0.05, 0.15, 0.30, 0.60, 0.99]:
            score = pd_para_score(pd_val)
            assert 300 <= score <= 1000, f"Score fora do range para PD={pd_val}: {score}"
