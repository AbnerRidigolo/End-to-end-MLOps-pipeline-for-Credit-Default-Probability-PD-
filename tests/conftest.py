"""
Fixtures compartilhadas para todos os testes.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# app/main.py lê API_TOKEN no import e levanta RuntimeError se ausente
os.environ.setdefault("API_TOKEN", "credito-api-token-2024")

# Garante que o projeto está no path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def sample_dataset():
    """Dataset sintético pequeno para testes (sem dependência de arquivo)."""
    np.random.seed(42)
    n = 200

    renda = np.random.lognormal(8.17, 0.5, n).clip(1320, 50000)
    limite = renda * np.random.uniform(1, 5, n)
    saldo = limite * np.random.beta(2, 4, n)
    score = np.random.randint(300, 1000, n)

    df = pd.DataFrame({
        "contrato_id": [f"CTR{i:07d}" for i in range(n)],
        "cliente_id": [f"CLI{i:07d}" for i in range(n)],
        "safra": np.random.choice(
            [f"2024-{m:02d}" for m in range(1, 13)], n
        ),
        "data_contrato": pd.date_range("2023-01-01", periods=n, freq="D")[:n],
        "idade": np.random.randint(18, 75, n),
        "uf": np.random.choice(["SP", "RJ", "MG", "RS"], n),
        "estado_civil": np.random.choice(["solteiro", "casado"], n),
        "escolaridade": np.random.choice(["medio", "superior"], n),
        "renda_mensal": renda.round(2),
        "score_interno": score,
        "score_serasa": np.clip(score + np.random.randint(-50, 50, n), 0, 1000),
        "produto": np.random.choice(["pessoal", "consignado", "cartao_credito"], n),
        "limite_credito": limite.round(2),
        "saldo_devedor": saldo.round(2),
        "num_parcelas": np.random.randint(1, 48, n),
        "valor_parcela": (saldo / np.random.randint(1, 48, n)).round(2),
        "idade_contrato": np.random.randint(1, 36, n),
        "historico_atrasos_30d": np.random.poisson(0.3, n),
        "historico_atrasos_60d": np.random.poisson(0.1, n),
        "historico_atrasos_90d": np.random.poisson(0.03, n),
        "dias_atraso_max": np.random.randint(0, 30, n),
        "tem_cpf_negativado": np.random.binomial(1, 0.1, n),
        "num_consultas_bureau_90d": np.random.poisson(1, n),
        "num_dependentes": np.random.choice([0, 1, 2], n, p=[0.4, 0.35, 0.25]),
        "tempo_relacionamento": np.random.randint(1, 120, n),
        "dti": (saldo / (renda * 12)).round(4),
        "ltv": (saldo / limite).round(4),
        "burden_ratio": (saldo / np.random.randint(1, 48, n) / renda).round(4),
        "utilizacao_limite": (saldo / limite).round(4),
        "inadimplente": np.random.binomial(1, 0.08, n),
        "prob_default_real": np.random.uniform(0, 0.3, n).round(4),
    })
    return df


@pytest.fixture
def sample_cliente_dict():
    """Dict com features de um cliente típico para testar a API."""
    return {
        "idade": 35,
        "renda_mensal": 5000.0,
        "score_interno": 650,
        "score_serasa": 620,
        "limite_credito": 15000.0,
        "saldo_devedor": 6000.0,
        "num_parcelas": 24,
        "valor_parcela": 250.0,
        "idade_contrato": 12,
        "historico_atrasos_30d": 0,
        "historico_atrasos_60d": 0,
        "historico_atrasos_90d": 0,
        "dias_atraso_max": 0,
        "tem_cpf_negativado": 0,
        "num_consultas_bureau_90d": 2,
        "num_dependentes": 1,
        "tempo_relacionamento": 36,
        "dti": 0.10,
        "ltv": 0.40,
        "burden_ratio": 0.05,
        "utilizacao_limite": 0.40,
    }
