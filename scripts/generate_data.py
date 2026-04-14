"""
Geração de Dados Sintéticos — Portfólio Risco de Crédito
========================================================
Gera ~10.000 registros realistas de contratos de crédito brasileiro
com ~8% taxa de inadimplência (imbalance realista para PD models).

Features baseadas em variáveis típicas de bureaus BR (Serasa, SPC, Quod):
  - Demográficas: idade, uf, estado_civil, escolaridade
  - Financeiras: renda_mensal, score_interno, score_serasa
  - Contrato: produto, limite_credito, saldo_devedor, num_parcelas
  - Comportamentais: historico_atrasos, dias_atraso_max, utilizacao_limite
  - Derivadas: DTI (Debt-to-Income), LTV (Loan-to-Value), idade_contrato

Uso:
    python scripts/generate_data.py
    python scripts/generate_data.py --n 50000 --seed 99 --output data/raw/clientes_50k.parquet
"""

import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constantes Brasil ─────────────────────────────────────────────────────────
UFS = [
    "SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO",
    "PE", "CE", "DF", "ES", "PA", "MT", "MS", "AM",
]
UF_PESOS = [
    0.22, 0.14, 0.11, 0.07, 0.06, 0.05, 0.05, 0.04,
    0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
]

PRODUTOS = ["pessoal", "consignado", "cartao_credito", "financiamento_auto", "cheque_especial"]
PRODUTO_PESOS = [0.30, 0.25, 0.20, 0.15, 0.10]

ESTADOS_CIVIS = ["solteiro", "casado", "divorciado", "viuvo", "uniao_estavel"]
ESCOLARIDADES = ["fundamental", "medio", "superior", "pos_graduacao"]

# ─────────────────────────────────────────────────────────────────────────────

def gerar_dataset(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Gera DataFrame com features realistas de crédito brasileiro."""
    rng = np.random.default_rng(seed)

    # ── IDs e Dados Demográficos ──────────────────────────────────────────────
    cliente_ids = [str(uuid.uuid4())[:8].upper() for _ in range(n)]
    contrato_ids = [f"CTR{str(i).zfill(7)}" for i in range(1, n + 1)]

    # Datas de criação: últimos 5 anos
    data_referencia = pd.Timestamp("2024-12-31")
    dias_offset = rng.integers(0, 5 * 365, n)
    data_contrato = pd.to_datetime(
        (data_referencia - pd.to_timedelta(dias_offset, unit="d")).values
    )
    # Safra mensal (AAAA-MM) — útil para temporal CV
    safra = data_contrato.to_period("M").astype(str)

    idade = rng.integers(18, 75, n)
    uf = rng.choice(UFS, n, p=UF_PESOS)
    estado_civil = rng.choice(ESTADOS_CIVIS, n, p=[0.35, 0.40, 0.12, 0.05, 0.08])
    escolaridade = rng.choice(ESCOLARIDADES, n, p=[0.15, 0.45, 0.30, 0.10])
    produto = rng.choice(PRODUTOS, n, p=PRODUTO_PESOS)

    # ── Finanças: Renda e Scores ───────────────────────────────────────────────
    # Renda log-normal: mediana ~R$3.500, P90 ~R$15k (realista Brasil formal)
    renda_mensal = np.exp(rng.normal(8.17, 0.85, n)).clip(1_320, 80_000)
    renda_mensal = np.round(renda_mensal, -1)  # arredonda dezena

    # Score interno (300-1000): correlacionado negativamente com inadimplência
    score_base = 600 - (renda_mensal < 3000) * 60 + rng.normal(0, 80, n)
    score_interno = score_base.clip(300, 1000).astype(int)

    # Score Serasa (0-1000): ligeiramente diferente do interno
    score_serasa = (score_interno * 0.85 + rng.normal(0, 50, n)).clip(0, 1000).astype(int)

    # ── Contrato ─────────────────────────────────────────────────────────────
    # Limite de crédito: múltiplo da renda (consignado tem limite maior)
    fator_limite = np.where(
        produto == "consignado", rng.uniform(8, 24, n),
        np.where(produto == "financiamento_auto", rng.uniform(12, 36, n),
                 rng.uniform(1, 10, n))
    )
    limite_credito = (renda_mensal * fator_limite / 12).clip(500, 200_000)
    limite_credito = np.round(limite_credito, 2)

    # Saldo devedor: Beta(2,4) → tendência de utilização moderada (~30-40%)
    utilizacao_raw = rng.beta(2, 4, n)
    saldo_devedor = (limite_credito * utilizacao_raw).round(2)

    num_parcelas = rng.integers(1, 60, n)
    valor_parcela = np.where(
        num_parcelas > 0,
        (saldo_devedor / num_parcelas).round(2),
        saldo_devedor.round(2)
    )

    # Idade do contrato em meses
    idade_contrato = np.clip(
        ((data_referencia - data_contrato) / np.timedelta64(30, "D")).to_numpy().astype(int),
        1, 60
    )

    # ── Histórico Comportamental ───────────────────────────────────────────────
    # Número de atrasos históricos (Poisson, maioria zero)
    historico_atrasos_30d = rng.poisson(0.4, n)
    historico_atrasos_60d = rng.poisson(0.15, n)
    historico_atrasos_90d = rng.poisson(0.05, n)
    dias_atraso_max = (historico_atrasos_30d * 15 + historico_atrasos_60d * 30 +
                       historico_atrasos_90d * 45 + rng.integers(0, 10, n))

    # CPF negativado (15% da base — realista BR)
    tem_cpf_negativado = rng.binomial(1, 0.15, n)
    num_consultas_bureau_90d = rng.poisson(1.5, n).clip(0, 20)

    # Dependentes financeiros
    num_dependentes = rng.choice([0, 1, 2, 3, 4], n, p=[0.30, 0.25, 0.25, 0.12, 0.08])

    # ── Features Derivadas (Risk) ─────────────────────────────────────────────
    # DTI: Debt-to-Income (saldo anualizado / renda anual)
    dti = (saldo_devedor / (renda_mensal * 12)).round(4)

    # LTV: Loan-to-Value (aproximado para crédito pessoal)
    ltv = utilizacao_raw.round(4)

    # Burden ratio: parcela / renda mensal
    burden_ratio = (valor_parcela / renda_mensal).round(4)

    # ── Target: Inadimplente (PD) ─────────────────────────────────────────────
    # Modelo logístico realista: ~8% default rate
    log_odds = (
        -3.2                                               # intercept
        - 0.005 * (score_interno - 600)                    # score reduz risco
        + 3.0  * dti.clip(0, 1)                            # DTI principal driver
        + 0.5  * historico_atrasos_30d.clip(0, 5)          # histórico atrasos
        + 0.8  * historico_atrasos_60d.clip(0, 3)
        + 1.5  * tem_cpf_negativado                        # negativado = alto risco
        + 1.5  * ltv.clip(0, 1)                            # alta utilização
        + 0.3  * burden_ratio.clip(0, 2)                   # comprometimento renda
        - 0.008 * (tempo_rel := rng.integers(1, 240, n))   # tempo rel reduz risco
        + 0.3  * (idade < 25).astype(int)                  # jovens: maior risco
        + 0.15 * num_dependentes                           # dependentes: pressão financeira
        + 0.08 * num_consultas_bureau_90d                  # consultas = busca crédito
        + rng.normal(0, 0.4, n)                            # ruído
    )
    prob_default = (1 / (1 + np.exp(-log_odds)))
    inadimplente = (rng.uniform(0, 1, n) < prob_default).astype(int)

    print(f"[INFO] Taxa de inadimplência gerada: {inadimplente.mean():.2%}")

    # ── Assemblar DataFrame ───────────────────────────────────────────────────
    df = pd.DataFrame({
        # IDs
        "cliente_id":              cliente_ids,
        "contrato_id":             contrato_ids,
        "data_contrato":           data_contrato,
        "safra":                   safra,
        # Demográficos
        "idade":                   idade,
        "uf":                      uf,
        "estado_civil":            estado_civil,
        "escolaridade":            escolaridade,
        # Financeiros
        "renda_mensal":            renda_mensal.round(2),
        "score_interno":           score_interno,
        "score_serasa":            score_serasa,
        # Contrato
        "produto":                 produto,
        "limite_credito":          limite_credito,
        "saldo_devedor":           saldo_devedor,
        "num_parcelas":            num_parcelas,
        "valor_parcela":           valor_parcela,
        "idade_contrato":          idade_contrato,
        # Comportamentais
        "historico_atrasos_30d":   historico_atrasos_30d,
        "historico_atrasos_60d":   historico_atrasos_60d,
        "historico_atrasos_90d":   historico_atrasos_90d,
        "dias_atraso_max":         dias_atraso_max,
        "tem_cpf_negativado":      tem_cpf_negativado,
        "num_consultas_bureau_90d": num_consultas_bureau_90d,
        "num_dependentes":         num_dependentes,
        "tempo_relacionamento":    tempo_rel,
        # Features derivadas
        "dti":                     dti,
        "ltv":                     ltv,
        "burden_ratio":            burden_ratio,
        "utilizacao_limite":       utilizacao_raw.round(4),
        # Target
        "inadimplente":            inadimplente,
        "prob_default_real":       prob_default.round(4),  # ground truth (remover em prod)
    })

    return df


def main():
    parser = argparse.ArgumentParser(description="Gera dataset sintético de crédito BR")
    parser.add_argument("--n", type=int, default=10_000, help="Número de registros")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", type=str,
        default="data/raw/clientes_contratos.parquet",
        help="Caminho de saída (.csv ou .parquet)"
    )
    args = parser.parse_args()

    print(f"[INFO] Gerando {args.n:,} registros com seed={args.seed}...")
    df = gerar_dataset(n=args.n, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False, engine="pyarrow")
    else:
        df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"[OK] Dataset salvo em: {output_path}")
    print(f"     Shape: {df.shape}")
    print(f"     Inadimplência: {df['inadimplente'].mean():.2%}")
    print(f"     Distribuição por produto:\n{df['produto'].value_counts().to_string()}")
    print(f"\nPrimeiras linhas:")
    print(df[["contrato_id", "renda_mensal", "score_interno", "dti", "inadimplente"]].head())

    # Também salva CSV para facilitar importação no Power BI
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[OK] CSV salvo em: {csv_path}")


if __name__ == "__main__":
    main()
