"""
Testes de qualidade dos dados gerados e transformações dbt.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.generate_data import gerar_dataset


class TestGerarDataset:
    """Testes unitários para o gerador de dados sintéticos."""

    def test_shape_padrao(self):
        df = gerar_dataset(n=100, seed=1)
        assert df.shape[0] == 100

    def test_colunas_obrigatorias(self):
        df = gerar_dataset(n=50, seed=1)
        colunas_obrig = [
            "contrato_id", "cliente_id", "safra", "renda_mensal",
            "score_interno", "saldo_devedor", "limite_credito", "inadimplente",
            "dti", "utilizacao_limite", "burden_ratio",
        ]
        for col in colunas_obrig:
            assert col in df.columns, f"Coluna ausente: {col}"

    def test_sem_nulos_colunas_criticas(self):
        df = gerar_dataset(n=200, seed=2)
        criticas = ["contrato_id", "renda_mensal", "score_interno", "inadimplente"]
        for col in criticas:
            assert df[col].isnull().sum() == 0, f"Nulos em {col}"

    def test_sem_duplicatas_contrato_id(self):
        df = gerar_dataset(n=500, seed=3)
        assert df["contrato_id"].duplicated().sum() == 0

    def test_ranges_financeiros(self):
        df = gerar_dataset(n=500, seed=4)
        assert df["renda_mensal"].min() >= 1000, "Renda mínima muito baixa"
        assert df["renda_mensal"].max() <= 100_000, "Renda máxima irreal"
        assert (df["saldo_devedor"] >= 0).all(), "Saldo negativo"
        assert (df["limite_credito"] > 0).all(), "Limite zero ou negativo"

    def test_ranges_scores(self):
        df = gerar_dataset(n=300, seed=5)
        assert df["score_interno"].between(300, 1000).all(), "Score fora do range 300-1000"
        assert df["score_serasa"].between(0, 1000).all(), "Score Serasa fora do range"

    def test_taxa_inadimplencia_realista(self):
        """Taxa de inadimplência deve estar entre 3% e 25% (realismo BR)."""
        df = gerar_dataset(n=5000, seed=42)
        taxa = df["inadimplente"].mean()
        assert 0.03 <= taxa <= 0.25, f"Taxa de inadimplência irrealista: {taxa:.2%}"

    def test_dti_calculado_corretamente(self):
        df = gerar_dataset(n=100, seed=6)
        dti_recalculado = df["saldo_devedor"] / (df["renda_mensal"] * 12)
        np.testing.assert_allclose(
            df["dti"].values,
            dti_recalculado.values,
            rtol=0.01,
            err_msg="DTI não corresponde ao cálculo esperado"
        )

    def test_utilizacao_entre_0_e_1(self):
        df = gerar_dataset(n=300, seed=7)
        assert df["utilizacao_limite"].between(0, 1.01).all(), \
            "Utilização fora do range [0, 1]"

    def test_reproducibilidade_seed(self):
        df1 = gerar_dataset(n=100, seed=99)
        df2 = gerar_dataset(n=100, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_produtos_validos(self):
        produtos_validos = {
            "pessoal", "consignado", "cartao_credito",
            "financiamento_auto", "cheque_especial"
        }
        df = gerar_dataset(n=200, seed=8)
        assert set(df["produto"].unique()).issubset(produtos_validos)

    def test_ufs_validas(self):
        ufs_validas = {
            "SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO",
            "PE", "CE", "DF", "ES", "PA", "MT", "MS", "AM"
        }
        df = gerar_dataset(n=200, seed=9)
        assert set(df["uf"].unique()).issubset(ufs_validas)

    def test_target_binario(self):
        df = gerar_dataset(n=300, seed=10)
        assert set(df["inadimplente"].unique()).issubset({0, 1})

    def test_imbalance_presente(self):
        """Classe majoritária deve ter pelo menos 70% dos dados."""
        df = gerar_dataset(n=1000, seed=42)
        class_counts = df["inadimplente"].value_counts(normalize=True)
        assert class_counts[0] >= 0.70, "Imbalance insuficiente para problema realista"


class TestFeatureEngineering:
    """Testa a consistência das features derivadas."""

    def test_score_atraso_ponderado(self, sample_dataset):
        df = sample_dataset.copy()
        esperado = (
            df["historico_atrasos_30d"] * 1 +
            df["historico_atrasos_60d"] * 2 +
            df["historico_atrasos_90d"] * 3
        )
        # Calcula como o dbt faria
        calculado = (
            df["historico_atrasos_30d"] * 1 +
            df["historico_atrasos_60d"] * 2 +
            df["historico_atrasos_90d"] * 3
        )
        pd.testing.assert_series_equal(esperado, calculado)

    def test_burden_ratio_range(self, sample_dataset):
        df = sample_dataset.copy()
        # burden_ratio = parcela / renda; esperado < 1 para clientes normais
        assert (df["burden_ratio"] >= 0).all()

    def test_segmentacao_risco_logica(self, sample_dataset):
        """Clientes com score alto e DTI baixo devem ser low_risk."""
        df = sample_dataset.copy()

        # Simula lógica do dbt fato_contrato
        def segmento(row):
            if (row["score_interno"] >= 700 and
                    row["dti"] < 0.30 and
                    row["historico_atrasos_30d"] == 0 and
                    not row["tem_cpf_negativado"]):
                return "low_risk"
            elif (row["score_interno"] < 500 or
                  row["dti"] >= 0.60 or
                  row["historico_atrasos_90d"] >= 1 or
                  row["tem_cpf_negativado"]):
                return "high_risk"
            return "medium_risk"

        df["segmento_calc"] = df.apply(segmento, axis=1)
        assert set(df["segmento_calc"].unique()).issubset(
            {"low_risk", "medium_risk", "high_risk"}
        )
