-- ============================================================
-- gold/features_risco.sql
-- Feature Store para Modelos de Risco de Crédito (PD)
-- ============================================================
-- Esta tabela é o INPUT direto do treinamento ML e serving API.
-- Features normalizadas, sem data leakage, prontas para sklearn/XGBoost.
-- Colunas: apenas features + target + IDs (sem raw data)
-- ============================================================

WITH

fato AS (
    SELECT * FROM {{ ref('fato_contrato') }}
),

atrasos AS (
    SELECT
        contrato_id,
        bucket_atraso,
        em_default_regulatorio
    FROM {{ ref('atrasos') }}
),

-- ── Feature Engineering Avançado ────────────────────────────────────────────
features AS (
    SELECT
        f.contrato_id,
        f.cliente_id,
        f.safra,
        f.data_contrato,

        -- ── Features Demográficas ──────────────────────────────────────────
        f.idade,
        CASE
            WHEN f.idade < 25 THEN 1 ELSE 0
        END AS flag_jovem,                          -- jovens: maior risco
        CASE
            WHEN f.idade > 60 THEN 1 ELSE 0
        END AS flag_senior,

        -- ── Features de Renda ──────────────────────────────────────────────
        f.renda_mensal,
        LOG(GREATEST(f.renda_mensal, 1))            AS log_renda,  -- reduz skew
        CASE
            WHEN f.renda_mensal < 3000 THEN 1 ELSE 0
        END AS flag_baixa_renda,

        -- ── Scores ────────────────────────────────────────────────────────
        f.score_interno,
        f.score_serasa,
        (f.score_interno + f.score_serasa) / 2.0    AS score_medio_bureaus,
        ABS(f.score_interno - f.score_serasa)       AS divergencia_scores,

        -- ── Features de Contrato ──────────────────────────────────────────
        f.limite_credito,
        LOG(GREATEST(f.limite_credito, 1))          AS log_limite,
        f.saldo_devedor,
        LOG(GREATEST(f.saldo_devedor, 1))           AS log_saldo,
        f.num_parcelas,
        f.valor_parcela,
        f.idade_contrato,

        -- ── Features de Endividamento (core do modelo PD) ─────────────────
        f.dti,                                      -- Debt-to-Income
        f.utilizacao_limite,                        -- Credit Utilization
        f.burden_ratio,                             -- Parcela / Renda
        f.dti * f.utilizacao_limite                 AS dti_x_utilizacao,  -- interação

        -- ── Histórico de Atrasos (maior poder preditivo) ──────────────────
        f.historico_atrasos_30d,
        f.historico_atrasos_60d,
        f.historico_atrasos_90d,
        f.dias_atraso_max,
        -- Feature composta: score de comportamento
        (f.historico_atrasos_30d * 1 +
         f.historico_atrasos_60d * 2 +
         f.historico_atrasos_90d * 3)               AS score_atraso_ponderado,
        CASE
            WHEN f.historico_atrasos_30d > 0 OR
                 f.historico_atrasos_60d > 0 OR
                 f.historico_atrasos_90d > 0
            THEN 1 ELSE 0
        END AS flag_qualquer_atraso,

        -- ── Features de Bureau ────────────────────────────────────────────
        f.tem_cpf_negativado::INTEGER               AS tem_cpf_negativado,
        f.num_consultas_bureau_90d,
        CASE
            WHEN f.num_consultas_bureau_90d >= 5 THEN 1 ELSE 0
        END AS flag_muitas_consultas,               -- busca ativa de crédito = risco

        -- ── Features de Relacionamento ────────────────────────────────────
        f.tempo_relacionamento,
        f.num_dependentes,
        f.num_dependentes * f.valor_parcela         AS pressao_familiar,

        -- ── Features de Produto (one-hot encoding parcial) ────────────────
        CASE WHEN f.produto = 'consignado'         THEN 1 ELSE 0 END AS prod_consignado,
        CASE WHEN f.produto = 'pessoal'            THEN 1 ELSE 0 END AS prod_pessoal,
        CASE WHEN f.produto = 'cartao_credito'     THEN 1 ELSE 0 END AS prod_cartao,
        CASE WHEN f.produto = 'financiamento_auto' THEN 1 ELSE 0 END AS prod_auto,
        CASE WHEN f.produto = 'cheque_especial'    THEN 1 ELSE 0 END AS prod_cheque,

        -- ── Segmento de Risco (label) ─────────────────────────────────────
        f.segmento_risco,
        CASE f.segmento_risco
            WHEN 'low_risk'    THEN 0
            WHEN 'medium_risk' THEN 1
            WHEN 'high_risk'   THEN 2
        END AS segmento_risco_num,

        -- ── Target ────────────────────────────────────────────────────────
        f.inadimplente,                             -- binário: 0/1
        a.em_default_regulatorio,                   -- definição Basel III

        -- Metadados (não usar como feature!)
        f._dbt_loaded_at

    FROM fato f
    LEFT JOIN atrasos a ON f.contrato_id = a.contrato_id
),

-- ── Sanity checks e remoção de casos extremos ─────────────────────────────────
filtrado AS (
    SELECT *
    FROM features
    WHERE
        renda_mensal > 0
        AND limite_credito > 0
        AND dti BETWEEN 0 AND 5       -- DTI > 500% é erro ou fraude
        AND score_interno BETWEEN 300 AND 1000
)

SELECT * FROM filtrado
ORDER BY safra, contrato_id
