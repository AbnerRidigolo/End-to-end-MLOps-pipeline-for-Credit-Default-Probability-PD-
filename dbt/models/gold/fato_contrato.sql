-- ============================================================
-- gold/fato_contrato.sql
-- Fato Contrato: tabela analítica principal da carteira de crédito
-- ============================================================
-- Particionada por safra (AAAA-MM) para análises temporais
-- Enriquecida com: segmentos de risco, faixas de score, bandas de renda
-- Materialização: TABLE (Gold — persistida para BI e ML)
-- ============================================================

WITH

base AS (
    SELECT * FROM {{ ref('stg_contratos') }}
),

-- Enriquecimento: segmentação e faixas
enriquecido AS (
    SELECT
        *,

        -- ── Faixas de Score Interno ──────────────────────────────────────
        CASE
            WHEN score_interno >= 800 THEN 'A - Excelente'
            WHEN score_interno >= 700 THEN 'B - Bom'
            WHEN score_interno >= 600 THEN 'C - Regular'
            WHEN score_interno >= 500 THEN 'D - Ruim'
            ELSE                           'E - Muito Ruim'
        END AS faixa_score,

        -- ── Faixas de Renda ───────────────────────────────────────────────
        CASE
            WHEN renda_mensal >= 20000 THEN 'Alta (>20k)'
            WHEN renda_mensal >= 10000 THEN 'Média-Alta (10k-20k)'
            WHEN renda_mensal >=  5000 THEN 'Média (5k-10k)'
            WHEN renda_mensal >=  2500 THEN 'Média-Baixa (2.5k-5k)'
            ELSE                            'Baixa (<2.5k)'
        END AS faixa_renda,

        -- ── Faixas de DTI (endividamento) ────────────────────────────────
        CASE
            WHEN dti >= 0.60 THEN 'Crítico (≥60%)'
            WHEN dti >= 0.40 THEN 'Alto (40-60%)'
            WHEN dti >= 0.20 THEN 'Moderado (20-40%)'
            ELSE                   'Baixo (<20%)'
        END AS faixa_dti,

        -- ── Segmento de Risco Composto ────────────────────────────────────
        -- Lógica: combina score + DTI + histórico para scoring rápido
        CASE
            WHEN score_interno >= 700
             AND dti < 0.30
             AND historico_atrasos_30d = 0
             AND NOT tem_cpf_negativado
            THEN 'low_risk'
            WHEN score_interno < 500
              OR dti >= 0.60
              OR historico_atrasos_90d >= 1
              OR tem_cpf_negativado
            THEN 'high_risk'
            ELSE 'medium_risk'
        END AS segmento_risco,

        -- ── Faixa de Utilização ───────────────────────────────────────────
        CASE
            WHEN utilizacao_limite >= 0.90 THEN 'Crítica (≥90%)'
            WHEN utilizacao_limite >= 0.70 THEN 'Alta (70-90%)'
            WHEN utilizacao_limite >= 0.40 THEN 'Moderada (40-70%)'
            ELSE                                 'Baixa (<40%)'
        END AS faixa_utilizacao,

        -- ── Vintage (safra em meses desde início) ────────────────────────
        DATE_DIFF('month',
            strptime(safra || '-01', '%Y-%m-%d'),
            CURRENT_DATE
        ) AS vintage_meses,

        -- ── Valor em Risco (EAD simplificado) ────────────────────────────
        ROUND(saldo_devedor * 0.45, 2) AS valor_em_risco,  -- LGD ~45%

        -- ── Exposição Total ───────────────────────────────────────────────
        ROUND(
            saldo_devedor *
            CASE segmento_risco
                WHEN 'high_risk'   THEN 0.25   -- PD esperada alto risco
                WHEN 'medium_risk' THEN 0.10
                WHEN 'low_risk'    THEN 0.02
                ELSE 0.10
            END,
        2) AS perda_esperada

    FROM base
)

SELECT
    -- Chaves
    contrato_id,
    cliente_id,
    data_contrato,
    safra,

    -- Demográficos
    idade,
    uf,
    estado_civil,
    escolaridade,

    -- Financeiros
    renda_mensal,
    score_interno,
    score_serasa,
    faixa_score,
    faixa_renda,

    -- Contrato
    produto,
    limite_credito,
    saldo_devedor,
    valor_parcela,
    num_parcelas,
    idade_contrato,

    -- Comportamentais
    historico_atrasos_30d,
    historico_atrasos_60d,
    historico_atrasos_90d,
    dias_atraso_max,
    tem_cpf_negativado,
    num_consultas_bureau_90d,
    num_dependentes,
    tempo_relacionamento,

    -- Features derivadas
    dti,
    utilizacao_limite,
    burden_ratio,
    faixa_dti,
    faixa_utilizacao,

    -- Segmentação
    segmento_risco,
    vintage_meses,
    valor_em_risco,
    perda_esperada,

    -- Target
    inadimplente,

    -- Metadados
    _dbt_loaded_at

FROM enriquecido

{% if is_incremental() %}
-- Incremental: só processa safras novas
WHERE safra > (SELECT MAX(safra) FROM {{ this }})
{% endif %}
