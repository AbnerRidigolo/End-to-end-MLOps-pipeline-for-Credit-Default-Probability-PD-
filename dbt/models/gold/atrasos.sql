-- ============================================================
-- gold/atrasos.sql
-- Análise de atrasos e roll-rates da carteira
-- ============================================================
-- Calcula: bucket de atraso, roll-rate por safra, curva de atraso,
--          PD por vintage, matriz de migração de risco
-- Materialização: TABLE — snapshot mensal da carteira
-- ============================================================

WITH

fato AS (
    SELECT * FROM {{ ref('fato_contrato') }}
),

-- ── Classificação em Buckets de Atraso ───────────────────────────────────────
buckets AS (
    SELECT
        *,
        CASE
            WHEN dias_atraso_max = 0                          THEN 'Current'
            WHEN dias_atraso_max BETWEEN 1  AND 30            THEN 'Bucket 1 (1-30d)'
            WHEN dias_atraso_max BETWEEN 31 AND 60            THEN 'Bucket 2 (31-60d)'
            WHEN dias_atraso_max BETWEEN 61 AND 90            THEN 'Bucket 3 (61-90d)'
            WHEN dias_atraso_max BETWEEN 91 AND 180           THEN 'Bucket 4 (91-180d)'
            ELSE                                                   'Bucket 5 (>180d / Write-off)'
        END AS bucket_atraso,

        -- Definição de "default" regulatória (Basel III: >90d)
        CASE
            WHEN dias_atraso_max > {{ var("cutoff_atraso_default") }}
            THEN 1 ELSE 0
        END AS em_default_regulatorio

    FROM fato
),

-- ── Resumo por Safra e Bucket ─────────────────────────────────────────────────
resumo_safra AS (
    SELECT
        safra,
        bucket_atraso,
        COUNT(*)                              AS n_contratos,
        SUM(saldo_devedor)                    AS saldo_total,
        SUM(inadimplente)                     AS n_inadimplentes,
        ROUND(AVG(inadimplente), 4)           AS taxa_inadimplencia,
        ROUND(AVG(dti), 4)                    AS dti_medio,
        ROUND(AVG(score_interno), 0)          AS score_medio,
        ROUND(AVG(saldo_devedor), 2)          AS ticket_medio
    FROM buckets
    GROUP BY safra, bucket_atraso
),

-- ── Roll-Rate: % de contratos que migram entre buckets ───────────────────────
-- (simplificado com lag — em prod usaria join com período anterior)
roll_rate_safra AS (
    SELECT
        safra,
        produto,
        uf,
        segmento_risco,
        COUNT(*)                              AS n_total,
        SUM(inadimplente)                     AS n_default,
        ROUND(AVG(inadimplente), 4)           AS pd_observada,
        ROUND(SUM(saldo_devedor), 2)          AS exposicao_total,
        ROUND(SUM(valor_em_risco), 2)         AS risco_total,
        ROUND(SUM(perda_esperada), 2)         AS perda_esperada_total,
        ROUND(AVG(score_interno), 0)          AS score_medio,
        ROUND(AVG(dti), 4)                    AS dti_medio,
        ROUND(AVG(burden_ratio), 4)           AS burden_ratio_medio,
        -- Concentração de high_risk
        ROUND(
            SUM(CASE WHEN segmento_risco = 'high_risk' THEN 1.0 ELSE 0 END) /
            COUNT(*),
        4) AS pct_high_risk
    FROM buckets
    GROUP BY safra, produto, uf, segmento_risco
),

-- ── Curva de Atraso por Vintage (aging) ───────────────────────────────────────
curva_vintage AS (
    SELECT
        safra                                 AS vintage,
        vintage_meses,
        COUNT(*)                              AS n_contratos,
        ROUND(AVG(inadimplente), 4)           AS pd_observada,
        ROUND(AVG(dias_atraso_max), 2)        AS dias_atraso_medio,
        ROUND(SUM(saldo_devedor), 2)          AS saldo_total,
        ROUND(SUM(perda_esperada), 2)         AS perda_esperada
    FROM buckets
    GROUP BY safra, vintage_meses
    ORDER BY safra, vintage_meses
)

-- ── Output Final ─────────────────────────────────────────────────────────────
SELECT
    b.contrato_id,
    b.cliente_id,
    b.safra,
    b.produto,
    b.uf,
    b.segmento_risco,

    -- Atraso
    b.dias_atraso_max,
    b.bucket_atraso,
    b.em_default_regulatorio,

    -- Histórico
    b.historico_atrasos_30d,
    b.historico_atrasos_60d,
    b.historico_atrasos_90d,

    -- Exposição
    b.saldo_devedor,
    b.valor_em_risco,
    b.perda_esperada,

    -- Safra métricas
    rs.n_contratos        AS n_contratos_safra,
    rs.taxa_inadimplencia AS taxa_inadimplencia_safra,
    rs.dti_medio          AS dti_medio_safra,

    -- Target
    b.inadimplente,
    b._dbt_loaded_at

FROM buckets b
LEFT JOIN resumo_safra rs
    ON b.safra = rs.safra
    AND b.bucket_atraso = rs.bucket_atraso
