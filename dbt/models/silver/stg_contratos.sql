-- ============================================================
-- silver/stg_contratos.sql
-- Staging: limpeza, cast, dedup dos dados Bronze
-- ============================================================
-- Origem: data/raw/clientes_contratos.parquet
-- Owner: data-eng-credito
-- Materialização: view (recalculada a cada query Gold)
-- ============================================================

WITH

-- Lê diretamente o Parquet Bronze via DuckDB
raw AS (
    SELECT *
    FROM read_parquet('{{ var("data_lake_path") }}/raw/clientes_contratos.parquet')
),

-- Remove duplicatas: mantém registro mais recente por contrato
deduplicado AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY contrato_id
            ORDER BY data_contrato DESC
        ) AS rn
    FROM raw
),

-- Limpeza e casting
limpo AS (
    SELECT
        -- IDs
        contrato_id::VARCHAR                            AS contrato_id,
        cliente_id::VARCHAR                             AS cliente_id,

        -- Datas
        data_contrato::DATE                             AS data_contrato,
        safra::VARCHAR                                  AS safra,

        -- Demográficos (validação de ranges)
        CASE
            WHEN CAST(idade AS INTEGER) BETWEEN 18 AND 100
            THEN CAST(idade AS INTEGER)
            ELSE NULL
        END                                             AS idade,
        UPPER(TRIM(uf::VARCHAR))                        AS uf,
        LOWER(TRIM(estado_civil::VARCHAR))              AS estado_civil,
        LOWER(TRIM(escolaridade::VARCHAR))              AS escolaridade,

        -- Financeiros (clip em valores absurdos)
        GREATEST(
            CAST(renda_mensal AS DECIMAL(12,2)),
            {{ var("min_renda_valida") }}
        )                                               AS renda_mensal,
        CAST(score_interno AS INTEGER)
            BETWEEN 300 AND 1000                        AS score_interno_valido,
        LEAST(GREATEST(
            CAST(score_interno AS INTEGER), 300
        ), 1000)                                        AS score_interno,
        LEAST(GREATEST(
            CAST(score_serasa AS INTEGER), 0
        ), 1000)                                        AS score_serasa,

        -- Contrato
        LOWER(TRIM(produto::VARCHAR))                   AS produto,
        CAST(limite_credito AS DECIMAL(12,2))           AS limite_credito,
        CAST(saldo_devedor AS DECIMAL(12,2))            AS saldo_devedor,
        CAST(num_parcelas AS INTEGER)                   AS num_parcelas,
        CAST(valor_parcela AS DECIMAL(10,2))            AS valor_parcela,
        CAST(idade_contrato AS INTEGER)                 AS idade_contrato,

        -- Comportamentais
        CAST(historico_atrasos_30d AS INTEGER)          AS historico_atrasos_30d,
        CAST(historico_atrasos_60d AS INTEGER)          AS historico_atrasos_60d,
        CAST(historico_atrasos_90d AS INTEGER)          AS historico_atrasos_90d,
        CAST(dias_atraso_max AS INTEGER)                AS dias_atraso_max,
        CAST(tem_cpf_negativado AS BOOLEAN)             AS tem_cpf_negativado,
        CAST(num_consultas_bureau_90d AS INTEGER)       AS num_consultas_bureau_90d,
        CAST(num_dependentes AS INTEGER)                AS num_dependentes,
        CAST(tempo_relacionamento AS INTEGER)           AS tempo_relacionamento,

        -- Features derivadas (recalculadas para garantir consistência)
        ROUND(
            CAST(saldo_devedor AS DECIMAL(12,2)) /
            NULLIF(CAST(renda_mensal AS DECIMAL(12,2)) * 12, 0),
        4)                                              AS dti,
        ROUND(
            CAST(saldo_devedor AS DECIMAL(12,2)) /
            NULLIF(CAST(limite_credito AS DECIMAL(12,2)), 0),
        4)                                              AS utilizacao_limite,
        ROUND(
            CAST(valor_parcela AS DECIMAL(10,2)) /
            NULLIF(CAST(renda_mensal AS DECIMAL(12,2)), 0),
        4)                                              AS burden_ratio,

        -- Target
        CAST(inadimplente AS INTEGER)                   AS inadimplente,

        -- Metadados
        CURRENT_TIMESTAMP                               AS _dbt_loaded_at

    FROM deduplicado
    WHERE rn = 1
      AND contrato_id IS NOT NULL
      AND cliente_id IS NOT NULL
      AND renda_mensal > 0
      AND limite_credito > 0
)

SELECT * FROM limpo
