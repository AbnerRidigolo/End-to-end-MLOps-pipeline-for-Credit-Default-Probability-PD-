"""
DAG: Pipeline de Ingestão de Dados de Crédito
===============================================
Orquestra: Geração → Bronze (Raw) → dbt Silver → dbt Gold → Trigger ML

Schedule: Diário às 6h (horário de Brasília)
Owner: data-engineering-credito
SLA: 3 horas máximo

Fluxo:
    [gerar_dados] → [validar_schema] → [salvar_bronze] →
    [dbt_silver]  → [dbt_gold]       → [trigger_treinamento]
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule

# ── Configurações Default ─────────────────────────────────────────────────────
DEFAULT_ARGS = {
    "owner": "data-eng-credito",
    "depends_on_past": False,
    "email": ["data-alerts@credito.br"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(hours=2),
}

DATALAKE_PATH = "/opt/airflow/data"
DBT_PROJECT_DIR = "/opt/airflow/dbt"
SCRIPTS_DIR = "/opt/airflow/scripts"


# ── Funções Python ────────────────────────────────────────────────────────────

def gerar_ou_ingerir_dados(**context) -> str:
    """
    Em produção: leria da fonte real (API banco, S3, CDC).
    Em demo: gera dados sintéticos e salva em Bronze.
    Retorna caminho do arquivo gerado.
    """
    import subprocess
    import sys
    from pathlib import Path

    output_path = f"{DATALAKE_PATH}/raw/clientes_contratos.parquet"
    Path(f"{DATALAKE_PATH}/raw").mkdir(parents=True, exist_ok=True)

    # Gera dados sintéticos (ou copia de fonte externa)
    result = subprocess.run(
        [sys.executable, f"{SCRIPTS_DIR}/generate_data.py",
         "--n", "10000", "--seed", "42",
         "--output", output_path],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Falha ao gerar dados: {result.stderr}")

    print(result.stdout)

    # Pusha caminho para XCom → próximas tasks
    context["ti"].xcom_push(key="raw_path", value=output_path)
    return output_path


def validar_schema_bronze(**context) -> bool:
    """
    Valida schema e qualidade básica dos dados bronze.
    ShortCircuit: retorna False para abortar pipeline se dados inválidos.
    """
    import pandas as pd

    raw_path = context["ti"].xcom_pull(key="raw_path", task_ids="gerar_dados")

    df = pd.read_parquet(raw_path)

    # Validações obrigatórias
    colunas_obrigatorias = [
        "contrato_id", "cliente_id", "renda_mensal", "score_interno",
        "saldo_devedor", "limite_credito", "inadimplente", "safra"
    ]
    colunas_faltando = [c for c in colunas_obrigatorias if c not in df.columns]
    if colunas_faltando:
        raise ValueError(f"[ERRO] Colunas faltando: {colunas_faltando}")

    # Sem duplicatas de contrato
    n_duplicatas = df["contrato_id"].duplicated().sum()
    if n_duplicatas > 0:
        raise ValueError(f"[ERRO] {n_duplicatas} contratos duplicados!")

    # Valores nulos críticos
    nulos_criticos = df[colunas_obrigatorias].isnull().sum()
    if nulos_criticos.any():
        raise ValueError(f"[ERRO] Nulos em colunas críticas:\n{nulos_criticos[nulos_criticos > 0]}")

    # Sanidade: taxa de default entre 3% e 25%
    default_rate = df["inadimplente"].mean()
    if not (0.03 <= default_rate <= 0.25):
        print(f"[WARN] Taxa de default fora do range esperado: {default_rate:.2%}")

    print(f"[OK] Validação Bronze: {len(df):,} registros | Default: {default_rate:.2%}")
    context["ti"].xcom_push(key="n_registros", value=len(df))
    return True


def salvar_metadados_bronze(**context) -> None:
    """Salva metadados da ingestão (audit trail)."""
    import json
    from pathlib import Path

    n_registros = context["ti"].xcom_pull(key="n_registros", task_ids="validar_schema")
    execution_date = context["ds"]

    metadata = {
        "execution_date": execution_date,
        "n_registros": n_registros,
        "camada": "bronze",
        "status": "success",
        "dag_run_id": context["run_id"],
    }

    meta_path = Path(f"{DATALAKE_PATH}/raw/metadata_{execution_date}.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"[OK] Metadados salvos: {meta_path}")


def checar_novos_dados(**context) -> bool:
    """
    Verifica se há novos dados desde última execução.
    ShortCircuit: pula restante se não há dados novos.
    """
    from pathlib import Path

    raw_path = f"{DATALAKE_PATH}/raw/clientes_contratos.parquet"
    if not Path(raw_path).exists():
        print("[WARN] Arquivo bronze não encontrado — abortando pipeline.")
        return False

    print("[OK] Novos dados detectados → continuando pipeline.")
    return True


# ── DAG Definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="pipeline_ingestion_credito",
    description="Ingestão diária + dbt transformações para carteira de crédito",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 6 * * *",   # Todo dia às 06:00
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["credito", "bronze", "dbt", "producao"],
    doc_md="""
    ## Pipeline de Ingestão — Risco de Crédito

    **Fluxo:** Bronze → Silver → Gold (dbt)

    **SLA:** 3 horas | **Owner:** data-eng-credito

    ### Camadas:
    - **Bronze:** dados raw (Parquet) sem transformação
    - **Silver:** limpeza, casting, deduplicação (dbt)
    - **Gold:** features analíticas para ML e BI (dbt)
    """,
) as dag:

    # ── Task 1: Ingestão Bronze ───────────────────────────────────────────────
    gerar_dados = PythonOperator(
        task_id="gerar_dados",
        python_callable=gerar_ou_ingerir_dados,
        doc_md="Gera dados sintéticos ou ingestão de fonte real → Bronze/Raw",
    )

    # ── Task 2: Validação Schema ──────────────────────────────────────────────
    validar_schema = ShortCircuitOperator(
        task_id="validar_schema",
        python_callable=validar_schema_bronze,
        doc_md="Valida schema obrigatório + qualidade básica. Abort se falhar.",
    )

    # ── Task 3: Metadados ─────────────────────────────────────────────────────
    salvar_meta = PythonOperator(
        task_id="salvar_metadados",
        python_callable=salvar_metadados_bronze,
        doc_md="Salva audit trail da ingestão (n_registros, execution_date, etc.)",
    )

    # ── Task 4: dbt deps + Silver ─────────────────────────────────────────────
    dbt_silver = BashOperator(
        task_id="dbt_run_silver",
        bash_command=(
            f"cd {DBT_PROJECT_DIR} && "
            "dbt deps --profiles-dir . --no-version-check && "
            "dbt run --select silver --profiles-dir . --no-version-check && "
            "dbt test --select silver --profiles-dir . --no-version-check"
        ),
        doc_md="dbt: deps + modelos Silver (limpeza, cast, dedup) + testes de qualidade",
    )

    # ── Task 5: dbt Gold ──────────────────────────────────────────────────────
    dbt_gold = BashOperator(
        task_id="dbt_run_gold",
        bash_command=(
            f"cd {DBT_PROJECT_DIR} && "
            "dbt run --select gold --profiles-dir . --no-version-check && "
            "dbt test --select gold --profiles-dir . --no-version-check"
        ),
        doc_md="dbt: modelos Gold (features_risco, fato_contrato, atrasos) para ML + BI",
    )

    # ── Task 6: Gera docs dbt ────────────────────────────────────────────────
    dbt_docs = BashOperator(
        task_id="dbt_generate_docs",
        bash_command=(
            f"cd {DBT_PROJECT_DIR} && "
            "dbt docs generate --profiles-dir . --no-version-check"
        ),
        doc_md="Atualiza documentação dbt (lineage, schema)",
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # ── Task 7: Trigger Treinamento ───────────────────────────────────────────
    trigger_ml = BashOperator(
        task_id="trigger_treinamento_ml",
        bash_command=(
            f"python {SCRIPTS_DIR}/train_model.py "
            f"--data {DATALAKE_PATH}/raw/clientes_contratos.parquet "
            "--run-name airflow_scheduled_run"
        ),
        doc_md="Dispara retreinamento com novos dados. Falha interrompe o pipeline.",
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # ── Dependências ─────────────────────────────────────────────────────────
    gerar_dados >> validar_schema >> salvar_meta >> dbt_silver >> dbt_gold >> [dbt_docs, trigger_ml]
