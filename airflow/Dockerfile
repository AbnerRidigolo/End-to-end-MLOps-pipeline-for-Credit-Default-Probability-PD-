# Imagem customizada do Airflow com todas as dependências de ML/dbt
FROM apache/airflow:2.9.3-python3.11

# Instala dependências de sistema como root
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Volta para usuário airflow para instalar pacotes Python
USER airflow

# Instala todas as dependências de ML + dbt + MLflow
RUN pip install --no-cache-dir \
    "dbt-core==1.8.7" \
    "dbt-duckdb==1.8.3" \
    "duckdb==1.1.3" \
    "pandas==2.2.3" \
    "numpy==1.26.4" \
    "pyarrow==15.0.2" \
    "scikit-learn==1.5.2" \
    "xgboost==2.1.3" \
    "lightgbm==4.5.0" \
    "shap==0.46.0" \
    "mlflow==2.14.3" \
    "boto3==1.35.86" \
    "scipy==1.14.1" \
    "matplotlib==3.9.3"
