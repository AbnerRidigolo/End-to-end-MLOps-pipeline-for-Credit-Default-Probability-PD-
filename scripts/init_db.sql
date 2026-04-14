-- Inicializa banco separado para MLflow (mesmo servidor Postgres)
CREATE DATABASE mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO airflow;
