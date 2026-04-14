FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mlflow==2.15.1 psycopg2-binary boto3

EXPOSE 5000

CMD sh -c "mlflow server \
     --backend-store-uri postgresql://${POSTGRES_USER:-airflow}:${POSTGRES_PASSWORD:-airflow}@postgres/mlflow \
     --default-artifact-root s3://mlflow-artifacts \
     --host 0.0.0.0 \
     --port 5000 \
     --serve-artifacts"
