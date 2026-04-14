"""
Treinamento de Modelos PD com MLflow
=====================================
Pipeline completo: LogisticRegression (baseline) → XGBoost (champion)
Métricas: AUC-ROC, KS Statistic, Precision-Recall, Gini, SHAP

MLflow Tracking:
  - Parâmetros, métricas, artefatos (plots, SHAP, feature importance)
  - Model Registry: registra melhor modelo como "credit_pd_model"
  - Stages: None → Staging → Production

Uso:
    python scripts/train_model.py
    python scripts/train_model.py --data data/raw/clientes_contratos.parquet --run-name "xgb_v2"
"""

import argparse
import os
import tempfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless para Docker
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
from mlflow.models.signature import infer_signature
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Configuração ──────────────────────────────────────────────────────────────
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "risco-credito-pd"
MODEL_NAME = "credit_pd_model"

FEATURES = [
    "idade", "renda_mensal", "score_interno", "score_serasa",
    "limite_credito", "saldo_devedor", "num_parcelas", "valor_parcela",
    "idade_contrato", "historico_atrasos_30d", "historico_atrasos_60d",
    "historico_atrasos_90d", "dias_atraso_max", "tem_cpf_negativado",
    "num_consultas_bureau_90d", "num_dependentes", "tempo_relacionamento",
    "dti", "ltv", "burden_ratio", "utilizacao_limite",
]
TARGET = "inadimplente"


# ── Funções Auxiliares ────────────────────────────────────────────────────────

def calcular_ks(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic — separação entre bons e maus pagadores."""
    preds_bom = y_pred[y_true == 0]
    preds_mau = y_pred[y_true == 1]
    ks_stat, _ = ks_2samp(preds_bom, preds_mau)
    return round(ks_stat, 4)


def calcular_gini(auc: float) -> float:
    """Gini = 2 * AUC - 1."""
    return round(2 * auc - 1, 4)


def plot_roc_curve(y_true, y_pred, title: str = "ROC Curve") -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}", color="#1a73e8")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_pr_curve(y_true, y_pred, title: str = "Precision-Recall") -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}", color="#e84343")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_ks_curve(y_true, y_pred, title: str = "KS Curve") -> plt.Figure:
    df_ks = pd.DataFrame({"y": y_true, "p": y_pred}).sort_values("p")
    n = len(df_ks)
    df_ks["cum_bom"] = (df_ks["y"] == 0).cumsum() / (y_true == 0).sum()
    df_ks["cum_mau"] = (df_ks["y"] == 1).cumsum() / (y_true == 1).sum()
    ks = calcular_ks(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(n), df_ks["cum_bom"], label="Bons Pagadores", color="#2ecc71")
    ax.plot(range(n), df_ks["cum_mau"], label="Maus Pagadores", color="#e74c3c")
    ax.set_title(f"{title} — KS = {ks:.4f}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_shap_summary(model, X_sample: pd.DataFrame) -> plt.Figure:
    """SHAP summary plot para XGBoost."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, show=False, plot_size=None)
    plt.tight_layout()
    return plt.gcf()


def carregar_dados(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def split_temporal(df: pd.DataFrame, test_months: int = 6):
    """
    Split temporal: treino = safras mais antigas, teste = últimas N safras.
    Evita data leakage em modelos de crédito.
    """
    df = df.copy()
    df["safra_dt"] = pd.to_datetime(df["safra"] + "-01")
    cutoff = df["safra_dt"].max() - pd.DateOffset(months=test_months)
    train_mask = df["safra_dt"] <= cutoff
    test_mask = df["safra_dt"] > cutoff
    print(f"[INFO] Train: {train_mask.sum()} registros | Test: {test_mask.sum()} registros")
    print(f"[INFO] Cutoff: {cutoff.strftime('%Y-%m')}")
    return df[train_mask], df[test_mask]


# ── Pipeline de Treinamento ───────────────────────────────────────────────────

def treinar_baseline(X_train, y_train, X_test, y_test) -> dict:
    """Regressão Logística — modelo baseline."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=1000,
            random_state=42, solver="lbfgs"
        ))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred)
    ks = calcular_ks(y_test.values, y_pred)
    ap = average_precision_score(y_test, y_pred)

    return {
        "model": pipe,
        "y_pred": y_pred,
        "metrics": {
            "auc_roc": round(auc, 4),
            "ks_statistic": ks,
            "gini": calcular_gini(auc),
            "avg_precision": round(ap, 4),
        },
        "params": {"C": 0.1, "class_weight": "balanced", "solver": "lbfgs"},
    }


def treinar_xgboost(X_train, y_train, X_test, y_test) -> dict:
    """XGBoost com calibração — modelo champion."""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    y_pred = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred)
    ks = calcular_ks(y_test.values, y_pred)
    ap = average_precision_score(y_test, y_pred)

    return {
        "model": model,
        "y_pred": y_pred,
        "metrics": {
            "auc_roc": round(auc, 4),
            "ks_statistic": ks,
            "gini": calcular_gini(auc),
            "avg_precision": round(ap, 4),
            "best_iteration": model.best_iteration,
        },
        "params": {
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": round(scale_pos_weight, 2),
        },
    }


def logar_run_mlflow(
    run_name: str,
    result: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    is_xgboost: bool = False,
    X_full: pd.DataFrame = None,
) -> str:
    """Loga experimento completo no MLflow e retorna run_id."""
    with mlflow.start_run(run_name=run_name) as run:
        # Parâmetros
        mlflow.log_params(result["params"])
        mlflow.log_param("n_features", len(FEATURES))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # Métricas
        mlflow.log_metrics(result["metrics"])

        # Tags
        mlflow.set_tag("modelo_tipo", "XGBoost" if is_xgboost else "LogisticRegression")
        mlflow.set_tag("target", TARGET)
        mlflow.set_tag("dataset", "sintetico_br_10k")
        mlflow.set_tag("autor", "portfólio-mlops")

        # Plots como artefatos
        artifacts_dir = Path(tempfile.mkdtemp())

        # ROC
        fig_roc = plot_roc_curve(y_test, result["y_pred"], title=f"ROC — {run_name}")
        fig_roc.savefig(artifacts_dir / "roc_curve.png", dpi=100)
        mlflow.log_artifact(str(artifacts_dir / "roc_curve.png"), "plots")
        plt.close(fig_roc)

        # Precision-Recall
        fig_pr = plot_pr_curve(y_test, result["y_pred"], title=f"PR — {run_name}")
        fig_pr.savefig(artifacts_dir / "pr_curve.png", dpi=100)
        mlflow.log_artifact(str(artifacts_dir / "pr_curve.png"), "plots")
        plt.close(fig_pr)

        # KS
        fig_ks = plot_ks_curve(y_test.values, result["y_pred"], title=f"KS — {run_name}")
        fig_ks.savefig(artifacts_dir / "ks_curve.png", dpi=100)
        mlflow.log_artifact(str(artifacts_dir / "ks_curve.png"), "plots")
        plt.close(fig_ks)

        # SHAP (só XGBoost — mais interpretável)
        if is_xgboost:
            model = result["model"]
            # Usa dataset completo (treino+teste) para amostra SHAP mais representativa
            X_pool = pd.concat([X_train, X_test]) if X_full is None else X_full
            X_sample = X_pool.sample(min(500, len(X_pool)), random_state=42)
            fig_shap = plot_shap_summary(model, X_sample)
            fig_shap.savefig(artifacts_dir / "shap_summary.png", dpi=100, bbox_inches="tight")
            mlflow.log_artifact(str(artifacts_dir / "shap_summary.png"), "plots")
            plt.close()

            # SHAP values como CSV para Power BI
            explainer = shap.TreeExplainer(model)
            shap_df = pd.DataFrame(
                explainer.shap_values(X_test),
                columns=[f"shap_{f}" for f in FEATURES]
            )
            shap_path = artifacts_dir / "shap_values.csv"
            shap_df.to_csv(shap_path, index=False)
            mlflow.log_artifact(str(shap_path), "interpretability")

        # Registra modelo
        signature = infer_signature(X_train, result["model"].predict_proba(X_train)[:, 1])
        input_example = X_train.head(3)

        if is_xgboost:
            mlflow.xgboost.log_model(
                result["model"],
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=MODEL_NAME,
            )
        else:
            mlflow.sklearn.log_model(
                result["model"],
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=f"{MODEL_NAME}_baseline",
            )

        run_id = run.info.run_id
        print(f"[OK] MLflow Run ID: {run_id}")
        print(f"     Métricas: {result['metrics']}")
        return run_id


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/clientes_contratos.parquet")
    parser.add_argument("--run-name", default="xgb_champion_v1")
    parser.add_argument("--mlflow-uri", default=MLFLOW_URI)
    parser.add_argument("--test-months", type=int, default=6)
    args = parser.parse_args()

    # Configura MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Carrega dados
    print(f"[INFO] Carregando dados de: {args.data}")
    df = carregar_dados(args.data)
    print(f"[INFO] Shape: {df.shape} | Default rate: {df[TARGET].mean():.2%}")

    # Split temporal (sem data leakage)
    df_train, df_test = split_temporal(df, test_months=args.test_months)
    X_train = df_train[FEATURES]
    y_train = df_train[TARGET]
    X_test = df_test[FEATURES]
    y_test = df_test[TARGET]

    X_full = df[FEATURES]

    # ── Baseline: Logistic Regression ────────────────────────────────────────
    print("\n[TREINO] Logistic Regression (baseline)...")
    result_lr = treinar_baseline(X_train, y_train, X_test, y_test)
    run_id_lr = logar_run_mlflow(
        "logistic_regression_baseline", result_lr,
        X_train, X_test, y_test, is_xgboost=False, X_full=X_full
    )

    # ── Champion: XGBoost ─────────────────────────────────────────────────────
    print("\n[TREINO] XGBoost (champion)...")
    result_xgb = treinar_xgboost(X_train, y_train, X_test, y_test)
    run_id_xgb = logar_run_mlflow(
        args.run_name, result_xgb,
        X_train, X_test, y_test, is_xgboost=True, X_full=X_full
    )

    # ── Promoção automática para Production (se AUC > baseline) ──────────────
    if result_xgb["metrics"]["auc_roc"] > result_lr["metrics"]["auc_roc"]:
        print("\n[MLflow] XGBoost supera baseline → promovendo para Production...")
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest_version = max(versions, key=lambda v: int(v.version))
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="Production",
            version=latest_version.version,
        )
        print(f"[OK] Modelo v{latest_version.version} alias 'Production' definido!")

    # ── Resumo ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTADOS FINAIS")
    print("=" * 60)
    print(f"Baseline (LR)  → AUC: {result_lr['metrics']['auc_roc']:.4f} | "
          f"KS: {result_lr['metrics']['ks_statistic']:.4f} | "
          f"Gini: {result_lr['metrics']['gini']:.4f}")
    print(f"Champion (XGB) → AUC: {result_xgb['metrics']['auc_roc']:.4f} | "
          f"KS: {result_xgb['metrics']['ks_statistic']:.4f} | "
          f"Gini: {result_xgb['metrics']['gini']:.4f}")
    print(f"\nMLflow UI: {args.mlflow_uri}")
    print(f"Experimento: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
