"""
Dashboard Streamlit — Credit Risk ML Pipeline
=============================================
Páginas:
  1. Dashboard    — saúde do sistema e métricas do modelo
  2. Predição     — análise individual de cliente
  3. Explicação   — SHAP + GenAI Copilot em PT-BR
  4. Lote         — upload CSV e predição em batch
"""

import os
import io
import json
import requests
import pandas as pd
import streamlit as st

# ── Configuração ──────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://fastapi:8000")
API_TOKEN = os.getenv("API_TOKEN", "credito-api-token-2024")

HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
}

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e2a3a;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    border-left: 4px solid #0194E2;
    margin-bottom: 0.5rem;
}
.risk-low    { color: #22c55e; font-weight: bold; font-size: 1.2rem; }
.risk-medium { color: #f59e0b; font-weight: bold; font-size: 1.2rem; }
.risk-high   { color: #ef4444; font-weight: bold; font-size: 1.2rem; }
.explain-box {
    background: #0f1923;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    border-left: 3px solid #7C3AED;
    font-size: 0.95rem;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def api_get(path):
    try:
        r = requests.get(f"{API_BASE}{path}", headers=HEADERS, timeout=10)
        return r.json(), r.status_code
    except Exception as e:
        return {"error": str(e)}, 0


def api_post(path, payload):
    try:
        r = requests.post(f"{API_BASE}{path}", headers=HEADERS,
                          data=json.dumps(payload), timeout=30)
        return r.json(), r.status_code
    except Exception as e:
        return {"error": str(e)}, 0


def badge_risco(segmento):
    mapa = {
        "low_risk":    ('<span class="risk-low">● BAIXO</span>', "🟢"),
        "medium_risk": ('<span class="risk-medium">● MÉDIO</span>', "🟡"),
        "high_risk":   ('<span class="risk-high">● ALTO</span>', "🔴"),
    }
    return mapa.get(segmento, ('<span>—</span>', "⚪"))


def score_color(score):
    if score >= 700:
        return "#22c55e"
    elif score >= 500:
        return "#f59e0b"
    else:
        return "#ef4444"


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
st.sidebar.title("Credit Risk ML")
st.sidebar.caption("Pipeline MLOps · Risco de Crédito")
st.sidebar.divider()

pagina = st.sidebar.radio(
    "Navegação",
    ["🏠 Dashboard", "🔍 Predição Individual", "🤖 Explicação GenAI", "📦 Predição em Lote"],
    label_visibility="collapsed",
)

st.sidebar.divider()

# Status rápido na sidebar
health, code = api_get("/health")
if code == 200 and health.get("modelo_carregado"):
    st.sidebar.success("API online · Modelo carregado")
elif code == 200:
    st.sidebar.warning("API online · Modelo não carregado")
else:
    st.sidebar.error("API offline")

st.sidebar.caption(f"Endpoint: `{API_BASE}`")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "🏠 Dashboard":
    st.title("🏦 Credit Risk ML Pipeline")
    st.caption("Dashboard de monitoramento — Probabilidade de Default (PD)")
    st.divider()

    # Status do sistema
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status = "✅ Healthy" if health.get("modelo_carregado") else "⚠️ Degraded"
        st.metric("Status da API", status)

    with col2:
        st.metric("Versão da API", health.get("versao_api", "—"))

    with col3:
        model_uri = health.get("modelo_uri", "N/A")
        st.metric("Modelo ativo", model_uri.split("/")[-1] if model_uri != "N/A" else "—")

    with col4:
        carregado = health.get("carregado_em", "—")
        st.metric("Carregado em", carregado[:10] if carregado != "N/A" else "—")

    st.divider()

    # Info do modelo
    if health.get("modelo_carregado"):
        info, _ = api_get("/model/info")
        if "nome" in info:
            st.subheader("📋 Modelo em Produção")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Nome", info.get("nome", "—"))
            col_b.metric("Stage", info.get("stage", "—"))
            col_c.metric("Versão", info.get("version", "—"))

            with st.expander("Ver features do modelo"):
                features = info.get("features", [])
                cols = st.columns(3)
                for i, f in enumerate(features):
                    cols[i % 3].markdown(f"- `{f}`")
    else:
        st.warning("⚠️ Modelo não carregado. Verifique se o alias **Production** foi definido no MLflow e reinicie a FastAPI.")

    st.divider()

    # Métricas de referência
    st.subheader("📊 Métricas de Performance (Holdout Temporal)")
    df_metrics = pd.DataFrame({
        "Modelo": ["LR Baseline", "XGBoost Champion"],
        "AUC-ROC": [0.77, 0.88],
        "KS Stat": [0.42, 0.58],
        "Gini": [0.54, 0.76],
        "Avg Precision": [0.28, 0.45],
    })
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔗 Links do Stack")
    col1, col2, col3 = st.columns(3)
    col1.link_button("Airflow UI", "http://localhost:8080", use_container_width=True)
    col2.link_button("MLflow UI", "http://localhost:5000", use_container_width=True)
    col3.link_button("MinIO Console", "http://localhost:9001", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — PREDIÇÃO INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🔍 Predição Individual":
    st.title("🔍 Predição Individual de Risco")
    st.caption("Preencha os dados do cliente para obter a probabilidade de default (PD)")
    st.divider()

    with st.form("form_predict"):
        st.subheader("Dados Demográficos e Financeiros")
        col1, col2, col3 = st.columns(3)

        with col1:
            idade = st.number_input("Idade", 18, 100, 35)
            renda_mensal = st.number_input("Renda Mensal (R$)", 500.0, 500000.0, 5000.0, step=500.0)
            num_dependentes = st.number_input("Nº de Dependentes", 0, 15, 1)
            tempo_relacionamento = st.number_input("Tempo de Relacionamento (meses)", 0, 600, 24)

        with col2:
            score_interno = st.slider("Score Interno", 300, 1000, 650)
            score_serasa = st.slider("Score Serasa", 0, 1000, 620)
            limite_credito = st.number_input("Limite de Crédito (R$)", 500.0, 1000000.0, 15000.0, step=1000.0)
            saldo_devedor = st.number_input("Saldo Devedor (R$)", 0.0, 1000000.0, 6000.0, step=500.0)

        with col3:
            num_parcelas = st.number_input("Nº de Parcelas", 1, 360, 24)
            valor_parcela = st.number_input("Valor da Parcela (R$)", 0.0, 50000.0, 250.0, step=50.0)
            idade_contrato = st.number_input("Idade do Contrato (meses)", 0, 600, 12)

        st.subheader("Comportamento de Crédito")
        col4, col5, col6 = st.columns(3)

        with col4:
            historico_atrasos_30d = st.number_input("Atrasos 1-30 dias", 0, 50, 0)
            historico_atrasos_60d = st.number_input("Atrasos 31-60 dias", 0, 50, 0)

        with col5:
            historico_atrasos_90d = st.number_input("Atrasos 61-90 dias", 0, 50, 0)
            dias_atraso_max = st.number_input("Máx. dias de atraso", 0, 365, 0)

        with col6:
            tem_cpf_negativado = st.selectbox("CPF Negativado?", [0, 1], format_func=lambda x: "Sim" if x else "Não")
            num_consultas_bureau_90d = st.number_input("Consultas bureau (90d)", 0, 50, 2)

        retornar_shap = st.checkbox("Incluir SHAP values na resposta", value=True)
        submitted = st.form_submit_button("🔮 Calcular Risco", use_container_width=True, type="primary")

    if submitted:
        payload = {
            "cliente": {
                "idade": idade,
                "renda_mensal": renda_mensal,
                "score_interno": score_interno,
                "score_serasa": score_serasa,
                "limite_credito": limite_credito,
                "saldo_devedor": saldo_devedor,
                "num_parcelas": num_parcelas,
                "valor_parcela": valor_parcela,
                "idade_contrato": idade_contrato,
                "historico_atrasos_30d": historico_atrasos_30d,
                "historico_atrasos_60d": historico_atrasos_60d,
                "historico_atrasos_90d": historico_atrasos_90d,
                "dias_atraso_max": dias_atraso_max,
                "tem_cpf_negativado": tem_cpf_negativado,
                "num_consultas_bureau_90d": num_consultas_bureau_90d,
                "num_dependentes": num_dependentes,
                "tempo_relacionamento": tempo_relacionamento,
            },
            "retornar_shap": retornar_shap,
        }

        with st.spinner("Calculando..."):
            result, status_code = api_post("/predict", payload)

        if status_code == 200:
            st.divider()
            st.subheader("Resultado")

            pd_val = result["probabilidade_default"]
            score = result["score_credito"]
            segmento = result["segmento_risco"]
            badge_html, badge_emoji = badge_risco(segmento)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Prob. de Default", f"{pd_val * 100:.1f}%")
            col2.metric("Score de Crédito", score)
            col3.markdown(f"**Risco**<br>{badge_html}", unsafe_allow_html=True)
            col4.metric("Latência", f"{result['latencia_ms']} ms")

            st.info(f"**Recomendação:** {result['recomendacao']}")

            if result.get("shap_top_features"):
                st.subheader("Top Features (SHAP)")
                shap_df = pd.DataFrame([
                    {"Feature": k, "Impacto SHAP": v, "Direção": "↑ Aumenta risco" if v > 0 else "↓ Reduz risco"}
                    for k, v in sorted(result["shap_top_features"].items(), key=lambda x: abs(x[1]), reverse=True)
                ])
                st.dataframe(shap_df, use_container_width=True, hide_index=True)
        else:
            st.error(f"Erro {status_code}: {result.get('detail', result)}")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — EXPLICAÇÃO GENAI
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🤖 Explicação GenAI":
    st.title("🤖 Explicação GenAI Copilot")
    st.caption("SHAP values + LLM via HuggingFace — explicação em linguagem natural para gerentes de crédito")
    st.divider()

    with st.form("form_explain"):
        col1, col2, col3 = st.columns(3)

        with col1:
            idade = st.number_input("Idade", 18, 100, 28)
            renda_mensal = st.number_input("Renda Mensal (R$)", 500.0, 500000.0, 3000.0, step=500.0)
            num_dependentes = st.number_input("Nº de Dependentes", 0, 15, 2)
            tempo_relacionamento = st.number_input("Tempo de Relacionamento (meses)", 0, 600, 8)

        with col2:
            score_interno = st.slider("Score Interno", 300, 1000, 420)
            score_serasa = st.slider("Score Serasa", 0, 1000, 380)
            limite_credito = st.number_input("Limite de Crédito (R$)", 500.0, 1000000.0, 10000.0, step=1000.0)
            saldo_devedor = st.number_input("Saldo Devedor (R$)", 0.0, 1000000.0, 9500.0, step=500.0)

        with col3:
            num_parcelas = st.number_input("Nº de Parcelas", 1, 360, 48)
            valor_parcela = st.number_input("Valor da Parcela (R$)", 0.0, 50000.0, 280.0, step=50.0)
            idade_contrato = st.number_input("Idade do Contrato (meses)", 0, 600, 6)
            historico_atrasos_30d = st.number_input("Atrasos 1-30 dias", 0, 50, 3)
            tem_cpf_negativado = st.selectbox("CPF Negativado?", [0, 1], format_func=lambda x: "Sim" if x else "Não", index=1)

        submitted = st.form_submit_button("🤖 Gerar Explicação com GenAI", use_container_width=True, type="primary")

    if submitted:
        payload = {
            "cliente": {
                "idade": idade,
                "renda_mensal": renda_mensal,
                "score_interno": score_interno,
                "score_serasa": score_serasa,
                "limite_credito": limite_credito,
                "saldo_devedor": saldo_devedor,
                "num_parcelas": num_parcelas,
                "valor_parcela": valor_parcela,
                "idade_contrato": idade_contrato,
                "historico_atrasos_30d": historico_atrasos_30d,
                "tem_cpf_negativado": tem_cpf_negativado,
                "num_dependentes": num_dependentes,
                "tempo_relacionamento": tempo_relacionamento,
            }
        }

        with st.spinner("Consultando modelo + GenAI Copilot..."):
            result, status_code = api_post("/explain", payload)

        if status_code == 200:
            st.divider()

            pd_val = result["probabilidade_default"]
            score = result["score_credito"]
            segmento = result["segmento_risco"]
            badge_html, _ = badge_risco(segmento)

            col1, col2, col3 = st.columns(3)
            col1.metric("Prob. de Default", f"{pd_val * 100:.1f}%")
            col2.metric("Score de Crédito", score)
            col3.markdown(f"**Risco**<br>{badge_html}", unsafe_allow_html=True)

            st.divider()

            # Explicação GenAI
            explicacao = result.get("explicacao_llm", "")
            if explicacao:
                st.subheader("📝 Explicação do Copilot")
                st.markdown(f'<div class="explain-box">{explicacao}</div>', unsafe_allow_html=True)

            # Ações recomendadas
            acoes = result.get("acoes_recomendadas", [])
            if acoes:
                st.subheader("✅ Ações Recomendadas")
                for acao in acoes:
                    st.markdown(f"- {acao}")

            # SHAP
            shap_top = result.get("shap_top_features", {})
            if shap_top:
                st.subheader("📊 Fatores de Risco (SHAP)")
                shap_df = pd.DataFrame([
                    {
                        "Feature": k,
                        "Impacto": v,
                        "Direção": "↑ Aumenta risco" if v > 0 else "↓ Reduz risco",
                        "Magnitude": abs(v),
                    }
                    for k, v in sorted(shap_top.items(), key=lambda x: abs(x[1]), reverse=True)
                ])
                st.dataframe(shap_df[["Feature", "Impacto", "Direção"]], use_container_width=True, hide_index=True)
                st.bar_chart(shap_df.set_index("Feature")["Magnitude"])
        else:
            st.error(f"Erro {status_code}: {result.get('detail', result)}")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — PREDIÇÃO EM LOTE
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📦 Predição em Lote":
    st.title("📦 Predição em Lote")
    st.caption("Envie um arquivo CSV com múltiplos clientes e obtenha as predições de PD")
    st.divider()

    st.subheader("Formato esperado do CSV")
    exemplo = pd.DataFrame([{
        "idade": 35, "renda_mensal": 5000, "score_interno": 650,
        "score_serasa": 620, "limite_credito": 15000, "saldo_devedor": 6000,
        "num_parcelas": 24, "valor_parcela": 250, "idade_contrato": 12,
        "historico_atrasos_30d": 0, "tem_cpf_negativado": 0,
        "num_dependentes": 1, "tempo_relacionamento": 36,
    }])
    st.dataframe(exemplo, use_container_width=True, hide_index=True)

    # Download template
    csv_template = exemplo.to_csv(index=False)
    st.download_button(
        "⬇️ Baixar template CSV",
        data=csv_template,
        file_name="template_clientes.csv",
        mime="text/csv",
    )

    st.divider()
    uploaded = st.file_uploader("📂 Enviar arquivo CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"**{len(df)} clientes carregados**")
        st.dataframe(df.head(5), use_container_width=True, hide_index=True)

        campos_obrigatorios = [
            "idade", "renda_mensal", "score_interno", "score_serasa",
            "limite_credito", "saldo_devedor", "num_parcelas", "valor_parcela",
            "idade_contrato",
        ]
        faltando = [c for c in campos_obrigatorios if c not in df.columns]
        if faltando:
            st.error(f"Colunas obrigatórias ausentes: {faltando}")
        else:
            if st.button("🚀 Processar Lote", type="primary", use_container_width=True):
                clientes = []
                for _, row in df.iterrows():
                    cliente = {c: row[c] for c in campos_obrigatorios}
                    for campo_opt in ["historico_atrasos_30d", "historico_atrasos_60d",
                                      "historico_atrasos_90d", "dias_atraso_max",
                                      "tem_cpf_negativado", "num_consultas_bureau_90d",
                                      "num_dependentes", "tempo_relacionamento"]:
                        if campo_opt in df.columns:
                            cliente[campo_opt] = row[campo_opt]
                    clientes.append({"cliente": cliente})

                payload = {"clientes": clientes}

                with st.spinner(f"Processando {len(clientes)} clientes..."):
                    result, status_code = api_post("/predict/batch", payload)

                if status_code == 200:
                    resultados = result.get("resultados", [])
                    df_result = pd.DataFrame(resultados)

                    st.success(f"✅ {result['n_predicoes']} predições concluídas em {result['latencia_ms']} ms")

                    if result["n_erros"] > 0:
                        st.warning(f"⚠️ {result['n_erros']} erros")

                    # Resumo
                    col1, col2, col3 = st.columns(3)
                    if "segmento_risco" in df_result.columns:
                        contagem = df_result["segmento_risco"].value_counts()
                        col1.metric("🟢 Baixo Risco", contagem.get("low_risk", 0))
                        col2.metric("🟡 Médio Risco", contagem.get("medium_risk", 0))
                        col3.metric("🔴 Alto Risco", contagem.get("high_risk", 0))

                    st.dataframe(df_result, use_container_width=True, hide_index=True)

                    # Download resultado
                    csv_result = df_result.to_csv(index=False)
                    st.download_button(
                        "⬇️ Baixar resultados CSV",
                        data=csv_result,
                        file_name="predicoes_resultado.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.error(f"Erro {status_code}: {result.get('detail', result)}")
