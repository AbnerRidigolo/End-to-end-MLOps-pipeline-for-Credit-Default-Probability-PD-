"""
GenAI Copilot de Risco de Crédito
===================================
Usa LLM via HuggingFace Inference API (gratuito) para gerar explicações
em PT-BR natural das predições de PD para gerentes de crédito.

Funcionalidades:
  - Traduz SHAP values em linguagem de negócio
  - Sugere ações concretas (aprovar, renegociar, reduzir limite)
  - Fallback: template rule-based se LLM indisponível
  - Modelo padrão: mistralai/Mistral-7B-Instruct-v0.3 (gratuito no HF)

Setup:
  1. Crie conta em huggingface.co (gratuito)
  2. Gere token em hf.co/settings/tokens → New token → Read
  3. Adicione ao .env: HF_TOKEN=hf_SEU_TOKEN
"""

import asyncio
import json
import os
import re
from functools import partial
from typing import Optional

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")

# Mapeamento de features para linguagem de negócio
FEATURE_LABELS_PT = {
    "dti":                      "relação dívida/renda (DTI)",
    "score_interno":            "score interno de crédito",
    "score_serasa":             "score Serasa",
    "historico_atrasos_30d":    "histórico de atrasos curtos (1-30 dias)",
    "historico_atrasos_60d":    "histórico de atrasos médios (31-60 dias)",
    "historico_atrasos_90d":    "histórico de atrasos longos (61-90 dias)",
    "tem_cpf_negativado":       "restrição de CPF/negativação",
    "utilizacao_limite":        "percentual de utilização do limite",
    "burden_ratio":             "comprometimento da renda com parcelas",
    "renda_mensal":             "renda mensal",
    "saldo_devedor":            "saldo devedor atual",
    "num_consultas_bureau_90d": "consultas ao bureau nos últimos 90 dias",
    "score_atraso_ponderado":   "score de comportamento de atraso",
    "tempo_relacionamento":     "tempo de relacionamento com a instituição",
    "num_dependentes":          "número de dependentes financeiros",
    "idade":                    "faixa etária",
    "limite_credito":           "limite de crédito concedido",
    "log_renda":                "renda (escala logarítmica)",
    "flag_jovem":               "perfil jovem (< 25 anos)",
    "ltv":                      "loan-to-value (relação saldo/limite)",
    "valor_parcela":            "valor da parcela",
    "num_parcelas":             "número de parcelas",
    "idade_contrato":           "tempo do contrato em meses",
    "score_medio_bureaus":      "score médio dos bureaus",
}


def feature_para_pt(feature: str) -> str:
    return FEATURE_LABELS_PT.get(feature, feature.replace("_", " "))


def gerar_explicacao_template(
    pd_value: float,
    segmento: str,
    shap_top: dict[str, float],
    cliente_data: dict[str, object],
) -> tuple[str, list[str]]:
    """
    Fallback: gera explicação rule-based sem LLM.
    Usado quando HF_TOKEN não está configurado ou API falha.
    """
    pct = f"{pd_value * 100:.1f}%"
    nivel = {
        "low_risk": "BAIXO",
        "medium_risk": "MÉDIO",
        "high_risk": "ALTO",
    }.get(segmento, "MÉDIO")

    _sorted = sorted(shap_top.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = [_sorted[i] for i in range(min(3, len(_sorted)))]

    fatores = []
    for feat, val in top_features:
        label = feature_para_pt(feat)
        direcao = "aumenta" if val > 0 else "reduz"
        fatores.append(f"{label} ({direcao} o risco)")

    fatores_txt = ", ".join(fatores) if fatores else "múltiplos fatores"

    explicacao = (
        f"O cliente apresenta probabilidade de inadimplência de {pct}, "
        f"classificado como risco {nivel}. "
        f"Os principais fatores que determinam essa avaliação são: {fatores_txt}. "
    )

    if pd_value > 0.30:
        explicacao += (
            "O perfil de risco é elevado, com múltiplos indicadores negativos "
            "que sugerem dificuldade de pagamento no curto prazo."
        )
    elif pd_value > 0.10:
        explicacao += (
            "O perfil é moderado — recomenda-se monitoramento periódico "
            "e possível renegociação preventiva das condições."
        )
    else:
        explicacao += (
            "O cliente apresenta bom perfil de crédito, "
            "com histórico consistente e endividamento controlado."
        )

    acoes = _gerar_acoes(pd_value, segmento, shap_top, cliente_data)
    return explicacao, acoes


def _gerar_acoes(
    pd_value: float,
    segmento: str,
    shap_top: dict,
    cliente_data: dict,
) -> list[str]:
    """Gera ações recomendadas baseadas no perfil."""
    acoes = []

    dti = cliente_data.get("dti", 0)
    score = cliente_data.get("score_interno", 600)
    negativado = cliente_data.get("tem_cpf_negativado", 0)
    atrasos = cliente_data.get("historico_atrasos_30d", 0)

    if segmento == "low_risk":
        acoes.append("Aprovar crédito nas condições solicitadas")
        if score > 750:
            acoes.append("Considerar aumento de limite em até 30%")
        acoes.append("Oferecer produtos complementares (seguro, investimento)")
    elif segmento == "medium_risk":
        acoes.append("Aprovar com limite reduzido em 20-30%")
        acoes.append("Solicitar comprovante de renda atualizado")
        if dti > 0.40:
            acoes.append("Propor renegociação do prazo para reduzir parcela")
        acoes.append("Monitorar conta com alertas mensais")
    else:  # high_risk
        if negativado:
            acoes.append("Reprovar — cliente com restrição ativa no bureau")
            acoes.append("Iniciar contato para regularização do CPF")
        elif atrasos >= 2:
            acoes.append("Reprovar — histórico de atrasos recorrentes")
            acoes.append("Oferecer programa de renegociação de dívidas")
        else:
            acoes.append("Reprovar ou exigir garantias reais")
            acoes.append("Encaminhar para análise manual da área de risco")
        acoes.append("Não conceder novos produtos até regularização")

    return acoes


class CopilotRisco:
    """
    Copilot de Risco: usa HuggingFace Inference API (gratuito) para
    explicações em PT-BR. Fallback automático para template se indisponível.

    Modelos recomendados (gratuitos):
      - mistralai/Mistral-7B-Instruct-v0.3  (padrão, ótimo PT-BR)
      - HuggingFaceH4/zephyr-7b-beta        (sem restrição de licença)
      - microsoft/Phi-3-mini-4k-instruct    (leve e rápido)
    """

    def __init__(self):
        self.use_llm = bool(HF_TOKEN)
        self.client = None
        self.model = HF_MODEL

        if self.use_llm:
            try:
                from huggingface_hub import InferenceClient
                self.client = InferenceClient(token=HF_TOKEN)
                print(f"[Copilot] HuggingFace API configurada. Modelo: {self.model}")
            except ImportError:
                print("[Copilot] WARN: huggingface_hub não instalado. Usando fallback.")
                self.use_llm = False
        else:
            print("[Copilot] HF_TOKEN não configurado. Usando template.")

    async def explicar_predicao(
        self,
        cliente,
        pd_value: float,
        segmento: str,
        shap_top: dict,
    ) -> tuple[str, list[str]]:
        """
        Gera explicação da predição em PT-BR.
        Usa HuggingFace API se disponível, senão usa template.
        """
        cliente_data = {
            "dti": getattr(cliente, "dti", None),
            "score_interno": getattr(cliente, "score_interno", None),
            "tem_cpf_negativado": getattr(cliente, "tem_cpf_negativado", 0),
            "historico_atrasos_30d": getattr(cliente, "historico_atrasos_30d", 0),
            "renda_mensal": getattr(cliente, "renda_mensal", None),
            "saldo_devedor": getattr(cliente, "saldo_devedor", None),
            "utilizacao_limite": getattr(cliente, "utilizacao_limite", None),
            "burden_ratio": getattr(cliente, "burden_ratio", None),
        }

        if self.use_llm and self.client:
            return await self._explicar_via_hf(pd_value, segmento, shap_top, cliente_data)
        else:
            return gerar_explicacao_template(pd_value, segmento, shap_top, cliente_data)

    async def _explicar_via_hf(
        self,
        pd_value: float,
        segmento: str,
        shap_top: dict,
        cliente_data: dict,
    ) -> tuple[str, list[str]]:
        """Gera explicação via HuggingFace Inference API (gratuito).

        Usa run_in_executor para não bloquear o event loop do FastAPI.
        Retry automático quando o modelo está em cold start (loading).
        """
        if self.client is None:
            return gerar_explicacao_template(pd_value, segmento, shap_top, cliente_data)

        shap_texto = "\n".join([
            f"  - {feature_para_pt(feat)}: impacto {'+' if val > 0 else ''}{val:.3f} "
            f"({'aumenta' if val > 0 else 'reduz'} o risco)"
            for feat, val in sorted(shap_top.items(), key=lambda x: abs(x[1]), reverse=True)
        ])

        nivel = {"low_risk": "BAIXO", "medium_risk": "MÉDIO", "high_risk": "ALTO"}.get(segmento, "MÉDIO")

        prompt = f"""Você é um analista sênior de risco de crédito em uma instituição financeira brasileira.

Um modelo de machine learning calculou a seguinte avaliação de crédito:

DADOS DO CLIENTE:
- Renda mensal: R$ {cliente_data.get('renda_mensal', 'N/A'):,.0f}
- Score interno: {cliente_data.get('score_interno', 'N/A')}
- DTI (dívida/renda): {(cliente_data.get('dti', 0) or 0) * 100:.1f}%
- Utilização do limite: {(cliente_data.get('utilizacao_limite', 0) or 0) * 100:.1f}%
- CPF negativado: {'Sim' if cliente_data.get('tem_cpf_negativado') else 'Não'}
- Histórico de atrasos 30d: {cliente_data.get('historico_atrasos_30d', 0)}

RESULTADO DO MODELO:
- Probabilidade de inadimplência: {pd_value * 100:.1f}%
- Nível de risco: {nivel}

PRINCIPAIS FATORES (SHAP values):
{shap_texto}

Por favor, em 2-3 parágrafos curtos em PT-BR:
1. Explique em linguagem acessível para um gerente de agência POR QUE esse cliente tem esse perfil de risco
2. Destaque os 2-3 fatores mais relevantes e o que eles significam para o negócio
3. Seja objetivo e profissional — evite jargão técnico de ML

Após a explicação, liste exatamente 3 ações recomendadas em formato JSON:
{{"acoes": ["ação 1", "ação 2", "ação 3"]}}"""

        _MAX_RETRIES = 3
        _RETRY_DELAY = 25  # segundos — cold start típico de modelos 7B no tier gratuito

        loop = asyncio.get_running_loop()
        fn = partial(
            self.client.chat_completion,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            timeout=60,
        )

        response = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await loop.run_in_executor(None, fn)
                break
            except Exception as e:
                if "loading" in str(e).lower() and attempt < _MAX_RETRIES - 1:
                    print(
                        f"[Copilot] Modelo carregando "
                        f"({attempt + 1}/{_MAX_RETRIES}), aguardando {_RETRY_DELAY}s..."
                    )
                    await asyncio.sleep(_RETRY_DELAY)
                else:
                    print(f"[Copilot] Erro na HuggingFace API: {e}. Usando fallback.")
                    return gerar_explicacao_template(pd_value, segmento, shap_top, cliente_data)

        if response is None:
            return gerar_explicacao_template(pd_value, segmento, shap_top, cliente_data)

        full_text = response.choices[0].message.content

        # Extrai parte JSON das ações
        json_match = re.search(r'\{"acoes":\s*\[.*?\]\}', full_text, re.DOTALL)
        acoes = []
        explicacao = full_text

        if json_match:
            try:
                acoes_data = json.loads(json_match.group())
                acoes = acoes_data.get("acoes", [])
                explicacao = full_text[:json_match.start()].strip()
            except json.JSONDecodeError:
                pass

        if not acoes:
            acoes = _gerar_acoes(pd_value, segmento, shap_top, cliente_data)

        return explicacao, acoes
