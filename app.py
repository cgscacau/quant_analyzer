# app.py
import streamlit as st
from core.ui import app_header, footer_note, ensure_session_defaults

st.set_page_config(
    page_title="Quant Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Garante variáveis padrão na sessão
ensure_session_defaults()

# Cabeçalho
app_header("Quant Analyzer", "Suite Quantitativa — Streamlit + Yahoo Finance")

st.markdown(
    """
    **Bem-vindo!** Este é o *esqueleto* de um analisador quant baseado em:
    - **Yahoo Finance** para dados
    - **Módulos** separados por páginas
    - **Arquitetura** organizada em `core/` (dados, UI, indicadores)

    > Fluxo sugerido: Dados → Visualização → Screener → Backtest → ML → Risco → Portfólio → Monte Carlo.
    """
)

st.info("Use o menu **Pages** (barra lateral) para navegar entre as abas. Vamos implementar cada uma por etapas, após sua aprovação.")

# Rodapé
footer_note()
