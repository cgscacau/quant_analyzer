# core/ui.py
import streamlit as st
from typing import List
import pandas as pd
from core.data import load_watchlists

def ensure_session_defaults():
    if "watchlists" not in st.session_state:
        st.session_state["watchlists"] = load_watchlists()
    if "asset_class" not in st.session_state:
        st.session_state["asset_class"] = "BR_STOCKS"   # padrão Brasil
    if "default_symbol" not in st.session_state:
        # primeiro da classe padrão
        wl = st.session_state["watchlists"][st.session_state["asset_class"]]
        st.session_state["default_symbol"] = wl[0] if wl else None

def app_header(title: str, subtitle: str = ""):
    st.markdown(f"# {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()

def footer_note():
    st.divider()
    st.caption("© 2025 Quant Analyzer — esqueleto inicial para iteração.")

def next_steps_card(items: List[str]):
    with st.expander("Próximos passos", expanded=True):
        for i in items:
            st.write("•", i)

def ticker_selector() -> str:
    st.sidebar.markdown("### Ativo")

    classes = {
        "Brasil (Ações B3)": "BR_STOCKS",
        "EUA (Ações US)": "US_STOCKS",
        "Criptos": "CRYPTO"
    }
    # seletor de classe
    label_to_key = {k: v for k, v in classes.items()}
    key_to_label = {v: k for k, v in classes.items()}

    current = st.session_state.get("asset_class", "BR_STOCKS")
    chosen_label = st.sidebar.selectbox(
        "Classe",
        list(classes.keys()),
        index=list(classes.values()).index(current) if current in classes.values() else 0
    )
    asset_class = label_to_key[chosen_label]
    st.session_state["asset_class"] = asset_class

    # lista de tickers da classe
    wl = st.session_state["watchlists"].get(asset_class, [])
    if not wl:
        st.sidebar.warning("Lista vazia para esta classe.")
        return None

    default_symbol = st.session_state.get("default_symbol", wl[0])
    # se o default não pertencer à classe, ajusta
    if default_symbol not in wl:
        default_symbol = wl[0]

    symbol = st.sidebar.selectbox("Símbolo (Yahoo Finance)", wl, index=wl.index(default_symbol))
    st.session_state["default_symbol"] = symbol
    return symbol

def data_status_badge(df: pd.DataFrame):
    if df is None or df.empty:
        st.error("Sem dados (vazio). Verifique o ticker, período/intervalo ou conexão.")
    else:
        st.success(f"Linhas: {len(df)} | Colunas: {list(df.columns)}")
