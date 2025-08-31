# pages/9_⚙️_Settings.py
from __future__ import annotations

import time
import json
from typing import Dict, List

import pandas as pd
import streamlit as st

from core.data import (
    read_watchlists_file,   # lê SEMPRE do arquivo (cacheado 24h)
    load_watchlists,        # usa override (memória) ou arquivo
    clear_data_cache,       # helper para limpar caches
)
from core.watchlists_builder import rebuild_watchlists


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
SHOW_KEYS = [
    "BR_STOCKS", "BR_FIIS", "BR_DIVIDEND", "BR_BLUE_CHIPS", "BR_SMALL_CAPS",
    "US_STOCKS", "US_DIVIDEND", "US_BLUE_CHIPS", "US_SMALL_CAPS",
    "CRYPTO",
]

def _sizes(d: dict) -> Dict[str, int]:
    out = {}
    for k in SHOW_KEYS:
        out[k] = len(d.get(k, []))
    return out

def _fmt_sizes(d: dict) -> str:
    return (
        f"BR:{len(d.get('BR_STOCKS', []))} | "
        f"FIIs:{len(d.get('BR_FIIS', []))} | "
        f"BR Div:{len(d.get('BR_DIVIDEND', []))} | "
        f"US:{len(d.get('US_STOCKS', []))} | "
        f"Cripto:{len(d.get('CRYPTO', []))}"
    )

def _is_nonempty_min(d: dict) -> bool:
    """Considera 'válido' quando pelo menos BR/US/CRYPTO não estão todos vazios."""
    base_keys = ["BR_STOCKS", "US_STOCKS", "CRYPTO"]
    return sum(len(d.get(k, [])) for k in base_keys) > 0


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("Settings")
st.subheader("Watchlists (atualizar online)")

ver = int(st.session_state.get("watchlists_version", 0))
using_override = "watchlists_override" in st.session_state
origem = "override (memória)" if using_override else "arquivo"

current = load_watchlists()
st.caption(f"Origem atual: **{origem}** • v{ver} • {_fmt_sizes(current)}")

with st.expander("Ver contagens detalhadas (origem atual)", expanded=False):
    st.json(_sizes(current))

debug = st.toggle("Modo debug (mostrar motivo por ticker descartado)", value=False, help="Mais lento; útil para diagnosticar por que um ticker não entrou.")

colA, colB, colC = st.columns([2, 1, 1])

with colA:
    if st.button("🔄 Atualizar watchlists (últimos 60 dias)", use_container_width=True):
        with st.spinner("Reconstruindo listas via Yahoo Finance..."):
            base = read_watchlists_file()  # SEMPRE do arquivo como semente

            try:
                if debug:
                    fresh, report = rebuild_watchlists(base, debug=True)
                else:
                    fresh = rebuild_watchlists(base, debug=False)
                    report = []
            except Exception as e:
                fresh, report = {}, []
                st.error(f"Erro durante a reconstrução: {type(e).__name__}: {e}")

            # Aceitamos override somente se não vier tudo vazio
            if _is_nonempty_min(fresh):
                st.session_state["watchlists_override"] = fresh
                st.session_state["watchlists_version"] = time.time()
                # limpa caches de dados/consultas
                clear_data_cache()
                src = "override (memória)"
                st.success(
                    f"Watchlists atualizadas em {src}! {_fmt_sizes(fresh)}",
                    icon="✅",
                )
                with st.expander("Ver contagens (override novo)", expanded=False):
                    st.json(_sizes(fresh))
                # debug opcional
                if debug and report:
                    st.info("Motivos para tickers descartados (amostra):")
                    df_rep = pd.DataFrame(report, columns=["ticker", "motivo"])
                    st.dataframe(df_rep, use_container_width=True, hide_index=True)
            else:
                # Fallback: mantém arquivo
                st.warning(
                    "Reconstrução retornou vazia. Mantendo listas do arquivo (sem override).",
                    icon="⚠️",
                )
                st.session_state.pop("watchlists_override", None)
                st.session_state["watchlists_version"] = time.time()
                clear_data_cache()
                st.success(
                    f"Watchlists atualizadas em arquivo! {_fmt_sizes(base)}",
                    icon="ℹ️",
                )
                with st.expander("Ver contagens (arquivo)", expanded=False):
                    st.json(_sizes(base))

with colB:
    if st.button("🧹 Reverter para arquivo (remover override)", use_container_width=True):
        st.session_state.pop("watchlists_override", None)
        st.session_state["watchlists_version"] = time.time()
        clear_data_cache()
        base = read_watchlists_file()
        st.info(f"Override removido. Lendo do arquivo. {_fmt_sizes(base)}", icon="🗑️")

with colC:
    if st.button("♻️ Limpar caches", help="Limpa caches de dados e força recarregamento", use_container_width=True):
        clear_data_cache()
        st.success("Caches limpos!", icon="🧼")

st.divider()

# Visualização opcional das listas
with st.expander("Pré-visualizar listas (origem atual)", expanded=False):
    wl_now = load_watchlists()
    tabs = st.tabs([
        "BR Ações", "BR FIIs", "BR Dividendos", "BR Blue Chips", "BR Small Caps",
        "US Ações", "US Dividendos", "US Blue Chips", "US Small Caps",
        "Criptos"
    ])
    key_to_tab = {
        0: "BR_STOCKS", 1: "BR_FIIS", 2: "BR_DIVIDEND", 3: "BR_BLUE_CHIPS", 4: "BR_SMALL_CAPS",
        5: "US_STOCKS", 6: "US_DIVIDEND", 7: "US_BLUE_CHIPS", 8: "US_SMALL_CAPS",
        9: "CRYPTO"
    }
    for i, t in enumerate(tabs):
        with t:
            k = key_to_tab[i]
            lst: List[str] = sorted(wl_now.get(k, []))
            st.caption(f"{k} • {len(lst)} itens")
            if lst:
                st.code(", ".join(lst), language="text")
            else:
                st.write("— vazio —")

st.caption("Obs.: no Streamlit Cloud, alterações são mantidas **em memória** (override). O arquivo no GitHub não é modificado daqui.")
