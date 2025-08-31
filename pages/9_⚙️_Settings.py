# pages/⚙️_Settings.py
import streamlit as st
from core.data import load_watchlists
from core.watchlists_builder import rebuild_watchlists

st.title("Settings")

st.markdown("### Watchlists (atualizar online)")
if st.button("🔄 Atualizar watchlists (últimos 60 dias)"):
    with st.spinner("Buscando no Yahoo Finance e reconstruindo listas..."):
        base = load_watchlists()  # usa o que está em arquivo ou override atual
        fresh = rebuild_watchlists(base)
        st.session_state["watchlists_override"] = fresh
        st.cache_data.clear()  # garante recarregar em outras páginas
    st.success("Watchlists atualizadas em memória! Volte às páginas e recarregue.")
