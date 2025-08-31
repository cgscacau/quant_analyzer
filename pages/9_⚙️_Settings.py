# pages/⚙️_Settings.py
import time
import streamlit as st
from core.data import load_watchlists
from core.watchlists_builder import rebuild_watchlists

st.title("Settings")

st.markdown("### Watchlists (atualizar online)")
if st.button("🔄 Atualizar watchlists (últimos 60 dias)"):
    with st.spinner("Buscando no Yahoo Finance e reconstruindo listas..."):
        base = load_watchlists()                   # usa o arquivo atual como universo
        fresh = rebuild_watchlists(base)           # dicionário novo (filtrado e classificado)
        st.session_state["watchlists_override"] = fresh
        st.session_state["watchlists_version"] = time.time()  # <-- NOVO: versão p/ quebrar caches
        st.cache_data.clear()                      # limpa caches de outras páginas
    st.success("Watchlists atualizadas em memória!")
