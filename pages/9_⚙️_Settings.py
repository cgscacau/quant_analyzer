# pages/⚙️_Settings.py (trecho)
import streamlit as st
from core.data import load_watchlists, _load_watchlists_file
from core.watchlists_builder import rebuild_watchlists

st.title("Settings")

st.markdown("### Watchlists (atualizar online)")
if st.button("🔄 Atualizar watchlists (últimos 60 dias)"):
    with st.spinner("Buscando dados no Yahoo Finance e reconstruindo listas..."):
        base = _load_watchlists_file()           # universo base do arquivo
        fresh = rebuild_watchlists(base)         # gera dicionário novo
        st.session_state["watchlists_override"] = fresh
        st.cache_data.clear()                    # força outras páginas a recarregar
    st.success(
        "Watchlists atualizadas em memória! "
        "Abra as páginas (Price Charts/Screener/etc.) para ver as novas classes."
    )

st.caption("Obs.: no Streamlit Cloud, alterações em disco **não persistem**. "
           "Este botão mantém as listas atualizadas em memória/cache do servidor.")
