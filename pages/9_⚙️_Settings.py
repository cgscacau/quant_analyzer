import time
import streamlit as st
from core.data import read_watchlists_file, load_watchlists
from core.watchlists_builder import rebuild_watchlists

st.title("Settings")
st.markdown("### Watchlists (atualizar online)")

if st.button("🔄 Atualizar watchlists (últimos 60 dias)"):
    with st.spinner("Buscando no Yahoo Finance e reconstruindo listas..."):
        base = read_watchlists_file()          # <-- SEMPRE do ARQUIVO como semente
        fresh = rebuild_watchlists(base)       # filtra/ classifica

        # failsafe: se por qualquer motivo vier tudo vazio, não grava um override vazio
        if sum(len(fresh.get(k, [])) for k in ["BR_STOCKS","US_STOCKS","CRYPTO"]) == 0:
            st.warning("Reconstrução retornou vazia. Mantendo listas do arquivo (sem override).")
            if "watchlists_override" in st.session_state:
                del st.session_state["watchlists_override"]
        else:
            st.session_state["watchlists_override"] = fresh
            st.session_state["watchlists_version"] = time.time()  # quebra caches dependentes

        st.cache_data.clear()  # invalida caches de outras páginas
    src = "override (memória)" if "watchlists_override" in st.session_state else "arquivo"
    sizes = fresh if "watchlists_override" in st.session_state else base
    st.success(
        f"Watchlists atualizadas em {src}! "
        f"BR:{len(sizes.get('BR_STOCKS',[]))} | US:{len(sizes.get('US_STOCKS',[]))} | "
        f"Cripto:{len(sizes.get('CRYPTO',[]))}"
    )

# utilitário opcional: limpar override e voltar ao arquivo
if st.button("🧹 Reverter para arquivo (remover override)"):
    st.session_state.pop("watchlists_override", None)
    st.session_state["watchlists_version"] = time.time()
    st.cache_data.clear()
    st.info("Override removido. Lendo direto do arquivo.")
