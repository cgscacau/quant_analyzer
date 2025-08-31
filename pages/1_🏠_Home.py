# pages/0_🏠_Home.py
from __future__ import annotations
import streamlit as st
from datetime import datetime

from core.ui import app_header, next_steps_card

# --------------------------------------------------------------------
# Config & Header
# --------------------------------------------------------------------
st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")  # docs: st.set_page_config
app_header("🏠 Home", "Visão geral do projeto & Manual Operacional")

st.success("Este é o *esqueleto* do Quant Analyzer. Nada aqui é recomendação; use para estudo e prototipagem.")

# --------------------------------------------------------------------
# Manual (Markdown central + componentes visuais nas abas)
# --------------------------------------------------------------------
MANUAL_MD = f"""
# Manual Operacional — Quant Analyzer

Última atualização: **{datetime.now().strftime("%Y-%m-%d %H:%M")}**

## 1) Visão geral
- App multipáginas em Streamlit; cada página define `st.set_page_config` no topo.
- Dados vindos do Yahoo Finance via camada `core.data` (tratando `Close/Adj Close`, MultiIndex).
- Gráficos com Plotly; tema claro/escuro controlado nas páginas.
- Cache: `st.cache_data` para downloads e transformações; limpe quando mudar versão do data layer.

## 2) Estrutura de páginas
- **Screener**: seleção de ativos; salva no `st.session_state`. Use chaves: `screener_selected` **ou** `screener_selection`.
- **Portfolio (Monte Carlo)**: cenários MVN/Bootstrap, fan chart, probabilidades.
- **Portfolio (Markowitz)**: fronteira eficiente, MVP, tangência, comparação de carteiras.
- **Portfolio Backtest**: buy&hold com rebalance, custos e benchmark.

## 3) Fluxo rápido
1. Marque os ativos no **Screener**.
2. Nas páginas, ative “Usar seleção do Screener”.
3. Ajuste período/intervalo e **pesos** (Equal-Weight, Máx. Sharpe, Manual).
4. Leia KPIs (CAGR, Vol, Sharpe, MaxDD) e exporte CSVs (pesos/equity/percentis).

## 4) Convenções de dados
- Períodos: `6mo`, `1y`, `2y`, `5y` | Intervalos: `1d`, `1wk`.
- Preferir `Close`; se ausente, cair para `Adj Close`.
- Alinhamento por **interseção** de datas (inner join).
- Retornos: `pct_change().dropna()`; anualização: média × 252, vol × √252.

## 5) Cache & Estado
- `@st.cache_data(ttl=600)` em funções de I/O e pré-processo.
- `st.session_state` para seleção do screener e preferências por sessão.

## 6) UI/UX
- Use **abas** para seções longas, **expanders** para detalhes.
- Ofereça **download** (CSV/MD) onde fizer sentido.
- Evite bloquear a UI: envolva operações pesadas em `st.spinner(...)`.

## 7) Erros comuns & correções
- *“If using all scalar values, you must pass an index”*: monte DataFrames com `pd.concat([...], axis=1)` a partir de **Series** nomeadas.
- *Colunas MultiIndex do provedor*: extraia `Close`/`Adj Close` para **Series 1-D** antes de `pct_change()`.
- *Dados insuficientes*: verifique `period/interval` e ativos; mostre aviso e prossiga com os válidos.

"""

# Abas principais para navegação visual (doc oficial: st.tabs)
tab_intro, tab_fluxo, tab_paginas, tab_dados, tab_uiux, tab_erros = st.tabs(
    ["📘 Introdução", "🧭 Fluxo operacional", "🗂️ Páginas", "🗃️ Dados & Cache", "🎛️ UI/UX", "🧯 Erros comuns"]
)

with tab_intro:
    st.markdown("### Introdução")
    st.markdown(MANUAL_MD.split("## 2) Estrutura de páginas")[0])  # só a seção 1 aqui
    next_steps_card([
        "Definir lista de ativos padrão (BR/US/Cripto).",
        "Padronizar funções de dados (Yahoo Finance).",
        "Estabelecer tema visual e componentes de UI.",
    ])
    st.download_button(
        "⬇️ Baixar Manual (Markdown)",
        data=MANUAL_MD.encode("utf-8"),
        file_name="manual_operacional_quant_analyzer.md",
        mime="text/markdown",
        use_container_width=True,  # docs: st.download_button
    )

with tab_fluxo:
    with st.expander("Passo a passo — do Screener ao resultado", expanded=True):  # docs: st.expander
        st.markdown("""
1. **Screener** → selecione tickers.
2. Nas páginas de análise, ative **Usar seleção do Screener**.
3. Escolha **pesos** (Equal-Weight / Máx. Sharpe / Manual).
4. Ajuste **período/intervalo**, rebalance, custos e aportes.
5. Analise **fan chart**, **distribuição final** e **KPIs**; exporte CSVs.
""")
    st.markdown("#### Boas práticas")
    st.markdown("""
- Prefira períodos mais longos para estabilidade de estimativas.
- Documente hipóteses (rf, w_max, rebalance) no relatório/CSV exportado.
""")

with tab_paginas:
    st.markdown("### Estrutura de páginas")
    st.markdown("""
- **Screener**: salva seleção no `session_state` (`screener_selected`/`screener_selection`).
- **Monte Carlo**: cenários (MVN/Bootstrap), probabilidades e export.
- **Markowitz**: fronteira eficiente, MVP, tangência, comparação com Monte Carlo.
- **Backtest**: buy&hold com rebalance, custos e benchmark (SPY, BOVA11.SA, etc.).
""")

with tab_dados:
    st.markdown("### Dados & Cache")
    st.markdown("""
- Extraia `Close` (ou `Adj Close`) e converta para **Series 1-D** (prove dores podem retornar **MultiIndex**).
- Monte DataFrames com `pd.concat(..., axis=1, join="inner")` e `dropna()`.
- Use `@st.cache_data(ttl=600)` em downloads/transformações; limpe ao mudar o data layer.
""")

with tab_uiux:
    st.markdown("### UI/UX")
    st.markdown("""
- Navegue conteúdo longo com **abas**; detalhe técnico em **expanders**.
- Ofereça **st.download_button** para CSV/MD.
- Ajuste tema claro/escuro nos gráficos Plotly via `template`.
""")

with tab_erros:
    st.markdown("### Erros comuns")
    st.markdown("""
- **DataFrame com escalares** → use `pd.concat` de **Series** nomeadas.
- **MultiIndex inesperado** → selecione nível `'Close'`/`'Adj Close'` e converta para Series.
- **Dados insuficientes** → perí odos curtos, ativos sem histórico ou fuso/feriados; trate e informe ao usuário.
""")

# Rodapé compacto do manual completo
with st.expander("📄 Ver o Manual completo (Markdown)", expanded=False):
    st.markdown(MANUAL_MD)
