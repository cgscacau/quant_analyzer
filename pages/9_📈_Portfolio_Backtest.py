# pages/9_📈_Portfolio_Backtest.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core.ui import app_header
from core.data import load_watchlists, download_bulk
from core.portfolio import mc_frontier, extract_weights
from core.portfolio_bt import backtest_portfolio, bench_equity, summarize_equities

# -----------------------------------------------------------------------------
# Config & Header
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Portfolio Backtest", page_icon="📈", layout="wide")
app_header("📈 Portfolio Backtest", "Buy&Hold com rebalanceamento, custos, taxas e benchmark")

# -----------------------------------------------------------------------------
# Helpers de dados (robustos a MultiIndex/duplicadas)
# -----------------------------------------------------------------------------
def _get_close_series(df: pd.DataFrame) -> pd.Series | None:
    """Extrai uma Series 1-D de preços 'Close' em diferentes formatos."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    # Caso 1: coluna simples "Close" é Series
    if "Close" in df.columns and not isinstance(df["Close"], pd.DataFrame):
        sr = pd.to_numeric(df["Close"], errors="coerce").dropna()
        return sr if not sr.empty else None

    # Caso 2: "Close" existe mas virou DataFrame (duplicadas)
    if "Close" in df.columns and isinstance(df["Close"], pd.DataFrame):
        sub = df["Close"]
        for col in sub.columns:
            sr = pd.to_numeric(sub[col], errors="coerce").dropna()
            if not sr.empty:
                sr.name = "Close"
                return sr

    # Caso 3: MultiIndex nas colunas (yfinance moderno)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            if "Close" in df.columns.get_level_values(0):
                sub = df.xs("Close", axis=1, level=0, drop_level=False)
                first = sub.columns[0]  # pega primeira série válida
                sr = pd.to_numeric(sub[first], errors="coerce").dropna()
                sr.name = "Close"
                return sr if not sr.empty else None
        except Exception:
            pass
        # fallback preferindo ajustado
        try:
            if "Adj Close" in df.columns.get_level_values(0):
                sub = df.xs("Adj Close", axis=1, level=0, drop_level=False)
                first = sub.columns[0]
                sr = pd.to_numeric(sub[first], errors="coerce").dropna()
                sr.name = "Adj Close"
                return sr if not sr.empty else None
        except Exception:
            pass

    return None

@st.cache_data(ttl=600, show_spinner=False)
def _cached_prices(symbols_tuple: tuple[str, ...], period: str, interval: str) -> pd.DataFrame:
    """
    Baixa dados com download_bulk e retorna DataFrame alinhado de preços de fechamento.
    Colunas = símbolos; linhas = datas (interseção).
    """
    data = download_bulk(list(symbols_tuple), period=period, interval=interval)
    series_list: list[pd.Series] = []
    bad: list[str] = []

    for s, df in (data or {}).items():
        try:
            sr = _get_close_series(df)
            if sr is None or sr.dropna().empty:
                bad.append(str(s)); continue
            sr.name = str(s)
            series_list.append(sr.astype(float))
        except Exception:
            bad.append(str(s))

    if not series_list:
        return pd.DataFrame()

    # concat por colunas, somente datas comuns
    prices = pd.concat(series_list, axis=1, join="inner").dropna(how="any")
    prices = prices.loc[:, ~prices.columns.duplicated()].copy()

    if bad:
        st.caption(f"⚠️ Sem preços válidos para: {', '.join(bad)}")

    return prices

# -----------------------------------------------------------------------------
# Seleção de ativos (aceita 2 chaves do Screener)
# -----------------------------------------------------------------------------
watch = load_watchlists()
all_syms = sorted(set(watch.get("BR_STOCKS", []) + watch.get("US_STOCKS", []) + watch.get("CRYPTO", [])))

_sel1 = st.session_state.get("screener_selected")
_sel2 = st.session_state.get("screener_selection")
sel_from_screener = [str(x) for x in ((_sel1 or []) or (_sel2 or [])) if isinstance(x, (str, bytes))]
sel_from_screener = sorted(set(sel_from_screener))  # dedup

use_sel = st.toggle("Usar seleção do Screener (se houver)", value=bool(sel_from_screener))
default_list = [s for s in ["AAPL", "MSFT", "NVDA", "PETR4.SA"] if s in all_syms] or all_syms[:3]

if use_sel and len(sel_from_screener) >= 2:
    symbols = sel_from_screener
    st.caption(f"Usando seleção do Screener ({len(symbols)} ativos).")
else:
    if use_sel and len(sel_from_screener) < 2:
        st.warning("Seleção do Screener tem menos de 2 ativos. Escolha manualmente abaixo.")
    symbols = st.multiselect("Ativos do portfólio", options=all_syms, default=default_list)

c1, c2, c3, c4 = st.columns(4)
period   = c1.selectbox("Período",   ["6mo", "1y", "2y", "5y"], index=2)
interval = c2.selectbox("Intervalo", ["1d", "1wk"], index=0)
reb      = c3.selectbox("Rebalanceamento", ["Mensal", "Trimestral", "Anual", "none"], index=0)
dark     = c4.toggle("Tema escuro", value=True)

if len(symbols) < 2:
    st.warning("Selecione pelo menos 2 ativos.")
    st.stop()

# -----------------------------------------------------------------------------
# Baixar dados (cache)
# -----------------------------------------------------------------------------
with st.spinner("📥 Baixando dados..."):
    prices = _cached_prices(tuple(symbols), period, interval)

if prices.shape[1] < 2:
    st.error("Dados insuficientes nesse período/intervalo (tente outros ativos/período).")
    st.stop()

symbols_aligned = list(prices.columns)

# -----------------------------------------------------------------------------
# Pesos do portfólio
# -----------------------------------------------------------------------------
st.subheader("Pesos do portfólio")
mode = st.radio("Como definir pesos?", ["Equal-Weight", "Máx. Sharpe (histórico)", "Manual"], horizontal=True)

if mode == "Equal-Weight":
    weights = np.ones(len(symbols_aligned)) / len(symbols_aligned)

elif mode == "Máx. Sharpe (histórico)":
    sims = st.slider("Simulações para achar Máx. Sharpe", 5_000, 50_000, 20_000, step=1_000)
    wmax = st.slider("Peso máximo por ativo", 0.10, 1.00, 0.35, step=0.05)
    rets_df = prices.pct_change().dropna()
    with st.spinner("🔎 Buscando pesos de Máx. Sharpe..."):
        front = mc_frontier(rets_df, interval=interval, rf_annual=0.0, n_sims=sims, w_max=wmax, seed=42)
        best  = front.iloc[front["sharpe"].idxmax()]
        weights = extract_weights(best, rets_df.columns.tolist()).values

else:
    sliders = []
    cols = st.columns(min(6, len(symbols_aligned)))
    for i, s in enumerate(symbols_aligned):
        sliders.append(cols[i % len(cols)].slider(s, 0.0, 1.0, 1.0/len(symbols_aligned), step=0.01))
    w = np.array(sliders, dtype=float)
    weights = (w if w.sum() > 0 else np.ones_like(w)) / (w.sum() if w.sum() > 0 else len(w))

# -----------------------------------------------------------------------------
# Custos, taxas e benchmark
# -----------------------------------------------------------------------------
st.subheader("Custos, taxas e benchmark")
k1, k2, k3, k4 = st.columns(4)
init_cap  = k1.number_input("Capital inicial (R$)", min_value=100.0, value=100_000.0, step=1_000.0)
mgmt_fee  = k2.number_input("Taxa adm. (% a.a.)", min_value=0.0, value=0.0, step=0.1)
tc_bps    = k3.number_input("Custo transação (bps por rebalance)", min_value=0.0, value=5.0, step=1.0)
contrib_m = k4.number_input("Aporte mensal (R$)", min_value=0.0, value=0.0, step=100.0)

bench = st.text_input("Benchmark (opcional — exemplo: SPY, BOVA11.SA)", value="SPY").strip()

# -----------------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------------
with st.spinner("🧪 Rodando backtest..."):
    eq, info = backtest_portfolio(
        prices, weights,
        interval=interval, rebalance=reb, init_cap=init_cap,
        mgmt_fee_annual=mgmt_fee, tc_bps=tc_bps, contrib_monthly=contrib_m
    )

# Benchmark (opcional)
bench_eq = pd.Series(dtype=float)
if bench:
    try:
        bdict = download_bulk([bench], period=period, interval=interval)
        bdf = bdict.get(bench)
        bsr = _get_close_series(bdf) if bdf is not None else None
        if bsr is not None and not bsr.empty:
            bsr = bsr.reindex(eq.index).dropna()
            if not bsr.empty:
                bench_eq = bench_equity(bsr, init_cap=init_cap).reindex(eq.index).ffill()
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Gráficos
# -----------------------------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Portfólio", line=dict(width=2)))
if not bench_eq.empty:
    fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq.values, name=bench, line=dict(width=1.6, dash="dash")))
fig.update_layout(
    template="plotly_dark" if dark else "plotly_white",
    title="Curva de patrimônio",
    xaxis_title="Data", yaxis_title="Equity (R$)",
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------
summ = summarize_equities(eq, bench_eq if not bench_eq.empty else None, interval=interval)
c1, c2, c3, c4 = st.columns(4)
c1.metric("CAGR",          f"{summ['portfolio']['cagr']*100:,.2f}%")
c2.metric("Volatilidade",  f"{summ['portfolio']['vol']*100:,.2f}%")
c3.metric("Sharpe",        f"{summ['portfolio']['sharpe']:.2f}")
c4.metric("Max Drawdown",  f"{summ['portfolio']['maxdd']*100:,.2f}%")

st.download_button(
    "⬇️ Baixar série de equity (CSV)",
    data=eq.to_frame(name="equity").to_csv().encode("utf-8"),
    file_name="portfolio_equity.csv",
    mime="text/csv",
)

st.divider()

# -----------------------------------------------------------------------------
# Resumo Executivo (card)
# -----------------------------------------------------------------------------
weights_df = pd.DataFrame({"Symbol": symbols_aligned, "Weight": (weights / weights.sum())})
top_w = ", ".join([f"{r.Symbol} {r.Weight:.0%}" for r in weights_df.sort_values("Weight", ascending=False).head(5).itertuples()])

bench_line = ""
if 'benchmark' in summ:
    b = summ['benchmark']
    bench_line = f"<br>Benchmark <b>{bench}</b> — CAGR {b['cagr']*100:,.2f}% • Sharpe {b['sharpe']:.2f} • MaxDD {b['maxdd']*100:,.2f}%"

bg1 = "#0b1220" if dark else "#f7fafc"
bg2 = "#121826" if dark else "#ffffff"
text= "#e8edf7" if dark else "#1a202c"
mut = "#a3adc2" if dark else "#4a5568"
border = "#1f2a44" if dark else "#e2e8f0"

st.markdown(f"""
<style>
.bt-card{{background:linear-gradient(135deg,{bg1},{bg2});border:1px solid {border};
        border-radius:14px;padding:16px}}
.bt-card h3{{margin:0 0 8px;color:{text}}}
.bt-card p{{margin:6px 0;color:{mut}}}
</style>
<div class="bt-card">
  <h3>Resumo geral</h3>
  <p><b>Alocação</b>: {top_w}</p>
  <p><b>Resultados</b>: CAGR <b>{summ['portfolio']['cagr']*100:,.2f}%</b> • Sharpe <b>{summ['portfolio']['sharpe']:.2f}</b> • Vol <b>{summ['portfolio']['vol']*100:,.2f}%</b> • MaxDD <b>{summ['portfolio']['maxdd']*100:,.2f}%</b>{bench_line}</p>
  <p><b>Nota</b>: valide os pesos no período escolhido; ajuste rebalance, custos (bps) e taxa de adm. conforme sua corretora/fundo.</p>
</div>
""", unsafe_allow_html=True)
