# pages/9_📈_Portfolio_Backtest.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.ui import app_header
from core.data import load_watchlists, download_bulk
from core.portfolio import mc_frontier, extract_weights
from core.portfolio_bt import backtest_portfolio, bench_equity, summarize_equities

st.set_page_config(page_title="Portfolio Backtest", page_icon="📈", layout="wide")
app_header("📈 Portfolio Backtest", "Buy&Hold com rebalanceamento, custos, taxas e benchmark")

# ---------------- seleção de ativos ----------------
watch = load_watchlists()
all_syms = sorted(set(watch["BR_STOCKS"] + watch["US_STOCKS"] + watch["CRYPTO"]))

sel_from_screener = st.session_state.get("screener_selection", [])
use_sel = st.toggle("Usar seleção do Screener (se houver)", value=bool(sel_from_screener))
default_list = [s for s in ["AAPL","MSFT","NVDA","PETR4.SA"] if s in all_syms] or all_syms[:3]

if use_sel and len(sel_from_screener) >= 2:
    symbols = sel_from_screener
    st.caption(f"Usando seleção do Screener ({len(symbols)} ativos).")
else:
    if use_sel and len(sel_from_screener) < 2:
        st.warning("Seleção do Screener tem menos de 2 ativos. Escolha manualmente abaixo.")
    symbols = st.multiselect("Ativos do portfólio", options=all_syms, default=default_list)

c1,c2,c3,c4 = st.columns(4)
period   = c1.selectbox("Período", ["6mo","1y","2y","5y"], index=2)
interval = c2.selectbox("Intervalo", ["1d","1wk"], index=0)
reb      = c3.selectbox("Rebalanceamento", ["Mensal","Trimestral","Anual","none"], index=0)
dark     = c4.toggle("Tema escuro", value=True)

if len(symbols) < 2:
    st.warning("Selecione pelo menos 2 ativos.")
    st.stop()

# ---------------- pesos ----------------
st.subheader("Pesos do portfólio")
mode = st.radio("Como definir pesos?", ["Equal-Weight","Máx. Sharpe (histórico)","Manual"], horizontal=True)

if mode == "Equal-Weight":
    weights = np.ones(len(symbols))/len(symbols)
elif mode == "Máx. Sharpe (histórico)":
    sims = st.slider("Simulações para achar Máx. Sharpe", 5000, 50000, 20000, step=1000)
    wmax = st.slider("Peso máximo por ativo", 0.10, 1.00, 0.35, step=0.05)
else:
    weights = np.array([st.slider(s, 0.0, 1.0, 1.0/len(symbols), step=0.01) for s in symbols], dtype=float)
    if weights.sum()==0: weights = np.ones_like(weights)
    weights = weights / weights.sum()

# ---------------- custos e taxas ----------------
st.subheader("Custos, taxas e benchmark")
k1,k2,k3,k4 = st.columns(4)
init_cap   = k1.number_input("Capital inicial (R$)", min_value=100.0, value=100_000.0, step=1000.0)
mgmt_fee   = k2.number_input("Taxa adm. (% a.a.)", min_value=0.0, value=0.0, step=0.1)
tc_bps     = k3.number_input("Custo transação (bps por rebalance)", min_value=0.0, value=5.0, step=1.0)
contrib_m  = k4.number_input("Aporte mensal (R$)", min_value=0.0, value=0.0, step=100.0)

bench = st.text_input("Benchmark (opcional — exemplo: SPY, BOVA11.SA)", value="SPY")

# ---------------- dados (cache) ----------------
@st.cache_data(ttl=600)
def _bulk(period, interval, symbols_tuple):
    return download_bulk(list(symbols_tuple), period=period, interval=interval)

with st.spinner("📥 Baixando dados..."):
    data = _bulk(period, interval, tuple(symbols))

closes = {s: df["Close"] for s,df in data.items() if df is not None and not df.empty and "Close" in df.columns}
if len(closes) < 2:
    st.error("Dados insuficientes nesse período/intervalo.")
    st.stop()

# alinhar datas
prices = pd.concat(closes, axis=1).dropna()
symbols_aligned = list(prices.columns)

# se precisar, calcular Máx. Sharpe agora
if mode == "Máx. Sharpe (histórico)":
    from core.portfolio import mc_frontier
    rets_df = prices.pct_change().dropna()
    with st.spinner("🔎 Buscando pesos de Máx. Sharpe..."):
        front = mc_frontier(rets_df, interval=interval, rf_annual=0.0, n_sims=sims, w_max=wmax, seed=42)
        best = front.iloc[front["sharpe"].idxmax()]
        weights = np.array([best[f"w_{s}"] for s in rets_df.columns])

# ---------------- backtest ----------------
with st.spinner("🧪 Rodando backtest..."):
    eq, info = backtest_portfolio(
        prices, weights,
        interval=interval, rebalance=reb, init_cap=init_cap,
        mgmt_fee_annual=mgmt_fee, tc_bps=tc_bps, contrib_monthly=contrib_m
    )

# benchmark
bench_eq = pd.Series(dtype=float)
if bench.strip():
    try:
        bench_data = download_bulk([bench], period=period, interval=interval).get(bench)
        if bench_data is not None and not bench_data.empty:
            close_b = bench_data["Close"].reindex(eq.index).dropna()
            if not close_b.empty:
                bench_eq = bench_equity(close_b, init_cap=init_cap).reindex(eq.index).fillna(method="ffill")
    except Exception:
        pass

# ---------------- gráficos ----------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Portfólio", line=dict(width=2)))
if not bench_eq.empty:
    fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq.values, name=bench, line=dict(width=1.6, dash="dash")))
fig.update_layout(
    template="plotly_dark" if dark else "plotly_white",
    title="Curva de patrimônio",
    xaxis_title="Data", yaxis_title="Equity (R$)",
    margin=dict(l=10,r=10,t=50,b=10)
)
st.plotly_chart(fig, use_container_width=True)

# ---------------- métricas ----------------
summ = summarize_equities(eq, bench_eq if not bench_eq.empty else None, interval=interval)
c1,c2,c3,c4 = st.columns(4)
c1.metric("CAGR", f"{summ['portfolio']['cagr']*100:,.2f}%")
c2.metric("Volatilidade", f"{summ['portfolio']['vol']*100:,.2f}%")
c3.metric("Sharpe", f"{summ['portfolio']['sharpe']:.2f}")
c4.metric("Max Drawdown", f"{summ['portfolio']['maxdd']*100:,.2f}%")

st.download_button(
    "⬇️ Baixar série de equity (CSV)",
    data=eq.to_frame().to_csv().encode("utf-8"),
    file_name="portfolio_equity.csv",
    mime="text/csv"
)

st.divider()

# ---------------- resumo executivo (card único) ----------------
weights_df = pd.DataFrame({"Symbol": symbols_aligned, "Weight": (weights/weights.sum())}, index=None)
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
  <p><b>Recomendação</b>: valide estes pesos no período completo escolhido; ajuste o
     <i>rebalance</i>, custos (bps) e taxa de adm. conforme sua corretora/fundo. Se o Sharpe estiver
     inferior ao benchmark, teste pesos alternativos (Máx. Sharpe) ou reduza concentração.</p>
</div>
""", unsafe_allow_html=True)
