# pages/7_💼_Portfolio.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from core.ui import app_header
from core.data import load_watchlists, download_bulk
from core.portfolio import mc_frontier, build_equal_weight_returns, ann_stats, extract_weights

st.set_page_config(page_title="Portfolio", page_icon="💼", layout="wide")
@st.cache_data(ttl=600)
def _cached_returns(symbols_tuple, period, interval):
    from core.data import download_bulk
    data = download_bulk(list(symbols_tuple), period=period, interval=interval)
    rets = {s: df["Close"].pct_change().dropna()
            for s, df in data.items() if (df is not None and not df.empty and "Close" in df.columns)}
    return pd.DataFrame(rets).dropna()

app_header("💼 Portfolio", "Fronteira eficiente (Monte Carlo), Equal-Weight e pesos ótimos")

# ------------- Tema escuro opcional (leve) -------------
def inject_dark():
    st.markdown("""
    <style>
      :root{ --bg:#0b0f16; --panel:#121826; --text:#e8edf7; --muted:#a3adc2; --border:#1f2a44; --accent:#38bdf8; }
      .stApp{ background:var(--bg); color:var(--text); }
      [data-testid="stMarkdownContainer"], label, p, span, h1,h2,h3,h4,h5{ color:var(--text)!important; }
      [data-baseweb="select"]>div{ background:var(--panel)!important; border-color:var(--border)!important; }
      [data-testid="stMetric"]{ background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:12px; }
    </style>
    """, unsafe_allow_html=True)

# ------------- Seleção de ativos -------------
watch = load_watchlists()
all_syms = sorted(set(watch["BR_STOCKS"] + watch["US_STOCKS"] + watch["CRYPTO"]))

# --- seleção integrada com Screener ---
sel_from_screener = st.session_state.get("screener_selection", [])
use_sel = st.toggle("Usar seleção do Screener (se houver)", value=bool(sel_from_screener))
sel_count = len(sel_from_screener)

# sugestão de default caso precise escolher manualmente
suggest = ["AAPL","MSFT","NVDA"]
default_list = [s for s in suggest if s in all_syms] or all_syms[:3]

if use_sel and sel_count >= 2:
    symbols = sel_from_screener
    preview = ", ".join(symbols[:8]) + ("…" if sel_count > 8 else "")
    st.caption(f"Usando **{sel_count}** ativos do Screener: {preview}")
else:
    if use_sel and sel_count < 2:
        st.warning("A seleção do Screener tem menos de **2** ativos. Escolha abaixo ou volte ao Screener e marque mais.")
    symbols = st.multiselect(
        "Ativos para montar o portfólio",
        options=all_syms,
        default=default_list
    )


col1, col2, col3, col4 = st.columns(4)
period   = col1.selectbox("Período", ["6mo","1y","2y","5y"], index=1)
interval = col2.selectbox("Intervalo", ["1d","1wk"], index=0)
rf_pct   = col3.number_input("Taxa livre de risco anual (%)", min_value=-5.0, value=6.0, step=0.5)
dark     = col4.toggle("Tema escuro", value=True)
if dark: inject_dark()

col5, col6, col7 = st.columns(3)
n_sims  = int(col5.slider("Simulações (Monte Carlo)", 2000, 50000, 15000, step=1000))
w_max   = float(col6.slider("Peso máximo por ativo", 0.10, 1.00, 0.35, step=0.05))
seed    = int(col7.number_input("Semente (reprodutibilidade)", value=42, step=1))

if len(symbols) < 2:
    st.warning("Selecione pelo menos **2** ativos.")
    st.stop()

# ------------- Download & retornos -------------
bulk = download_bulk(symbols, period=period, interval=interval)
rets = {}
for sym, df in bulk.items():
    if not df.empty and "Close" in df.columns:
        rets[sym] = df["Close"].pct_change().dropna()
# ------------- Download & retornos (cacheado) -------------
with st.spinner("🔄 Baixando dados e calculando retornos..."):
    rets_df = _cached_returns(tuple(symbols), period, interval)

if rets_df.shape[1] < 2 or rets_df.empty:
    st.error("Retornos insuficientes para montar o portfólio nesse período/intervalo.")
    st.stop()

# ------------- Equal-Weight e Fronteira (rápidos) -------------
with st.spinner(f"🎲 Simulando {n_sims:,} carteiras..."):
    eqw_ret = build_equal_weight_returns(rets_df)
    eqw_stats = ann_stats(eqw_ret, interval, rf_annual=rf_pct/100.0)
    front = mc_frontier(
        rets_df, interval=interval, rf_annual=rf_pct/100.0,
        n_sims=n_sims, w_max=w_max, seed=seed
    )


# ------------- Equal-Weight e Fronteira -------------
eqw_ret = build_equal_weight_returns(rets_df)
eqw_stats = ann_stats(eqw_ret, interval, rf_annual=rf_pct/100.0)

front = mc_frontier(
    rets_df, interval=interval, rf_annual=rf_pct/100.0,
    n_sims=n_sims, w_max=w_max, seed=seed
)

# Portfólios de interesse
best_row = front.iloc[front["sharpe"].idxmax()]
minv_row = front.iloc[front["vol_ann"].idxmin()]
names = rets_df.columns.tolist()
w_best = extract_weights(best_row, names)
w_minv = extract_weights(minv_row, names)

# ------------- KPIs rápidos -------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Equal-Weight (CAGR)", f"{eqw_stats['cagr']*100:,.2f}%")
c2.metric("Equal-Weight (Sharpe)", f"{eqw_stats['sharpe']:.2f}")
c3.metric("Máx. Sharpe (ret/vol)", f"{best_row['ret_ann']*100:,.2f}% / {best_row['vol_ann']*100:,.2f}%")
c4.metric("Mín. Vol (vol)", f"{minv_row['vol_ann']*100:,.2f}%")

st.divider()

# ------------- Scatter: Fronteira -------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=front["vol_ann"]*100, y=front["ret_ann"]*100,
    mode="markers", name="Portfólios (MC)",
    marker=dict(size=5, color=front["sharpe"], colorscale="Viridis", showscale=True, colorbar=dict(title="Sharpe"))
))
fig.add_trace(go.Scatter(
    x=[eqw_stats["vol_ann"]*100], y=[eqw_stats["ret_ann"]*100],
    mode="markers+text", name="Equal-Weight",
    marker=dict(size=12, symbol="diamond", line=dict(width=1, color="#333")),
    text=["EW"], textposition="top center"
))
fig.add_trace(go.Scatter(
    x=[best_row["vol_ann"]*100], y=[best_row["ret_ann"]*100],
    mode="markers+text", name="Máx. Sharpe",
    marker=dict(size=12, symbol="star", line=dict(width=1, color="#333")),
    text=["MS"], textposition="bottom center"
))
fig.add_trace(go.Scatter(
    x=[minv_row["vol_ann"]*100], y=[minv_row["ret_ann"]*100],
    mode="markers+text", name="Mín. Vol",
    marker=dict(size=12, symbol="x", line=dict(width=1, color="#333")),
    text=["MV"], textposition="bottom center"
))
fig.update_layout(
    template="plotly_dark" if dark else "plotly_white",
    title="Fronteira (retorno × volatilidade) — cor = Sharpe",
    xaxis_title="Volatilidade anualizada (%)",
    yaxis_title="Retorno anualizado (%)",
    margin=dict(l=10, r=10, t=60, b=10)
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ------------- Inspeção de pesos -------------
st.subheader("Pesos dos portfólios")
choice = st.radio("Qual portfólio inspecionar?", ["Máx. Sharpe","Mín. Vol","Equal-Weight"], horizontal=True)

if choice == "Máx. Sharpe":
    w_sel = w_best
elif choice == "Mín. Vol":
    w_sel = w_minv
else:
    w_sel = pd.Series(np.ones(len(names))/len(names), index=names, dtype=float)

# gráfico de barras
bar = go.Figure()
bar.add_trace(go.Bar(x=w_sel.index, y=(w_sel.values*100)))
bar.update_layout(
    template="plotly_dark" if dark else "plotly_white",
    title=f"Pesos — {choice}",
    yaxis_title="Peso (%)",
    margin=dict(l=10, r=10, t=50, b=10)
)
st.plotly_chart(bar, use_container_width=True)

# tabela e export
weights_df = pd.DataFrame({"Symbol": w_sel.index, "Weight": w_sel.values})
colA, colB = st.columns([3,1])
with colA:
    st.dataframe(weights_df.style.format({"Weight":"{:.2%}"}), use_container_width=True, height=300)
with colB:
    st.download_button(
        "⬇️ Exportar pesos (CSV)",
        data=weights_df.to_csv(index=False).encode("utf-8"),
        file_name=f"weights_{choice.replace(' ','_').lower()}.csv",
        mime="text/csv"
    )

st.divider()

# ------------- Top carteiras (tabelas rápidas) -------------
topN = 10
top_sharpe = front.nlargest(topN, "sharpe")[["ret_ann","vol_ann","sharpe"]].copy()
top_minv  = front.nsmallest(topN, "vol_ann")[["ret_ann","vol_ann","sharpe"]].copy()
top_sharpe[["ret_ann","vol_ann","sharpe"]] = top_sharpe[["ret_ann","vol_ann","sharpe"]].astype(float)
top_minv[["ret_ann","vol_ann","sharpe"]]   = top_minv[["ret_ann","vol_ann","sharpe"]].astype(float)

cL, cR = st.columns(2)
with cL:
    st.markdown("#### Top Sharpe")
    st.dataframe(
        top_sharpe.style.format({"ret_ann":"{:.2%}","vol_ann":"{:.2%}","sharpe":"{:.2f}"}),
        use_container_width=True, height=300
    )
with cR:
    st.markdown("#### Menor Volatilidade")
    st.dataframe(
        top_minv.style.format({"ret_ann":"{:.2%}","vol_ann":"{:.2%}","sharpe":"{:.2f}"}),
        use_container_width=True, height=300
    )
