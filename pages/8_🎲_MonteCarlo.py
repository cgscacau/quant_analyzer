# pages/8_🎲_MonteCarlo.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.ui import app_header
from core.data import load_watchlists, download_bulk
from core.portfolio import mc_frontier, extract_weights
from core.montecarlo import (
    simulate_mvn_portfolio,
    simulate_bootstrap_portfolio,
    fan_percentiles,
    paths_summary,
    probability_targets,
)

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Monte Carlo", page_icon="🎲", layout="wide")
app_header("🎲 Monte Carlo", "Cenários do portfólio: fan chart, distribuição final e probabilidades")

# --------------------------------------------------------------------------------------
# Cache de retornos
# --------------------------------------------------------------------------------------
@st.cache_data(ttl=600)
@st.cache_data(ttl=600)
def _cached_returns(symbols_tuple, period, interval):
    data = download_bulk(list(symbols_tuple), period=period, interval=interval)

    series_list = []
    for s, df in (data or {}).items():
        if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
            # garante tipo numérico e série de retornos com nome
            sr = pd.to_numeric(df["Close"], errors="coerce").pct_change().dropna()
            if not sr.empty:
                sr.name = str(s)
                series_list.append(sr)

    if not series_list:
        # retorna DF vazio e deixa o chamador tratar
        return pd.DataFrame()

    # concatena por coluna, alinhando por índice e limpando NaNs
    rets_df = pd.concat(series_list, axis=1, join="inner").dropna(how="any")
    return rets_df


# --------------------------------------------------------------------------------------
# Seleção de ativos
# --------------------------------------------------------------------------------------
watch = load_watchlists()
all_syms = sorted(set(watch["BR_STOCKS"] + watch["US_STOCKS"] + watch["CRYPTO"]))

sel_from_screener = st.session_state.get("screener_selection", [])
use_sel = st.toggle("Usar seleção do Screener (se houver)", value=bool(sel_from_screener))
sel_count = len(sel_from_screener)

default_list = [s for s in ["AAPL", "MSFT", "NVDA"] if s in all_syms] or all_syms[:3]

if use_sel and sel_count >= 2:
    symbols = sel_from_screener
    preview = ", ".join(symbols[:8]) + ("…" if sel_count > 8 else "")
    st.caption(f"Usando **{sel_count}** ativos do Screener: {preview}")
else:
    if use_sel and sel_count < 2:
        st.warning("A seleção do Screener tem menos de **2** ativos. Escolha abaixo ou volte ao Screener e marque mais.")
    symbols = st.multiselect("Ativos do portfólio", options=all_syms, default=default_list)

c1, c2, c3, c4 = st.columns(4)
period   = c1.selectbox("Período histórico", ["6mo", "1y", "2y", "5y"], index=1)
interval = c2.selectbox("Intervalo", ["1d", "1wk"], index=0)
h_months = c3.selectbox("Horizonte (meses)", [3, 6, 12, 24, 36, 60], index=2)
n_sims   = int(c4.slider("Nº de simulações", 1000, 50000, 10000, step=1000))

c5, c6, c7 = st.columns(3)
init_cap = float(c5.number_input("Capital inicial (R$)", min_value=100.0, value=100000.0, step=1000.0))
model    = c6.radio("Modelo", ["MVN (normal multivariado)", "Bootstrap histórico"], horizontal=True)
dark     = c7.toggle("Tema escuro", value=True)

if len(symbols) < 2:
    st.warning("Selecione pelo menos **2** ativos.")
    st.stop()

# --------------------------------------------------------------------------------------
# Retornos (cache)
# --------------------------------------------------------------------------------------
with st.spinner("🔄 Preparando retornos..."):
    rets_df = _cached_returns(tuple(symbols), period, interval)

if rets_df.shape[1] < 2:
    st.error("Retornos insuficientes para simular.")
    st.stop()

# --------------------------------------------------------------------------------------
# Pesos do portfólio
# --------------------------------------------------------------------------------------
st.subheader("Pesos do portfólio")
choice = st.radio("Modo de pesos", ["Equal-Weight", "Máx. Sharpe (via Monte Carlo)", "Personalizado"], horizontal=True)

if choice == "Equal-Weight":
    weights = np.ones(len(symbols)) / len(symbols)
    st.caption("Pesos uniformes (somam 1).")

elif choice == "Máx. Sharpe (via Monte Carlo)":
    sims_for_weights = int(st.slider("Simulações para achar Máx. Sharpe", 5000, 50000, 15000, step=1000))
    w_max = float(st.slider("Peso máximo por ativo", 0.10, 1.00, 0.40, step=0.05, help="Restrição de peso na busca."))
    with st.spinner("📈 Buscando portfólio de Máx. Sharpe..."):
        front = mc_frontier(
            rets_df, interval=interval, rf_annual=0.0,
            n_sims=sims_for_weights, w_max=w_max, seed=42
        )
        best = front.iloc[front["sharpe"].idxmax()]
        weights = extract_weights(best, rets_df.columns.tolist()).values
    st.caption("Pesos calculados por simulação (sem short).")

else:  # Personalizado
    sliders = []
    cols = st.columns(min(6, len(symbols)))
    for i, sym in enumerate(rets_df.columns):
        sliders.append(cols[i % len(cols)].slider(f"{sym}", 0.0, 1.0, 1.0/len(symbols), step=0.01))
    w = np.array(sliders, dtype=float)
    if w.sum() == 0:
        w = np.ones_like(w)
    weights = w / w.sum()
    st.caption("Pesos normalizados para somar 1.")

w_df = pd.DataFrame({"Symbol": rets_df.columns, "Weight": weights})
st.dataframe(w_df.style.format({"Weight": "{:.2%}"}), use_container_width=True, height=220)

st.divider()

# --------------------------------------------------------------------------------------
# Parâmetros operacionais (rebalance e aportes)
# --------------------------------------------------------------------------------------
# passos por mês conforme intervalo
steps_per_month = 21 if interval == "1d" else 4
steps = int(round(h_months * steps_per_month))

colA, colB = st.columns(2)
rebalance_label = colA.selectbox("Rebalanceamento", ["Sem", "Mensal", "Trimestral", "Anual"], index=1)
aporte_mensal  = colB.number_input("Aporte mensal (R$)", min_value=0.0, value=0.0, step=100.0)
st.markdown("### Taxas e saques")
cF1, cF2, cF3, cF4 = st.columns(4)
mgmt_fee_annual = cF1.number_input("Taxa de administração (% a.a.)", min_value=0.0, value=0.0, step=0.1)
perf_fee_pct    = cF2.number_input("Taxa de performance (% sobre lucro)", min_value=0.0, value=0.0, step=1.0)
hurdle_annual   = cF3.number_input("Hurdle anual (% a.a.)", min_value=0.0, value=0.0, step=0.5, help="Meta mínima antes de cobrar performance")
saque_mensal    = cF4.number_input("Saque mensal (R$)", min_value=0.0, value=0.0, step=100.0)

withdraw_per_step = (saque_mensal / steps_per_month) if saque_mensal > 0 else 0.0


if interval == "1d":
    rb_map = {"Sem": 0, "Mensal": 21, "Trimestral": 63, "Anual": 252}
else:  # 1wk
    rb_map = {"Sem": 0, "Mensal": 4, "Trimestral": 13, "Anual": 52}

rebalance_every = rb_map[rebalance_label]
contrib_per_step = (aporte_mensal / steps_per_month) if aporte_mensal > 0 else 0.0

st.caption(
    f"Parâmetros: **{rebalance_label}** rebalance • "
    f"Aporte mensal **R$ {aporte_mensal:,.0f}** (≈ **R$ {contrib_per_step:,.0f}**/passo) • "
    f"Adm **{mgmt_fee_annual:.2f}% a.a.** • Perf **{perf_fee_pct:.0f}%** (hurdle **{hurdle_annual:.2f}% a.a.**) • "
    f"Saque mensal **R$ {saque_mensal:,.0f}** (≈ **R$ {withdraw_per_step:,.0f}**/passo)"
)


# --------------------------------------------------------------------------------------
# Simulação
# --------------------------------------------------------------------------------------
with st.spinner(f"🎲 Simulando {n_sims:,} cenários por {h_months} meses..."):
    if model.startswith("MVN"):
        res = simulate_mvn_portfolio(
            rets_df, weights,
            steps=steps, interval=interval, n_sims=n_sims, start_equity=init_cap,
            rebalance_every=rebalance_every, contrib_per_step=contrib_per_step,
            mgmt_fee_annual=mgmt_fee_annual, perf_fee_pct=perf_fee_pct,
            hurdle_annual=hurdle_annual, withdraw_per_step=withdraw_per_step,
        )
    else:
        block = int(st.slider("Tamanho do bloco (bootstrap)", 1, 21 if interval == "1d" else 4, 5, step=1))
        res = simulate_bootstrap_portfolio(
            rets_df, weights,
            steps=steps, n_sims=n_sims, start_equity=init_cap, block=block,
            rebalance_every=rebalance_every, contrib_per_step=contrib_per_step,
            mgmt_fee_annual=mgmt_fee_annual, perf_fee_pct=perf_fee_pct,
            hurdle_annual=hurdle_annual, withdraw_per_step=withdraw_per_step,
        )


equity = res["equity"]  # shape: (T+1, S)

# --------------------------------------------------------------------------------------
# Fan chart
# --------------------------------------------------------------------------------------
fan = fan_percentiles(equity, percs=(5, 25, 50, 75, 95))
x = np.arange(fan.shape[0])

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=fan["p95"], line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=x, y=fan["p75"], fill="tonexty", name="75–95%", mode="lines"))
fig.add_trace(go.Scatter(x=x, y=fan["p50"], fill="tonexty", name="50–75%", mode="lines"))
fig.add_trace(go.Scatter(x=x, y=fan["p25"], fill="tonexty", name="25–50%", mode="lines"))
fig.add_trace(go.Scatter(x=x, y=fan["p5"],  fill="tonexty", name="5–25%",  mode="lines"))
fig.update_layout(
    template="plotly_dark" if dark else "plotly_white",
    title="Fan chart — percentis do patrimônio ao longo do tempo",
    xaxis_title=f"Passos ({'dias' if interval=='1d' else 'semanas'})",
    yaxis_title="Equity (R$)",
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------------------
# Distribuição final + probabilidades
# --------------------------------------------------------------------------------------
colL, colR = st.columns([2, 1])

with colL:
    summary = paths_summary(equity, interval=interval)
    hist = go.Figure()
    hist.add_trace(go.Histogram(x=summary["terminal"], nbinsx=40))
    hist.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        title="Distribuição do patrimônio ao final do horizonte",
        xaxis_title="Equity final (R$)",
        yaxis_title="Cenários",
    )
    st.plotly_chart(hist, use_container_width=True)

with colR:
    st.markdown("### Probabilidades")
    tgt = st.number_input("Meta anual (CAGR)", min_value=-50.0, max_value=200.0, value=15.0, step=1.0) / 100.0
    dd_lim = st.slider("Drawdown limite (%)", -80.0, -1.0, -30.0, step=1.0) / 100.0
    probs = probability_targets(equity, interval=interval, target_cagr=tgt, dd_thresh=dd_lim)
    c1, c2 = st.columns(2)
    c1.metric("P(atingir meta)", f"{probs['p_target']*100:,.1f}%")
    c2.metric("P(DD ≤ limite)", f"{probs['p_dd']*100:,.1f}%")

st.divider()

# --------------------------------------------------------------------------------------
# KPIs + export
# --------------------------------------------------------------------------------------
summ_all = paths_summary(equity, interval=interval)
med_cagr = np.nanmedian(summ_all["cagr"]) * 100
med_dd   = np.nanmedian(summ_all["maxdd"]) * 100
p05_final = float(np.percentile(equity[-1, :], 5))
p50_final = float(np.percentile(equity[-1, :], 50))
p95_final = float(np.percentile(equity[-1, :], 95))

k1, k2, k3 = st.columns(3)
k1.metric("CAGR mediano", f"{med_cagr:,.2f}%")
k2.metric("MaxDD mediano", f"{med_dd:,.2f}%")
k3.metric("Equity final (P5 / P50 / P95)", f"R$ {p05_final:,.0f} / {p50_final:,.0f} / {p95_final:,.0f}")

# Exportar percentis do fan chart
csv = fan.assign(step=np.arange(fan.shape[0]))
st.download_button(
    "⬇️ Exportar percentis (CSV)",
    data=csv.to_csv(index=False).encode("utf-8"),
    file_name=f"mc_fanchart_{h_months}m_{n_sims}sim.csv",
    mime="text/csv",
)
