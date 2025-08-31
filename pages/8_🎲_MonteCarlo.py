# pages/8_🎲_MonteCarlo.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
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
# Helpers
# --------------------------------------------------------------------------------------
def _get_close_series(df: pd.DataFrame) -> pd.Series | None:
    """
    Extrai uma Series 1-D de preços 'Close' de diferentes formatos retornados por provedores.
    Trata:
      - DataFrame simples com coluna "Close"
      - DataFrame com MultiIndex nas colunas (ex: ('Close', 'AAPL'))
      - Coluna "Close" que veio como DataFrame (ex: colunas duplicadas): pega a primeira série válida
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    # Caso 1: colunas simples
    if "Close" in df.columns and not isinstance(df["Close"], pd.DataFrame):
        sr = pd.to_numeric(df["Close"], errors="coerce")
        return sr.dropna()

    # Caso 2: "Close" existe mas virou DataFrame (colunas duplicadas)
    if "Close" in df.columns and isinstance(df["Close"], pd.DataFrame):
        # pega a primeira coluna válida
        sub = df["Close"]
        for col in sub.columns:
            sr = pd.to_numeric(sub[col], errors="coerce").dropna()
            if not sr.empty:
                sr.name = "Close"
                return sr

    # Caso 3: MultiIndex nas colunas (mutações recentes do yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        # tenta usar o nível 0 como tipo de preço
        try:
            if "Close" in df.columns.get_level_values(0):
                sub = df.xs("Close", axis=1, level=0, drop_level=False)
                # se ainda for 2-D, pega a primeira coluna
                if isinstance(sub, pd.DataFrame):
                    first = sub.columns[0]
                    sr = pd.to_numeric(sub[first], errors="coerce").dropna()
                    sr.name = "Close"
                    return sr
        except Exception:
            pass

        # fallback: se houver "Adj Close" preferir ajustado
        try:
            if "Adj Close" in df.columns.get_level_values(0):
                sub = df.xs("Adj Close", axis=1, level=0, drop_level=False)
                if isinstance(sub, pd.DataFrame):
                    first = sub.columns[0]
                    sr = pd.to_numeric(sub[first], errors="coerce").dropna()
                    sr.name = "Adj Close"
                    return sr
        except Exception:
            pass

    return None

# --------------------------------------------------------------------------------------
# Cache de retornos (robusto a MultiIndex e entradas inválidas)
# --------------------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def _cached_returns(symbols_tuple: tuple[str, ...], period: str, interval: str) -> pd.DataFrame:
    """
    Baixa preços em bloco via `download_bulk`, extrai 'Close' como Series 1-D,
    calcula retornos e concatena por colunas (inner join).
    """
    data = download_bulk(list(symbols_tuple), period=period, interval=interval)

    series_list: list[pd.Series] = []
    bad = []

    for s, df in (data or {}).items():
        try:
            sr = _get_close_series(df)
            if sr is None or sr.dropna().empty:
                bad.append(s)
                continue
            sr = pd.to_numeric(sr, errors="coerce").dropna()
            if sr.empty:
                bad.append(s)
                continue
            sr = sr.astype(float).pct_change().dropna()
            if sr.empty:
                bad.append(s)
                continue
            sr.name = str(s)
            series_list.append(sr)
        except Exception:
            bad.append(s)

    if not series_list:
        return pd.DataFrame()

    # Concatena alinhando por índice e exigindo interseção (dados comuns)
    rets_df = pd.concat(series_list, axis=1, join="inner").dropna(how="any")
    rets_df = rets_df.loc[:, ~rets_df.columns.duplicated()].copy()
    rets_df = rets_df.astype(float)

    # Adiciona diagnóstico ao cache (opcional)
    if bad:
        st.caption(f"⚠️ Sem retornos válidos para: {', '.join(map(str, bad))}")

    return rets_df

# --------------------------------------------------------------------------------------
# Seleção de ativos
# --------------------------------------------------------------------------------------
watch = load_watchlists()
all_syms = sorted(set(watch.get("BR_STOCKS", []) + watch.get("US_STOCKS", []) + watch.get("CRYPTO", [])))

# Aceita ambas as chaves usadas nas outras páginas (uniformiza o app)
_sel1 = st.session_state.get("screener_selected")
_sel2 = st.session_state.get("screener_selection")
sel_from_screener = [str(x) for x in ((_sel1 or []) or (_sel2 or [])) if isinstance(x, (str, bytes))]
sel_from_screener = sorted(set(sel_from_screener))  # de-dup

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
    st.error("Retornos insuficientes para simular (tente outro período/intervalo ou ajuste a lista de ativos).")
    st.stop()

# --------------------------------------------------------------------------------------
# Pesos do portfólio
# --------------------------------------------------------------------------------------
st.subheader("Pesos do portfólio")
choice = st.radio("Modo de pesos", ["Equal-Weight", "Máx. Sharpe (via Monte Carlo)", "Personalizado"], horizontal=True)

if choice == "Equal-Weight":
    weights = np.ones(len(rets_df.columns)) / len(rets_df.columns)
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
    cols = st.columns(min(6, len(rets_df.columns)))
    for i, sym in enumerate(rets_df.columns):
        sliders.append(cols[i % len(cols)].slider(f"{sym}", 0.0, 1.0, 1.0/len(rets_df.columns), step=0.01))
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

#================================================================================================================================================================================================
#================================================================================================================================================================================================
#================================================================================================================================================================================================
# ============================================================
# 🔽 RESUMO DETALHADO — Monte Carlo (plug-and-play)  [v2]
# ============================================================
import io
import numpy as np
import pandas as pd
import streamlit as st

def _g(name, default=None):
    return globals().get(name, default)

# (⚠️ fix) capturar equity sem usar "or" com np.ndarray
equity_obj = _g("equity", None)
if equity_obj is None:
    res_obj = _g("res", None)
    if isinstance(res_obj, dict):
        equity_obj = res_obj.get("equity", None)

# segue igual…
if isinstance(equity_obj, pd.DataFrame):
    EQ = equity_obj.to_numpy()
elif isinstance(equity_obj, np.ndarray):
    EQ = equity_obj
else:
    EQ = None
