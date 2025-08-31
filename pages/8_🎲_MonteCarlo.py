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
# 🔽 RESUMO DETALHADO — Monte Carlo (plug-and-play, seguro)  [v2]
# Cole no FINAL da página de Monte Carlo
# ============================================================
import io
import numpy as np
import pandas as pd
import streamlit as st

# ---------- pegar variáveis já existentes, com fallback seguro ----------
def _g(name, default=None):
    return globals().get(name, default)

# ⚠️ Captura de `equity` sem usar "or" (evita ValueError do NumPy)
equity_obj = _g("equity", None)
if equity_obj is None:
    res_obj = _g("res", None)
    if isinstance(res_obj, dict):
        equity_obj = res_obj.get("equity", None)

if isinstance(equity_obj, pd.DataFrame):
    EQ = equity_obj.to_numpy()
elif isinstance(equity_obj, np.ndarray):
    EQ = equity_obj
else:
    EQ = None

if EQ is None or EQ.ndim != 2 or EQ.shape[1] < 1:
    st.warning("Resumo Monte Carlo: não encontrei `equity` (matriz T×S).")
else:
    # ---------------- parâmetros básicos ----------------
    interval = str(_g("interval", "1d")).strip().lower()
    steps_total = EQ.shape[0] - 1
    sims = int(EQ.shape[1])
    steps_year = 252 if interval == "1d" else 52
    steps_month = 21 if interval == "1d" else 4
    months = _g("h_months", round(steps_total / max(steps_month, 1)))

    init_cap = _g("init_cap", None)
    if init_cap is None or not np.isfinite(init_cap):
        init_cap = float(np.nanmedian(EQ[0, :]))

    # ---------------- métricas por cenário ----------------
    final = EQ[-1, :].astype(float)                       # equity final por sim
    base  = EQ[0, :].astype(float)                        # equity inicial por sim
    mult  = final / np.maximum(base, 1e-12)               # multiplicador total
    cagr  = np.power(mult, steps_year / max(steps_total, 1)) - 1.0

    # Max Drawdown por caminho (vetorizado)
    roll_max = np.maximum.accumulate(EQ, axis=0)
    dd = EQ / np.maximum(roll_max, 1e-12) - 1.0           # negativo
    maxdd = dd.min(axis=0)

    # ---------------- percentis & estatísticas ----------------
    def _pctile(a, q): 
        try: return float(np.percentile(a, q))
        except Exception: return np.nan

    P5, P25, P50, P75, P95 = [_pctile(final, q) for q in (5, 25, 50, 75, 95)]
    cagr_med = float(np.nanmedian(cagr)) * 100
    dd_med   = float(np.nanmedian(maxdd)) * 100
    prob_loss = float(np.mean(final < base)) if np.all(np.isfinite(base)) else np.nan

    # metas do usuário (se existirem)
    tgt    = _g("tgt", np.nan)      # ex.: 0.15 (15% a.a.)
    dd_lim = _g("dd_lim", np.nan)   # ex.: -0.30 (-30%)
    p_target = float(np.mean(cagr >= float(tgt)))    if np.isfinite(tgt)    else np.nan
    p_dd_ok  = float(np.mean(maxdd >= float(dd_lim))) if np.isfinite(dd_lim) else np.nan

    # Expected Shortfall (média dos 5% piores finais)
    k = max(1, int(0.05 * sims))
    es_5 = float(np.sort(final)[:k].mean()) if sims >= 20 else np.nan

    # ---------------- parâmetros operacionais (opcionais) ----------------
    rebalance_label   = _g("rebalance_label", None)
    aporte_mensal     = _g("aporte_mensal", None)
    saque_mensal      = _g("saque_mensal", None)
    mgmt_fee_annual   = _g("mgmt_fee_annual", None)
    perf_fee_pct      = _g("perf_fee_pct", None)
    hurdle_annual     = _g("hurdle_annual", None)
    n_sims            = int(_g("n_sims", sims))

    # ---------------- helpers de formatação ----------------
    def _fmt_money(x):
        try:
            s = f"R$ {float(x):,.0f}"
            return s.replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return "—"

    def _fmt_pct(x):
        try:
            return f"{float(x)*100:,.1f}%"
        except Exception:
            return "—"

    def _fmt_pct_abs(x):
        try:
            return f"{float(x):,.2f}%"
        except Exception:
            return "—"

    # ---------------- CARD grande ----------------
    card = f"""
    <div style="border-radius:14px;padding:14px 18px;margin:12px 0;
                background:linear-gradient(135deg,#0B3,#095);color:#fff">
      <div style="font-weight:700;font-size:18px;margin-bottom:6px">
        Resumo da Simulação — {n_sims:,} cenários • horizonte {months} meses • intervalo {interval.upper()}
      </div>
      <div style="display:flex;gap:24px;flex-wrap:wrap;font-size:15px">
        <div><b>Equity final (P5 / P50 / P95)</b><br>
            <span style="font-size:20px">{_fmt_money(P5)}</span> /
            <span style="font-size:20px">{_fmt_money(P50)}</span> /
            <span style="font-size:20px">{_fmt_money(P95)}</span></div>
        <div><b>CAGR mediano</b><br><span style="font-size:20px">{_fmt_pct_abs(cagr_med)}</span></div>
        <div><b>MaxDD mediano</b><br><span style="font-size:20px">{_fmt_pct_abs(dd_med)}</span></div>
        <div><b>P(perder dinheiro)</b><br><span style="font-size:20px">{_fmt_pct(prob_loss)}</span></div>
        <div><b>ES (5% piores)</b><br><span style="font-size:20px">{_fmt_money(es_5)}</span></div>
      </div>
    </div>
    """
    st.markdown(card, unsafe_allow_html=True)

    # ---------------- Tabela-resumo ----------------
    rows = [
        ["Meta (CAGR)",   "—" if not np.isfinite(tgt)    else _fmt_pct(tgt),   "P(atingir meta)", "—" if not np.isfinite(p_target) else _fmt_pct(p_target)],
        ["Limite de DD",  "—" if not np.isfinite(dd_lim) else _fmt_pct(dd_lim),"P(DD ≤ limite)",  "—" if not np.isfinite(p_dd_ok)   else _fmt_pct(p_dd_ok)],
        ["Prob. perder",  "",                                                   "P(final < início)", _fmt_pct(prob_loss)],
        ["Terminal (P5)", _fmt_money(P5),                                       "Terminal (P50)",    _fmt_money(P50)],
        ["Terminal (P95)",_fmt_money(P95),                                      "ES 5% (média piores)", _fmt_money(es_5)],
        ["CAGR mediano",  _fmt_pct_abs(cagr_med),                               "MaxDD mediano",     _fmt_pct_abs(dd_med)],
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Métrica A","Valor A","Métrica B","Valor B"]),
                 use_container_width=True, hide_index=True)

    # ---------------- Observações operacionais ----------------
    bullets = []
    if rebalance_label: bullets.append(f"Rebalance: **{rebalance_label}**.")
    if isinstance(aporte_mensal, (int, float)) and aporte_mensal: bullets.append(f"Aporte mensal: **R$ {aporte_mensal:,.0f}**.")
    if isinstance(saque_mensal, (int, float)) and saque_mensal: bullets.append(f"Saque mensal: **R$ {saque_mensal:,.0f}**.")
    if isinstance(mgmt_fee_annual, (int, float)) and mgmt_fee_annual: bullets.append(f"Taxa adm.: **{mgmt_fee_annual:.2f}% a.a.**.")
    if isinstance(perf_fee_pct, (int, float)) and perf_fee_pct:
        if isinstance(hurdle_annual, (int, float)) and hurdle_annual:
            bullets.append(f"Taxa de performance: **{perf_fee_pct:.0f}%** (hurdle **{hurdle_annual:.2f}% a.a.**).")
        else:
            bullets.append(f"Taxa de performance: **{perf_fee_pct:.0f}%**.")
    if bullets:
        st.markdown("**Parâmetros considerados**  \n- " + "\n- ".join(bullets))

    # ---------------- Exportar relatório (Markdown) ----------------
    md = io.StringIO()
    md.write(f"# Resumo Monte Carlo\n\n")
    md.write(f"- Cenários: **{n_sims:,}**\n")
    md.write(f"- Horizonte: **{months} meses** (intervalo {interval.upper()})\n")
    md.write(f"- Capital inicial (amostra): **{_fmt_money(init_cap)}**\n\n")
    md.write(f"## Resultados\n")
    md.write(f"- Equity final (P5 / P50 / P95): **{_fmt_money(P5)} / {_fmt_money(P50)} / {_fmt_money(P95)}**\n")
    md.write(f"- CAGR mediano: **{_fmt_pct_abs(cagr_med)}** • MaxDD mediano: **{_fmt_pct_abs(dd_med)}**\n")
    md.write(f"- P(perder dinheiro): **{_fmt_pct(prob_loss)}** • ES (5% piores): **{_fmt_money(es_5)}**\n")
    if np.isfinite(tgt):    md.write(f"- Meta (CAGR): **{_fmt_pct(tgt)}** → P(atingir): **{_fmt_pct(p_target)}**\n")
    if np.isfinite(dd_lim): md.write(f"- Limite DD: **{_fmt_pct(dd_lim)}** → P(DD ≤ limite): **{_fmt_pct(p_dd_ok)}**\n")
    if bullets:
        md.write(f"\n## Parâmetros\n")
        for b in bullets: md.write(f"- {b}\n")

    st.download_button(
        "⬇️ Baixar resumo (Markdown)",
        data=md.getvalue().encode("utf-8"),
        file_name="resumo_monte_carlo.md",
        mime="text/markdown",
        use_container_width=True,
    )






