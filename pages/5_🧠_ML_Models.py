# pages/5_🤖_ML_Models.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.ui import app_header
from core.data import load_watchlists, download_bulk
from core.ml import FeatureConfig, ModelConfig, train_and_eval

st.set_page_config(page_title="ML Models", page_icon="🤖", layout="wide")
app_header("🤖 ML Models", "Classificação de próxima barra (subida/queda) com Logistic/RF")

# ---------------- Seleção de ativos ----------------
watch = load_watchlists()
all_syms = sorted(set(watch["BR_STOCKS"] + watch["US_STOCKS"] + watch["CRYPTO"]))

sel_from_screener = st.session_state.get("screener_selection", [])
use_sel = st.toggle("Usar seleção do Screener (se houver)", value=bool(sel_from_screener))
sel_count = len(sel_from_screener)

default_list = [s for s in ["AAPL", "MSFT", "NVDA", "PETR4.SA"] if s in all_syms] or all_syms[:4]

if use_sel and sel_count >= 1:
    symbols = sel_from_screener
    preview = ", ".join(symbols[:10]) + ("…" if len(symbols) > 10 else "")
    st.caption(f"Usando {len(symbols)} ativos do Screener: {preview}")
else:
    symbols = st.multiselect("Ativos para treinar", options=all_syms, default=default_list)

c1, c2, c3 = st.columns(3)
period   = c1.selectbox("Período", ["6mo","1y","2y","5y"], index=1)
interval = c2.selectbox("Intervalo", ["1d","1wk"], index=0)
dark     = c3.toggle("Tema escuro", value=True)

# ---------------- Config de Features ----------------
st.markdown("### Features")
f1, f2, f3, f4 = st.columns(4)
rsi_len  = f1.slider("RSI length", 3, 30, 14, step=1)
sma_fast = f2.slider("SMA rápida", 5, 50, 10, step=1)
sma_slow = f3.slider("SMA lenta", 10, 200, 30, step=1)
n_lags   = f4.slider("Lags de retorno", 1, 10, 5, step=1)
vol_win  = st.slider("Janela de volatilidade", 10, 60, 20, step=1, help="Desvio-padrão dos retornos")

fcfg = FeatureConfig(rsi_len=rsi_len, sma_fast=sma_fast, sma_slow=sma_slow, n_lags=n_lags, vol_win=vol_win)

# ---------------- Config de Modelo ----------------
st.markdown("### Modelo")
mcol1, mcol2, mcol3 = st.columns(3)
model_kind = mcol1.radio("Tipo", ["RandomForest", "Logistic"], horizontal=True)
prob_th = mcol2.slider("Limiar de compra (prob. de alta)", 0.50, 0.80, 0.55, step=0.01)

if model_kind == "RandomForest":
    n_estimators = mcol3.slider("n_estimators (RF)", 100, 800, 300, step=50)
    max_depth = st.slider("max_depth (RF)", 0, 30, 0, step=1, help="0 = None (sem limite)")
    mcfg = ModelConfig(kind="RandomForest", n_estimators=int(n_estimators), max_depth=(None if max_depth==0 else int(max_depth)))
else:
    C = mcol3.slider("C (Logistic)", 0.1, 5.0, 1.0, step=0.1)
    mcfg = ModelConfig(kind="Logistic", C=float(C))

# ---------------- Treino ----------------
if not symbols:
    st.warning("Selecione ao menos 1 ativo.")
    st.stop()

@st.cache_data(ttl=600)
def _bulk(period, interval, symbols_tuple):
    return download_bulk(list(symbols_tuple), period=period, interval=interval)

with st.spinner("🔄 Baixando dados e treinando modelos..."):
    data = _bulk(period, interval, tuple(symbols))

rows = []
details = {}
for sym, df in data.items():
    if df is None or df.empty or "Close" not in df.columns:
        rows.append(dict(Symbol=sym, AUC=np.nan, F1=np.nan, Acc=np.nan, ProbUp=np.nan, Signal="-", Obs="sem dados"))
        continue
    try:
        out = train_and_eval(df, fcfg, mcfg, prob_threshold=prob_th)
        if "error" in out:
            rows.append(dict(Symbol=sym, AUC=np.nan, F1=np.nan, Acc=np.nan, ProbUp=np.nan, Signal="-", Obs=out["error"]))
        else:
            rows.append(dict(
                Symbol=sym,
                AUC=out["auc"],
                F1=out["f1"],
                Acc=out["acc"],
                ProbUp=out["proba_last"],
                Signal=("BUY" if out["signal"]==1 else "HOLD")
            ))
            details[sym] = out
    except Exception as e:
        rows.append(dict(Symbol=sym, AUC=np.nan, F1=np.nan, Acc=np.nan, ProbUp=np.nan, Signal="-", Obs=str(e)))

tbl = pd.DataFrame(rows).sort_values("ProbUp", ascending=False)

st.markdown("### Resultados por ativo")
st.dataframe(
    tbl.style.format({"AUC":"{:.3f}","F1":"{:.3f}","Acc":"{:.3f}","ProbUp":"{:.3f}"}),
    use_container_width=True, height=360
)

# ---------------- Enviar para Screener ----------------
st.markdown("### Ações")
left, right = st.columns([2,1])
topN = int(left.slider("Enviar Top-N (por ProbUp ≥ limiar)", 1, max(1, len(tbl)), min(5, len(tbl))))
if right.button("➡️ Enviar para Screener"):
    selected = tbl.loc[(tbl["ProbUp"] >= prob_th), "Symbol"].head(topN).tolist()
    st.session_state["screener_selection"] = selected
    st.success(f"Enviado! {len(selected)} símbolos setados no Screener: {', '.join(selected)}")

st.divider()

# ---------------- Detalhes por ativo ----------------
st.markdown("### Importância das features (por ativo)")
for sym in tbl["Symbol"].tolist():
    if sym not in details:
        continue
    imp = details[sym]["importances"]
    fig = go.Figure(go.Bar(x=imp.values[::-1], y=imp.index[::-1], orientation="h"))
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        height=320, title=f"{sym} — importâncias / coeficientes",
        margin=dict(l=10,r=10,t=40,b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================== RESUMO EXECUTIVO (CARD ÚNICO) ====================
st.divider()

# agregados
valid = tbl.dropna(subset=["ProbUp"])
n_total = len(tbl)
n_valid = len(valid)
n_buy = int((valid["ProbUp"] >= prob_th).sum())

avg_auc = valid["AUC"].mean()
avg_f1  = valid["F1"].mean()
avg_acc = valid["Acc"].mean()

# top candidatos (acima do limiar), máx 5
top_candidates = (
    valid.loc[valid["ProbUp"] >= prob_th]
         .nlargest(5, "ProbUp")[["Symbol", "ProbUp"]]
)
recs = ", ".join([f"{r.Symbol} ({r.ProbUp:.0%})" for r in top_candidates.itertuples()]) or "—"

# helpers p/ formatação segura
def f3(x): return "—" if pd.isna(x) else f"{x:.3f}"

# tema dinâmico (combina com o toggle 'dark')
bg1   = "#0b1220" if dark else "#f7fafc"
bg2   = "#121826" if dark else "#ffffff"
text  = "#e8edf7" if dark else "#1a202c"
muted = "#a3adc2" if dark else "#4a5568"
border= "#1f2a44" if dark else "#e2e8f0"
badge_bg   = "#0b1729" if dark else "#edf2f7"
badge_text = "#9cc9ff" if dark else "#2b6cb0"

st.markdown(f"""
<style>
.ml-card{{background:linear-gradient(135deg,{bg1},{bg2});border:1px solid {border};
         border-radius:14px;padding:16px;}}
.ml-card h3{{margin:0 0 8px;color:{text};}}
.ml-card p{{margin:6px 0;color:{muted};}}
.ml-badges{{display:flex;gap:10px;flex-wrap:wrap;margin-top:6px}}
.ml-badge{{padding:4px 10px;border-radius:999px;border:1px solid {border};
          color:{badge_text};background:{badge_bg};font-size:0.85rem}}
</style>

<div class="ml-card">
  <h3>Resumo & Recomendações</h3>
  <div class="ml-badges">
    <span class="ml-badge">Ativos: {n_total}</span>
    <span class="ml-badge">Modelos válidos: {n_valid}</span>
    <span class="ml-badge">Limiar: {prob_th:.2f}</span>
    <span class="ml-badge">Sinais BUY: {n_buy}</span>
  </div>
  <p>Métricas médias (válidos): AUC <b>{f3(avg_auc)}</b> • F1 <b>{f3(avg_f1)}</b> • Acc <b>{f3(avg_acc)}</b></p>
  <p><b>Recomendação:</b> priorize os ativos com probabilidade ≥ limiar.</p>
  <p><b>Top candidatos</b>: {recs}</p>
  <p style="font-size:0.85rem;opacity:0.8;">Observação: previsões para a próxima barra; ajuste o limiar para equilibrar
     precisão vs. cobertura. Considere confirmar com o Screener e Backtest.</p>
</div>
""", unsafe_allow_html=True)

