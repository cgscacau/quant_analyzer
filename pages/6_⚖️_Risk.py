# pages/6_⚖️_Risk.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from core.ui import app_header
from core.data import load_watchlists, download_bulk, download_history
from core.risk import var_es_hist, ann_vol, atr, position_size, kelly_fraction

st.set_page_config(page_title="Risk", page_icon="⚖️", layout="wide")
app_header("⚖️ Risk", "VaR/ES, correlação, sizing por risco e Kelly")

# ======================
# Seleção de ativos
# ======================
watch = load_watchlists()
all_syms = sorted(set(watch["BR_STOCKS"] + watch["US_STOCKS"] + watch["CRYPTO"]))

sel_from_screener = st.session_state.get("screener_selection", [])
use_sel = st.toggle("Usar seleção do Screener (se houver)", value=bool(sel_from_screener))
symbols = sel_from_screener if use_sel and sel_from_screener else st.multiselect(
    "Ativos para análise de risco",
    options=all_syms,
    default=["AAPL"] if "AAPL" in all_syms else all_syms[:1]
)

c1, c2, c3 = st.columns(3)
period   = c1.selectbox("Período", ["6mo","1y","2y","5y"], index=1)
interval = c2.selectbox("Intervalo", ["1d","1wk"], index=0)
alpha    = c3.selectbox("Confiança VaR/ES", ["95%","99%"], index=0)
alpha_val = 0.95 if alpha=="95%" else 0.99

if not symbols:
    st.warning("Nenhum ativo selecionado.")
    st.stop()

# ======================
# Download e métricas por ativo
# ======================
bulk = download_bulk(symbols, period=period, interval=interval)

rows = []
retn_dict = {}
for sym, df in bulk.items():
    if df.empty or "Close" not in df.columns:
        rows.append(dict(Symbol=sym, Price=np.nan, VolAnnPct=np.nan, VaR=np.nan, ES=np.nan, MaxDD=np.nan))
        continue

    close = df["Close"].astype(float)
    ret = close.pct_change().dropna()
    retn_dict[sym] = ret

    vol_ann = ann_vol(ret, freq=interval) * 100.0
    var1, es1 = var_es_hist(ret, alpha=alpha_val, horizon_days=1)  # 1 dia
    eq = (1 + ret).cumprod()
    maxdd = (eq/eq.cummax()-1).min()*100.0

    rows.append(dict(
        Symbol=sym,
        Price=float(close.iloc[-1]),
        VolAnnPct=float(vol_ann),
        VaR=float(var1*100.0),   # em %
        ES=float(es1*100.0),     # em %
        MaxDD=float(maxdd)
    ))

riskdf = pd.DataFrame(rows).sort_values("VolAnnPct", ascending=False)

# Exibe tabela
st.markdown("### Risco por ativo (1 dia, histórico)")
st.dataframe(
    riskdf.style.format({
        "Price":"{:,.2f}",
        "VolAnnPct":"{:.1f}%",
        "VaR":"{:.2f}%",
        "ES":"{:.2f}%",
        "MaxDD":"{:.2f}%"
    }),
    use_container_width=True,
    height=420
)
st.download_button(
    "⬇️ Exportar tabela de risco (CSV)",
    data=riskdf.to_csv(index=False).encode("utf-8"),
    file_name=f"risk_{period}_{interval}.csv",
    mime="text/csv"
)

st.divider()

# ======================
# Correlação entre ativos
# ======================
if len(retn_dict) >= 2:
    rets = pd.DataFrame(retn_dict).dropna(how="all")
    corr = rets.corr().fillna(0.0)

    heat = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        zmin=-1, zmax=1, colorscale="RdBu"
    ))
    heat.update_layout(
        title="Correlação (retornos)",
        template="plotly_white",
        margin=dict(l=10,r=10,t=40,b=10),
        yaxis_autorange="reversed"
    )
    st.plotly_chart(heat, use_container_width=True)

st.divider()

# ======================
# Sizing por risco (ATR ou %)
# ======================
st.markdown("## Position sizing (por risco)")

col_left, col_right = st.columns([1,1])

with col_left:
    sym_for_size = st.selectbox("Ativo (sizing)", symbols, index=0)
    df_sz = download_history(sym_for_size, period=period, interval=interval)
    if df_sz.empty or "Close" not in df_sz.columns:
        st.warning("Sem dados para o sizing.")
    else:
        last_close = float(df_sz["Close"].iloc[-1])
        st.metric("Preço atual (Close)", f"{last_close:,.2f}")

        mode = st.radio("Base do stop", ["ATR múltiplos","Percentual"], horizontal=True)
        if mode == "ATR múltiplos":
            l1, l2 = st.columns(2)
            atr_len = l1.slider("ATR length", 5, 50, 14, step=1)
            atr_mult = l2.slider("ATR múltiplos", 0.5, 5.0, 2.0, step=0.1)
            series_atr = atr(df_sz, atr_len)
            stop_dist = float(series_atr.iloc[-1] * atr_mult) if not series_atr.empty else last_close*0.05
        else:
            pct = st.slider("Stop (%)", 0.5, 20.0, 5.0, step=0.5)
            stop_dist = last_close * (pct/100.0)

        stop_price = last_close - stop_dist
        st.caption(f"Stop calculado: **{stop_price:,.2f}** (distância {stop_dist:,.2f})")

        eq_val = st.number_input("Capital (R$)", min_value=100.0, value=100000.0, step=1000.0)
        risk_pct = st.slider("Risco por trade (%)", 0.1, 5.0, 1.0, step=0.1)
        lot_step = st.number_input("Passo de lote (ex.: 1 ação; cripto pode 0.0001)", min_value=0.0001, value=1.0, step=0.0001, format="%.4f")

        sz = position_size(eq_val, last_close, stop_price, risk_pct=risk_pct, lot_step=lot_step)
        st.session_state["risk_sizing_summary"] = {
            "asset": sym_for_size,
            "qty": sz["qty"],
            "notional": sz["notional"],
            "expected_loss": sz["expected_loss"],
            "risk_pct": risk_pct,
            "stop_price": stop_price
        }

        c1, c2, c3 = st.columns(3)
        c1.metric("Quantidade", f"{sz['qty']:,.4f}")
        c2.metric("Notional (R$)", f"{sz['notional']:,.2f}")
        c3.metric("Perda esperada", f"{sz['expected_loss']:,.2f}")

with col_right:
    st.markdown("## Kelly (fracionado)")
    st.caption("Informe **Win Rate (%)** e **Payoff** (ganho médio / perda média).")
    w = st.slider("Win Rate (%)", 10.0, 90.0, 50.0, step=1.0)
    b = st.slider("Payoff Ratio (b = ganho/ perda)", 0.2, 5.0, 1.5, step=0.1)
    k = kelly_fraction(w, b)
    c1, c2, c3 = st.columns(3)
    c1.metric("Kelly*", f"{k*100:,.2f}%")
    c2.metric("½ Kelly", f"{k*50:,.2f}%")
    c3.metric("¼ Kelly", f"{k*25:,.2f}%")
    st.caption("Regra prática: operar **½ Kelly** costuma ser mais estável.")

# ======================
# 📊 Resumo executivo (moderno)
# ======================
st.divider()
st.markdown("## 📊 Resumo executivo")

# --- prepara métricas de portfólio (equal-weight) e correlação ---
rets_df = pd.DataFrame(retn_dict).dropna()
has_port = not rets_df.empty
if has_port:
    # equal-weight simples
    port_ret = rets_df.mean(axis=1)
    port_var, port_es = var_es_hist(port_ret, alpha=alpha_val, horizon_days=1)
    port_vol = ann_vol(port_ret, freq=interval) * 100.0
else:
    port_var = port_es = port_vol = np.nan

avg_corr = np.nan
min_pair = ("-", "-", np.nan)
max_pair = ("-", "-", np.nan)
if has_port and rets_df.shape[1] >= 2:
    corr_mx = rets_df.corr().values
    names = rets_df.columns.tolist()
    pairs = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            pairs.append((names[i], names[j], float(corr_mx[i, j])))
    if pairs:
        avg_corr = float(np.mean([p[2] for p in pairs]))
        min_pair = min(pairs, key=lambda x: x[2])
        max_pair = max(pairs, key=lambda x: x[2])

# piores/maiores por ativo a partir da tabela de risco
riskdf_nonempty = not riskdf.empty
if riskdf_nonempty:
    worst_var_row = riskdf.loc[riskdf["VaR"].idxmin()]   # mais negativo
    worst_dd_row  = riskdf.loc[riskdf["MaxDD"].idxmin()] # mais negativo
    hi_vol_row    = riskdf.loc[riskdf["VolAnnPct"].idxmax()]
else:
    worst_var_row = worst_dd_row = hi_vol_row = pd.Series(dtype=float)

# --- estilo moderno (cards) ---
st.markdown("""
<style>
.risk-grid{display:grid; grid-template-columns:repeat(5, minmax(0,1fr)); gap:12px;}
.risk-card{background:linear-gradient(135deg,#0b1220,#121826); border:1px solid #1f2a44; border-radius:14px; padding:14px;}
.risk-card h4{margin:0 0 6px 0; font-size:0.90rem; color:#a3adc2;}
.risk-card .v{font-size:1.35rem; font-weight:800; color:#e8edf7;}
.badge{display:inline-block; margin-top:6px; padding:2px 10px; border-radius:999px; font-size:.75rem;
       border:1px solid #1f2a44; color:#9cc9ff; background:#0b1729;}
@media (max-width:1200px){ .risk-grid{grid-template-columns:repeat(2, minmax(0,1fr));} }
</style>
""", unsafe_allow_html=True)

def fmt_pct(x, digits=2):
    return "—" if x is None or np.isnan(x) else f"{x:.{digits}f}%"

def card(label, value, badge=None):
    html = f'<div class="risk-card"><h4>{label}</h4><div class="v">{value}</div>'
    if badge: html += f'<div class="badge">{badge}</div>'
    html += "</div>"
    return html

n_assets = len(retn_dict)
tail_ratio = (abs(port_es)/abs(port_var)) if (has_port and np.isfinite(port_var) and port_var != 0) else np.nan
div_tag = "baixa" if np.isfinite(avg_corr) and avg_corr < 0.25 else ("média" if np.isfinite(avg_corr) and avg_corr < 0.6 else "alta")

cards1 = ""
cards1 += card("Ativos analisados", f"{n_assets}", "equal-weight")
cards1 += card("VaR 1D (portfólio)", fmt_pct(port_var*100), f"α={int(alpha_val*100)}%")
cards1 += card("ES 1D (portfólio)", fmt_pct(port_es*100), "perda média na cauda")
cards1 += card("Vol anualizada (portfólio)", fmt_pct(port_vol, 1))
cards1 += card("Correlação média", "—" if np.isnan(avg_corr) else f"{avg_corr:.2f}", f"Diversificação {div_tag}")

cards2 = ""
if riskdf_nonempty:
    cards2 += card("Mais volátil", f"{hi_vol_row['Symbol']}", fmt_pct(hi_vol_row['VolAnnPct'],1))
    cards2 += card("Pior VaR (1D)", f"{worst_var_row['Symbol']}", fmt_pct(worst_var_row['VaR']))
    cards2 += card("Maior MaxDD", f"{worst_dd_row['Symbol']}", fmt_pct(worst_dd_row['MaxDD']))
else:
    cards2 += card("Mais volátil", "—")
    cards2 += card("Pior VaR (1D)", "—")
    cards2 += card("Maior MaxDD", "—")
# pares de correlação
mp = min_pair; xp = max_pair
cards2 += card("Menor correlação (par)", f"{mp[0]} × {mp[1]}", "—" if np.isnan(mp[2]) else f"{mp[2]:.2f}")
cards2 += card("Maior correlação (par)", f"{xp[0]} × {xp[1]}", "—" if np.isnan(xp[2]) else f"{xp[2]:.2f}")

# sizing (se calculado)
sz = st.session_state.get("risk_sizing_summary")
cards3 = ""
if sz:
    cards3 += card("Sizing selecionado", sz["asset"], f"qty {sz['qty']:,.4f}")
    cards3 += card("Notional", f"R$ {sz['notional']:,.2f}")
    cards3 += card("Perda esperada", f"R$ {sz['expected_loss']:,.2f}", f"{sz['risk_pct']:.1f}% do capital")
    cards3 += card("Stop", f"R$ {sz['stop_price']:,.2f}")
else:
    cards3 += card("Sizing selecionado", "—", "calcule acima")
    cards3 += card("Notional", "—")
    cards3 += card("Perda esperada", "—")
    cards3 += card("Stop", "—")

st.markdown(f'<div class="risk-grid">{cards1}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="risk-grid" style="margin-top:10px">{cards2}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="risk-grid" style="margin-top:10px">{cards3}</div>', unsafe_allow_html=True)

# --- narrativa em bullets ---
st.markdown("### 🧭 Narrativa")
bullets = []
if has_port:
    bullets.append(f"- Com **{n_assets}** ativos, o *equal-weight* tem **VaR(1D)** de **{fmt_pct(port_var*100)}** e **ES(1D)** de **{fmt_pct(port_es*100)}** (α={int(alpha_val*100)}%).")
    if np.isfinite(tail_ratio):
        bullets.append(f"- A **cauda** está {('mais pesada' if tail_ratio>1.3 else 'moderada' if tail_ratio>1.1 else 'leve')} (|ES|/|VaR| ≈ **{tail_ratio:.2f}**).")
    if np.isfinite(avg_corr):
        bullets.append(f"- **Correlação média = {avg_corr:.2f}** → diversificação **{div_tag}**.")
if riskdf_nonempty:
    bullets.append(f"- **Driver de risco**: {hi_vol_row['Symbol']} (vol {hi_vol_row['VolAnnPct']:.1f}%). "
                   f"Pior *tail* (VaR): {worst_var_row['Symbol']} ({worst_var_row['VaR']:.2f}%).")
if not np.isnan(min_pair[2]):
    bullets.append(f"- **Hedge/diversificação**: par menos correlacionado **{min_pair[0]} × {min_pair[1]}** (ρ={min_pair[2]:.2f}).")
if sz:
    bullets.append(f"- **Sizing**: {sz['asset']} com **{sz['qty']:,.4f}** {('unid.' if sz['qty']>=1 else 'lotes')}, "
                   f"perda esperada **R$ {sz['expected_loss']:,.2f}** ({sz['risk_pct']:.1f}% do capital), stop **R$ {sz['stop_price']:,.2f}**.")

for b in bullets:
    st.markdown(b)


