# pages/4_📊_Backtest.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from core.ui import app_header
from core.data import load_watchlists, download_history
from core.backtest import run_sma_cross, run_rsi_meanrev

st.set_page_config(page_title="Backtest", page_icon="📊", layout="wide")
app_header("📊 Backtest", "Comparação rápida de estratégias por ativo")

# (opcional) obtém um universo de tickers para popular o multiselect.
def _load_watchlists_safe():
    try:
        from core.data import load_watchlists as _lw  # usa o projeto, se existir
        return _lw()
    except Exception:
        # fallback mínimo – ajuste se quiser
        return {
            "BR_STOCKS": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA"],
            "US_STOCKS": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"],
            "CRYPTO":    ["BTC-USD", "ETH-USD", "SOL-USD"],
        }

def _flatten_unique(lst_of_lists):
    out = []
    seen = set()
    for lst in lst_of_lists:
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out

_watch = _load_watchlists_safe()
_universe = _flatten_unique(_watch.values())  # lista única para o multiselect

# 1) Lê a seleção que veio do Screener (se houver)
_screener_sel = st.session_state.get("screener_selected", [])

# 2) Toggle para usar (ou não) a seleção do Screener
use_screener = st.toggle(
    "Usar seleção do Screener (se houver)",
    value=bool(_screener_sel),  # liga por padrão se tiver algo salvo
    key="use_screener_toggle",
)

# 3) Multiselect para edição manual (fica desabilitado quando o toggle está ON)
symbols_manual = st.multiselect(
    "Escolha os ativos (caso não use a seleção do Screener)",
    options=_universe,
    default=(_screener_sel or ["AAPL"]),  # pré-preenche com a seleção do screener se existir
    disabled=use_screener,
)

# 4) Lista efetiva de símbolos que o seu backtest vai usar
symbols = _screener_sel if (use_screener and _screener_sel) else symbols_manual

# 5) Segurança: se a lista estiver vazia, avisa e interrompe cedo
if not symbols:
    st.info("Nenhum ativo selecionado. Selecione no Screener e clique em **Usar seleção no Backtest** ou escolha manualmente acima.")
    st.stop()

# (opcional) debug rápido
with st.expander("Debug seleção", expanded=False):
    st.write("screener_selected (session):", _screener_sel)
    st.write("use_screener:", use_screener)
    st.write("symbols (efetivos):", symbols)


def inject_dark_theme():
    import streamlit as st
    st.markdown("""
    <style>
    :root{
      --bg:#0b0f16; --panel:#121826; --panel2:#0f172a;
      --text:#e8edf7; --muted:#a3adc2; --border:#1f2a44; --accent:#38bdf8;
    }
    /* fundo + texto padrão */
    .stApp{ background:var(--bg); color:var(--text); }
    .block-container{ padding-top:1.5rem; }

    /* textos/headers/labels */
    [data-testid="stMarkdownContainer"], h1,h2,h3,h4,h5,h6,
    [data-testid="stWidgetLabel"], label, p, span { color:var(--text) !important; }
    [data-testid="stCaptionContainer"], small { color:var(--muted) !important; }

    /* -------- SIDEBAR / MENU DE PÁGINAS -------- */
    [data-testid="stSidebar"]{
      background: var(--panel2) !important;
      border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] *{ color: var(--text) !important; }
    /* container do nav (versões novas do Streamlit) */
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] ul{ padding: .25rem .5rem; }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a{
      display:block; padding:.45rem .65rem; border-radius:10px;
      color: var(--muted) !important; text-decoration:none;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover{
      background:#111827; color: var(--text) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"]{
      background:#0b1220; color: var(--accent) !important;
      border:1px solid var(--border);
      box-shadow: 0 0 0 1px rgba(56,189,248,.25) inset;
    }
    /* fallback para versões antigas (ancoras genéricas) */
    [data-testid="stSidebar"] nav a{ color: var(--muted) !important; }
    [data-testid="stSidebar"] nav a[aria-current="page"]{
      color: var(--accent) !important; background:#0b1220;
      border-radius:10px; padding:.45rem .65rem;
    }

    /* widgets (select, slider, radio, tabs) */
    [data-baseweb="select"]>div{ background:var(--panel) !important; border-color:var(--border) !important; }
    [data-baseweb="select"] *{ color:var(--text) !important; }
    [data-baseweb="slider"] div{ color:var(--text) !important; }
    [role="radiogroup"] label{ color:var(--text) !important; }
    [role="tab"] p{ color:var(--text) !important; }
    [role="tab"][aria-selected="true"] p{ color:var(--accent) !important; }

    /* métricas (cards) */
    [data-testid="stMetric"]{
      background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:12px;
    }
    [data-testid="stMetricValue"]{ color:var(--text) !important; }
    [data-testid="stMetricLabel"]{ color:var(--muted) !important; }

    /* expanders e tabelas */
    [data-testid="stExpander"] div[role="button"]{
      background:var(--panel); color:var(--text); border:1px solid var(--border);
    }
    [data-testid="stDataFrame"]{ background:var(--panel2); }
    .stTable, .stDataFrame td, .stDataFrame th{ color:var(--text) !important; }

    /* botões */
    .stButton>button{ background:var(--panel); color:var(--text); border:1px solid var(--border); }
    </style>
    """, unsafe_allow_html=True)


# -----------------------
# Entrada de símbolos
# -----------------------
watch = load_watchlists()
all_syms = sorted(set(watch["BR_STOCKS"] + watch["US_STOCKS"] + watch["CRYPTO"]))

sel_from_screener = st.session_state.get("screener_selection", [])
use_sel = st.toggle("Usar seleção do Screener (se houver)", value=bool(sel_from_screener))
symbols = sel_from_screener if use_sel and sel_from_screener else st.multiselect(
    "Escolha os ativos (caso não use a seleção do Screener)",
    options=all_syms,
    default=["AAPL"] if "AAPL" in all_syms else all_syms[:1]
)

colp, coli, cold = st.columns(3)
period = colp.selectbox("Período", ["6mo","1y","2y","5y"], index=1)
interval = coli.selectbox("Intervalo", ["1d","1wk"], index=0)
dark = cold.toggle("Tema escuro (gráficos)", value=True)

# Estilo dark para a página (leve)
if dark:
    inject_dark_theme()


# -----------------------
# Estratégia e parâmetros
# -----------------------
st.subheader("Estratégia")
mode = st.radio("Tipo", ["Momentum (SMA Cross)", "Mean Reversion (RSI)"], horizontal=True)

c_params = st.container()
if mode == "Momentum (SMA Cross)":
    c1, c2 = c_params.columns(2)
    fast = c1.slider("SMA Rápida", 5, 100, 20, step=1)
    slow = c2.slider("SMA Lenta", 20, 300, 50, step=1)
else:
    c1, c2, c3 = c_params.columns(3)
    rsi_len = c1.slider("RSI length", 2, 50, 14, step=1)
    buy_below = c2.slider("Compra se RSI <", 5, 50, 30, step=1)
    exit_mid = c3.slider("Zera se RSI >", 30, 70, 50, step=1)

st.markdown("### Custos, slippage & sizing")
c1, c2, c3 = st.columns(3)
fee_bps = c1.number_input("Custo por lado (bps)", min_value=0.0, value=2.0, step=0.5,
                          help="Ex.: 2 bps = 0,02% por entrada ou por saída.")
slip_bps = c2.number_input("Slippage por lado (bps)", min_value=0.0, value=3.0, step=0.5,
                           help="Ex.: 3 bps = 0,03% por entrada ou por saída.")
alloc_pct = c3.slider("Alocação do capital (%)", 1, 100, 100, step=1)
alloc = alloc_pct / 100.0


if not symbols:
    st.warning("Nenhum símbolo selecionado.")
    st.stop()

# -----------------------
# Primeira passada: roda e coleta métricas (para resumo & comparativo)
# -----------------------
results = []
for sym in symbols:
    df = download_history(sym, period=period, interval=interval)
    if df.empty or "Close" not in df.columns:
        results.append(dict(symbol=sym, error=True))
        continue

    if mode == "Momentum (SMA Cross)":
        res = run_sma_cross(
            df, fast=fast, slow=slow, freq=interval,
            alloc=alloc, fee_bps_side=fee_bps, slippage_bps_side=slip_bps
        )
        ptxt = f"SMA{fast}/{slow}"
    else:
        res = run_rsi_meanrev(
            df, rsi_len=rsi_len, buy_below=buy_below, exit_mid=exit_mid, freq=interval,
            alloc=alloc, fee_bps_side=fee_bps, slippage_bps_side=slip_bps
        )
        ptxt = f"RSI len={rsi_len}, buy<{buy_below}, exit>{exit_mid}"

    results.append(dict(symbol=sym, df=df, res=res, params_text=ptxt))

valid = [r for r in results if not r.get("error")]
if valid:
    # Card-resumo (igual) + tabela comparativa
    met = [(r["symbol"], r["res"]["metrics"], r["res"]["bench_metrics"]) for r in valid]
    total_trades = int(sum(m["Trades"] for _, m, _ in met))
    avg_sharpe = float(np.mean([m["Sharpe"] for _, m, _ in met]))
    avg_cagr   = float(np.mean([m["CAGR"] for _, m, _ in met]))
    avg_win    = float(np.mean([m["WinRate"] for _, m, _ in met]))
    best_sym, best_m, _ = max(met, key=lambda x: x[1]["CAGR"])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ativos válidos", f"{len(valid)}")
    c2.metric("Trades (total)", f"{total_trades}")
    c3.metric("CAGR médio", f"{avg_cagr*100:,.2f}%")
    c4.metric("Sharpe médio", f"{avg_sharpe:.2f}")
    c5.metric("Win Rate média", f"{avg_win:,.1f}%")
    st.caption(f"Melhor por CAGR: **{best_sym}** ({best_m['CAGR']*100:,.2f}%)")

    # ---- Tabela comparativa (Estratégia vs Buy&Hold) ----
    rows = []
    for s, m, mb in met:
        rows.append(dict(
            Symbol=s,
            Strat_CAGR=f"{m['CAGR']*100:.2f}%",
            Strat_Sharpe=f"{m['Sharpe']:.2f}",
            Strat_MaxDD=f"{m['MaxDD']*100:.2f}%",
            Trades=int(m["Trades"]),
            WinRate=f"{m['WinRate']:.1f}%",
            BH_CAGR=f"{mb['CAGR']*100:.2f}%",
            BH_MaxDD=f"{mb['MaxDD']*100:.2f}%",
            Outperf=f"{(m['CAGR']-mb['CAGR'])*100:.2f}%"
        ))
    comp = pd.DataFrame(rows)
    st.markdown("#### Comparativo (Estratégia × Buy & Hold)")
    st.dataframe(comp, use_container_width=True)
    st.download_button(
        "⬇️ Exportar comparativo (CSV)",
        data=comp.to_csv(index=False).encode("utf-8"),
        file_name=f"backtest_compare_{mode.split()[0].lower()}_{period}_{interval}.csv",
        mime="text/csv"
    )
else:
    st.warning("Nenhum ativo com dados válidos para este período/intervalo.")

st.divider()


# -----------------------
# Abas por ativo (equity + trades)
# -----------------------
tabs = st.tabs([r["symbol"] for r in results])

for i, r in enumerate(results):
    sym = r["symbol"]
    with tabs[i]:
        if r.get("error"):
            st.error("Sem dados para este ativo/período.")
            continue

        res = r["res"]
        eq = res["equity"]
        m = res["metrics"]
        params_text = r["params_text"]

        # KPIs (cards)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("CAGR", f"{m['CAGR']*100:,.2f}%")
        c2.metric("Sharpe", f"{m['Sharpe']:.2f}")
        c3.metric("Max Drawdown", f"{m['MaxDD']*100:,.2f}%")
        c4.metric("Trades", f"{m['Trades']}")
        c5.metric("Win Rate", f"{m['WinRate']:.1f}%")

        # Equity curve (dark moderno)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Equity",
                                 line=dict(width=2), line_shape="spline"))
        template = "plotly_dark" if dark else "plotly_white"
        fig.update_layout(
            template=template,
            title=f"{sym} — {mode} ({params_text})",
            yaxis_title="Equity (R$ virtual)",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        if dark:
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")

        st.plotly_chart(fig, use_container_width=True)

        # Log de trades (novo!)
        with st.expander("📜 Trades (log)"):
            trades = res.get("trades", pd.DataFrame())
            if trades.empty:
                st.info("Sem trades para este período/parâmetros.")
            else:
                st.dataframe(
                    trades.style.format({"EntryPx":"{:,.2f}","ExitPx":"{:,.2f}","Ret%":"{:+.2f}%"}),
                    use_container_width=True, height=300
                )

        # Dados de retornos/equity (continua disponível)
        with st.expander("Ver dados (retornos/equity)"):
            df_out = pd.DataFrame({"returns": res["returns"], "equity": res["equity"]})
            st.dataframe(df_out.tail(20), use_container_width=True)
