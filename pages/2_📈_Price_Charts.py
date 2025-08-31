# pages/2_📈_Price_Charts.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# utilidades do app
from core.ui import app_header, data_status_badge
from core.data import download_history, load_watchlists
from core.indicators import sma, ema, rsi

# -------------------------------------------------------------------
# Config & Header
# -------------------------------------------------------------------
st.set_page_config(page_title="Price Charts", page_icon="📈", layout="wide")
app_header("📈 Price Charts", "Candles + MAs + Volume + RSI")

# -------------------------------------------------------------------
# Helpers: seleção segura + normalização de OHLCV
# -------------------------------------------------------------------
def _safe_watchlists() -> dict[str, list[str]]:
    """Busca watchlists; nunca levanta KeyError."""
    # tenta carregar dos dados (se existir) e mescla com session_state
    wl_file = {}
    try:
        wl_file = load_watchlists()
    except Exception:
        pass
    wl_state = st.session_state.get("watchlists", {}) or {}
    merged = {**wl_file, **wl_state}
    # defaults mínimos
    merged.setdefault("BR_STOCKS", ["PETR4.SA", "VALE3.SA", "ITUB4.SA"])
    merged.setdefault("US_STOCKS", ["AAPL", "MSFT", "NVDA", "SPY"])
    merged.setdefault("CRYPTO", ["BTC-USD", "ETH-USD"])
    return merged

def _sidebar_symbol_selector() -> str | None:
    """Seletor de classe e ticker (sem KeyError, com fallback a texto)."""
    wls = _safe_watchlists()
    classes = list(wls.keys())
    default_class = st.session_state.get("asset_class", classes[0] if classes else "US_STOCKS")

    st.sidebar.markdown("### Ativo")
    asset_class = st.sidebar.selectbox("Classe", classes, index=max(classes.index(default_class), 0) if classes else 0)
    st.session_state["asset_class"] = asset_class  # persistir escolha

    options = wls.get(asset_class, [])
    symbol = st.sidebar.selectbox("Ticker", options, index=0 if options else None) if options else None
    custom = st.sidebar.text_input("Ou digite manualmente (ex.: SPY, BOVA11.SA)", value="")

    return (custom.strip() or symbol or None)

def _normalize_ohlcv(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Garante um DataFrame com colunas: Open, High, Low, Close, Volume (Series 1-D).
    Trata casos de MultiIndex/duplicadas que alguns provedores retornam.
    """
    if df_raw is None or not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    def _get_col(name: str) -> pd.Series | None:
        # 1) coluna simples
        if name in df.columns and not isinstance(df[name], pd.DataFrame):
            return pd.to_numeric(df[name], errors="coerce").dropna()
        # 2) coluna virou DataFrame (duplicadas)
        if name in df.columns and isinstance(df[name], pd.DataFrame):
            sub = df[name]
            for c in sub.columns:
                sr = pd.to_numeric(sub[c], errors="coerce").dropna()
                if not sr.empty:
                    return sr
        # 3) MultiIndex (nível 0 com OHLCV)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                if name in df.columns.get_level_values(0):
                    sub = df.xs(name, axis=1, level=0, drop_level=False)
                    first = sub.columns[0]
                    sr = pd.to_numeric(sub[first], errors="coerce").dropna()
                    return sr
            except Exception:
                pass
        return None

    out = pd.DataFrame(index=df.index)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        sr = _get_col(col)
        if sr is not None:
            out[col] = sr

    out = out.dropna(how="any")
    return out

# -------------------------------------------------------------------
# Sidebar / Controles
# -------------------------------------------------------------------
symbol = _sidebar_symbol_selector()
col_period, col_interval = st.sidebar.columns(2)
period = col_period.selectbox("Período", ["1mo","3mo","6mo","1y","2y","5y","ytd","max"], index=2)
interval = col_interval.selectbox("Intervalo", ["1d","1h","1wk"], index=0, help="Para intraday confiável use 1h; diário 1d; semanal 1wk.")

st.sidebar.markdown("### Médias móveis")
ma_sma_len = st.sidebar.slider("SMA", 5, 200, 20, step=1)
ma_ema_len = st.sidebar.slider("EMA", 5, 200, 50, step=1)

st.sidebar.markdown("### RSI")
rsi_len = st.sidebar.slider("RSI length", 2, 50, 14, step=1)
rsi_show_bands = st.sidebar.checkbox("Mostrar bandas 30/70", value=True)

show_volume = st.sidebar.checkbox("Mostrar Volume", value=True)
show_rangeslider = st.sidebar.checkbox("Range Slider", value=False)

# -------------------------------------------------------------------
# Download de dados
# -------------------------------------------------------------------
if not symbol:
    st.warning("Escolha um ativo na barra lateral.")
    st.stop()

df0 = download_history(symbol, period=period, interval=interval)
data_status_badge(df0)  # badge informativo (core.ui)

df = _normalize_ohlcv(df0)
if df.empty or "Close" not in df.columns:
    st.error("Sem dados suficientes para plotar. Tente outro período/intervalo.")
    st.stop()

# -------------------------------------------------------------------
# Cálculos
# -------------------------------------------------------------------
df = df.sort_index().copy()
df["SMA"] = sma(df["Close"], ma_sma_len)
df["EMA"] = ema(df["Close"], ma_ema_len)
rsi_series = rsi(df["Close"], rsi_len)

# -------------------------------------------------------------------
# KPIs simples
# -------------------------------------------------------------------
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
pct = (last_close / prev_close - 1.0) * 100.0

c1, c2, c3 = st.columns(3)
c1.metric("Preço (Close)", f"{last_close:,.2f}", f"{pct:+.2f}%")
c2.metric("SMA", f"{df['SMA'].iloc[-1]:,.2f}")
c3.metric("EMA", f"{df['EMA'].iloc[-1]:,.2f}")

# -------------------------------------------------------------------
# Gráfico
# -------------------------------------------------------------------
rows = 3 if show_volume else 2
row_heights = [0.55, 0.20, 0.25] if show_volume else [0.65, 0.35]

fig = make_subplots(
    rows=rows, cols=1, shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=row_heights
)

# Row 1: Candles + MAs
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df.get("Open"), high=df.get("High"), low=df.get("Low"), close=df["Close"],
        name="Candles", showlegend=False
    ),
    row=1, col=1
)
fig.add_trace(go.Scatter(x=df.index, y=df["SMA"], mode="lines", name=f"SMA {ma_sma_len}"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["EMA"], mode="lines", name=f"EMA {ma_ema_len}"), row=1, col=1)

# Row 2: Volume (opcional)
row_vol = 2 if show_volume else None
row_rsi = 3 if show_volume else 2
if show_volume and "Volume" in df.columns:
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.4), row=row_vol, col=1)

# Row RSI
fig.add_trace(go.Scatter(x=df.index, y=rsi_series, mode="lines", name=f"RSI {rsi_len}"), row=row_rsi, col=1)
if rsi_show_bands:
    # add_hline aceita row/col; default já é "all", mas especificamos para evitar confusão
    fig.add_hline(y=70, line_dash="dot", line_width=1, row=row_rsi, col=1)
    fig.add_hline(y=30, line_dash="dot", line_width=1, row=row_rsi, col=1)

# Layout
fig.update_layout(
    template="plotly_white",
    title=dict(text=f"{symbol} — Price Charts", x=0.01, xanchor="left"),
    xaxis_rangeslider_visible=show_rangeslider,
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)

# Eixos
fig.update_yaxes(title_text="Preço", row=1, col=1)
if show_volume:
    fig.update_yaxes(title_text="Volume", row=row_vol, col=1, showgrid=False)
fig.update_yaxes(title_text="RSI", row=row_rsi, col=1, range=[0, 100])

st.plotly_chart(fig, use_container_width=True)

# Tabela opcional (head)
with st.expander("Pré-visualização de dados (head)"):
    st.dataframe(df.head())
