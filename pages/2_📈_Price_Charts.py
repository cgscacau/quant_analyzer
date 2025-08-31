# pages/2_📈_Price_Charts.py
from __future__ import annotations

import math
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# App helpers
from core.ui import app_header, data_status_badge
from core.data import download_history, load_watchlists
from core.indicators import sma, ema, rsi  # usa seus indicadores base

# -----------------------------------------------------------------------------
# Config & Header
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Price Charts", page_icon="📈", layout="wide")
app_header("📈 Price Charts", "Candles + MAs + Volume + RSI")

# -----------------------------------------------------------------------------
# Helpers: watchlists seguras, normalização de OHLCV, cache e retornos
# -----------------------------------------------------------------------------
def _safe_watchlists() -> dict[str, list[str]]:
    """Combina watchlists do arquivo com o session_state, sem KeyError."""
    wl_file = {}
    try:
        wl_file = load_watchlists() or {}
    except Exception:
        wl_file = {}
    wl_state = st.session_state.get("watchlists", {}) or {}
    merged = {**wl_file, **wl_state}
    merged.setdefault("BR_STOCKS", ["PETR4.SA", "VALE3.SA", "B3SA3.SA"])
    merged.setdefault("US_STOCKS", ["AAPL", "MSFT", "NVDA", "SPY"])
    merged.setdefault("CRYPTO", ["BTC-USD", "ETH-USD"])
    return merged

def _normalize_ohlcv(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Garante DataFrame com colunas 1-D: Open, High, Low, Close, Adj Close (se houver), Volume.
    Trata casos com MultiIndex e colunas duplicadas (comportamento recente do yfinance).
    """
    if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    def _get_one(name: str) -> pd.Series | None:
        # 1) coluna simples
        if name in df.columns and not isinstance(df[name], pd.DataFrame):
            return pd.to_numeric(df[name], errors="coerce").dropna()
        # 2) coluna "virou DataFrame" (duplicadas)
        if name in df.columns and isinstance(df[name], pd.DataFrame):
            sub = df[name]
            for col in sub.columns:
                sr = pd.to_numeric(sub[col], errors="coerce").dropna()
                if not sr.empty:
                    return sr
        # 3) MultiIndex (nível 0 = OHLCV)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                if name in df.columns.get_level_values(0):
                    sub = df.xs(name, axis=1, level=0, drop_level=False)
                    first = sub.columns[0]
                    sr = pd.to_numeric(sub[first], errors="coerce").dropna()
                    return sr
            except Exception:
                return None
        return None

    out = pd.DataFrame(index=df.index)
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        sr = _get_one(c)
        if sr is not None:
            out[c] = sr
    out = out.dropna(how="any")
    return out

@st.cache_data(ttl=600, show_spinner=False)
def _cached_download(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Encapsula download + normalização; retorna DF pronto para cálculo."""
    data = download_history(symbol, period=period, interval=interval)
    return _normalize_ohlcv(data)

def _first_non_empty(*dfs: pd.DataFrame) -> pd.DataFrame:
    for d in dfs:
        if isinstance(d, pd.DataFrame) and not d.empty:
            return d
    return pd.DataFrame()

def _pct(from_series: pd.Series, periods: int) -> float | None:
    """Retorno percentual em 'periods' períodos (segura a falta de dados)."""
    if not isinstance(from_series, pd.Series) or len(from_series) <= periods:
        return None
    try:
        a = float(from_series.iloc[-periods-1])
        b = float(from_series.iloc[-1])
        if a and math.isfinite(a) and math.isfinite(b):
            return b / a - 1.0
    except Exception:
        return None
    return None

# -----------------------------------------------------------------------------
# Sidebar — seleção robusta
# -----------------------------------------------------------------------------
wls = _safe_watchlists()
classes = list(wls.keys())
default_class = st.session_state.get("asset_class", "US_STOCKS" if "US_STOCKS" in wls else classes[0])

st.sidebar.markdown("### Ativo")
asset_class = st.sidebar.selectbox("Classe", classes, index=max(classes.index(default_class), 0))
st.session_state["asset_class"] = asset_class
options = wls.get(asset_class, [])
symbol_sel = st.sidebar.selectbox("Ticker", options, index=0 if options else None) if options else None
symbol_manual = st.sidebar.text_input("Ou digite manualmente (ex.: SPY, BOVA11.SA)", value="")
symbol = (symbol_manual.strip() or symbol_sel or "").strip()

col_period, col_interval = st.sidebar.columns(2)
period = col_period.selectbox("Período", ["1mo","3mo","6mo","1y","2y","5y","ytd","max"], index=2)
interval = col_interval.selectbox("Intervalo", ["1d","1h","1wk"], index=0, help="Intraday: 1h • Diário: 1d • Semanal: 1wk")

st.sidebar.markdown("### Médias móveis")
ma_sma_len = st.sidebar.slider("SMA", 5, 200, 20, step=1)
ma_ema_len = st.sidebar.slider("EMA", 5, 200, 50, step=1)
st.sidebar.markdown("### RSI")
rsi_len = st.sidebar.slider("RSI length", 2, 50, 14, step=1)
rsi_show_bands = st.sidebar.checkbox("Mostrar bandas 30/70", value=True)

st.sidebar.markdown("### Extras")
show_volume = st.sidebar.checkbox("Mostrar Volume", value=True)
show_rangeslider = st.sidebar.checkbox("Range Slider", value=False)
use_adjclose_for_ma = st.sidebar.checkbox("Usar Adj Close nas MAs/RSI (se existir)", value=False)
show_bbands = st.sidebar.checkbox("Bollinger Bands", value=False)

# -----------------------------------------------------------------------------
# Download de dados com fallback
# -----------------------------------------------------------------------------
if not symbol:
    st.warning("Escolha um ativo na barra lateral.")
    st.stop()

with st.spinner("📥 Baixando dados..."):
    df = _cached_download(symbol, period, interval)
    data_status_badge(df)  # badge informativo

# fallback: se vazio, tentar combinações úteis (provider instável/sem histórico)
if df.empty:
    alt_periods = ["6mo", "1y", "2y"] if period in ("ytd", "max", "1mo", "3mo") else ["6mo", "1y"]
    alt_intervals = ["1d"] if interval == "1wk" else ["1wk", "1d"]
    tried = [f"{period}/{interval}"]
    for p in alt_periods:
        for itv in alt_intervals:
            df_try = _cached_download(symbol, p, itv)
            tried.append(f"{p}/{itv}")
            if not df_try.empty:
                st.info(f"Sem dados para **{period}/{interval}**. Usando fallback **{p}/{itv}**.")
                df, period, interval = df_try, p, itv
                break
        if not df.empty:
            break

if df.empty or "Close" not in df.columns:
    st.error("Sem dados suficientes para plotar. Tente outro ticker/período/intervalo.")
    st.stop()

# -----------------------------------------------------------------------------
# Cálculos
# -----------------------------------------------------------------------------
df = df.sort_index().copy()

# preço-base para indicadores (Close ou Adj Close)
price_col = "Adj Close" if (use_adjclose_for_ma and "Adj Close" in df.columns) else "Close"

df["SMA"] = sma(df[price_col], ma_sma_len)
df["EMA"] = ema(df[price_col], ma_ema_len)
rsi_series = rsi(df[price_col], rsi_len)

if show_bbands:
    # Bollinger Bands simples (20/2 se usuário não mudou SMA/EMA)
    bb_len = ma_sma_len
    bb_std = 2.0
    px = df[price_col].rolling(bb_len)
    df["BB_MID"] = px.mean()
    df["BB_UP"] = df["BB_MID"] + bb_std * px.std(ddof=0)
    df["BB_DN"] = df["BB_MID"] - bb_std * px.std(ddof=0)

# -----------------------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------------------
last_close = float(df[price_col].iloc[-1])
prev_close = float(df[price_col].iloc[-2]) if len(df) > 1 else last_close
pct = (last_close / prev_close - 1.0) * 100.0

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Preço ({price_col})", f"{last_close:,.2f}", f"{pct:+.2f}%")
c2.metric("SMA", f"{df['SMA'].iloc[-1]:,.2f}")
c3.metric("EMA", f"{df['EMA'].iloc[-1]:,.2f}")
c4.metric("Última data", df.index[-1].strftime("%Y-%m-%d"))

# Tabela de retornos
def _fmt(x):
    return "—" if x is None or not math.isfinite(x) else f"{x*100:,.2f}%"

rets = {
    "1M": _pct(df[price_col], 21),
    "3M": _pct(df[price_col], 63),
    "6M": _pct(df[price_col], 126),
    "YTD": (df[price_col].iloc[-1] / df[price_col][df.index.year == datetime.now().year].iloc[0] - 1.0) if any(df.index.year == datetime.now().year) else None,
    "1Y": _pct(df[price_col], 252),
}
st.caption("**Retornos aproximados** (baseados em sessões ~21/mes).")
st.dataframe(
    pd.DataFrame([{"Período": k, "Retorno": _fmt(v)} for k, v in rets.items()]),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------------------------------------------------------
# Gráfico
# -----------------------------------------------------------------------------
rows = 3 if show_volume else 2
row_heights = [0.55, 0.20, 0.25] if show_volume else [0.65, 0.35]
fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)

# Row 1: Candles + MAs (+ BBands)
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df.get("Open"),
        high=df.get("High"),
        low=df.get("Low"),
        close=df["Close"],  # candles sempre com Close "padrão"
        name="Candles",
        showlegend=False,
    ),
    row=1, col=1,
)
fig.add_trace(go.Scatter(x=df.index, y=df["SMA"], mode="lines", name=f"SMA {ma_sma_len}"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["EMA"], mode="lines", name=f"EMA {ma_ema_len}"), row=1, col=1)

if show_bbands and {"BB_MID", "BB_UP", "BB_DN"}.issubset(df.columns):
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_MID"], mode="lines", name="BB mid", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_UP"], mode="lines", name="BB up", line=dict(width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_DN"], mode="lines", name="BB dn", line=dict(width=1, dash="dot")), row=1, col=1)

# Row 2: Volume (opcional)
row_vol = 2 if show_volume else None
row_rsi = 3 if show_volume else 2
if show_volume and "Volume" in df.columns:
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.35), row=row_vol, col=1)

# Row RSI
fig.add_trace(go.Scatter(x=df.index, y=rsi_series, mode="lines", name=f"RSI {rsi_len}"), row=row_rsi, col=1)
if rsi_show_bands:
    # add_hline aceita row/col; default é "all", mas especificamos para este subplot
    fig.add_hline(y=70, line_dash="dot", line_width=1, row=row_rsi, col=1)
    fig.add_hline(y=30, line_dash="dot", line_width=1, row=row_rsi, col=1)

# Layout
fig.update_layout(
    template="plotly_white",
    title=dict(text=f"{symbol} — Price Charts", x=0.01, xanchor="left"),
    xaxis_rangeslider_visible=show_rangeslider,
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
fig.update_yaxes(title_text="Preço", row=1, col=1)
if show_volume:
    fig.update_yaxes(title_text="Volume", row=row_vol, col=1, showgrid=False)
fig.update_yaxes(title_text="RSI", row=row_rsi, col=1, range=[0, 100])

st.plotly_chart(fig, use_container_width=True)

# Tabela opcional (dados)
with st.expander("Pré-visualização de dados (head)"):
    st.dataframe(df.head(), use_container_width=True)
