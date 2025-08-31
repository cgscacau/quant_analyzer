# pages/2_📈_Price_Charts.py
from __future__ import annotations

import math
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Helpers do app (existentes no seu projeto)
from core.ui import app_header, data_status_badge
from core.data import load_watchlists, download_history
from core.indicators import sma, ema, rsi

# =============================================================================
# Config & Header
# =============================================================================
st.set_page_config(page_title="Price Charts", page_icon="📈", layout="wide")
app_header("📈 Price Charts", "Candles + MAs + Volume + RSI")

# =============================================================================
# Utilidades locais (robustez contra provider / MultiIndex / estado)
# =============================================================================
def _safe_watchlists() -> dict[str, list[str]]:
    """Combina watchlists persistidas com as do session_state, sem KeyError."""
    wl_file = {}
    try:
        wl_file = load_watchlists() or {}
    except Exception:
        wl_file = {}
    wl_state = st.session_state.get("watchlists", {}) or {}
    merged = {**wl_file, **wl_state}
    # defaults mínimos para não ficar vazio
    merged.setdefault("BR_STOCKS", ["PETR4.SA", "VALE3.SA", "B3SA3.SA"])
    merged.setdefault("US_STOCKS", ["AAPL", "MSFT", "NVDA", "SPY"])
    merged.setdefault("CRYPTO", ["BTC-USD", "ETH-USD"])
    # classes extras podem existir (ex.: BR_DIVIDEND). Não filtramos nada aqui.
    return merged

def _normalize_ohlcv(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Garante DataFrame com colunas 1-D: Open, High, Low, Close, Adj Close (se houver), Volume.
    Trata MultiIndex e colunas duplicadas (mudanças recentes de provedores).
    """
    if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    def _get_one(name: str) -> pd.Series | None:
        # 1) coluna simples
        if name in df.columns and not isinstance(df[name], pd.DataFrame):
            return pd.to_numeric(df[name], errors="coerce").dropna()
        # 2) coluna duplicada (DataFrame)
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
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        sr = _get_one(c)
        if sr is not None:
            out[c] = sr

    out = out.dropna(how="any")
    # normaliza índice (sem timezone) para evitar warnings em concat/plot
    if not out.empty and getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    return out.sort_index()

@st.cache_data(ttl=600, show_spinner=False)
def _cached_download(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Wrapper cacheado: baixa e normaliza OHLCV."""
    df = download_history(symbol, period=period, interval=interval)
    return _normalize_ohlcv(df)

def _pct(series: pd.Series, steps: int) -> float | None:
    """Retorno percentual aproximado em 'steps' pregões (21≈1M, 63≈3M...)."""
    if not isinstance(series, pd.Series) or len(series) <= steps:
        return None
    a = float(series.iloc[-steps-1])
    b = float(series.iloc[-1])
    if not math.isfinite(a) or not math.isfinite(b) or a == 0:
        return None
    return b / a - 1.0

# =============================================================================
# Sidebar — seleção robusta
# =============================================================================
wls = _safe_watchlists()
classes = list(wls.keys())
default_class = st.session_state.get("asset_class", "US_STOCKS" if "US_STOCKS" in wls else (classes[0] if classes else "US_STOCKS"))

st.sidebar.markdown("### Ativo")
asset_class = st.sidebar.selectbox("Classe", classes, index=max(classes.index(default_class), 0) if classes else 0)
st.session_state["asset_class"] = asset_class

tickers = wls.get(asset_class, [])
symbol_sel = st.sidebar.selectbox("Ticker", tickers, index=0 if tickers else None) if tickers else None
symbol_manual = st.sidebar.text_input("Ou digite manualmente (ex.: SPY, BOVA11.SA)")
symbol = (symbol_manual.strip() or symbol_sel or "").strip()

col_p, col_i = st.sidebar.columns(2)
period = col_p.selectbox("Período", ["1mo","3mo","6mo","1y","2y","5y","ytd","max"], index=2)
interval = col_i.selectbox("Intervalo", ["1d","1h","1wk"], index=0, help="Intraday: 1h • Diário: 1d • Semanal: 1wk")

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

if not symbol:
    st.warning("Escolha um ativo na barra lateral.")
    st.stop()

# =============================================================================
# Download + fallback
# =============================================================================
with st.spinner("📥 Baixando dados..."):
    df = _cached_download(symbol, period, interval)
data_status_badge(df)  # badge informativo existente

# fallback automático (provider sem histórico na combinação pedida)
if df.empty:
    alt_periods = ["6mo", "1y", "2y"] if period in ("max", "ytd", "1mo", "3mo") else ["6mo", "1y"]
    alt_intervals = ["1d"] if interval == "1wk" else ["1wk", "1d"]
    for p in alt_periods:
        for itv in alt_intervals:
            df_try = _cached_download(symbol, p, itv)
            if not df_try.empty:
                st.info(f"Sem dados em **{period}/{interval}**. Usando fallback **{p}/{itv}**.")
                df, period, interval = df_try, p, itv
                break
        if not df.empty:
            break

if df.empty or "Close" not in df.columns:
    st.error("Sem dados suficientes para plotar. Tente outro ticker/período/intervalo.")
    st.stop()

# =============================================================================
# Cálculos de indicadores
# =============================================================================
df = df.sort_index().copy()
price_col = "Adj Close" if (use_adjclose_for_ma and "Adj Close" in df.columns) else "Close"

df["SMA"] = sma(df[price_col], ma_sma_len)
df["EMA"] = ema(df[price_col], ma_ema_len)
rsi_series = rsi(df[price_col], rsi_len)

if show_bbands:
    bb_len = ma_sma_len
    bb_std = 2.0
    roll = df[price_col].rolling(bb_len)
    df["BB_MID"] = roll.mean()
    std = roll.std(ddof=0)
    df["BB_UP"] = df["BB_MID"] + bb_std * std
    df["BB_DN"] = df["BB_MID"] - bb_std * std

# =============================================================================
# KPIs + retornos rápidos
# =============================================================================
last_px = float(df[price_col].iloc[-1])
prev_px = float(df[price_col].iloc[-2]) if len(df) > 1 else last_px
d1 = (last_px / prev_px - 1.0) * 100.0

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Preço ({price_col})", f"{last_px:,.2f}", f"{d1:+.2f}%")
c2.metric("SMA", f"{df['SMA'].iloc[-1]:,.2f}")
c3.metric("EMA", f"{df['EMA'].iloc[-1]:,.2f}")
c4.metric("Última data", df.index[-1].strftime("%Y-%m-%d"))

def _fmt_pct(x: float | None) -> str:
    return "—" if x is None or not math.isfinite(x) else f"{x*100:,.2f}%"

ytd_ret = None
try:
    this_year = df.index.year == datetime.now().year
    if any(this_year):
        ytd_series = df.loc[this_year, price_col]
        if len(ytd_series) >= 2:
            ytd_ret = float(ytd_series.iloc[-1] / ytd_series.iloc[0] - 1.0)
except Exception:
    ytd_ret = None

rets_tbl = pd.DataFrame(
    [
        {"Período": "1M", "Retorno": _fmt_pct(_pct(df[price_col], 21))},
        {"Período": "3M", "Retorno": _fmt_pct(_pct(df[price_col], 63))},
        {"Período": "6M", "Retorno": _fmt_pct(_pct(df[price_col], 126))},
        {"Período": "YTD", "Retorno": _fmt_pct(ytd_ret)},
        {"Período": "1Y", "Retorno": _fmt_pct(_pct(df[price_col], 252))},
    ]
)
st.caption("**Retornos aproximados** (≈21 pregões/mês).")
st.dataframe(rets_tbl, hide_index=True, use_container_width=True)

# =============================================================================
# Gráfico
# =============================================================================
rows = 3 if show_volume else 2
row_heights = [0.55, 0.20, 0.25] if show_volume else [0.65, 0.35]

fig = make_subplots(
    rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights
)

# Row 1: candles + MAs (+ BB)
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df.get("Open"),
        high=df.get("High"),
        low=df.get("Low"),
        close=df["Close"],  # candles sempre no Close "padrão"
        name="Candles",
        showlegend=False,
    ),
    row=1, col=1,
)
fig.add_trace(go.Scatter(x=df.index, y=df["SMA"], mode="lines", name=f"SMA {ma_sma_len}"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["EMA"], mode="lines", name=f"EMA {ma_ema_len}"), row=1, col=1)

if show_bbands and {"BB_MID", "BB_UP", "BB_DN"}.issubset(df.columns):
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_MID"], mode="lines", name="BB mid", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_UP"],  mode="lines", name="BB up",  line=dict(width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_DN"],  mode="lines", name="BB dn",  line=dict(width=1, dash="dot")), row=1, col=1)

# Row 2: volume (opcional)
row_vol = 2 if show_volume else None
row_rsi = 3 if show_volume else 2
if show_volume and "Volume" in df.columns:
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.35), row=row_vol, col=1)

# Row 3/2: RSI
fig.add_trace(go.Scatter(x=df.index, y=rsi_series, mode="lines", name=f"RSI {rsi_len}"), row=row_rsi, col=1)
if rsi_show_bands:
    fig.add_hline(y=70, line_dash="dot", line_width=1, row=row_rsi, col=1)
    fig.add_hline(y=30, line_dash="dot", line_width=1, row=row_rsi, col=1)

# Layout & eixos
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

# =============================================================================
# Dados (prévia)
# =============================================================================
with st.expander("Pré-visualização de dados (head)"):
    st.dataframe(df.head(), use_container_width=True)
