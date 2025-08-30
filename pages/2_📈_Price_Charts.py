# pages/2_📈_Price_Charts.py
import streamlit as st
from core.ui import app_header, ticker_selector, data_status_badge
from core.data import download_history
from core.indicators import sma, ema, rsi

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

st.set_page_config(page_title="Price Charts", page_icon="📈", layout="wide")
app_header("📈 Price Charts", "Candles + MAs + Volume + RSI")
st.caption(f"Classe selecionada: **{st.session_state.get('asset_class','?')}**")


# =======================
# Sidebar / Controles
# =======================
symbol = ticker_selector()

col_period, col_interval = st.sidebar.columns(2)
period = col_period.selectbox(
    "Período",
    ["1mo","3mo","6mo","1y","2y","5y","ytd","max"],
    index=2
)
interval = col_interval.selectbox(
    "Intervalo",
    ["1d","1h","1wk"],
    index=0,
    help="Para intraday confiável use 1h; diário 1d; semanal 1wk."
)

st.sidebar.markdown("### Médias móveis")
ma_sma_len = st.sidebar.slider("SMA", 5, 200, 20, step=1)
ma_ema_len = st.sidebar.slider("EMA", 5, 200, 50, step=1)

st.sidebar.markdown("### RSI")
rsi_len = st.sidebar.slider("RSI length", 2, 50, 14, step=1)
rsi_show_bands = st.sidebar.checkbox("Mostrar bandas 30/70", value=True)

show_volume = st.sidebar.checkbox("Mostrar Volume", value=True)
show_rangeslider = st.sidebar.checkbox("Range Slider", value=False)

# =======================
# Download de dados
# =======================
if not symbol:
    st.warning("Escolha um ativo na barra lateral.")
    st.stop()

df = download_history(symbol, period=period, interval=interval)
data_status_badge(df)
if df is None or df.empty or "Close" not in df.columns:
    st.error("Sem dados suficientes para plotar. Tente outro período/intervalo.")
    st.stop()

# =======================
# Cálculos
# =======================
df = df.copy()
df["SMA"] = sma(df["Close"], ma_sma_len)
df["EMA"] = ema(df["Close"], ma_ema_len)
rsi_series = rsi(df["Close"], rsi_len)

# =======================
# KPIs simples
# =======================
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
pct = (last_close/prev_close - 1.0) * 100.0
c1, c2, c3 = st.columns(3)
c1.metric("Preço (Close)", f"{last_close:,.2f}", f"{pct:+.2f}%")
c2.metric("SMA", f"{df['SMA'].iloc[-1]:,.2f}")
c3.metric("EMA", f"{df['EMA'].iloc[-1]:,.2f}")

# =======================
# Gráfico
# =======================
rows = 3 if show_volume else 2
row_heights = [0.65, 0.35] if not show_volume else [0.55, 0.20, 0.25]

fig = make_subplots(
    rows=rows, cols=1, shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=row_heights
)

# Row 1: Candles + MAs
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"] if "Open" in df.columns else None,
        high=df["High"] if "High" in df.columns else None,
        low=df["Low"] if "Low" in df.columns else None,
        close=df["Close"],
        name="Candles",
        showlegend=False
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df.index, y=df["SMA"], mode="lines", name=f"SMA {ma_sma_len}"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df.index, y=df["EMA"], mode="lines", name=f"EMA {ma_ema_len}"),
    row=1, col=1
)

# Row 2: Volume (opcional)
row_vol = 2 if show_volume else None
row_rsi = 3 if show_volume else 2

if show_volume and "Volume" in df.columns:
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.4),
        row=row_vol, col=1
    )

# Row RSI
fig.add_trace(
    go.Scatter(x=df.index, y=rsi_series, mode="lines", name=f"RSI {rsi_len}"),
    row=row_rsi, col=1
)
if rsi_show_bands:
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
fig.update_yaxes(title_text="RSI", row=row_rsi, col=1, range=[0,100])

st.plotly_chart(fig, use_container_width=True)

# Tabela opcional (head)
with st.expander("Pré-visualização de dados (head)"):
    st.dataframe(df.head())
