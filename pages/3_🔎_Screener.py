# pages/3_🔎_Screener.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Dependências internas do app
#   - load_watchlists: lê as listas (override em memória OU arquivo /data/watchlists.json)
#   - download_bulk: baixa vários tickers de uma vez e retorna {ticker: DataFrame}
# ---------------------------------------------------------------------
from core.data import load_watchlists, download_bulk

# ---------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------
st.title("🔎 Screener")
st.caption("Triagem multi-ativos (BR/US/Cripto) com métricas e filtros")

# Sidebar – Classe / Janela / Filtros
st.sidebar.header("Classe")
CLASS_LABELS = {
    "BR_STOCKS": "Brasil (Ações B3)",
    "BR_FIIS": "Brasil (FIIs)",
    "BR_DIVIDEND": "Brasil — Dividendos",
    "BR_BLUE_CHIPS": "Brasil — Blue Chips",
    "BR_SMALL_CAPS": "Brasil — Small Caps",
    "US_STOCKS": "EUA (Ações US)",
    "US_BLUE_CHIPS": "US — Blue Chips",
    "US_SMALL_CAPS": "US — Small Caps",
    "CRYPTO": "Criptos",
}

# ---------------------------------------------------------------------
# Carrega watchlists (arquivo ou override em memória se existir)
# ---------------------------------------------------------------------
WATCH = load_watchlists()  # dict[str, list[str]]

# Mapeia para o rótulo “bonito” na interface
def _choices_with_counts(watch: dict[str, list[str]]) -> list[tuple[str, str]]:
    items = []
    for k, lbl in CLASS_LABELS.items():
        n = len(watch.get(k, []))
        items.append((k, f"{lbl} · {n}"))
    return items

choices = _choices_with_counts(WATCH)
default_key = "BR_STOCKS" if len(WATCH.get("BR_STOCKS", [])) else choices[0][0]
selected_key = st.sidebar.selectbox(
    "Classe", options=[c[0] for c in choices], format_func=lambda k: dict(choices)[k], index=[c[0] for c in choices].index(default_key)
)

st.sidebar.header("Janela")
period = st.sidebar.selectbox("Período", ["3mo", "6mo", "1y", "2y", "5y"], index=1)
interval = st.sidebar.selectbox("Intervalo", ["1d", "1wk", "1h"], index=0)

st.sidebar.header("Filtros")
min_price = st.sidebar.number_input("Preço mínimo", value=1.00, step=0.5, format="%.2f")
min_avgvol = st.sidebar.number_input("Volume médio mín. (unid.)", value=100_000, step=10_000, format="%d")
trend_only = st.sidebar.toggle("Somente tendência de alta (SMA50 > SMA200)", value=False)
max_to_process = st.sidebar.slider("Máx. de ativos processados", 10, 200, 60, 1)

# Botão para forçar “break” do cache de dados
colA, colB = st.sidebar.columns([1, 1])
with colA:
    refresh = st.button("🔄 Recarregar dados")
with colB:
    st.write("")

# Session key para seleção
if "screener_selected" not in st.session_state:
    st.session_state["screener_selected"] = []

# Versão do cache (incrementa quando usuário quer forçar reload)
if "screener_cache_ver" not in st.session_state:
    st.session_state["screener_cache_ver"] = 0
if refresh:
    st.session_state["screener_cache_ver"] += 1
ver = st.session_state["screener_cache_ver"]

# ---------------------------------------------------------------------
# Funções de indicadores
# ---------------------------------------------------------------------
def _close_col(df: pd.DataFrame) -> pd.Series:
    for c in ["Adj Close", "AdjClose", "Close", "close"]:
        if c in df.columns:
            return df[c].astype(float)
    # fallback: primeira coluna numérica
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        return df[num_cols[0]].astype(float)
    return pd.Series(dtype=float)

def _volume_col(df: pd.DataFrame) -> pd.Series:
    for c in ["Volume", "volume", "Vol", "vol"]:
        if c in df.columns:
            return df[c].astype(float)
    return pd.Series(dtype=float)

def _pct(df: pd.DataFrame, bars: int) -> float:
    c = _close_col(df)
    if len(c) < bars + 1:
        return np.nan
    return (c.iloc[-1] / c.iloc[-(bars + 1)] - 1.0) * 100.0

def _rsi14(df: pd.DataFrame) -> float:
    c = _close_col(df)
    if len(c) < 15:
        return np.nan
    delta = c.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def _sma(df: pd.DataFrame, n: int) -> float:
    c = _close_col(df)
    if len(c) < n:
        return np.nan
    return float(c.rolling(n).mean().iloc[-1])

def _vol_ann(df: pd.DataFrame, interval: str) -> float:
    c = _close_col(df).pct_change().dropna()
    if c.empty:
        return np.nan
    if interval == "1d":
        scale = math.sqrt(252)
    elif interval == "1wk":
        scale = math.sqrt(52)
    elif interval == "1h":
        # aproximado (6h úteis/dia * 252 dias ~ 1512 barras)
        scale = math.sqrt(1512)
    else:
        scale = 1.0
    return float(c.std() * 100.0 * scale)

def _avgvol(df: pd.DataFrame, lookback=60) -> float:
    v = _volume_col(df)
    if v.empty:
        return np.nan
    x = v.tail(lookback)
    if x.empty:
        return np.nan
    return float(x.mean())

# ---------------------------------------------------------------------
# Cache de dados (wrapper com parâmetros “hasháveis”)
# ATENÇÃO: não passe função/callback para o cache (daria UnhashableParamError)
# ---------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_bulk(period: str, interval: str, symbols_tuple: tuple[str, ...], _ver: int) -> dict[str, pd.DataFrame]:
    # _ver é propositalmente não usado: serve só para invalidar o cache quando muda
    symbols = list(symbols_tuple)
    return download_bulk(symbols, period=period, interval=interval)

# ---------------------------------------------------------------------
# Monta universo e baixa dados
# ---------------------------------------------------------------------
symbols_all = WATCH.get(selected_key, [])[:]
total_in_class = len(symbols_all)

st.markdown(
    f"Processando **{min(total_in_class, max_to_process)}** de **{total_in_class}** ativos desta classe…"
)

symbols = symbols_all[:max_to_process]

# Baixa tudo de uma vez (mais rápido do que 1 a 1)
with st.spinner("Baixando séries..."):
    data_dict = _fetch_bulk(period, interval, tuple(symbols), ver)

# ---------------------------------------------------------------------
# Calcula métricas por ativo
# ---------------------------------------------------------------------
@dataclass
class Row:
    Symbol: str
    Price: float
    D1: float
    D5: float
    M1: float
    M6: float
    Y1: float
    VolAnn: float
    AvgVol: float
    RSI14: float
    TrendUp: bool
    Score: float

rows: list[Row] = []

prog = st.progress(0.0, text="Processando métricas...")
n = len(symbols)
for i, s in enumerate(symbols, start=1):
    df = data_dict.get(s)
    if df is None or len(df) < 5:
        prog.progress(i / max(1, n), text=f"Sem dados: {s}")
        continue

    close = _close_col(df)
    price = float(close.iloc[-1]) if len(close) else np.nan

    d1 = _pct(df, 1)
    d5 = _pct(df, 5)
    m1 = _pct(df, 21)
    m6 = _pct(df, 126)
    y1 = _pct(df, 252)

    volann = _vol_ann(df, interval)
    avgvol = _avgvol(df, 60)
    rsi = _rsi14(df)

    sma50 = _sma(df, 50)
    sma200 = _sma(df, 200)
    trend = bool(pd.notna(sma50) and pd.notna(sma200) and sma50 > sma200)

    # Score simples (normalização robusta por percentuais + bônus de tendência)
    comps = [d1, d5, m1, m6, y1]
    vals = [x for x in comps if pd.notna(x)]
    score = float(np.nan) if not vals else float(np.nanmean(vals))
    if trend and not math.isnan(score):
        score += 2.0

    rows.append(
      Row(
        Symbol=s, Price=price, D1=d1, D5=d5, M1=m1, M6=m6, Y1=y1,
        VolAnn=volann, AvgVol=avgvol, RSI14=rsi, TrendUp=trend, Score=score
      )
    )
    prog.progress(i / max(1, n), text=f"Processando {s} ({i}/{n})")

prog.empty()

if not rows:
    st.warning("Nenhum ativo processado. Ajuste filtros/consulta ou atualize as watchlists em **Settings**.")
    st.stop()

df = pd.DataFrame([r.__dict__ for r in rows])

# ---------------------------------------------------------------------
# Aplica filtros
# ---------------------------------------------------------------------
mask = pd.Series(True, index=df.index)
mask &= df["Price"] >= float(min_price)
mask &= (df["AvgVol"].fillna(0.0) >= float(min_avgvol))
if trend_only:
    mask &= df["TrendUp"].fillna(False)

df = df[mask].copy()

# ---------------------------------------------------------------------
# Ordenação
# ---------------------------------------------------------------------
st.subheader("Ordenar por")
order_cols = {
    "Score": "Score",
    "Preço": "Price",
    "D1%": "D1",
    "D5%": "D5",
    "M1%": "M1",
    "M6%": "M6",
    "Y1%": "Y1",
    "VolAnn%": "VolAnn",
    "RSI14": "RSI14",
    "AvgVol": "AvgVol",
    "Tendência": "TrendUp",
}
col_s, col_dir = st.columns([3, 1])
with col_s:
    order_by = st.selectbox("Coluna", list(order_cols.keys()), index=0)
with col_dir:
    asc = st.toggle("Ordem crescente", value=False)

df.sort_values(order_cols[order_by], ascending=asc, inplace=True, na_position="last")

# ---------------------------------------------------------------------
# Exibição (com um pouco de cor nas variações e no score)
# ---------------------------------------------------------------------
def _fmt_pct(x):
    return "" if pd.isna(x) else f"{x:,.2f}%"

def _fmt_price(x):
    return "" if pd.isna(x) else f"{x:,.2f}"

def _fmt_int(x):
    return "" if pd.isna(x) else f"{int(x):,}".replace(",", ".")

formatters = {
    "Price": _fmt_price,
    "D1": _fmt_pct,
    "D5": _fmt_pct,
    "M1": _fmt_pct,
    "M6": _fmt_pct,
    "Y1": _fmt_pct,
    "VolAnn": _fmt_pct,
    "AvgVol": _fmt_int,
    "RSI14": lambda x: "" if pd.isna(x) else f"{x:,.1f}",
    "Score": lambda x: "" if pd.isna(x) else f"{x:,.2f}",
}

def _color_pct(v):
    if pd.isna(v): 
        return ""
    c = "#2ecc71" if v >= 0 else "#e74c3c"
    return f"color:{c}"

def _color_score(v):
    if pd.isna(v):
        return ""
    c = "#2ecc71" if v >= 0 else "#e74c3c"
    return f"font-weight:600;color:{c}"

styled = (
    df.rename(columns={
        "Symbol": "Symbol",
        "Price": "Price",
        "D1": "D1%",
        "D5": "D5%",
        "M1": "M1%",
        "M6": "M6%",
        "Y1": "Y1%",
        "VolAnn": "VolAnn%",
        "AvgVol": "AvgVol",
        "RSI14": "RSI14",
        "TrendUp": "TrendUp",
        "Score": "Score",
    })
      .style
      .map(_color_pct, subset=["D1%","D5%","M1%","M6%","Y1%"])
      .map(_color_score, subset=["Score"])
      .format({
          "Price": _fmt_price,
          "D1%": _fmt_pct, "D5%": _fmt_pct, "M1%": _fmt_pct, "M6%": _fmt_pct, "Y1%": _fmt_pct,
          "VolAnn%": _fmt_pct, "AvgVol": _fmt_int, "RSI14": lambda x: "" if pd.isna(x) else f"{x:,.1f}",
          "Score": lambda x: "" if pd.isna(x) else f"{x:,.2f}",
      })
)

st.dataframe(styled, height=480, width="stretch")

# ---------------------------------------------------------------------
# Seleção para envio ao Backtest
# ---------------------------------------------------------------------
st.subheader("Marque os ativos que deseja enviar para o Backtest")

# Sugere seleção padrão (os 10 melhores por Score)
suggest = df.sort_values("Score", ascending=False).head(10)["Symbol"].tolist()

selected = st.multiselect(
    "Selecione tickers",
    options=df["Symbol"].tolist(),
    default=st.session_state.get("screener_selected", suggest),
)

if st.button("Usar seleção no Backtest"):
    st.session_state["screener_selected"] = selected
    st.success(
        f"Seleção salva: {len(selected)} ativos. "
        "Use a opção de Backtest para continuar."
    )
