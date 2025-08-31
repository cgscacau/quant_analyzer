# pages/3_🔎_Screener.py
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Tentamos usar utilidades internas do projeto (se existirem)
# ============================================================
_dl_bulk = None
_get_watchlists = None

try:
    # baixe em lote (dict[str, pd.DataFrame]) usando a infra do projeto
    from core.data import download_bulk as _dl_bulk  # type: ignore
except Exception:
    _dl_bulk = None

try:
    # carrega watchlists (dict com chaves BR_STOCKS, BR_FIIS, ..., US_STOCKS, CRYPTO etc.)
    from core.data import load_watchlists as _get_watchlists  # type: ignore
except Exception:
    _get_watchlists = None

# Fallback de download com yfinance (caso não exista o do projeto)
def _fallback_download_bulk(symbols: list[str], period: str = "6mo", interval: str = "1d") -> dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except Exception:
        st.error("yfinance não está disponível e não há downloader interno (core.data).")
        return {}

    out: dict[str, pd.DataFrame] = {}
    if not symbols:
        return out

    raw = yf.download(
        tickers=list(dict.fromkeys(symbols)),
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    # yfinance muda o formato quando é 1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        for s in symbols:
            if s in raw.columns.get_level_values(0):
                df = raw[s].copy()
                df.columns = [c.title() for c in df.columns]  # Open/High/Low/Close/Adj Close/Volume
                out[s] = df
    else:
        # um único ticker
        df = raw.copy()
        df.columns = [c.title() for c in df.columns]
        out[symbols[0]] = df

    return out


# ============================================================
# Helpers de formatação e estilos (robustos p/ Series e escalar)
# ============================================================
def _fmt_price(x):
    try:
        return "" if pd.isna(x) else f"{float(x):,.2f}"
    except Exception:
        return ""

def _fmt_int(x):
    try:
        return "" if pd.isna(x) else f"{float(x):,.0f}"
    except Exception:
        return ""

def _fmt_pct(x):
    try:
        return "" if pd.isna(x) else f"{float(x):+.2f}%"
    except Exception:
        return ""

def _color_pct(v):
    """
    Aceita escalar ou Series. Verde p/ >=0, vermelho p/ <0, vazio p/ NaN.
    Retorna string (quando escalar) ou Series de strings (quando Series).
    """
    if isinstance(v, pd.Series):
        s = pd.to_numeric(v, errors="coerce")
        out = np.where(s >= 0, "color:#22cc71", "color:#e74c3c")
        out = np.where(s.isna(), "", out)
        return pd.Series(out, index=s.index)

    try:
        f = float(v)
    except Exception:
        return ""
    if np.isnan(f):
        return ""
    return "color:#22cc71" if f >= 0 else "color:#e74c3c"

def _color_score(v):
    """
    Aceita escalar ou Series. Fundo verde/ vermelho conforme score.
    """
    def _one(f):
        if np.isnan(f): 
            return ""
        if f >= 1:
            return "background-color: rgba(34,204,113,.15)"   # verde mais forte
        if f >= 0:
            return "background-color: rgba(34,204,113,.05)"   # verde leve
        return "background-color: rgba(231,76,60,.08)"        # vermelho

    if isinstance(v, pd.Series):
        s = pd.to_numeric(v, errors="coerce")
        out = s.apply(_one)
        out[s.isna()] = ""
        return out

    try:
        f = float(v)
    except Exception:
        return ""
    return _one(f)


# ============================================================
# Cálculos técnicos
# ============================================================
def _rsi14(close: pd.Series) -> float | np.floating | float:
    close = pd.to_numeric(close, errors="coerce").dropna()
    if len(close) < 15:
        return np.nan
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    roll_up = gain.ewm(alpha=1/14, adjust=False).mean()
    roll_down = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi.iloc[-1])

def _annualized_vol(ret: pd.Series, periods_per_year: int = 252) -> float:
    ret = pd.to_numeric(ret, errors="coerce").dropna()
    if len(ret) == 0:
        return np.nan
    return float(ret.std(ddof=0) * math.sqrt(periods_per_year) * 100.0)

def _sma(series: pd.Series, n: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rolling(n).mean()

def _trend_up(close: pd.Series) -> bool | None:
    if len(close) < 200:
        return None
    sma50 = _sma(close, 50).iloc[-1]
    sma200 = _sma(close, 200).iloc[-1]
    if pd.isna(sma50) or pd.isna(sma200):
        return None
    return bool(sma50 > sma200)

def _pct_change(series: pd.Series, lookback: int) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= lookback:
        return np.nan
    try:
        return float((s.iloc[-1] / s.iloc[-1 - lookback] - 1.0) * 100.0)
    except Exception:
        return np.nan

def _to_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return np.nan


# ============================================================
# UI – Controles
# ============================================================
st.title("🔎 Screener")
st.caption("Triagem multi-ativos (BR/US/Cripto) com métricas e filtros")

# Watchlists
if _get_watchlists is not None:
    wl = _get_watchlists()
else:
    wl = {
        "BR_STOCKS": ["PETR4.SA", "VALE3.SA", "ITUB4.SA"],
        "BR_FIIS": ["MXRF11.SA", "HGLG11.SA"],
        "US_STOCKS": ["AAPL", "MSFT", "NVDA"],
        "CRYPTO": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "BR_DIVIDEND": [],
        "BR_BLUE_CHIPS": [],
        "BR_SMALL_CAPS": [],
        "US_BLUE_CHIPS": [],
        "US_SMALL_CAPS": [],
    }

classes = {
    "Brasil (Ações B3)": ("BR_STOCKS", wl.get("BR_STOCKS", [])),
    "Brasil (FIIs)": ("BR_FIIS", wl.get("BR_FIIS", [])),
    "Brasil — Dividendos": ("BR_DIVIDEND", wl.get("BR_DIVIDEND", [])),
    "Brasil — Blue Chips": ("BR_BLUE_CHIPS", wl.get("BR_BLUE_CHIPS", [])),
    "Brasil — Small Caps": ("BR_SMALL_CAPS", wl.get("BR_SMALL_CAPS", [])),
    "EUA (Ações US)": ("US_STOCKS", wl.get("US_STOCKS", [])),
    "EUA — Blue Chips": ("US_BLUE_CHIPS", wl.get("US_BLUE_CHIPS", [])),
    "EUA — Small Caps": ("US_SMALL_CAPS", wl.get("US_SMALL_CAPS", [])),
    "Criptos": ("CRYPTO", wl.get("CRYPTO", [])),
}

with st.sidebar:
    st.header("Classe")
    classe_label = st.selectbox(
        "Classe",
        list(classes.keys()),
        index=0,
    )
    _, symbols = classes[classe_label]
    st.caption(f"Total na classe: **{len(symbols)}**")

    st.header("Janela")
    period = st.selectbox("Período", ["6mo", "1y", "2y", "5y"], index=0)
    interval = st.selectbox("Intervalo", ["1d", "1wk"], index=0)

    st.header("Filtros")
    price_min = st.number_input("Preço mínimo", value=1.0, step=0.5, format="%.2f")
    vol_min = st.number_input("Volume médio mínimo (unid.)", value=100_000.0, step=10_000.0, format="%.0f")
    only_trend = st.checkbox("Somente tendência de alta (SMA50>SMA200)", value=False)

# ============================================================
# Download em lote (com cache)
# ============================================================
@st.cache_data(show_spinner=True)
def _download_bulk_cached(_symbols: tuple[str, ...], _period: str, _interval: str):
    if _dl_bulk is not None:
        return _dl_bulk(list(_symbols), period=_period, interval=_interval)
    return _fallback_download_bulk(list(_symbols), period=_period, interval=_interval)

# ============================================================
# Processamento
# ============================================================
if not symbols:
    st.info("Nenhum ativo nesta classe. Atualize as watchlists na página **Settings**.")
    st.stop()

st.write(f"Processando **{len(symbols)}** ativos desta classe…")
data_dict = _download_bulk_cached(tuple(symbols), period, interval)

rows = []
for sym in symbols:
    df = data_dict.get(sym)
    if df is None or df.empty:
        continue

    # Padroniza nomes e garante índice como datetime
    cols = {c.lower(): c for c in df.columns}
    for need in ["open", "high", "low", "close", "adj close", "volume"]:
        if need not in cols:
            # se faltou alguma coluna essencial, pula
            continue
    df = df.copy()
    df.columns = [c.title() for c in df.columns]
    df = df.dropna(how="all")
    if df.empty:
        continue

    close = pd.to_numeric(df.get("Adj Close", df.get("Close")), errors="coerce")
    vol = pd.to_numeric(df.get("Volume"), errors="coerce")

    last_price = float(close.iloc[-1]) if not pd.isna(close.iloc[-1]) else np.nan
    avg_vol = float(vol.tail(30).mean()) if len(vol) >= 1 else np.nan

    # Retornos percentuais (aprox. 1d, 5d, 21d, 126d, 252d)
    d1 = _pct_change(close, 1)
    d5 = _pct_change(close, 5)
    m1 = _pct_change(close, 21)
    m6 = _pct_change(close, 126)
    y1 = _pct_change(close, 252)

    # Vol anualizada
    ret_daily = close.pct_change()
    vol_ann = _annualized_vol(ret_daily, periods_per_year=252)

    # RSI14
    rsi14 = _rsi14(close)

    # Tendência
    t_up = _trend_up(close)

    # Score robusto: média dos disponíveis + bônus de tendência
    comps = [_to_float(d1), _to_float(d5), _to_float(m1), _to_float(m6), _to_float(y1)]
    vals = [v for v in comps if not np.isnan(v)]
    score = np.nan if not vals else float(np.mean(vals))
    if only_trend and t_up is False:
        # se a pessoa pediu apenas tendência e não está em tendência, marque score NaN
        score = np.nan
    elif (t_up is True) and (not np.isnan(score)):
        score += 2.0

    # Filtros mínimos
    if not np.isnan(price_min) and not np.isnan(last_price) and last_price < price_min:
        continue
    if not np.isnan(vol_min) and not np.isnan(avg_vol) and avg_vol < vol_min:
        continue
    if only_trend and (t_up is not True):
        continue

    rows.append(
        {
            "Symbol": sym,
            "Price": last_price,
            "D1%": d1,
            "D5%": d5,
            "M1%": m1,
            "M6%": m6,
            "Y1%": y1,
            "VolAnn%": vol_ann,
            "AvgVol": avg_vol,
            "RSI14": rsi14,
            "TrendUp": t_up,
            "Score": score,
        }
    )

df = pd.DataFrame(rows)

if df.empty:
    st.warning("Nenhum ativo passou nos filtros. Ajuste os critérios.")
    st.stop()

# ============================================================
# Ordenação
# ============================================================
st.subheader("Ordenar por")
order_col = st.selectbox(
    "Ordenar por",
    ["Score", "D1%", "D5%", "M1%", "M6%", "Y1%", "Price", "VolAnn%", "AvgVol", "RSI14", "TrendUp", "Symbol"],
    index=0,
)
ascending = st.checkbox("Ordem crescente", value=False)
df = df.sort_values(by=order_col, ascending=ascending, kind="mergesort").reset_index(drop=True)

# ============================================================
# Tabela estilizada (cores robustas para Series/escalares)
# ============================================================
cols_pct = ["D1%", "D5%", "M1%", "M6%", "Y1%"]

styled = (
    df.rename(
        columns={
            "Symbol": "Symbol",
            "Price": "Price",
            "D1%": "D1%",
            "D5%": "D5%",
            "M1%": "M1%",
            "M6%": "M6%",
            "Y1%": "Y1%",
            "VolAnn%": "VolAnn%",
            "AvgVol": "AvgVol",
            "RSI14": "RSI14",
            "TrendUp": "TrendUp",
            "Score": "Score",
        }
    )
    .style
    .map(_color_pct, subset=cols_pct)
    .map(_color_score, subset=["Score"])
    .format(
        {
            "Price": _fmt_price,
            "D1%": _fmt_pct,
            "D5%": _fmt_pct,
            "M1%": _fmt_pct,
            "M6%": _fmt_pct,
            "Y1%": _fmt_pct,
            "VolAnn%": _fmt_pct,
            "AvgVol": _fmt_int,
            "RSI14": lambda x: "" if pd.isna(x) else f"{x:.1f}",
            "Score": lambda x: "" if pd.isna(x) else f"{x:.2f}",
        }
    )
)

st.dataframe(styled, height=480, use_container_width=True)

# ============================================================
# Seleção p/ Backtest (opcional – id na sessão)
# ============================================================
st.subheader("Marque os ativos que deseja enviar para o Backtest")
if "screener_selected" not in st.session_state:
    st.session_state["screener_selected"] = []

# (Opcional) Botão para disponibilizar a lista atual no estado
if st.button("Usar seleção no Backtest"):
    st.session_state["screener_selected"] = df["Symbol"].tolist()
    st.success("Seleção salva! Use-a na página **Backtest**.")
