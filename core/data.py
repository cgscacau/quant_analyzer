# core/data.py
from __future__ import annotations

# Stdlib
from pathlib import Path
import json
from typing import Dict, List

# Third-party
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st


# =============================================================================
# Watchlists (listas de ativos)
# =============================================================================

# Caminho do arquivo: <raiz do projeto>/data/watchlists.json
_WL_FILE = Path(__file__).resolve().parents[1] / "data" / "watchlists.json"

# Chaves esperadas (básicas + classes adicionais)
_WL_KEYS = [
    "BR_STOCKS", "US_STOCKS", "CRYPTO",
    "BR_FIIS", "BR_DIVIDEND", "BR_SMALL_CAPS", "BR_BLUE_CHIPS",
    "US_DIVIDEND", "US_SMALL_CAPS", "US_BLUE_CHIPS",
]

# Fallback mínimo se o arquivo não existir
_DEFAULT_WL = {
    "BR_STOCKS": ["PETR4.SA", "VALE3.SA", "ITUB4.SA"],
    "US_STOCKS": ["AAPL", "MSFT", "NVDA"],
    "CRYPTO":    ["BTC-USD", "ETH-USD", "SOL-USD"],
}


@st.cache_data(ttl=86400)
def read_watchlists_file() -> dict:
    """
    Lê SEMPRE do arquivo físico (cacheado 24h) e garante as chaves padrão.
    Use esta função na Settings como semente para reconstrução.
    """
    try:
        data = json.loads(_WL_FILE.read_text(encoding="utf-8"))
    except Exception:
        data = dict(_DEFAULT_WL)

    for k in _WL_KEYS:
        data.setdefault(k, [])
    return data


def load_watchlists() -> dict:
    """
    Fonte única para as páginas:
      - Se existir 'watchlists_override' em sessão (Settings), usa-o;
      - Caso contrário, usa o arquivo (via read_watchlists_file()).
    """
    return st.session_state.get("watchlists_override", read_watchlists_file())


# =============================================================================
# Dados de mercado (yfinance) + normalização OHLC
# =============================================================================

def _to_dtindex(idx) -> pd.DatetimeIndex:
    """Garante DatetimeIndex sem timezone (naïve)."""
    di = pd.to_datetime(idx)
    # remove timezone (yfinance pode retornar tz-aware)
    if getattr(di, "tz", None) is not None:
        di = di.tz_convert(None)
    return di


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomeia e padroniza colunas para: Open, High, Low, Close, Adj Close, Volume.
    Remove linhas completamente vazias e garante tipos numéricos.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # nomes normalizados (lower + espaço no lugar de '_')
    lower_map = {c: c.lower().replace("_", " ").strip() for c in df.columns}

    # dicionário de destino
    target = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adjclose": "Adj Close",
        "volume": "Volume",
    }

    # constrói mapa de renome
    rename_map: Dict[str, str] = {}
    for orig, low in lower_map.items():
        if low in target:
            rename_map[orig] = target[low]

    out = df.rename(columns=rename_map).copy()

    # mantém apenas as colunas de interesse, se existirem
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in out.columns]
    out = out[keep]

    # garante datetime index e ordenação
    out.index = _to_dtindex(out.index)
    out = out.sort_index()

    # converte para numérico
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # remove linhas totalmente vazias
    out = out.dropna(how="all")

    return out


@st.cache_data(ttl=600, show_spinner=False)
def download_history(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Baixa OHLCV de um único ativo via yfinance e normaliza as colunas.
    Cacheado por 10 minutos.
    """
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df.index = _to_dtindex(df.index)
        return normalize_ohlc(df)
    except Exception:
        # Não derruba a página por falha de 1 ativo
        return pd.DataFrame()



@st.cache_data(show_spinner=False)
def download_bulk(symbols: list[str], period: str, interval: str, ver: int = 0) -> dict[str, pd.DataFrame]:
    """Baixa dados para vários símbolos. `ver` só serve para invalidar o cache."""
    if not symbols:
        return {}
    result: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            df = yf.download(s, period=period, interval=interval, progress=False, auto_adjust=False)
        except Exception:
            df = pd.DataFrame()
        result[s] = df
    return result



# =============================================================================
# Utilidades
# =============================================================================

def clear_data_cache() -> None:
    """Limpa caches de dados (útil em Settings após atualizar watchlists)."""
    st.cache_data.clear()
