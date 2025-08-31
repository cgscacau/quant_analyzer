# core/data.py
# core/data.py (trecho)
from __future__ import annotations
from pathlib import Path
import json
import streamlit as st

_WL_FILE = Path(__file__).resolve().parents[1] / "data" / "watchlists.json"

_WL_KEYS = [
    "BR_STOCKS", "US_STOCKS", "CRYPTO",
    "BR_FIIS", "BR_DIVIDEND", "BR_SMALL_CAPS", "BR_BLUE_CHIPS",
    "US_DIVIDEND", "US_SMALL_CAPS", "US_BLUE_CHIPS",
]

_DEFAULT = {
    "BR_STOCKS": ["PETR4.SA", "VALE3.SA", "ITUB4.SA"],
    "US_STOCKS": ["AAPL", "MSFT", "NVDA"],
    "CRYPTO":    ["BTC-USD", "ETH-USD", "SOL-USD"],
}

@st.cache_data(ttl=86400)
def read_watchlists_file() -> dict:
    """Lê SEMPRE do arquivo (cacheado 24h) e garante as chaves padrão."""
    try:
        data = json.loads(_WL_FILE.read_text(encoding="utf-8"))
    except Exception:
        data = dict(_DEFAULT)
    for k in _WL_KEYS:
        data.setdefault(k, [])
    return data

def load_watchlists() -> dict:
    """
    Retorna o override em memória (se existir), caso contrário o arquivo.
    Use esta função nas páginas (Screener etc.).
    """
    return st.session_state.get("watchlists_override", read_watchlists_file())

# ----------------------------
# Helpers de normalização OHLC
# ----------------------------
def _flatten_col(col) -> str:
    """Converte colunas MultiIndex/tuplas para string simples e limpa o nome."""
    if isinstance(col, tuple):
        for part in reversed(col):
            if isinstance(part, str):
                return part.strip()
        return "_".join(str(p) for p in col)
    return str(col).strip()

def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza colunas para ['Open','High','Low','Close','Adj Close','Volume']."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open","High","Low","Close","Adj Close","Volume"])

    df = df.copy()

    # Achata possíveis MultiIndex
    try:
        df.columns = [_flatten_col(c) for c in df.columns]
    except Exception:
        df.columns = [str(c) for c in df.columns]

    # Renomeia variações para padrão
    rename_map = {}
    for c in df.columns:
        key = c.lower().replace("_", " ").strip()
        if key == "open":         rename_map[c] = "Open"
        elif key == "high":       rename_map[c] = "High"
        elif key == "low":        rename_map[c] = "Low"
        elif key == "close":      rename_map[c] = "Close"
        elif key in ("adj close", "adjclose", "adj close*"): rename_map[c] = "Adj Close"
        elif key == "volume":     rename_map[c] = "Volume"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Garante 'Close'
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    # Ordena e remove duplicados
    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]

    # Mantém apenas essenciais que existirem
    keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    return df[keep]

# ----------------------------
# Download de dados (Yahoo)
# ----------------------------
def download_history(symbol: str, period: str="1y", interval: str="1d") -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="ticker",   # evita MultiIndex por ticker
            threads=False,
        )
    except Exception:
        df = pd.DataFrame()

    df = normalize_ohlc(df)

    # Fallback simples se vier vazio ou sem Close
    if df.empty or "Close" not in df.columns:
        try:
            df = yf.download(symbol, period="1y", interval="1d", progress=False, threads=False)
            df = normalize_ohlc(df)
        except Exception:
            df = pd.DataFrame()

    return df

# ----------------------------
# Info do ticker (opcional)
# ----------------------------
def get_info(symbol: str) -> dict:
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        return dict(info)
    except Exception:
        return {}

# ======= Bulk helpers para Screener =======
def _norm_single_ticker(df_ticker: pd.DataFrame) -> pd.DataFrame:
    """Normaliza o DataFrame de um único ticker (já fatiado) para OHLC padrão."""
    return normalize_ohlc(df_ticker)

def download_bulk(symbols: list[str], period: str = "6mo", interval: str = "1d") -> dict[str, pd.DataFrame]:
    """
    Baixa vários tickers de uma vez e retorna {ticker: df_normalizado}.
    Usa yf.download com group_by='ticker' e faz fallback individual quando necessário.
    """
    result: dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in symbols}
    if not symbols:
        return result

    try:
        raw = yf.download(
            tickers=" ".join(symbols),
            period=period,
            interval=interval,
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=False,
        )
    except Exception:
        raw = pd.DataFrame()

    # Caso múltiplos → columns multiindex por ticker
    # Caso único → DataFrame simples
    if isinstance(raw.columns, pd.MultiIndex):
        for sym in symbols:
            if sym in raw.columns.get_level_values(0):
                df_sym = raw[sym]
                result[sym] = _norm_single_ticker(df_sym)
    else:
        # Pode ter vindo 1 só ou falha; tenta normalizar e aplicar a todos (?) — melhor refazer por ticker
        raw = normalize_ohlc(raw)
        if not raw.empty:
            result[symbols[0]] = raw

    # Fallback por símbolo vazio
    for s in symbols:
        if result[s].empty:
            try:
                df = yf.download(s, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
                result[s] = normalize_ohlc(df)
            except Exception:
                result[s] = pd.DataFrame()

    return result


