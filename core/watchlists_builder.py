# core/watchlists_builder.py
from __future__ import annotations
import time, re
from typing import Iterable, Dict, List, Tuple
import pandas as pd
import yfinance as yf

# ---------------------------- Parâmetros (ajustáveis) ----------------------------
LAST_DAYS       = 60      # janela que vamos olhar
MIN_TRADING_DAYS= 6       # mínimo de candles no período
MAX_LAST_GAP    = 30      # último candle tem que estar a <= X dias
DIV_YIELD_MIN   = 0.04    # 4% TTM para classificar como "dividendos"
TOP_PCT         = 0.30    # top 30% market cap = blue chips
BOTTOM_PCT      = 0.30    # bottom 30% market cap = small caps
SLEEP           = 0.20    # intervalo entre chamadas p/ não apanhar do Yahoo

# ---------------------------- Helpers ----------------------------
def _days_since(ts) -> int:
    """Retorna quantos dias se passaram desde ts até agora, lidando com tz."""
    now = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(None)
    t = pd.Timestamp(ts)
    try:
        # Se vier tz-aware, converte para naïve
        if getattr(t, "tzinfo", None) is not None:
            t = t.tz_convert(None)
    except Exception:
        # Alguns índices são "time-zone-naive" e tz_convert falha; garante naïve
        t = t.tz_localize(None)
    return int((now - t).days)

def _active_last_2m(tickers: Iterable[str], debug: bool=False) -> List[str] | Tuple[List[str], List[Tuple[str,str]]]:
    """
    Filtra tickers com dados nos últimos LAST_DAYS, gap <= MAX_LAST_GAP,
    e pelo menos MIN_TRADING_DAYS candles. Se debug=True, retorna (ativos, report).
    """
    active: List[str] = []
    report: List[Tuple[str,str]] = []

    for t in tickers:
        reason = ""
        try:
            df = yf.download(
                t, period=f"{LAST_DAYS}d", interval="1d",
                progress=False, auto_adjust=False
            )
            if df is None or df.empty or "Close" not in df.columns:
                reason = "sem dados"
            else:
                df = df.dropna()
                if len(df) < MIN_TRADING_DAYS:
                    reason = f"poucos candles ({len(df)})"
                else:
                    gap = _days_since(df.index[-1])
                    if gap > MAX_LAST_GAP:
                        reason = f"gap {gap}d > {MAX_LAST_GAP}d"
                    else:
                        # se houver coluna volume, exige >0 em pelo menos metade dos dias
                        if "Volume" in df.columns and (df["Volume"] > 0).sum() < max(1, MIN_TRADING_DAYS // 2):
                            reason = "volume baixo"
            if reason:
                if debug: report.append((t, reason))
            else:
                active.append(t)
        except Exception as e:
            if debug: report.append((t, f"erro: {type(e).__name__}"))
        time.sleep(SLEEP)

    return (active, report) if debug else active

def _fast_market_cap(t: str) -> float | None:
    try:
        tk = yf.Ticker(t)
        mc = None
        fi = getattr(tk, "fast_info", None)
        if fi:
            mc = fi.get("market_cap") or fi.get("marketCap")
        if mc is None:
            info = tk.get_info()
            mc = info.get("marketCap")
        return float(mc) if mc is not None else None
    except Exception:
        return None

def _ttm_div_yield(t: str) -> float:
    try:
        tk = yf.Ticker(t)
        div = tk.dividends
        if div is None or div.empty:
            return 0.0
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
        total = float(div[div.index >= cutoff].sum())
        px = float(tk.history(period="1d")["Close"].iloc[-1])
        return (total / px) if px > 0 else 0.0
    except Exception:
        return 0.0

def _split_caps(tickers: List[str]) -> Tuple[List[str], List[str]]:
    mcap: Dict[str, float] = {t: _fast_market_cap(t) or 0.0 for t in tickers}
    ordered = sorted(mcap.items(), key=lambda kv: kv[1], reverse=True)
    if not ordered:
        return [], []
    n = len(ordered)
    k_top = max(3, int(TOP_PCT * n))
    k_bot = max(3, int(BOTTOM_PCT * n))
    blue  = [t for t, _ in ordered[:k_top]]
    small = [t for t, _ in ordered[-k_bot:]]
    return blue, small

def _div_payers(tickers: List[str]) -> List[str]:
    out = []
    for t in tickers:
        if _ttm_div_yield(t) >= DIV_YIELD_MIN:
            out.append(t)
        time.sleep(SLEEP)
    return out

def _is_fii_brazil(t: str) -> bool:
    # Heurística comum para FIIs: final '11.SA'
    return t.endswith("11.SA") and re.search(r"\d{2}\.SA$", t) is not None

# ---------------------------- Builder principal ----------------------------
def rebuild_watchlists(base: dict, debug: bool=False) -> dict | Tuple[dict, List[Tuple[str,str]]]:
    """
    Recebe um dicionário-base com chaves BR_STOCKS / US_STOCKS / CRYPTO
    e retorna um novo dicionário com:
      - listas filtradas por atividade recente
      - classes: BR_FIIS, BR_DIVIDEND, BR_SMALL_CAPS, BR_BLUE_CHIPS,
                 US_DIVIDEND, US_SMALL_CAPS, US_BLUE_CHIPS
    Se debug=True, retorna (resultado, report) onde 'report' tem motivo por ticker.
    """
    br_base = base.get("BR_STOCKS", [])
    us_base = base.get("US_STOCKS", [])
    cr_base = base.get("CRYPTO", [])

    # 1) atividade
    br_active, rep_br = _active_last_2m(br_base, debug=True)
    us_active, rep_us = _active_last_2m(us_base,  debug=True)
    cr_active, rep_cr = _active_last_2m(cr_base,  debug=True)

    # 2) classes Brasil
    br_fiis     = [t for t in br_active if _is_fii_brazil(t)]
    br_equities = [t for t in br_active if t not in br_fiis]
    br_blue, br_small = _split_caps(br_equities)
    br_div      = _div_payers(br_equities)

    # 3) classes EUA
    us_blue, us_small = _split_caps(us_active)
    us_div      = _div_payers(us_active)

    out = {
        "BR_STOCKS":      sorted(br_equities),
        "US_STOCKS":      sorted(us_active),
        "CRYPTO":         sorted(cr_active),
        "BR_FIIS":        sorted(br_fiis),
        "BR_BLUE_CHIPS":  sorted(br_blue),
        "BR_SMALL_CAPS":  sorted(br_small),
        "BR_DIVIDEND":    sorted(br_div),
        "US_BLUE_CHIPS":  sorted(us_blue),
        "US_SMALL_CAPS":  sorted(us_small),
        "US_DIVIDEND":    sorted(us_div),
    }

    if debug:
        report = rep_br + rep_us + rep_cr
        return out, report
    return out
