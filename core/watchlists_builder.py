# core/watchlists_builder.py
from __future__ import annotations
import re, time
from typing import Iterable, Dict, List, Tuple
import pandas as pd
import yfinance as yf

LAST_DAYS = 60
MIN_TRADING_DAYS = 10
MAX_LAST_GAP = 15
DIV_YIELD_MIN = 0.04
TOP_PCT = 0.30
BOTTOM_PCT = 0.30
SLEEP = 0.15  # respeitar rate-limit

def _active_last_2m(tickers: Iterable[str]) -> List[str]:
    active = []
    for t in tickers:
        try:
            df = yf.download(t, period="2mo", interval="1d", progress=False, auto_adjust=False)
            if df is None or df.empty or "Close" not in df.columns: 
                continue
            df = df.dropna()
            if len(df) < MIN_TRADING_DAYS: 
                continue
            last_idx = df.index[-1]
            gap = (pd.Timestamp.utcnow() - (last_idx.tz_localize(None))).days
            if gap > MAX_LAST_GAP: 
                continue
            if ("Volume" in df.columns) and (df["Volume"] > 0).sum() < MIN_TRADING_DAYS // 2:
                continue
            active.append(t)
        except Exception:
            pass
        time.sleep(SLEEP)
    return active

def _fast_market_cap(t: str) -> float | None:
    try:
        tk = yf.Ticker(t)
        fi = getattr(tk, "fast_info", None)
        mc = None
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
    return t.endswith("11.SA") and re.search(r"\d{2}\.SA$", t) is not None

def rebuild_watchlists(base: dict) -> dict:
    br_base = base.get("BR_STOCKS", [])
    us_base = base.get("US_STOCKS", [])
    crypto_base = base.get("CRYPTO", [])

    # filtrados (negociados 2 meses)
    br_active = _active_last_2m(br_base)
    us_active = _active_last_2m(us_base)
    cr_active = _active_last_2m(crypto_base)

    # classes Brasil
    br_fiis = [t for t in br_active if _is_fii_brazil(t)]
    br_equities = [t for t in br_active if t not in br_fiis]
    br_blue, br_small = _split_caps(br_equities)
    br_div = _div_payers(br_equities)

    # classes EUA
    us_blue, us_small = _split_caps(us_active)
    us_div = _div_payers(us_active)

    return {
        "BR_STOCKS": sorted(br_equities),
        "US_STOCKS": sorted(us_active),
        "CRYPTO":    sorted(cr_active),
        "BR_FIIS":           sorted(br_fiis),
        "BR_BLUE_CHIPS":     sorted(br_blue),
        "BR_SMALL_CAPS":     sorted(br_small),
        "BR_DIVIDEND":       sorted(br_div),
        "US_BLUE_CHIPS":     sorted(us_blue),
        "US_SMALL_CAPS":     sorted(us_small),
        "US_DIVIDEND":       sorted(us_div),
    }
