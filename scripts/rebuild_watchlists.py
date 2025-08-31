# scripts/rebuild_watchlists.py
from __future__ import annotations
import json, time, re
from pathlib import Path
from typing import Iterable, Dict, List, Tuple

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
WL_PATH = ROOT / "data" / "watchlists.json"

# ---------------------------- Config ----------------------------
LAST_DAYS = 60                 # janela "negociados nos últimos 2 meses"
MIN_TRADING_DAYS = 10          # mínimo de candles no período
MAX_LAST_GAP = 15              # última cotação deve estar a <= X dias

DIV_YIELD_MIN = 0.04           # 4% TTM para "pagadoras de dividendos"
TOP_PCT = 0.30                 # top 30% por market cap = blue chips
BOTTOM_PCT = 0.30              # bottom 30% por market cap = small caps
SLEEP = 0.15                   # descanso entre chamadas para não ser bloqueado

# ---------------------------- Utils ----------------------------
def _active_last_2m(tickers: Iterable[str]) -> List[str]:
    """Retorna tickers com candles recentes (últimos 60d), volume>0 e >= MIN_TRADING_DAYS."""
    active = []
    for t in tickers:
        try:
            df = yf.download(t, period="2mo", interval="1d", progress=False, auto_adjust=False)
            if df is None or df.empty or "Close" not in df.columns:
                continue
            df = df.dropna()
            if len(df) < MIN_TRADING_DAYS:
                continue
            # último dia não muito distante
            last_idx = df.index[-1]
            gap = (pd.Timestamp.utcnow().tz_localize("UTC") - last_idx.tz_localize("UTC")).days if last_idx.tzinfo else \
                  (pd.Timestamp.utcnow() - last_idx).days
            if gap > MAX_LAST_GAP:
                continue
            # volume > 0 em parte dos dias
            if ("Volume" in df.columns) and (df["Volume"] > 0).sum() < MIN_TRADING_DAYS // 2:
                continue
            active.append(t)
        except Exception:
            pass
        time.sleep(SLEEP)
    return active

def _fast_market_cap(t: str) -> float | None:
    """Market cap via fast_info/ info (pode ser None)."""
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
    """Dividend yield TTM simples (soma últimos 365 dias / último preço)."""
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
    """Retorna (blue_chips, small_caps) por percentis de market cap."""
    mcap: Dict[str, float] = {t: _fast_market_cap(t) or 0.0 for t in tickers}
    # ordena por mcap desc
    ordered = sorted(mcap.items(), key=lambda kv: kv[1], reverse=True)
    if not ordered:
        return [], []
    n = len(ordered)
    k_top = max(3, int(TOP_PCT * n))
    k_bot = max(3, int(BOTTOM_PCT * n))
    blue = [t for t, _ in ordered[:k_top]]
    small = [t for t, _ in ordered[-k_bot:]]
    return blue, small

def _div_payers(tickers: List[str]) -> List[str]:
    out = []
    for t in tickers:
        y = _ttm_div_yield(t)
        if y >= DIV_YIELD_MIN:
            out.append(t)
        time.sleep(SLEEP)
    return out

def _is_fii_brazil(t: str) -> bool:
    """FIIs brasileiros costumam terminar com '11.SA'."""
    return bool(re.search(r"\d{2}\.SA$", t)) and t.endswith("11.SA")

# ---------------------------- Main ----------------------------
def main():
    # 1) lê a lista base existente (se não houver, usa dicionário vazio)
    base = {}
    if WL_PATH.exists():
        base = json.loads(WL_PATH.read_text(encoding="utf-8"))
    br_base = base.get("BR_STOCKS", [])
    us_base = base.get("US_STOCKS", [])
    crypto_base = base.get("CRYPTO", [])

    print(f"[1/5] Base: BR={len(br_base)} US={len(us_base)} CRYPTO={len(crypto_base)}")

    # 2) filtra quem negociou nos últimos 2 meses
    print("[2/5] Checando atividade (últimos 2 meses)...")
    br_active = _active_last_2m(br_base)
    us_active = _active_last_2m(us_base)
    cr_active = _active_last_2m(crypto_base)
    print(f"     Ativos: BR={len(br_active)} US={len(us_active)} CRYPTO={len(cr_active)}")

    # 3) classes Brasil
    print("[3/5] Classificando Brasil (FIIs, dividendos, caps)...")
    br_fiis = [t for t in br_active if _is_fii_brazil(t)]
    br_equities = [t for t in br_active if t not in br_fiis]
    br_blue, br_small = _split_caps(br_equities)
    br_div = _div_payers(br_equities)

    # 4) classes EUA
    print("[4/5] Classificando EUA (dividendos, caps)...")
    us_blue, us_small = _split_caps(us_active)
    us_div = _div_payers(us_active)

    # 5) escreve JSON final
    print("[5/5] Escrevendo data/watchlists.json ...")
    out = {
        # listas “base” já filtradas por atividade
        "BR_STOCKS": sorted(br_equities),
        "US_STOCKS": sorted(us_active),
        "CRYPTO":    sorted(cr_active),

        # novas classes
        "BR_FIIS":           sorted(br_fiis),
        "BR_BLUE_CHIPS":     sorted(br_blue),
        "BR_SMALL_CAPS":     sorted(br_small),
        "BR_DIVIDEND":       sorted(br_div),
        "US_BLUE_CHIPS":     sorted(us_blue),
        "US_SMALL_CAPS":     sorted(us_small),
        "US_DIVIDEND":       sorted(us_div),
    }
    WL_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK!")

if __name__ == "__main__":
    main()
