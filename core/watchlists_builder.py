# core/watchlists_builder.py
from __future__ import annotations

"""
Constrói/consolida watchlists a partir de um dicionário-base (arquivo),
aplicando filtros de atividade recente e classificações por FIIs, Dividendos,
Blue Chips e Small Caps.

- Não usa variáveis globais dependentes de `base`.
- Tolerante a timezone nos índices do yfinance.
- Possui fallbacks para evitar retorno completamente vazio quando a API limitar.
- Opcionalmente retorna um relatório de erros/motivos por ticker (debug=True).
"""

from typing import Dict, Iterable, List, Tuple
import re
import time

import pandas as pd
import yfinance as yf


# ============================ Parâmetros (ajustáveis) ==========================
LAST_DAYS: int        = 60    # janela (dias) usada para verificar atividade
MIN_TRADING_DAYS: int = 6     # mínimo de candles na janela
MAX_LAST_GAP: int     = 30    # último candle deve estar a <= X dias
DIV_YIELD_MIN: float  = 0.04  # 4% TTM para classificar como "dividendos"
TOP_PCT: float        = 0.30  # top 30% = blue chips
BOTTOM_PCT: float     = 0.30  # bottom 30% = small caps
SLEEP: float          = 0.15  # pausa entre chamadas ao Yahoo (anti-rate-limit)


# =============================== Helper functions =============================
def _days_since(ts) -> int:
    """Dias entre agora e o timestamp `ts`, tratando índices tz-aware/naïve."""
    now = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(None)
    t = pd.Timestamp(ts)
    try:
        # se vier tz-aware
        if getattr(t, "tzinfo", None) is not None:
            t = t.tz_convert(None)
    except Exception:
        # se for naïve, tz_convert falha; garante naïve
        t = t.tz_localize(None)
    return int((now - t).days)


def _active_last_2m(
    tickers: Iterable[str], debug: bool = False
) -> List[str] | Tuple[List[str], List[Tuple[str, str]]]:
    """
    Mantém ativos com:
      - dados nos últimos `LAST_DAYS`,
      - gap <= `MAX_LAST_GAP`,
      - pelo menos `MIN_TRADING_DAYS` candles.
    Se debug=True, retorna também uma lista [(ticker, motivo)].
    """
    active: List[str] = []
    report: List[Tuple[str, str]] = []

    for t in tickers:
        reason = ""
        try:
            df = yf.download(
                t,
                period=f"{LAST_DAYS}d",
                interval="1d",
                progress=False,
                auto_adjust=False,
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
                        if "Volume" in df.columns and (df["Volume"] > 0).sum() < max(
                            1, MIN_TRADING_DAYS // 2
                        ):
                            reason = "volume baixo"
            if reason:
                if debug:
                    report.append((t, reason))
            else:
                active.append(t)
        except Exception as e:
            if debug:
                report.append((t, f"erro: {type(e).__name__}"))
        time.sleep(SLEEP)

    return (active, report) if debug else active


def _fast_market_cap(t: str) -> float | None:
    """Obtém market cap via fast_info / info; retorna None em caso de falha."""
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
    """Dividend yield TTM aproximado: soma dividendos 12m / preço atual."""
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
    """Classifica por market cap em Blue (top) e Small (bottom)."""
    mcap: Dict[str, float] = {t: _fast_market_cap(t) or 0.0 for t in tickers}
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
    """Filtra pagadoras de dividendos por yield TTM >= DIV_YIELD_MIN."""
    out: List[str] = []
    for t in tickers:
        if _ttm_div_yield(t) >= DIV_YIELD_MIN:
            out.append(t)
        time.sleep(SLEEP)
    return out


def _is_fii_brazil(t: str) -> bool:
    """Heurística para FIIs B3: geralmente terminam com '11.SA'."""
    return t.endswith("11.SA") and re.search(r"\d{2}\.SA$", t) is not None


def _ensure_nonempty(lst: List[str], fallback: Iterable[str], n: int = 50) -> List[str]:
    """Se a lista filtrada ficar vazia (erro/limite API), devolve parte do fallback."""
    return lst if lst else list(fallback)[:n]


# ============================= Função principal ===============================
def rebuild_watchlists(
    base: dict, debug: bool = False
) -> dict | Tuple[dict, List[Tuple[str, str]]]:
    """
    Recebe um dicionário-base (tipicamente do arquivo `data/watchlists.json`)
    com chaves como BR_STOCKS / US_STOCKS / CRYPTO (e opcionalmente BR_FIIS)
    e retorna um novo dicionário com:

      - Listas filtradas por atividade recente (últimos 60 dias por padrão);
      - Classes derivadas:
          * BR_FIIS
          * BR_DIVIDEND
          * BR_BLUE_CHIPS / BR_SMALL_CAPS
          * US_DIVIDEND
          * US_BLUE_CHIPS / US_SMALL_CAPS

    Se `debug=True`, retorna também um relatório [(ticker, motivo)] para
    tickers descartados durante o processo de atividade.
    """
    # Universo Brasil considera BR_STOCKS ∪ BR_FIIS (caso BR_FIIS exista no arquivo)
    br_base = sorted(set(base.get("BR_STOCKS", [])) | set(base.get("BR_FIIS", [])))
    us_base = base.get("US_STOCKS", [])
    cr_base = base.get("CRYPTO", [])

    # 1) Filtro de atividade recente
    br_active, rep_br = _active_last_2m(br_base, debug=True)
    us_active, rep_us = _active_last_2m(us_base, debug=True)
    cr_active, rep_cr = _active_last_2m(cr_base, debug=True)

    # Fallbacks (evitam retorno vazio quando a API limita)
    br_active = _ensure_nonempty(br_active, br_base)
    us_active = _ensure_nonempty(us_active, us_base)
    cr_active = _ensure_nonempty(cr_active, cr_base)

    # 2) Classes Brasil
    br_fiis = [t for t in br_active if _is_fii_brazil(t)]
    # se após o filtro não restar nenhum FII, tenta pegar da base original
    br_fiis = _ensure_nonempty(br_fiis, [t for t in br_base if _is_fii_brazil(t)])

    br_equities = [t for t in br_active if t not in br_fiis]
    br_blue, br_small = _split_caps(br_equities)
    br_div = _div_payers(br_equities)

    # 3) Classes EUA
    us_blue, us_small = _split_caps(us_active)
    us_div = _div_payers(us_active)

    out = {
        # bases filtradas
        "BR_STOCKS": sorted(br_equities),
        "US_STOCKS": sorted(us_active),
        "CRYPTO": sorted(cr_active),
        # classes BR
        "BR_FIIS": sorted(br_fiis),
        "BR_DIVIDEND": sorted(br_div[:20]),
        "BR_BLUE_CHIPS": sorted(br_blue) or sorted(br_equities[:10]),
        "BR_SMALL_CAPS": sorted(br_small) or sorted(br_equities[-10:]),
        # classes US
        "US_DIVIDEND": sorted(us_div[:20]),
        "US_BLUE_CHIPS": sorted(us_blue) or sorted(us_active[:10]),
        "US_SMALL_CAPS": sorted(us_small) or sorted(us_active[-10:]),
    }

    if debug:
        report = rep_br + rep_us + rep_cr
        return out, report
    return out
