# pages/3_🔎_Screener.py
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from core.data import load_watchlists, download_bulk

pd.options.mode.copy_on_write = True

# =============================================================================
# Mapeamento de classes (rótulo UI -> chave das watchlists)
# =============================================================================
CLASS_MAP = {
    "Brasil (Ações B3)": "BR_STOCKS",
    "Brasil (FIIs)": "BR_FIIS",
    "Brasil - Dividendos": "BR_DIVIDEND",
    "Brasil - Blue Chips": "BR_BLUE_CHIPS",
    "Brasil - Small Caps": "BR_SMALL_CAPS",
    "EUA (Ações US)": "US_STOCKS",
    "EUA - Dividendos": "US_DIVIDEND",
    "EUA - Blue Chips": "US_BLUE_CHIPS",
    "EUA - Small Caps": "US_SMALL_CAPS",
    "Criptos": "CRYPTO",
}

# =============================================================================
# Utilidades de métricas
# =============================================================================
def _safe_pct(series: pd.Series, period: int) -> float:
    """Retorno percentual para 'period' barras (fechamento)."""
    try:
        if len(series) <= period:
            return np.nan
        return float(series.iloc[-1] / series.iloc[-1 - period] - 1.0) * 100.0
    except Exception:
        return np.nan


def _rsi_wilder(close: pd.Series, length: int = 14) -> float:
    """RSI de Wilder (retorna o último valor)."""
    try:
        if len(close) < length + 1:
            return np.nan
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)
        roll_up = up.ewm(alpha=1.0 / length, adjust=False).mean()
        roll_down = down.ewm(alpha=1.0 / length, adjust=False).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi.iloc[-1])
    except Exception:
        return np.nan


def _sma(series: pd.Series, n: int) -> float:
    try:
        if len(series) < n:
            return np.nan
        return float(series.iloc[-n:].mean())
    except Exception:
        return np.nan


def _ann_vol(close: pd.Series) -> float:
    """Volatilidade anualizada por retornos diários."""
    try:
        if len(close) < 3:
            return np.nan
        rets = close.pct_change().dropna()
        vol = float(rets.std() * math.sqrt(252)) * 100.0
        return vol
    except Exception:
        return np.nan


def _avg_volume(volume: pd.Series, n: int = 20) -> float:
    try:
        if volume is None:
            return np.nan
        return float(volume.tail(n).mean())
    except Exception:
        return np.nan


def _trend_up(close: pd.Series) -> bool:
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    if np.isnan(sma50) or np.isnan(sma200):
        return False
    return sma50 > sma200


def _score(row: pd.Series) -> float:
    """
    Score simples (0..100): tendência, retornos recentes, RSI e volatilidade.
    Ajuste pesos conforme preferir.
    """
    score = 0.0

    # tendência
    if row.get("TrendUp", False):
        score += 35.0

    # retornos recentes (cap 0..25)
    add = 0.0
    for k, w in [("D1%", 2.0), ("D5%", 6.0), ("M1%", 8.0), ("M6%", 9.0)]:
        v = row.get(k, 0.0)
        if not np.isnan(v):
            add += np.clip(v, -10, 10) * (w / 40.0)
    score += np.clip(add, -25, 25)

    # RSI
    rsi = row.get("RSI14", np.nan)
    if not np.isnan(rsi):
        if 45 <= rsi <= 70:
            score += 20.0
        elif rsi < 30 or rsi > 80:
            score -= 10.0

    # volatilidade
    vol = row.get("VolAnn%", np.nan)
    if not np.isnan(vol):
        if vol > 80:
            score -= 10.0
        elif vol < 20:
            score += 5.0

    return float(np.clip(score, 0.0, 100.0))


# =============================================================================
# UI auxiliar
# =============================================================================
def _ui_sidebar(wl: dict) -> Tuple[List[str], str, str, str, float, float, bool, int, bool]:
    with st.sidebar:
        st.markdown("### Classe")
        class_label = st.selectbox(
            "Classe", list(CLASS_MAP.keys()), index=0, label_visibility="collapsed"
        )
        key = CLASS_MAP[class_label]
        symbols = wl.get(key, [])
        st.caption(f"Total na classe: **{len(symbols)}**")

        st.markdown("### Janela")
        period = st.selectbox("Período", ["6mo", "1y", "2y", "5y"], index=0)
        interval = st.selectbox("Intervalo", ["1d", "1wk", "1mo"], index=0)

        st.markdown("### Filtros")
        price_min = float(st.number_input("Preço mínimo", value=1.00, step=0.5, format="%.2f"))
        vol_min = float(
            st.number_input("Volume médio mínimo (unid.)", value=100_000.0, step=10_000.0, format="%.0f")
        )
        trend_only = st.checkbox("Somente tendência de alta (SMA50>SMA200)", value=False)

        max_items = int(
            st.slider("Máx. de ativos processados", min_value=10, max_value=200, value=min(60, len(symbols)))
        )

        debug_download = st.toggle("Modo debug de download", value=False)

    return symbols, class_label, period, interval, price_min, vol_min, trend_only, max_items, debug_download


def _expander_criterios() -> None:
    with st.expander("📊 Critérios & métricas (clique para ver)", expanded=False):
        st.markdown(
            """
**Métricas calculadas**
- **Price**: último fechamento.
- **D1%, D5%, M1%, M6%, Y1%**: retornos percentuais (1 dia, 5 dias, ~21, ~126, ~252 barras).
- **VolAnn%**: volatilidade anualizada (desvio-padrão dos retornos diários × √252).
- **AvgVol**: média dos últimos 20 volumes.
- **RSI14**: RSI de Wilder com 14 períodos.
- **TrendUp**: `SMA50 > SMA200`.
- **Score**: composto simples (tendência, retornos recentes, RSI e volatilidade).

**Filtros**
- **Preço mínimo** e **Volume médio mínimo** atuam sobre `Price` e `AvgVol`.
- Se marcar **Somente tendência de alta**, apenas ativos com `SMA50>SMA200` passam.
"""
        )


def _calc_row_metrics(df: pd.DataFrame) -> Dict[str, float | bool]:
    """Extrai métricas do DataFrame de um símbolo."""
    if df is None or df.empty or "Close" not in df.columns:
        return {}

    close = df["Close"].astype(float)
    price = float(close.iloc[-1])

    d1 = _safe_pct(close, 1)
    d5 = _safe_pct(close, 5)
    m1 = _safe_pct(close, 21)
    m6 = _safe_pct(close, 126)
    y1 = _safe_pct(close, 252)

    vol_ann = _ann_vol(close)
    rsi14 = _rsi_wilder(close, 14)

    avgvol = _avg_volume(df["Volume"], 20) if "Volume" in df.columns else np.nan
    trend = _trend_up(close)

    row = {
        "Price": price,
        "D1%": d1,
        "D5%": d5,
        "M1%": m1,
        "M6%": m6,
        "Y1%": y1,
        "VolAnn%": vol_ann,
        "AvgVol": avgvol,
        "RSI14": rsi14,
        "TrendUp": trend,
    }
    row["Score"] = _score(pd.Series(row))
    return row


def _apply_filters(df: pd.DataFrame, price_min: float, vol_min: float, trend_only: bool) -> pd.DataFrame:
    out = df.copy()
    if "Price" in out.columns:
        out = out[out["Price"].fillna(0) >= price_min]
    if "AvgVol" in out.columns:
        out = out[out["AvgVol"].fillna(0) >= vol_min]
    if trend_only and "TrendUp" in out.columns:
        out = out[out["TrendUp"] == True]  # noqa: E712
    return out


def _data_editor_selection(df: pd.DataFrame) -> List[str]:
    """Tabela com checkbox para escolher papéis e enviar ao Backtest."""
    if df.empty:
        return []
    editable = df.copy()
    if "Select" not in editable.columns:
        editable.insert(0, "Select", False)
    edited = st.data_editor(
        editable,
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        column_config={"Select": st.column_config.CheckboxColumn(required=False)},
        key="screener_table_editor",
    )
    selected = edited.loc[edited["Select"] == True, "Symbol"].tolist() if "Select" in edited.columns else []  # noqa: E712
    return selected


# =============================================================================
# Página principal
# =============================================================================
def main() -> None:
    st.title("Screener")
    st.caption("Triagem multi-ativos (BR/US/Cripto) com métricas e filtros")

    # 1) Watchlists
    wl = load_watchlists()

    # 2) UI lateral
    (
        symbols,
        class_label,
        period,
        interval,
        price_min,
        vol_min,
        trend_only,
        max_items,
        debug_download,
    ) = _ui_sidebar(wl)

    _expander_criterios()

    if not symbols:
        st.warning("Nenhum ativo nesta classe. Atualize as watchlists em **Settings** ou reduza filtros.")
        st.stop()

    # limita quantidade
    symbols = symbols[:max_items]

    # 3) versão das watchlists para quebrar o cache
    ver = int(st.session_state.get("watchlists_version", 0))

    # 4) download com barra de progresso e contadores
    ok_count = 0
    empty_count = 0
    error_count = 0

    st.markdown(f"Processando **{len(symbols)}** ativos desta classe…")

    def _cb(done: int, total: int, sym: str, ok: bool, reason: str):
        nonlocal ok_count, empty_count, error_count
        prog.progress(done / total)
        if ok:
            ok_count += 1
            if debug_download:
                st.write(f"{done}/{total} ✓ {sym}")
        else:
            if reason == "empty":
                empty_count += 1
                if debug_download:
                    st.write(f"{done}/{total} ∅ {sym} (vazio)")
            else:
                error_count += 1
                if debug_download:
                    st.write(f"{done}/{total} ✗ {sym} ({reason})")

    # status com fallback se a função não aceitar callback
    with st.status(f"Baixando {len(symbols)} ativo(s)…", expanded=debug_download) as status:
        prog = st.progress(0.0)
        try:
            data_dict = download_bulk(
                symbols, period=period, interval=interval, ver=ver, callback=_cb
            )
        except TypeError:
            # versão antiga de download_bulk sem callback
            data_dict = download_bulk(symbols, period=period, interval=interval, ver=ver)
            # estimativa simples para contadores
            for s in symbols:
                df = data_dict.get(s)
                if df is None:
                    error_count += 1
                elif df.empty:
                    empty_count += 1
                else:
                    ok_count += 1
            prog.progress(1.0)

        status.update(
            label=f"Download concluído: {ok_count} OK, {empty_count} vazios, {error_count} erro(s).",
            state="complete",
        )

    # 5) métricas
    rows: List[Dict] = []
    for s in symbols:
        df = data_dict.get(s)
        metrics = _calc_row_metrics(df)
        if metrics:
            row = {"Symbol": s}
            row.update(metrics)
            rows.append(row)

    if not rows:
        st.warning(
            f"Nenhum ativo com dados utilizáveis. "
            f"OK: {ok_count} | Vazios: {empty_count} | Erros: {error_count}. "
            f"Tente outro período/intervalo ou reduzir filtros."
        )
        st.stop()

    base_df = pd.DataFrame(rows)

    # 6) filtros
    filtered = _apply_filters(base_df, price_min=price_min, vol_min=vol_min, trend_only=trend_only)

    # cards-resumo
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Baixados OK", ok_count)
    colB.metric("Vazios", empty_count)
    colC.metric("Erros", error_count)
    colD.metric("Passaram filtros", int(filtered.shape[0]))

    # 7) ordenação
    st.markdown("### Ordenar por")
    order_by = st.selectbox(
        "Ordenar por",
        options=["Score", "Price", "D1%", "D5%", "M1%", "M6%", "Y1%", "VolAnn%", "AvgVol", "RSI14", "TrendUp", "Symbol"],
        index=0,
        label_visibility="collapsed",
    )
    asc = st.checkbox("Ordem crescente", value=False)

    filtered = filtered.sort_values(order_by, ascending=asc, na_position="last")

    # 8) tabela
    st.dataframe(
        filtered[["Symbol", "Price", "D1%", "D5%", "M1%", "M6%", "Y1%", "VolAnn%", "AvgVol", "RSI14", "TrendUp", "Score"]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("### Marque os ativos que deseja enviar para o Backtest")

    # 9) seleção para backtest
    selected = _data_editor_selection(
        filtered[["Symbol", "Price", "D1%", "D5%", "M1%", "M6%", "Y1%", "VolAnn%", "AvgVol", "RSI14", "TrendUp", "Score"]]
    )
    if selected:
        st.success(f"{len(selected)} ativo(s) selecionado(s): {', '.join(selected[:10])}{'…' if len(selected) > 10 else ''}")

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Usar seleção no Backtest", use_container_width=True):
            st.session_state["screener_selected"] = selected
            st.success("Seleção enviada. Abra a página **Backtest** para usar os ativos.")
    with col2:
        st.caption("A seleção fica disponível em `st.session_state['screener_selected']`.")


if __name__ == "__main__":
    main()
