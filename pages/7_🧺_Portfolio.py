# >>> DEVE SER A PRIMEIRA LINHA ÚTIL DO ARQUIVO <<<
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import streamlit as st

# Plotly para gráficos
import plotly.express as px
import plotly.graph_objs as go

# yfinance é opcional – se não houver, a página continua mostrando instruções
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore


# ================================================================
# Utilidades
# ================================================================
def fmt_pct(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "N/A"
    v = float(x)
    if abs(v) <= 1.5:
        return f"{v*100:.2f}%"
    return f"{v:.2f}%"


def fmt_num(x: float | int | None, ndigits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "N/A"
    return f"{float(x):.{ndigits}f}"


# ------------------------------------------------------------
# 1) Estatísticas anuais robustas (aceita Series/array/vazio)
# ------------------------------------------------------------
def ann_stats(ret_series, periods_per_year: int = 252):
    r = pd.Series(ret_series).dropna().astype(float)
    if len(r) < 2:               # nada para medir
        return np.nan, np.nan, np.nan
    mu = r.mean() * periods_per_year
    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sh  = mu / vol if vol and np.isfinite(vol) and vol > 0 else np.nan
    return float(mu), float(vol), float(sh)



def max_drawdown(equity: pd.Series) -> float:
    """
    Max drawdown de uma curva de patrimônio.
    """
    if not isinstance(equity, pd.Series) or equity.empty:
        return float("nan")
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def screener_selection() -> list[str]:
    sel = st.session_state.get("screener_selected", None)
    if isinstance(sel, (list, tuple)) and len(sel):
        return [str(x) for x in sel if isinstance(x, (str, bytes))]
    return []


# ================================================================
# Dados
# ================================================================
@st.cache_data(show_spinner=False)
def dl_prices(symbols: tuple[str, ...], period: str, interval: str) -> dict[str, pd.DataFrame]:
    """
    Baixa OHLC para cada ativo.
    Retorna um dicionário {symbol: DataFrame} (Close, etc).
    """
    data: dict[str, pd.DataFrame] = {}
    if yf is None or len(symbols) == 0:
        return data

    for sym in symbols:
        try:
            df = yf.download(sym, period=period, interval=interval, auto_adjust=False, progress=False)  # type: ignore
            if isinstance(df, pd.DataFrame) and len(df):
                df = df.rename(columns=str.title)
                if "Close" in df.columns:
                    data[sym] = df.copy()
        except Exception:
            # Falha silenciosa para não travar a página
            pass

    return data


def build_returns_dict(
    prices_dict: dict[str, pd.DataFrame]
) -> dict[str, pd.Series]:
    """
    Constrói séries de retornos para cada ativo com base em Close.
    """
    rets: dict[str, pd.Series] = {}
    for sym, df in prices_dict.items():
        if "Close" not in df.columns:
            continue
        sr = df["Close"].dropna().pct_change().dropna()
        if len(sr) >= 10:
            rets[sym] = sr
    return rets


def align_returns(rets_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Alinha por data as séries de retorno.
    """
    if not rets_dict:
        return pd.DataFrame()
    df = pd.concat(rets_dict.values(), axis=1, join="inner")
    df.columns = list(rets_dict.keys())
    df = df.dropna()
    return df


# ================================================================
# Portfólio / Monte Carlo
# ================================================================
def random_weights(n: int, cap: float, rng: np.random.Generator) -> np.ndarray:
    """
    Gera pesos aleatórios (>=0, somam 1) com limite máximo por ativo (cap).
    Usa rotina de rejeição simples.
    """
    if cap <= 0:
        cap = 1.0
    cap = float(min(1.0, max(0.01, cap)))  # clamp

    for _ in range(20_000):  # limite de tentativas
        w = rng.random(n)
        w = w / w.sum()
        if (w <= cap + 1e-9).all():
            return w
        # Tenta “empurrar” excesso para os demais
        # (estratégia simples para aumentar taxa de aceitação)
        over = w > cap
        if over.any():
            excess = float((w[over] - cap).sum())
            w[over] = cap
            # distribui excesso
            under = ~over
            if under.any():
                w[under] += excess * (w[under] / w[under].sum())
            # normaliza de novo
            w = np.maximum(w, 0)
            s = w.sum()
            if s > 0:
                w = w / s
            if (w <= cap + 1e-9).all():
                return w
    # fallback: dirichlet normal, sem cap
    w = rng.dirichlet(np.ones(n))
    return w


def portfolio_stats(weights: np.ndarray, mu_vec: np.ndarray, cov: np.ndarray, rf: float) -> tuple[float, float, float]:
    """
    Retorno anual, volatilidade anual, Sharpe.
    """
    port_mu = float(weights @ mu_vec)
    port_vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (port_mu - rf) / (port_vol + 1e-12)
    return port_mu, port_vol, sharpe


def build_portfolios(
    rets: pd.DataFrame,
    rf: float,
    n_sims: int = 10_000,
    w_cap: float = 0.35,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Monte Carlo de portfólios com cap por ativo.
    Retorna:
      df_mc (mu, vol, sharpe, pesos...) e dict com carteiras especiais.
    """
    # Sanitização
    if rets.empty or rets.shape[1] < 2:
        return pd.DataFrame(), {}

    # Garante apenas colunas numéricas e limpa NaN
    rets = rets.select_dtypes(include=[np.number]).dropna(how="any")
    if rets.empty or rets.shape[1] < 2:
        return pd.DataFrame(), {}

    periods = 252
    mu_vec = rets.mean().to_numpy() * periods               # vetor de retornos anuais esperados
    cov = rets.cov().to_numpy() * periods                   # matriz de covariância anualizada

    rng = np.random.default_rng(seed)
    n_assets = rets.shape[1]
    columns = ["mu", "vol", "sharpe"] + [f"w_{c}" for c in rets.columns]
    out = np.empty((n_sims, 3 + n_assets), dtype=float)

    for i in range(n_sims):
        w = random_weights(n_assets, w_cap, rng)            # pesos >=0, somam 1, respeitando cap
        mu, vol, sh = portfolio_stats(w, mu_vec, cov, rf)
        out[i, 0] = mu
        out[i, 1] = vol
        out[i, 2] = sh
        out[i, 3:] = w

    df_mc = pd.DataFrame(out, columns=columns)

    # Carteiras especiais
    w_eq = np.ones(n_assets) / n_assets                     # Equal-Weight
    mu_eq, vol_eq, sh_eq = portfolio_stats(w_eq, mu_vec, cov, rf)

    # Máx. Sharpe
    best = df_mc.iloc[df_mc["sharpe"].idxmax()]
    w_best = best[3:].to_numpy()

    # “Mesmo risco” (proxy): maior retorno entre os pontos de vol mais próxima do target_vol
    target_vol = float(best["vol"])
    k = min(200, len(df_mc))
    order = (df_mc["vol"] - target_vol).abs().argsort().values[:k]
    df_close = df_mc.iloc[order].dropna(subset=["mu", "vol"])

    if not df_close.empty:
        idx = df_close["mu"].idxmax()       # rótulo
        row = df_close.loc[idx]             # usa .loc
        w_bal = row.iloc[3:].to_numpy()
    else:
        w_bal = w_best.copy()

    special = {
        "Equal-Weight": w_eq,
        "Max Sharpe": w_best,
        "Same Risk (Max Sharpe)": w_bal,
        "__mu_vec__": mu_vec,
        "__cov__": cov,
        "__cols__": rets.columns.to_list(),
    }
    return df_mc, special



def equity_curve(rets: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Curva patrimonial (base 1.0).
    """
    pr = (rets @ weights).fillna(0)
    eq = (1 + pr).cumprod()
    return eq


# ================================================================
# Página
# ================================================================
st.set_page_config(page_title="Portfolio (Monte Carlo)", page_icon="💰", layout="wide")
st.title("Portfolio (Monte Carlo)")

# ----------------- Seleção de Ativos -----------------
use_screener = st.toggle("Usar seleção do Screener (se houver)", value=True)

warn = st.container()
symbols = screener_selection() if use_screener else []

if not symbols:
    warn.info("A seleção do Screener tem menos de 2 ativos. Escolha abaixo ou volte ao Screener e marque mais.", icon="ℹ️")

all_input = st.multiselect(
    "Ativos para montar o portfólio",
    options=symbols if symbols else ["AAPL", "MSFT", "NVDA", "AVGO", "PETR4.SA", "VALE3.SA"],
    default=symbols if len(symbols) >= 2 else ["AAPL", "MSFT", "NVDA"],
)

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    period = st.selectbox("Período", ["6mo", "1y", "2y", "3y"], index=1)
with c2:
    interval = st.selectbox("Intervalo", ["1d", "1wk"], index=0)
with c3:
    rf_pct = st.number_input("Taxa livre de risco anual (%)", value=6.0, step=0.25, min_value=0.0)
    rf = rf_pct / 100.0
with c4:
    dark = st.toggle("Tema escuro", value=True)

c5, c6, c7 = st.columns([1, 1, 1])
with c5:
    n_sims = int(st.slider("Simulações (Monte Carlo)", 2_000, 50_000, 15_000, step=1_000))
with c6:
    w_cap = float(st.slider("Peso máximo por ativo", 0.05, 1.0, 0.35, step=0.01))
with c7:
    seed = int(st.number_input("Semente (reprodutibilidade)", value=42, step=1, min_value=0))

if len(all_input) < 2:
    st.stop()

# ----------------- Dados -----------------
data = dl_prices(tuple(all_input), period, interval)
rets_dict = build_returns_dict(data)
rets = align_returns(rets_dict)

if rets.empty or rets.shape[1] < 2:
    st.warning("Não foi possível obter dados suficientes para os ativos selecionados.", icon="⚠️")
    st.stop()

# ----------------- Monte Carlo -----------------
with st.spinner("Calculando portfólios..."):
    df_mc, special = build_portfolios(rets, rf, n_sims=n_sims, w_cap=w_cap, seed=seed)

if df_mc.empty:
    st.warning("Monte Carlo não gerou portfólios válidos. Ajuste parâmetros ou verifique os dados.", icon="⚠️")
    st.stop()

cols = special["__cols__"]  # type: ignore

# ----------------- Gráfico: Fronteira / Nuvem Monte Carlo -----------------
fig = px.scatter(
    df_mc,
    x="vol",
    y="mu",
    color="sharpe",
    color_continuous_scale="Viridis",
    title="Fronteira (Monte Carlo) — retorno vs risco",
    labels={"vol": "Vol anual", "mu": "Retorno anual"},
)
# marca carteiras especiais
mk = {
    "Equal-Weight": "diamond",
    "Max Sharpe": "star",
    "Same Risk (Max Sharpe)": "x",
}
for name in ("Equal-Weight", "Max Sharpe", "Same Risk (Max Sharpe)"):
    w = special[name]  # type: ignore
    mu, vol, sh = portfolio_stats(w, special["__mu_vec__"], special["__cov__"], rf)  # type: ignore
    fig.add_trace(
        go.Scatter(
            x=[vol],
            y=[mu],
            mode="markers+text",
            marker=dict(size=12, symbol=mk.get(name, "circle"), color="red"),
            text=[name],
            textposition="top center",
            name=name,
        )
    )

fig.update_layout(template="plotly_dark" if dark else "plotly_white")
st.plotly_chart(fig, use_container_width=True)

# ----------------- Tabelas e Pesos -----------------
def series_from_weights(rets: pd.DataFrame,
                        w: np.ndarray,
                        cols_ref: list[str] | None = None) -> pd.Series:
    """
    Retorna a série de retornos do portfólio alinhando os pesos às colunas.
    - rets: DataFrame (colunas = ativos; linhas = períodos)
    - w: vetor de pesos (mesmo tamanho de cols_ref)
    - cols_ref: lista de colunas usada para gerar w; se None, usa rets.columns
    """
    if rets.empty:
        return pd.Series(dtype=float)

    # só numéricos + limpa linhas com NaN
    rets = rets.select_dtypes(include=[np.number]).dropna(how="any")
    if rets.empty:
        return pd.Series(dtype=float)

    # referência de colunas que geraram w (as mesmas usadas no Monte Carlo)
    cols = list(cols_ref) if cols_ref is not None else list(rets.columns)

    # garante que só usemos colunas existentes em rets
    cols = [c for c in cols if c in rets.columns]
    if not cols:
        return pd.Series(dtype=float)

    # recorta/normaliza pesos para essas colunas
    w = np.asarray(w, dtype=float).flatten()
    if len(w) != len(cols):
        # Se tamanhos diferem, recortamos para o mínimo
        n = min(len(w), len(cols))
        w = w[:n]
        cols = cols[:n]

    # pesos não-negativos e normalizados
    w = np.clip(w, 0, 1)
    s = w.sum()
    if s <= 0:
        return pd.Series(dtype=float)
    w = w / s

    # monta a série de retorno
    R = rets[cols].to_numpy()  # (n_periodos, n_assets)
    r = R @ w                  # (n_periodos,)
    r = pd.Series(r, index=rets.index)
    return r.dropna()

# ---------------------------------------------------------------------
# 2) Linha da tabela de portfólio com alinhamento e normalização robustos
#    label  : nome a mostrar
#    w      : pesos (dict | Series | ndarray)
#    rets   : DataFrame de retornos (colunas = símbolos)
#    cols_ref: ordem/seleção de colunas usada para gerar os pesos
# ---------------------------------------------------------------------
def portfolio_table_row(label: str,
                        w,
                        rets: pd.DataFrame,
                        cols_ref: list[str]) -> list:
    # Apenas colunas numéricas e que existem no DF
    rets = rets.select_dtypes(include=[np.number]).copy()
    cols = [c for c in cols_ref if c in rets.columns]

    if len(cols) == 0 or rets.empty:
        return [label, np.nan, np.nan, np.nan, {}]

    # --------------------
    # Monta vetor de pesos
    # --------------------
    if isinstance(w, dict):
        w_vec = pd.Series(w, dtype=float).reindex(cols).fillna(0.0).to_numpy()
    elif isinstance(w, pd.Series):
        w_vec = w.reindex(cols).astype(float).fillna(0.0).to_numpy()
    else:
        w_arr = np.asarray(w, dtype=float).reshape(-1)
        if w_arr.size == len(cols):
            w_vec = w_arr
        else:
            # Se tiver um índice, tenta alinhar por ele
            if hasattr(w, "index"):
                try:
                    w_vec = pd.Series(w_arr, index=list(getattr(w, "index"))
                                      ).reindex(cols).fillna(0.0).to_numpy()
                except Exception:
                    # fallback: corta/preenche com zero
                    w_vec = np.zeros(len(cols), dtype=float)
                    n = min(len(cols), w_arr.size)
                    if n > 0:
                        w_vec[:n] = w_arr[:n]
            else:
                # fallback: corta/preenche com zero
                w_vec = np.zeros(len(cols), dtype=float)
                n = min(len(cols), w_arr.size)
                if n > 0:
                    w_vec[:n] = w_arr[:n]

    # Normaliza (garante soma = 1 se possível)
    s = float(np.nansum(w_vec))
    if not np.isfinite(s) or abs(s) < 1e-12:
        w_vec = np.zeros_like(w_vec)
    else:
        w_vec = w_vec / s

    # ----------------------------
    # Retorno da carteira (T x 1)
    # ----------------------------
    R = rets[cols].to_numpy(dtype=float)       # (T, N)
    r = R @ w_vec                              # (T,)
    r = pd.Series(r, index=rets.index).dropna()

    # ----------------------------
    # Estatísticas anuais
    # ----------------------------
    mu, vol, sh = ann_stats(r)

    # Dicionário de pesos (limpa números muito pequenos)
    weights_dict = {c: float(x) for c, x in zip(cols, w_vec) if abs(x) > 1e-6}

    return [label, mu, vol, sh, weights_dict]


# ================================================================
# Markowitz (analítico) — fronteira, MVP e Tangência
# (Colar este bloco depois de `portfolio_table_row`)
# ================================================================
def _pinv_psd(a: np.ndarray) -> np.ndarray:
    """Pseudoinversa estável para matrizes PSD (covariância)."""
    try:
        return np.linalg.pinv(a)
    except Exception:
        # fallback bem conservador
        eye = np.eye(a.shape[0], dtype=float)
        return eye

def mvo_unconstrained(mu_vec: np.ndarray,
                      cov: np.ndarray,
                      rf: float = 0.0):
    """
    Retorna: w_mvp, w_tan, frontier_fn
      - w_mvp: mínima variância global (sem restrições)
      - w_tan: tangência (máx Sharpe) com taxa rf (sem restrições)
      - frontier_fn(num, r_min, r_max) -> (retornos, vols, pesos) ao longo da fronteira
    """
    mu_vec = np.asarray(mu_vec, dtype=float).reshape(-1)
    cov    = np.asarray(cov,    dtype=float)
    n      = mu_vec.size

    inv = _pinv_psd(cov)
    ones = np.ones(n, dtype=float)

    A = float(ones @ inv @ ones)
    B = float(ones @ inv @ mu_vec)
    C = float(mu_vec @ inv @ mu_vec)
    D = float(A * C - B * B) if np.isfinite(A) and np.isfinite(B) and np.isfinite(C) else np.nan

    # Mínima Variância Global (MVP)
    w_mvp = inv @ ones
    s = float(ones @ w_mvp)
    w_mvp = w_mvp / (s if abs(s) > 1e-15 else 1.0)

    # Tangência (máx. Sharpe) com rf
    t = inv @ (mu_vec - rf * ones)
    s = float(ones @ t)
    if abs(s) > 1e-15:
        w_tan = t / s
    else:
        # fallback: usa equal-weight
        w_tan = np.ones(n) / n

    def efficient_frontier(num: int = 80,
                           r_min: float | None = None,
                           r_max: float | None = None):
        # Faixa de retornos-alvo
        if r_min is None:
            r_min = float(mu_vec.min())
        if r_max is None:
            r_max = float(mu_vec.max())
        if r_min > r_max:
            r_min, r_max = r_max, r_min

        rs  = np.linspace(r_min, r_max, num=num)
        vs  = np.zeros_like(rs, dtype=float)
        wss = np.zeros((num, n), dtype=float)

        # Fórmula fechada com A,B,C,D
        if not (np.isfinite(D) and abs(D) > 1e-18):
            # fronteira indisponível — retorna arrays vazios coerentes
            return rs, np.full_like(rs, np.nan), np.tile(np.nan, (num, n))

        inv1 = inv @ ones
        invm = inv @ mu_vec

        for i, r in enumerate(rs):
            w = ((C - r * B) / D) * inv1 + ((r * A - B) / D) * invm
            # risco (desvio padrão)
            vs[i]  = float(np.sqrt(max(w @ cov @ w, 0.0)))
            wss[i] = w
        return rs, vs, wss

    return w_mvp, w_tan, efficient_frontier

def _long_only_clip_renorm(w: np.ndarray) -> np.ndarray:
    """Heurística simples para long-only: zera negativos e renormaliza."""
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    return w / s if s > 1e-15 else w

# ----------------- UI: Markowitz + comparação de carteiras -----------------
st.markdown("### Markowitz (analítico) & Comparação de Carteiras")

cA, cB, cC = st.columns([1, 1, 1])
with cA:
    enable_mvo = st.toggle("Ativar Markowitz (analítico)", value=True, help="Usa solução fechada sem restrições.")
with cB:
    enforce_longonly = st.toggle("Forçar long-only (aproximação)", value=True,
                                 help="Heurística: zera pesos negativos e renormaliza.")
with cC:
    frontier_points = int(st.slider("Pontos da fronteira", 30, 200, 80, step=10))

if enable_mvo:
    mu_vec = np.asarray(special["__mu_vec__"], dtype=float)   # type: ignore
    cov    = np.asarray(special["__cov__"],    dtype=float)   # type: ignore
    cols   = list(special["__cols__"])                        # type: ignore

    w_mvp, w_tan, frontier_fn = mvo_unconstrained(mu_vec, cov, rf=rf)

    if enforce_longonly:
        w_mvp = _long_only_clip_renorm(w_mvp)
        w_tan = _long_only_clip_renorm(w_tan)

    # Fronteira eficiente (linha)
    r_line, v_line, w_line = frontier_fn(num=frontier_points)
    if enforce_longonly and np.all(np.isfinite(w_line)):
        # aplica heurística ponto-a-ponto (pode distorcer levemente a fronteira)
        w_line = np.apply_along_axis(_long_only_clip_renorm, 1, w_line)
        v_line = np.array([np.sqrt(max(w @ cov @ w, 0.0)) for w in w_line], dtype=float)
        r_line = np.array([float(mu_vec @ w) for w in w_line], dtype=float)

    # sobrepõe a fronteira no gráfico existente
    fig.add_trace(go.Scatter(
        x=v_line, y=r_line, mode="lines",
        name="Fronteira (MVO)",
        line=dict(width=2, dash="solid")
    ))
    # marca MVP e Tangência
    mu_mvp, vol_mvp, _ = portfolio_stats(w_mvp, mu_vec, cov, rf)
    mu_tan, vol_tan, _ = portfolio_stats(w_tan, mu_vec, cov, rf)
    fig.add_trace(go.Scatter(
        x=[vol_mvp], y=[mu_mvp], mode="markers+text",
        marker=dict(size=12, symbol="triangle-up"),
        text=["Mín. Variância"], textposition="bottom center",
        name="Mín. Variância"
    ))
    fig.add_trace(go.Scatter(
        x=[vol_tan], y=[mu_tan], mode="markers+text",
        marker=dict(size=12, symbol="triangle-right"),
        text=["Tangência"], textposition="bottom center",
        name="Tangência"
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ----------------- Tabela de pesos -----------------
    def _weights_to_df(named_weights: dict[str, np.ndarray]) -> pd.DataFrame:
        rows = []
        for name, w in named_weights.items():
            row = {"Carteira": name}
            for c, wi in zip(cols, w):
                row[c] = wi
            rows.append(row)
        dfw = pd.DataFrame(rows).set_index("Carteira")
        return dfw

    named_w = {
        "Equal-Weight": special["Equal-Weight"],                 # type: ignore
        "Max Sharpe (MC)": special["Max Sharpe"],                # type: ignore
        "Same Risk (MC)": special["Same Risk (Max Sharpe)"],     # type: ignore
        "Mín. Variância (MVO)": w_mvp,
        "Tangência (MVO)": w_tan,
    }
    df_weights = _weights_to_df(named_w)
    df_weights_fmt = df_weights.applymap(lambda x: fmt_pct(x))

    st.markdown("#### Pesos por carteira")
    st.dataframe(df_weights_fmt, use_container_width=True)

    # Download (CSV) dos pesos
    st.download_button(
        "Baixar pesos (CSV)",
        data=df_weights.to_csv(index=True).encode("utf-8"),
        file_name="portfolio_weights.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ----------------- Escolha e análise de uma carteira -----------------
    st.markdown("#### Análise detalhada de uma carteira")
    pick = st.selectbox(
        "Escolha a carteira para análise",
        list(named_w.keys()),
        index=1
    )
    w_pick = np.asarray(named_w[pick], dtype=float)

    # Série de retornos e curva patrimonial
    r_pick = series_from_weights(rets, w_pick, cols_ref=cols)
    if r_pick.dropna().empty:
        st.warning("Série de retornos vazia para a carteira escolhida.", icon="⚠️")
    else:
        eq = (1 + r_pick).cumprod()
        # métricas
        muP, volP, shP = ann_stats(r_pick)
        ddP = max_drawdown(eq)

        # cartões/resumo
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Retorno (anual)", fmt_pct(muP))
        with c2: st.metric("Vol (anual)",     fmt_pct(volP))
        with c3: st.metric("Sharpe",          fmt_num(shP, 3))
        with c4: st.metric("Max DD",          fmt_pct(ddP))

        # gráfico curva patrimônio
        fig_eq = px.line(eq.rename("Equity"), title=f"Curva de Patrimônio — {pick}")
        fig_eq.update_layout(template="plotly_dark" if dark else "plotly_white")
        st.plotly_chart(fig_eq, use_container_width=True)

        # gráfico barras de pesos
        df_bar = pd.DataFrame({"Ativo": cols, "Peso": w_pick})
        fig_w = px.bar(df_bar, x="Ativo", y="Peso", title=f"Pesos — {pick}")
        fig_w.update_layout(template="plotly_dark" if dark else "plotly_white")
        st.plotly_chart(fig_w, use_container_width=True)

else:
    # Se Markowitz desativado, ainda mostramos a nuvem MC renderizada acima.
    st.info("Markowitz (analítico) desativado. Ative a chave para ver fronteira, tangência e mínima variância.", icon="ℹ️")



