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


def ann_stats(returns: pd.Series | pd.DataFrame, periods_per_year: int = 252) -> tuple[float, float]:
    """
    Anualiza média e desvio de retornos (diários por padrão).
    """
    mu = returns.mean() * periods_per_year
    vol = returns.std(ddof=0) * (periods_per_year ** 0.5)
    return (float(mu), float(vol))


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
    if rets.empty or rets.shape[1] < 2:
        return pd.DataFrame(), {}

    periods_per_year = 252 if rets.index.freq is None else None  # anualização padrão
    mu_vec, _ = ann_stats(rets, periods_per_year=252)
    mu_vec = rets.mean().values * 252  # vetor
    cov = rets.cov().values * 252

    rng = np.random.default_rng(seed)
    n_assets = rets.shape[1]
    columns = ["mu", "vol", "sharpe"] + [f"w_{c}" for c in rets.columns]
    out = np.empty((n_sims, 3 + n_assets), dtype=float)

    for i in range(n_sims):
        w = random_weights(n_assets, w_cap, rng)
        mu, vol, sh = portfolio_stats(w, mu_vec, cov, rf)
        out[i, 0] = mu
        out[i, 1] = vol
        out[i, 2] = sh
        out[i, 3:] = w

    df_mc = pd.DataFrame(out, columns=columns)

    # Equal-Weight
    w_eq = np.ones(n_assets) / n_assets
    mu_eq, vol_eq, sh_eq = portfolio_stats(w_eq, mu_vec, cov, rf)

    # Máx. Sharpe
    best = df_mc.iloc[df_mc["sharpe"].idxmax()]
    w_best = best[3:].values

    # “Mesmo risco do Máx. Sharpe” → procura portfólio com vol próximo
    target_vol = float(best["vol"])
    df_close = df_mc.iloc[(df_mc["vol"] - target_vol).abs().argsort()[:200]]
    # dentre os mais próximos, pega o de maior retorno (proxy para “min risk” nesse universo)
    w_bal = df_close.iloc[df_close["mu"].idxmax()][3:].values

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
def portfolio_table_row(name: str, w: np.ndarray) -> dict:
    eq = equity_curve(rets, w)
    mu, vol, sh = ann_stats((rets @ w))
    mu = float(mu)
    vol = float(vol)
    sharpe = (mu - rf) / (vol + 1e-12)
    ddown = max_drawdown(eq)
    return {
        "Carteira": name,
        "Retorno (CAGR)": mu,
        "Vol (ann)": vol,
        "Sharpe": sharpe,
        "MaxDD": ddown,
    }

rows = []
weights_dict: dict[str, np.ndarray] = {}
for name in ("Equal-Weight", "Max Sharpe", "Same Risk (Max Sharpe)"):
    w = special[name]  # type: ignore
    weights_dict[name] = w
    rows.append(portfolio_table_row(name, w))

df_sum = pd.DataFrame(rows)
# formatação
show = df_sum.copy()
show["Retorno (CAGR)"] = show["Retorno (CAGR)"].map(fmt_pct)
show["Vol (ann)"] = show["Vol (ann)"].map(fmt_pct)
show["Sharpe"] = show["Sharpe"].map(lambda x: fmt_num(x, 2))
show["MaxDD"] = show["MaxDD"].map(fmt_pct)

st.subheader("Resumo das carteiras")
st.dataframe(show, use_container_width=True, hide_index=True)

# ----------------- Pesos / Pizza -----------------
st.subheader("Alocações (pesos)")
cc1, cc2, cc3 = st.columns(3)
for i, name in enumerate(("Equal-Weight", "Max Sharpe", "Same Risk (Max Sharpe)")):
    w = weights_dict[name]
    df_w = pd.DataFrame({"Ativo": cols, "Peso": w})
    figw = px.pie(df_w, names="Ativo", values="Peso", title=name)
    figw.update_layout(template="plotly_dark" if dark else "plotly_white")
    (cc1 if i == 0 else cc2 if i == 1 else cc3).plotly_chart(figw, use_container_width=True)

# ----------------- Histórico / Curvas -----------------
st.subheader("Curvas patrimoniais (base 1.0)")
eq_data = []
for name in ("Equal-Weight", "Max Sharpe", "Same Risk (Max Sharpe)"):
    w = weights_dict[name]
    eq = equity_curve(rets, w)
    eq_data.append(eq.rename(name))
eq_df = pd.concat(eq_data, axis=1)

figh = px.line(eq_df, title="Equity curves", labels={"value": "Equity", "index": "Data"})
figh.update_layout(template="plotly_dark" if dark else "plotly_white")
st.plotly_chart(figh, use_container_width=True)

# ----------------- Notas finais -----------------
with st.expander("Como interpretar", expanded=False):
    st.markdown(
        """
- **Nuvem Monte Carlo**: cada ponto é um portfólio aleatório respeitando o limite máximo por ativo  
  (e somando 100%). Cor representa **Sharpe**.
- **Equal-Weight**: pesos iguais; costuma ser um bom baseline.
- **Max Sharpe**: busca melhor relação retorno/risco para a taxa livre de risco informada.
- **Same Risk (Max Sharpe)**: dentro do universo simulado, portfólio com volatilidade mais próxima da do Max Sharpe e maior retorno.
- As curvas e métricas são **históricas** e não garantem resultados futuros.
        """
    )

st.caption("Conteúdo educacional; não é recomendação. Use gestão de risco.")
