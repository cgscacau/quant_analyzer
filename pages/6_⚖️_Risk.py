# pages/6_⚖️_Risk.py
# >>> DEVE SER A PRIMEIRA LINHA ÚTIL DO ARQUIVO <<<
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import streamlit as st

# yfinance só é usado aqui, para baixar OHLC quando necessário
try:
    import yfinance as yf  # type: ignore
except Exception:  # ambiente sem yfinance
    yf = None  # type: ignore


# ================================================================
# Utilidades
# ================================================================

def fmt_pct(x: float | int | None) -> str:
    """Formata percentuais com segurança."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "N/A"
    return f"{float(x)*100:.2f}%" if abs(x) <= 1.5 else f"{float(x):.2f}%"


def fmt_num(x: float | int | None, ndigits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "N/A"
    return f"{float(x):.{ndigits}f}"


def clamp(x: float, lo: float, hi: float) -> float:
    if not math.isfinite(x):
        return float("nan")
    return max(lo, min(hi, x))


def kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
    """
    Kelly clássico para aposta binária:
       f* = p - (1 - p) / b
    onde:
      p = win_rate (0..1)
      b = payoff_ratio = ganho_médio / perda_média (módulo)
    """
    if not (math.isfinite(win_rate) and math.isfinite(payoff_ratio)):
        return float("nan")
    if payoff_ratio <= 0 or not (0 <= win_rate <= 1):
        return float("nan")
    k = win_rate - (1 - win_rate) / payoff_ratio
    # opcional: clamp em 0..1 para operação prática
    return clamp(k, 0.0, 1.0)


@st.cache_data(show_spinner=False)
def dl_prices(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Baixa OHLCV via yfinance. Se não tiver yfinance disponível,
    devolve DataFrame vazio.
    """
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)  # type: ignore
        if isinstance(df, pd.DataFrame) and len(df):
            df = df.rename(columns=str.title)
        else:
            return pd.DataFrame()
        # Garante colunas padrão que usaremos
        for col in ("Open", "High", "Low", "Close"):
            if col not in df.columns:
                return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def wr_and_b_from_history(df: pd.DataFrame) -> tuple[float, float, int]:
    """
    Calcula win-rate e payoff-ratio (b) a partir das variações percentuais
    da coluna Close (bar-to-bar). Isso dá uma visão agregada máx simples.
    b = média dos ganhos / média (em módulo) das perdas

    Retorna: (win_rate, b, n_amostras_validas)
    """
    if not isinstance(df, pd.DataFrame) or "Close" not in df.columns or len(df) < 5:
        return (float("nan"), float("nan"), 0)

    rets = df["Close"].pct_change().dropna()
    if len(rets) < 5:
        return (float("nan"), float("nan"), 0)

    pos = rets[rets > 0]
    neg = rets[rets < 0]

    wr = float(len(pos)) / float(len(rets)) if len(rets) else float("nan")

    if len(pos) == 0 or len(neg) == 0:
        # Sem ganhos ou sem perdas -> b não é bem definido
        b = float("nan")
    else:
        avg_gain = float(pos.mean())
        avg_loss = float(abs(neg.mean()))
        b = avg_gain / avg_loss if avg_loss > 0 else float("nan")

    return (wr, b, len(rets))


def screener_selection() -> list[str]:
    """
    Lê seleção do Screener, se existir. Espera encontrar uma lista
    em st.session_state['screener_selected'].
    """
    sel = st.session_state.get("screener_selected", None)
    if isinstance(sel, (list, tuple)) and len(sel):
        # normaliza para string
        return [str(x) for x in sel if isinstance(x, (str, bytes))]
    return []


# ================================================================
# Página
# ================================================================

st.set_page_config(page_title="Kelly (fracionado) & Resumo", page_icon="⚖️", layout="wide")

st.title("Kelly (fracionado)")

# ----------------- Seletor do Ativo -----------------
use_screener = st.toggle("Usar seleção do Screener (se houver)", value=True)

symbols = screener_selection() if use_screener else []

col_left, col_right = st.columns([2, 1])

with col_left:
    if symbols:
        sym = st.selectbox("Ativo (sizing)", options=symbols, index=0, key="kelly_sym")
    else:
        # manual
        sym = st.text_input("Ativo (sizing)", value="AAPL").strip().upper()

with col_right:
    period = st.selectbox("Período", ["6mo", "1y", "2y", "3y"], index=1)
    interval = st.selectbox("Intervalo", ["1d", "1wk"], index=0)

df = dl_prices(sym, period, interval)

# ----------------- Bloco de entrada de parâmetros -----------------
st.subheader("Informe Win Rate (%) e Payoff (ganho médio / perda média).")

# : Se o usuário quiser puxar do histórico, basta clicar no botão:
wcol, bcol, btncol = st.columns([1, 1, 1])
with wcol:
    wr_pct = st.slider("Win Rate (%)", min_value=5.0, max_value=95.0, value=50.0, step=0.5)
with bcol:
    payoff = st.slider("Payoff Ratio (b = ganho/perda)", min_value=0.50, max_value=3.00, value=1.50, step=0.05)
with btncol:
    st.caption(" ")
    if st.button("Calcular a partir do histórico", use_container_width=True, disabled=df.empty):
        h_wr, h_b, n = wr_and_b_from_history(df)
        if math.isfinite(h_wr):
            wr_pct = h_wr * 100.0
        if math.isfinite(h_b):
            payoff = h_b
        if n == 0:
            st.warning("Histórico insuficiente para estimar Win Rate e Payoff.", icon="⚠️")

# ----------------- Kelly -----------------
wr = wr_pct / 100.0
k = kelly_fraction(wr, payoff)
k_half = clamp(k / 2.0, 0.0, 1.0) if math.isfinite(k) else float("nan")
k_quarter = clamp(k / 4.0, 0.0, 1.0) if math.isfinite(k) else float("nan")

m1, m2, m3 = st.columns(3)
m1.metric("Kelly*", fmt_pct(k))
m2.metric("½ Kelly", fmt_pct(k_half))
m3.metric("¼ Kelly", fmt_pct(k_quarter))

st.caption("**Regra prática:** operar **½ Kelly** costuma ser mais estável.")

# ================================================================
# Resumo executivo
# ================================================================
st.markdown("---")
st.header("Resumo executivo")

# Monta um “executivo” com segurança
summary_rows: list[dict] = []

row = {
    "Ativo": sym,
    "Período": period,
    "Intervalo": interval,
    "Win Rate (entrada)": wr_pct,
    "Payoff b (entrada)": payoff,
    "Kelly": k,
    "½ Kelly": k_half,
    "¼ Kelly": k_quarter,
}

# Se tivermos histórico, agrega números-resumo
if not df.empty:
    # preço atual seguro
    try:
        px = float(df["Close"].iloc[-1])
    except Exception:
        px = float("nan")

    h_wr, h_b, n_obs = wr_and_b_from_history(df)
    row.update(
        {
            "Preço": px,
            "WR (hist)": h_wr,
            "b (hist)": h_b,
            "Obs válidas": n_obs,
        }
    )
else:
    row.update({"Preço": float("nan"), "WR (hist)": float("nan"), "b (hist)": float("nan"), "Obs válidas": 0})

summary_rows.append(row)

# Exibe DataFrame com formatação segura
summary_df = pd.DataFrame(summary_rows)

if len(summary_df):
    # Formata colunas numéricas
    def _fmt_pct(x):
        return fmt_pct(x) if math.isfinite(x) else "N/A"

    fmt_cols_pct = ["Win Rate (entrada)", "Kelly", "½ Kelly", "¼ Kelly", "WR (hist)"]
    for c in fmt_cols_pct:
        if c in summary_df.columns:
            summary_df[c] = summary_df[c].apply(lambda v: float(v) if math.isfinite(float(v)) else float("nan"))
            summary_df[c] = summary_df[c].apply(_fmt_pct)

    if "Payoff b (entrada)" in summary_df.columns:
        summary_df["Payoff b (entrada)"] = summary_df["Payoff b (entrada)"].apply(lambda v: fmt_num(v, 2))
    if "b (hist)" in summary_df.columns:
        summary_df["b (hist)"] = summary_df["b (hist)"].apply(lambda v: fmt_num(v, 2))
    if "Preço" in summary_df.columns:
        summary_df["Preço"] = summary_df["Preço"].apply(lambda v: fmt_num(v, 2))

    st.dataframe(summary_df, use_container_width=True, hide_index=True)
else:
    st.info("Sem dados para o resumo executivo.")

# Aviso educacional
st.caption("**Conteúdo educacional; não é recomendação. Use gestão de risco.**")

#============================================================================================================================================================================================
#============================================================================================================================================================================================
#============================================================================================================================================================================================

# ============================================================
# 🔽 RESUMO DESCRITIVO — Kelly (fracionado)  [plug-and-play]
# Cole a partir daqui no FINAL da página Kelly
# ============================================================
import io
import math
import numpy as np
import pandas as pd
import streamlit as st

# -------- helpers: pegar variáveis já existentes (globals ou session_state) ----
def _get_any(names, default=None):
    for n in names:
        if n in globals(): return globals()[n]
        if n in st.session_state: return st.session_state[n]
    return default

def _as_prob(x):
    if x is None: return None
    try:
        x = float(x)
        return x/100.0 if x > 1.0000001 else x
    except Exception:
        return None

# tenta encontrar win rate (p) e payoff (b) usados na sua página
p = _as_prob(_get_any(["p","wr","win_rate","winrate","wr_input","p_input"]))
b = _get_any(["b","payoff","payoff_ratio","b_input","rr","r_risk_reward"])

# histórico (opcional, se sua página preencher)
wr_hist = _as_prob(_get_any(["wr_hist","win_rate_hist","hist_wr"]))
b_hist  = _get_any(["b_hist","hist_b"])
n_obs   = _get_any(["obs_validas","n_obs","n_valid"], None)

# -------- validação mínima --------
if p is None or b is None or b <= 0 or not (0 <= p <= 1):
    st.warning("Resumo Kelly: não encontrei `p` (win rate) e/ou `b` (payoff). "
               "Garanta que os controles estejam preenchidos.")
else:
    q = 1.0 - p
    # Kelly ótimo (f*)
    f_star = p - q / b
    f_half = 0.5 * f_star
    f_quar = 0.25 * f_star

    # win rate mínimo para não perder dinheiro (breakeven) e payoff mínimo dado p
    p_break = 1.0 / (1.0 + b)         # precisa de p > p_break para EV ≥ 0
    b_min   = (q / p) if p > 0 else np.inf  # precisa de b > b_min para EV ≥ 0

    # expectativa simples por trade normalizando a perda média como 1
    # (ganha b quando acerta; perde 1 quando erra)
    ev = p * b - q

    # crescimento log (por trade) em Kelly e em 1/2 Kelly
    def _g_log(frac: float) -> float | None:
        # evita log(negativo) quando frac for muito grande p/ b
        if frac is None or not np.isfinite(frac): return None
        if frac <= -0.999 or frac*b <= -0.999:  return None
        try:
            return p * math.log1p(frac*b) + q * math.log1p(-frac)
        except ValueError:
            return None

    g_kelly = _g_log(f_star)
    g_half  = _g_log(f_half)
    # classificação qualitativa do edge
    if f_star <= 0:
        label = "Desfavorável — não operar (Kelly ≤ 0)"
        color = "#b00020"
    elif f_star < 0.05:
        label = "Edge fraco — use frações pequenas (¼ Kelly)"
        color = "#d97706"
    elif f_star < 0.15:
        label = "Edge moderado — ½ Kelly costuma ser prudente"
        color = "#0ea5e9"
    else:
        label = "Edge forte — ½ Kelly ainda é prudente"
        color = "#16a34a"

    # --------- CARD descritivo grandão ---------
    def _fmt_pct(x, nd=2):
        try: return f"{float(x)*100:.{nd}f}%"
        except: return "—"

    st.markdown(f"""
    <div style="border-radius:14px;padding:14px 16px;margin:10px 0;
                background:linear-gradient(135deg,{color},#111827);color:#fff">
      <div style="font-weight:700;font-size:18px;margin-bottom:6px">Resumo — Kelly (fracionado)</div>
      <div style="font-size:15px">
        Com <b>Win Rate</b> = <b>{_fmt_pct(p)}</b> e <b>Payoff</b> = <b>{b:.2f}</b>,
        o Kelly ótimo é <b><span style="font-size:20px">{_fmt_pct(f_star)}</span></b>.
        <br/>Na prática, recomenda-se operar frações para reduzir drawdowns:
        ½ Kelly = <b>{_fmt_pct(f_half)}</b> • ¼ Kelly = <b>{_fmt_pct(f_quar)}</b>.
        <br/><br/>
        <b>Diagnóstico:</b> {label}.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # --------- Tabela de métricas úteis ---------
    rows = [
        ["Win Rate (entrada)", _fmt_pct(p),               "Payoff (entrada)",  f"{b:.2f}"],
        ["Kelly (f*)",         _fmt_pct(f_star),          "½ Kelly",           _fmt_pct(f_half)],
        ["¼ Kelly",            _fmt_pct(f_quar),          "EV por trade",      f"{ev:.4f} (unidades de perda)"],
        ["Win Rate mínimo p*", _fmt_pct(p_break),         "Payoff mínimo b*",  "∞" if b_min==np.inf else f"{b_min:.2f}"],
        ["gᵣ (Kelly)",         "—" if g_kelly is None else f"{g_kelly:.5f}",
         "gᵣ (½ Kelly)",       "—" if g_half  is None else f"{g_half:.5f}"],
    ]
    # comparar com histórico, se houver
    if wr_hist is not None and b_hist not in (None, 0):
        f_hist = wr_hist - (1.0 - wr_hist)/b_hist
        rows.append(["Win Rate (hist.)", _fmt_pct(wr_hist),
                     "Payoff (hist.)",   f"{b_hist:.2f}"])
        rows.append(["Kelly (hist.)", _fmt_pct(f_hist),
                     "Obs. válidas",    f"{int(n_obs):,}" if isinstance(n_obs, (int, float)) and n_obs==n_obs else "—"])

    df_k = pd.DataFrame(rows, columns=["Métrica A","Valor A","Métrica B","Valor B"])
    st.dataframe(df_k, use_container_width=True, hide_index=True)

    # --------- Texto interpretativo curto ---------
    bullet = []
    if f_star <= 0:
        bullet.append("**Kelly ≤ 0** indica expectativa negativa com os parâmetros atuais — evite operar ou revise stops/alvos.")
    else:
        bullet.append("Usar **½ Kelly** reduz a volatilidade e costuma preservar ~75–80% do crescimento esperado do Kelly completo.")
        bullet.append(f"Para **não perder dinheiro**, você precisa de **p > { _fmt_pct(p_break) }** "
                      f"(ou **b > { '∞' if b_min==np.inf else f'{b_min:.2f}' }** dado seu win rate).")
        bullet.append("Se o payoff vier de razão alvo/stop (R), então **b ≈ R** (ganho médio ≈ R × perda média).")
    if wr_hist is not None and b_hist not in (None, 0):
        bullet.append("Comparar **Entrada vs. Histórica** ajuda a validar suposições. Divergências grandes sinalizam "
                      "overfitting ou mudança de regime.")

    st.markdown("**Como ler estes números**  \n- " + "\n- ".join(bullet))

    # --------- Exportar resumo (Markdown) ---------
    md = io.StringIO()
    md.write("# Resumo — Kelly (fracionado)\n\n")
    md.write(f"- Win Rate (entrada): **{_fmt_pct(p)}**\n")
    md.write(f"- Payoff (entrada): **{b:.2f}**\n")
    md.write(f"- Kelly ótimo (f*): **{_fmt_pct(f_star)}** • ½ Kelly: **{_fmt_pct(f_half)}** • ¼ Kelly: **{_fmt_pct(f_quar)}**\n")
    md.write(f"- EV por trade (perda=1): **{ev:.4f}**\n")
    md.write(f"- Win Rate mínimo para EV≥0: **{_fmt_pct(p_break)}** • Payoff mínimo dado seu p: **{'∞' if b_min==np.inf else f'{b_min:.2f}'}**\n")
    if g_kelly is not None or g_half is not None:
        md.write(f"- Crescimento log por trade — Kelly: **{'—' if g_kelly is None else f'{g_kelly:.5f}'}**, ½ Kelly: **{'—' if g_half is None else f'{g_half:.5f}'}**\n")
    if wr_hist is not None and b_hist not in (None, 0):
        md.write("\n## Histórico\n")
        md.write(f"- Win Rate (hist.): **{_fmt_pct(wr_hist)}** • Payoff (hist.): **{b_hist:.2f}**")
        if n_obs is not None: md.write(f" • Obs. válidas: **{int(n_obs):,}**")
        md.write("\n")
    md.write("\n> Regra prática: operar **½ Kelly** costuma ser mais estável; ajuste conforme tolerância a drawdown.\n")

    st.download_button(
        "⬇️ Baixar resumo (Markdown)",
        data=md.getvalue().encode("utf-8"),
        file_name="resumo_kelly.md",
        mime="text/markdown",
        use_container_width=True,
    )
