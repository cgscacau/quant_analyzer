# core/indicators.py
import pandas as pd
import numpy as np

def sma(series: pd.Series, length: int = 14) -> pd.Series:
    return series.rolling(length, min_periods=1).mean()

def ema(series: pd.Series, length: int = 14) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=1).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    # RSI (Wilder)
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi
