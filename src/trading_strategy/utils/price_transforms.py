import numpy as np
import pandas as pd


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def sma(series, period):
    return series.rolling(window=period).mean()


# ---------- Momentum ----------
def momentum(series, period=10):
    return series.diff(period)


def roc(series, period=10):
    return (series.diff(period) / series.shift(period)) * 100


def rocp(series, period=10):
    return series.diff(period) / series.shift(period)


def rocr(series, period=10):
    return series / series.shift(period)


def rocr100(series, period=10):
    return (series / series.shift(period)) * 100


def rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=series.index).rolling(period).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def williams_r(high, low, close, period=14):
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    return -100 * ((highest_high - close) / (highest_high - lowest_low))


# ---------- DMI/ADX Family ----------
def true_range(high, low, close):
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def plus_dm(high, low):
    move_up = high.diff()
    move_down = low.diff().abs()
    return np.where((move_up > move_down) & (move_up > 0), move_up, 0)


def minus_dm(high, low):
    move_up = high.diff()
    move_down = low.diff().abs()
    return np.where((move_down > move_up) & (move_down > 0), move_down, 0)


def plus_di(high, low, close, period=14):
    tr = true_range(high, low, close).rolling(period).sum()
    pdm = pd.Series(plus_dm(high, low)).rolling(period).sum()
    return 100 * (pdm / tr)


def minus_di(high, low, close, period=14):
    tr = true_range(high, low, close).rolling(period).sum()
    mdm = pd.Series(minus_dm(high, low)).rolling(period).sum()
    return 100 * (mdm / tr)


def dx(high, low, close, period=14):
    pdi = plus_di(high, low, close, period)
    mdi = minus_di(high, low, close, period)
    return (abs(pdi - mdi) / (pdi + mdi)) * 100


def adx(high, low, close, period=14):
    return dx(high, low, close, period).rolling(period).mean()


def adxr(high, low, close, period=14):
    adx_series = adx(high, low, close, period)
    return (adx_series + adx_series.shift(period)) / 2


# ---------- Trend & Oscillators ----------
def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def apo(series, fast=12, slow=26):
    return ema(series, fast) - ema(series, slow)


def ppo(series, fast=12, slow=26):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    return (fast_ema - slow_ema) / slow_ema * 100


def cci(high, low, close, period=14):
    tp = (high + low + close) / 3
    sma_tp = sma(tp, period)
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    return (tp - sma_tp) / (0.015 * mad)


def cmo(close, period=14):
    diff = close.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    sum_up = up.rolling(period).sum()
    sum_down = down.rolling(period).sum()
    return 100 * (sum_up - sum_down) / (sum_up + sum_down)


def stoch(high, low, close, k=14, d=3):
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    k_line = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d_line = k_line.rolling(d).mean()
    return k_line, d_line


def stochf(high, low, close, k=5, d=3):
    return stoch(high, low, close, k, d)


def stochrsi(close, period=14, k=5, d=3):
    rsi_series = rsi(close, period)
    lowest = rsi_series.rolling(period).min()
    highest = rsi_series.rolling(period).max()
    stoch_rsi = (rsi_series - lowest) / (highest - lowest)
    k_line = stoch_rsi.rolling(k).mean()
    d_line = k_line.rolling(d).mean()
    return k_line, d_line


def trix(close, period=30):
    ema1 = ema(close, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return ema3.pct_change() * 100


def ultosc(high, low, close, s1=7, s2=14, s3=28):
    tr = true_range(high, low, close)
    bp = close - pd.concat([low, close.shift()], axis=1).min(axis=1)
    avg1 = bp.rolling(s1).sum() / tr.rolling(s1).sum()
    avg2 = bp.rolling(s2).sum() / tr.rolling(s2).sum()
    avg3 = bp.rolling(s3).sum() / tr.rolling(s3).sum()
    return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7


# ---------- Volume ----------
def obv(close, volume):
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()


def ad_line(high, low, close, volume):
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm * volume
    return mfv.cumsum()


def adosc(high, low, close, volume, fast=3, slow=10):
    ad = ad_line(high, low, close, volume)
    return ema(ad, fast) - ema(ad, slow)


# ---------- Volatility ----------
def atr(high, low, close, period=14):
    tr = true_range(high, low, close)
    return tr.rolling(period).mean()


def natr(high, low, close, period=14):
    return atr(high, low, close, period) / close * 100


# ---------- Price Transforms ----------
def avgprice(open_, high, low, close):
    return (open_ + high + low + close) / 4


def medprice(high, low):
    return (high + low) / 2


def typprice(high, low, close):
    return (high + low + close) / 3


def wclprice(high, low, close):
    return (high + low + 2 * close) / 4


# ---------- Candlestick Patterns ----------
def cdl_doji(open_, high, low, close, threshold=0.1):
    body = np.abs(close - open_)
    avg_range = (high - low).rolling(14).mean()
    return (body <= threshold * avg_range).astype(int)


def cdl_engulfing(open_, close):
    prev_open, prev_close = open_.shift(), close.shift()
    cond_bull = (
        (close > open_)
        & (prev_close < prev_open)
        & (close > prev_open)
        & (open_ < prev_close)
    )
    cond_bear = (
        (open_ > close)
        & (prev_open < prev_close)
        & (open_ > prev_close)
        & (close < prev_open)
    )
    return cond_bull.astype(int) - cond_bear.astype(int)


def cdl_hammer(open_, high, low, close):
    body = np.abs(close - open_)
    candle_range = high - low
    lower_shadow = (open_ - low).where(close >= open_, close - low)
    return ((lower_shadow >= 2 * body) & (body <= 0.3 * candle_range)).astype(int)


def cdl_shootingstar(open_, high, low, close):
    body = np.abs(close - open_)
    candle_range = high - low
    upper_shadow = (high - close).where(close >= open_, high - open_)
    return ((upper_shadow >= 2 * body) & (body <= 0.3 * candle_range)).astype(int)
