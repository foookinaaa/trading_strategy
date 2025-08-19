from typing import Tuple

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Creates an Exponentially Weighted Moving object
    EMA gives more weight to recent data points

    :param series: Pandas Series (e.g., closing prices of a stock)
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: Exponentially Weighted Moving Average
    """
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """
     Simple Moving Average
     It assigns equal weight to all values in the window

    :param series: Pandas Series (e.g., closing prices of a stock)
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: Simple Moving Average
    """
    return series.rolling(window=period).mean()


def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Measure of price change speed or trend

    :param series: Pandas Series (e.g., closing prices of a stock)
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: difference between the current value
        and the value period steps ago
    """
    return series.diff(period)


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Rate of Change

    :param series: Pandas Series (e.g., closing prices of a stock)
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: percentage change of the series over the given period
    """
    return (series.diff(period) / series.shift(period)) * 100


def rocp(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Rate of Change in proportion

    :param series: Pandas Series (e.g., closing prices of a stock)
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: same as roc, but without multiplying by 100
    """
    return series.diff(period) / series.shift(period)


def rocr(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Rate of Change Ratio: relative change expressed as a ratio

    :param series: Pandas Series (e.g., closing prices of a stock)
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: 1 → no change over the period;
        Greater than 1 → the series increased over the period
        Less than 1 → the series decreased over the period.
    """
    return series / series.shift(period)


def rocr100(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Rate of Change Ratio expressed as a percentage

    :param series: Pandas Series (e.g., closing prices of a stock)
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: same as rocr but scaled to percentage form
    """
    return (series / series.shift(period)) * 100


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index:
        smoothed measure of recent gains vs. recent losses
        over the specified period

    :param series: Pandas Series (e.g., closing prices of a stock)
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: [0, 100];
        Above 70 → asset may be overbought;
        Below 30 → asset may be oversold;
        Measures momentum rather than trend
    """
    delta = series.diff()  # change between consecutive values
    gain = np.where(delta > 0, delta, 0)  # positive changes
    loss = np.where(delta < 0, -delta, 0)  # negative changes turned positive
    avg_gain = (
        pd.Series(gain, index=series.index).rolling(period).mean()
    )  # average gain over 'period'
    avg_loss = (
        pd.Series(loss, index=series.index).rolling(period).mean()
    )  # average loss over 'period'
    rs = avg_gain / avg_loss  # relative strength
    return 100 - (100 / (1 + rs))


def williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Williams Percent Range:
        a momentum indicator that measures overbought and oversold levels

    :param high: high price
    :param low: low price
    :param close: close price
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: [0, -100]:
        Above -20 → asset may be overbought;
        Below -80 → asset may be oversold;
        It compares the current close to the recent high-low range,
            giving a sense of momentum and reversal potential.
    """
    highest_high = high.rolling(
        period
    ).max()  # highest high over the last 'period' bars
    lowest_low = low.rolling(period).min()  # lowest low over the last 'period' bars
    return -100 * ((highest_high - close) / (highest_high - lowest_low))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    True Range (TR):
        a key component in volatility indicators

    :param high: high price
    :param low: low price
    :param close: close price
    :return: largest of price difference in a day and prev day
    """
    high_low = high - low  # Range of the current period
    high_close = (high - close.shift()).abs()  # Gap from previous close to current high
    low_close = (low - close.shift()).abs()  # Gap from previous close to current low
    return pd.concat([high_low, high_close, low_close], axis=1).max(
        axis=1
    )  # Take the largest


def plus_dm(high: pd.Series, low: pd.Series) -> np.array:
    """
    Positive Directional Movement:
        only count upward movement if it is greater than
        downward movement AND positive

    :param high: high price
    :param low: low price
    :return: strong upward movements
    """
    move_up = (
        high.diff()
    )  # how much the price moved upward from the previous period’s high
    move_down = (
        low.diff().abs()
    )  # how much the price moved downward from the previous period’s low
    return np.where((move_up > move_down) & (move_up > 0), move_up, 0)


def minus_dm(high: pd.Series, low: pd.Series) -> np.array:
    """
    Negative Directional Movement:
        only count upward movement if it is lower than
        downward movement AND negative

    :param high: high price
    :param low: low price
    :return: strong downward movements
    """
    move_up = high.diff()
    move_down = low.diff().abs()
    return np.where((move_down > move_up) & (move_down > 0), move_down, 0)


def plus_di(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Positive Directional Indicator

    :param high: high price
    :param low: low price
    :param close: close price
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: +DI high → strong upward trend;
        +DI low → weak upward trend (or possibly downward trend).
    """
    tr = (
        true_range(high, low, close).rolling(period).sum()
    )  # Smoothed True Range over 'period'
    pdm = (
        pd.Series(plus_dm(high, low)).rolling(period).sum()
    )  # Smoothed Positive Directional Movement
    return 100 * (pdm / tr)  # Normalize to percentage scale


def minus_di(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Negative Directional Indicator

    :param high: high price
    :param low: low price
    :param close: close price
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: +DI high → strong downward trend;
        +DI low → weak downward trend (or possibly downward trend).
    """
    tr = true_range(high, low, close).rolling(period).sum()
    mdm = pd.Series(minus_dm(high, low)).rolling(period).sum()
    return 100 * (mdm / tr)


def dx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Directional Index:
        measures the strength of a trend by comparing
        the Positive Directional Indicator (+DI)
        and Negative Directional Indicator (−DI).
    :param high: high price
    :param low: low price
    :param close: close price
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: DX close to 0 → little diff between +DI and −DI → weak/sideways trend;
        DX closer to 100 → one direction dominates → strong trend.
    """
    pdi = plus_di(high, low, close, period)
    mdi = minus_di(high, low, close, period)
    return (abs(pdi - mdi) / (pdi + mdi)) * 100


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Average Directional Index: measures trend strength

    :param high: high price
    :param low: low price
    :param close: close price
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: 0–20 → Weak/No trend;
        20–40 → Developing/Moderate trend;
        40–60 → Strong trend;
        60+ → Very strong trend.
    """
    return dx(high, low, close, period).rolling(period).mean()


def adxr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Average Directional Movement Rating:
        measures trend strength with a bit of "lag" to reduce noise.
        often used to filter out false signals in choppy markets.

    :param high: high price
    :param low: low price
    :param close: close price
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: smoothed version of the ADX
    """
    adx_series = adx(high, low, close, period)
    return (adx_series + adx_series.shift(period)) / 2


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence

    :param series: Pandas Series (e.g., closing prices of a stock)
    :param fast: short-term EMA param
    :param slow: long-term EMA param
    :param signal: periods for EMA of the MACD line
    :return: MACD line crossing above Signal line → Bullish signal (momentum up).
        MACD line crossing below Signal line → Bearish signal (momentum down).
        Histogram → Expands when momentum strengthens, shrinks when momentum weakens.
    """
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema  # momentum/trend direction
    signal_line = ema(macd_line, signal)  # trigger line for buy/sell signals
    hist = (
        macd_line - signal_line
    )  # momentum strength (positive = bullish, negative = bearish)
    return macd_line, signal_line, hist


def apo(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """
    Absolute Price Oscillator (APO):
        a momentum indicator that’s very similar to MACD,
        but without the signal line and histogram

    :param series: Pandas Series (e.g., closing prices of a stock)
    :param fast: short-term EMA param
    :param slow: long-term EMA param
    :return: APO > 0 → Upward momentum (short EMA above long EMA).
        APO < 0 → Downward momentum (short EMA below long EMA).
        Crossing zero → Possible trend reversal signal.
    """
    return ema(series, fast) - ema(series, slow)


def ppo(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """
    Percentage Price Oscillator

    :param series: Pandas Series (e.g., closing prices of a stock)
    :param fast: short-term EMA param
    :param slow: long-term EMA param
    :return: PPO > 0 → Short-term EMA is above long-term EMA → bullish momentum;
        PPO < 0 → Short-term EMA is below long-term EMA → bearish momentum;
        Crossing zero line → Potential trend reversal.
    """
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    return (fast_ema - slow_ema) / slow_ema * 100


def cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Commodity Channel Index

    :param high: high price
    :param low: low price
    :param close: close price
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: CCI > +100 → Strong bullish signal (overbought / upward momentum).
        CCI < -100 → Strong bearish signal (oversold / downward momentum).
        Crossing ±100 → Often used as entry/exit signals.
    """
    tp = (high + low + close) / 3
    sma_tp = sma(tp, period)
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    return (tp - sma_tp) / (0.015 * mad)


def cmo(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Chande Momentum Oscillator:
        similar to RSI, but instead of using averages of gains and losses,
        it uses their sums, which makes it more sensitive to momentum changes.

    :param close: close price
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: [-100, +100]:
        CMO > +50 → Strong bullish momentum;
        CMO < -50 → Strong bearish momentum;
        CMO crossing above 0 → Possible start of an uptrend;
        CMO crossing below 0 → Possible start of a downtrend.
    """
    diff = close.diff()  # Price changes between periods
    up = diff.clip(lower=0)  # Positive changes (gains)
    down = -diff.clip(upper=0)  # Negative changes (losses, made positive)
    sum_up = up.rolling(period).sum()  # Total gains over period
    sum_down = down.rolling(period).sum()  # Total losses over period
    return 100 * (sum_up - sum_down) / (sum_up + sum_down)


def stoch(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator:
        identify overbought and oversold market conditions

    :param high: high price
    :param low: low price
    :param close: close price
    :param k: periods for measure current close relative to the recent high-low range
    :param d: moving average of k, used as a signal line
    :return: [0, 100]:
        %K or %D > 80 → Overbought (possible reversal down);
        %K or %D < 20 → Oversold (possible reversal up);
        Crossovers:
        %K crossing above %D → Bullish signal;
        %K crossing below %D → Bearish signal.
    """
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    k_line = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d_line = k_line.rolling(d).mean()
    return k_line, d_line


def stochf(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 5, d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator:
        identify overbought and oversold market conditions

    :param high: high price
    :param low: low price
    :param close: close price
    :param k: periods for measure current close relative to the recent high-low range
    :param d: moving average of k, used as a signal line
    :return: [0, 100]:
        %K or %D > 80 → Overbought (possible reversal down);
        %K or %D < 20 → Oversold (possible reversal up);
        Crossovers:
        %K crossing above %D → Bullish signal;
        %K crossing below %D → Bearish signal.
    """
    return stoch(high, low, close, k, d)


def stochrsi(
    close: pd.Series, period: int = 14, k: int = 5, d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic RSI:
        applies the stochastic oscillator formula to the RSI instead of price;
        more sensitive than regular RSI.

    :param close: close price
    :param period: lookback window (e.g., 12, 26, 50 days)
    :param k: periods for measure current close relative to the recent high-low range
    :param d: moving average of k, used as a signal line
    :return:  [0, 1]:
        > 0.8 → Overbought, < 0.2 → Oversold.
    """
    rsi_series = rsi(close, period)
    lowest = rsi_series.rolling(period).min()
    highest = rsi_series.rolling(period).max()
    stoch_rsi = (rsi_series - lowest) / (highest - lowest)
    k_line = stoch_rsi.rolling(k).mean()
    d_line = k_line.rolling(d).mean()
    return k_line, d_line


def trix(close: pd.Series, period: int = 30) -> pd.Series:
    """
    Triple-smoothed exponential moving average of the price

    :param close: close price
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: Positive → uptrend gaining strength
        Negative → downtrend gaining strength
    """
    ema1 = ema(close, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return ema3.pct_change() * 100


def ultosc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    s1: int = 7,
    s2: int = 14,
    s3: int = 28,
) -> pd.Series:
    """
    Ultimate Oscillator:
        capture momentum across three different timeframes to reduce false signals

    :param high: high price
    :param low: low price
    :param close: close price
    :param s1: short-term period
    :param s2: medium-term period
    :param s3: long-term period
    :return: overbought (>70) and oversold (<30) conditions
    """
    tr = true_range(high, low, close)  # True Range
    bp = close - pd.concat([low, close.shift()], axis=1).min(axis=1)  # Buying Pressure
    avg1 = bp.rolling(s1).sum() / tr.rolling(s1).sum()  # Short-term average
    avg2 = bp.rolling(s2).sum() / tr.rolling(s2).sum()  # Medium-term average
    avg3 = bp.rolling(s3).sum() / tr.rolling(s3).sum()  # Long-term average
    return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume: relates price movement to volume

    :param close: close price
    :param volume: volume
    :return: A rising OBV: volume is heavier on up days → potential buying pressure.
        A falling OBV: volume is heavier on down days → potential selling pressure.
    """
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()


def ad_line(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """
    Accumulation/Distribution (A/D) Line:
        detect whether a stock is being accumulated or distributed,
        even if the price itself isn’t moving much

    :param high: high price
    :param low: low price
    :param close: close price
    :param volume: volume
    :return: Rising A/D → accumulation (buying pressure),
        Falling A/D → distribution (selling pressure).
    """
    mfm = ((close - low) - (high - close)) / (high - low).replace(
        0, np.nan
    )  # where the close is in the high-low range
    mfv = mfm * volume  # money flow for that period
    return mfv.cumsum()


def adosc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast: int = 3,
    slow: int = 10,
) -> pd.Series:
    """
    Accumulation/Distribution Oscillator:
        measures the momentum of accumulation/distribution using volume and price

    :param high: high price
    :param low: low price
    :param close: close price
    :param volume: volume
    :param fast: short-term EMA param
    :param slow: long-term EMA param
    :return: Positive values → buying pressure (accumulation increasing)
        Negative values → selling pressure (distribution increasing)
        Crossovers of zero or the signal line: potential trend changes
    """
    ad = ad_line(high, low, close, volume)
    return ema(ad, fast) - ema(ad, slow)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Average True Range:
        how much the price typically moves in a given period,
        accounting for gaps and intraday swings

    :param high: high price
    :param low: low price
    :param close: close price
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: Higher ATR → higher price volatility
        Lower ATR → lower price volatility
    """
    tr = true_range(high, low, close)
    return tr.rolling(period).mean()


def natr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Normalized Average True Range:
        how volatile an asset is relative to its price

    :param high: high price
    :param low: low price
    :param close: close price
    :param period: lookback window (e.g., 12, 26, 50 days)
    :return: typical price movement over the period is about X% of the closing price.
    """
    return atr(high, low, close, period) / close * 100


def avgprice(
    open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """
    Average price of a candlestick (or bar) for each period

    :param open: opening price
    :param high: high price
    :param low: low price
    :param close: close price
    :return: typical price
    """
    return (open + high + low + close) / 4


def medprice(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Median price of a candlestick (or bar) for each period

    :param high: high price
    :param low: low price
    :return: simple central value of the price range for the period
    """
    return (high + low) / 2


def typprice(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Typical price for a given period

    :param high: high price
    :param low: low price
    :param close: close price
    :return: central value of the price
    """
    return (high + low + close) / 3


def wclprice(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Weighted close price for a given period.

    :param high: high price
    :param low: low price
    :param close: close price
    :return: weighted close gives more importance to the closing price
    """
    return (high + low + 2 * close) / 4


def cdl_doji(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    threshold: float = 0.1,
) -> int:
    """
    Doji candlestick patterns:
        A Doji is a candlestick where the open and close prices are almost equal.

    :param open: opening price
    :param high: high price
    :param low: low price
    :param close: close price
    :param threshold:
    :return: if the body is smaller than threshold of the average range -> Doji
    """
    body = np.abs(
        close - open
    )  # absolute size of the candle’s body (distance between open and close)
    avg_range = (
        (high - low).rolling(14).mean()
    )  # Average trading range of the last 14 periods
    return (body <= threshold * avg_range).astype(int)


def cdl_engulfing(open: pd.Series, close: pd.Series) -> int:
    """
    Engulfing candlestick patterns:
        identifies bullish and bearish engulfing patterns

    :param open: opening price
    :param close: close price
    :return: 1 → Bullish Engulfing
        -1 → Bearish Engulfing
        0 → No Engulfing pattern
    """
    prev_open, prev_close = open.shift(), close.shift()
    cond_bull = (
        (close > open)  # current candle is bullish
        & (prev_close < prev_open)  # previous candle was bearish
        & (close > prev_open)  # current close above previous open
        & (open < prev_close)  # current open below previous close
    )
    cond_bear = (
        (open > close)  # current candle is bearish
        & (prev_open < prev_close)  # previous candle was bullish
        & (open > prev_close)  # current open above previous close
        & (close < prev_open)  # current close below previous open
    )
    return cond_bull.astype(int) - cond_bear.astype(int)


def cdl_hammer(
    open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> int:
    """
    Hammer candlestick patterns:
        bullish reversal signals often appearing after a downtrend

    :param open: opening price
    :param high: high price
    :param low: low price
    :param close: close price
    :return: 1 if hammer pattern detected:
        (Lower shadow at least twice the body → long lower wick.
        Body is ≤ 30% of total candle range → small real body.)
        0 otherwise
    """
    body = np.abs(
        close - open
    )  # real body of the candle, i.e., difference between open and close.
    candle_range = high - low
    lower_shadow = (open - low).where(close >= open, close - low)
    return ((lower_shadow >= 2 * body) & (body <= 0.3 * candle_range)).astype(int)


def cdl_shootingstar(
    open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> int:
    """
    Shooting Star candlestick patterns:
        bearish reversal signal that usually appears after an uptrend,
        indicating potential selling pressure

    :param open: opening price
    :param high: high price
    :param low: low price
    :param close: close price
    :return: 1 if a Shooting Star is detected:
        (Upper shadow at least twice the body → long upper wick.
        Body ≤ 30% of total candle range → small real body.)
        0 otherwise.
    """
    body = np.abs(close - open)
    candle_range = high - low
    upper_shadow = (high - close).where(close >= open, high - open)
    return ((upper_shadow >= 2 * body) & (body <= 0.3 * candle_range)).astype(int)
