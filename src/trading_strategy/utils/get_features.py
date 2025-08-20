import pandas as pd
import pandas_datareader as pdr
import yfinance as yf


def get_growth(
    df: pd.DataFrame, column_name: str, drop_original_column: bool = True
) -> pd.DataFrame:
    """
    Get growth columns for different days

    :param df: dataframe with column_name column
    :param column_name: column for which we count growth
    :param drop_original_column: True if drop column column_name, else False
    :return: dataframe with columns like 'growth_{column_name}_{days}d'
        for [1, 3, 7, 30, 90, 252, 365] days
    """
    for days in [1, 3, 7, 30, 90, 252, 365]:
        df[f"growth_{column_name}_{days}d"] = df[column_name] / df[column_name].shift(
            days
        )
    if drop_original_column:
        df.drop(column_name, axis=1, inplace=True)
    return df


def get_macro_indicator_pdr(
    ticker: str, start_date: str, drop_original_column: bool = True
) -> pd.DataFrame:
    """
    Collect ticker from fred
    :param ticker: ticker name
    :param start_date: start date for collect
    :param drop_original_column: True if drop column 'ticker' with abs price, else False
    :return: dataframe with growth columns for ticker
        + [Date, year, month]
    """
    df = pdr.DataReader(ticker, "fred", start=start_date)
    df = get_growth(df, ticker, drop_original_column)
    df.reset_index(inplace=True)
    df.rename({"DATE": "Date"}, axis=1, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    return df


def get_ticker_growth_yf(ticker: str, start_date: str) -> pd.DataFrame:
    """
    Collect ticker and count price growth from yf
    :param ticker: ticker name
    :param start_date: start date for collect
    :return: dataframe with growth columns for ticker
        + [Date]
    """
    df = yf.Ticker(ticker).history(start=start_date)
    df.rename({"Close": ticker}, axis=1, inplace=True)
    df = get_growth(df, ticker)
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"].dt.date)
    return df
