from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trading_strategy.utils.get_features import (
    get_growth,
    get_macro_indicator_pdr,
    get_ticker_growth_yf,
)


def test_get_growth_drop_original_column():
    data = {"col1": [420, 380, 390], "col2": [50, 40, 45]}
    test_df = pd.DataFrame(data)
    df_resp = get_growth(test_df, "col1", drop_original_column=True)
    assert len(df_resp.columns) == 8
    assert "col1" not in df_resp.columns


def test_get_growth_not_drop_original_column():
    data = {"col1": [420, 380, 390], "col2": [50, 40, 45]}
    test_df = pd.DataFrame(data)
    df_resp = get_growth(test_df, "col1", drop_original_column=False)
    assert len(df_resp.columns) == 9
    assert "col1" in df_resp.columns


@pytest.fixture
def sample_df():
    """Fixture: mock dataframe returned by pdr.DataReader"""
    dates = pd.date_range("2020-01-01", periods=3, freq="M")
    df = pd.DataFrame({"TEST": [100, 105, 110]}, index=dates)
    df.index.name = "DATE"
    return df


@pytest.fixture
def growth_df():
    """Fixture: mock dataframe returned by get_growth"""
    dates = pd.date_range("2020-01-01", periods=3, freq="M")
    df = pd.DataFrame(
        {
            "TEST_growth": [0.0, 0.05, 0.047],
        },
        index=dates,
    )
    df.index.name = "DATE"
    return df


@pytest.fixture
def sample_yf_df():
    """Fixture: mock dataframe returned by yf.Ticker().history()"""
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame({"Close": [100, 102, 104]}, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture
def growth_yf_df():
    """Fixture: mock dataframe returned by get_growth"""
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame({"TEST_growth": [0.0, 0.02, 0.019]}, index=dates)
    df.index.name = "Date"
    return df


@patch("trading_strategy.utils.get_features.get_growth")
@patch("pandas_datareader.DataReader")
def test_get_macro_indicator_pdr(
    mock_datareader, mock_get_growth, sample_df, growth_df
):
    mock_datareader.return_value = sample_df
    mock_get_growth.return_value = growth_df

    result = get_macro_indicator_pdr("TEST", "2020-01-01")

    mock_datareader.assert_called_once_with("TEST", "fred", start="2020-01-01")
    mock_get_growth.assert_called_once_with(sample_df, "TEST", True)
    assert "Date" in result.columns
    assert "year" in result.columns
    assert "month" in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])
    assert all(result["year"] == result["Date"].dt.year)
    assert all(result["month"] == result["Date"].dt.month)
    assert "TEST_growth" in result.columns
    assert len(result) == len(growth_df)


@patch("trading_strategy.utils.get_features.get_growth")
@patch("yfinance.Ticker")
def test_get_ticker_growth_yf(
    mock_ticker_class, mock_get_growth, sample_yf_df, growth_yf_df
):
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = sample_yf_df
    mock_ticker_class.return_value = mock_ticker_instance
    mock_get_growth.return_value = growth_yf_df

    result = get_ticker_growth_yf("TEST", "2020-01-01")

    mock_ticker_class.assert_called_once_with("TEST")
    mock_ticker_instance.history.assert_called_once_with(start="2020-01-01")
    assert "Date" in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])
    assert "TEST_growth" in result.columns
    assert all(result["Date"] == pd.to_datetime(growth_yf_df["Date"]))
    assert len(result) == len(growth_yf_df)
