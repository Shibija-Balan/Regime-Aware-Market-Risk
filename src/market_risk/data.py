"""Market data loading and cleaning."""

from collections.abc import Sequence
import pandas as pd
import yfinance as yf


def download_adjusted_close(tickers: Sequence[str], start: str, end: str | None = None) -> pd.DataFrame:
    if not tickers:
        raise ValueError("At least one ticker is required.")
    raw = yf.download(
        list(tickers), start=start, end=end, auto_adjust=True,
        progress=False, group_by="column", threads=True,
    )
    if raw.empty:
        raise ValueError("No market data were returned for the requested period.")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Downloaded data do not contain Close prices.")
        prices = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise ValueError("Downloaded data do not contain Close prices.")
        prices = raw[["Close"]].copy()
        prices.columns = [str(tickers[0])]
    prices = prices.reindex(columns=list(tickers)).dropna(how="any").sort_index()
    if prices.empty:
        raise ValueError("No overlapping observations remain after alignment.")
    if (prices <= 0).any().any():
        raise ValueError("Non-positive prices detected.")
    return prices


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("Price data are empty.")
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    if returns.empty:
        raise ValueError("Not enough observations to calculate returns.")
    return returns
