"""Loading daily index prices and checking their quality before any test is run."""

from __future__ import annotations

import pandas as pd

MARKETS = {"sp500": "S&P 500", "nasdaq": "NASDAQ Composite"}
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_market(name: str) -> pd.DataFrame:
    """Daily OHLCV for an index, 1999 to 2018, as shipped with the `arch` package.

    The data originally comes from Yahoo Finance. Index levels exclude dividends.
    """
    from arch.data import nasdaq, sp500

    loaders = {"sp500": sp500, "nasdaq": nasdaq}
    if name not in loaders:
        raise ValueError(f"unknown market {name!r}; choose from {sorted(loaders)}")
    df = loaders[name].load()[PRICE_COLUMNS].astype(float)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def returns(df: pd.DataFrame) -> pd.DataFrame:
    """Simple returns: close to close, overnight (close to next open) and intraday (open to close)."""
    out = pd.DataFrame(index=df.index)
    out["cc"] = df["Close"].pct_change()
    out["overnight"] = df["Open"] / df["Close"].shift(1) - 1
    out["intraday"] = df["Close"] / df["Open"] - 1
    return out


def open_equals_prev_close(df: pd.DataFrame) -> pd.Series:
    """Share of days in each year where the open exactly equals the previous close.

    A real index almost never opens exactly at yesterday's close. A share near 1 means
    the vendor filled the open with the previous close, so open prices cannot be used.
    """
    same = df["Open"] == df["Close"].shift(1)
    return same.groupby(df.index.year).mean()


def open_prices_valid(df: pd.DataFrame, max_share: float = 0.05) -> bool:
    """True if no year has more than `max_share` of opens copied from the previous close."""
    return bool((open_equals_prev_close(df).dropna() <= max_share).all())
