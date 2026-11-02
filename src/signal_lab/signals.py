"""A family of classic daily signals.

Every position is decided at the close of day t using prices up to and including day t,
and is held over the next day's return. Positions are +1 (long), -1 (short) or 0 (flat).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MOMENTUM_LOOKBACKS = [5, 10, 21, 63, 126, 252]
REVERSAL_LOOKBACKS = [1, 2, 5]
MA_PAIRS = [(5, 20), (10, 50), (20, 100), (50, 200)]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def _sign(x: pd.Series) -> pd.Series:
    return np.sign(x).fillna(0.0)


def momentum(close: pd.Series, lookback: int) -> pd.Series:
    """Long if the trailing `lookback` day return is positive, short if negative."""
    return _sign(close / close.shift(lookback) - 1)


def reversal(close: pd.Series, lookback: int) -> pd.Series:
    """Short-term mean reversion: bet against the trailing `lookback` day move."""
    return -_sign(close / close.shift(lookback) - 1)


def ma_crossover(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Long when the fast moving average is above the slow one, short otherwise."""
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    pos = pd.Series(np.where(f > s, 1.0, -1.0), index=close.index)
    return pos.where(s.notna(), 0.0)


def day_of_week(index: pd.DatetimeIndex, weekday: int) -> pd.Series:
    """Long only over the next trading day if it falls on `weekday` (0 = Monday).

    The calendar is known in advance, so using the next day's weekday is not look-ahead.
    """
    next_day = index.to_series().shift(-1)
    pos = (next_day.dt.weekday == weekday).astype(float)
    return pd.Series(pos.values, index=index)


def position_signals(close: pd.Series) -> dict[str, pd.Series]:
    """All position-based signals, keyed by a short name."""
    sigs: dict[str, pd.Series] = {}
    for lb in MOMENTUM_LOOKBACKS:
        sigs[f"momentum_{lb}d"] = momentum(close, lb)
    for lb in REVERSAL_LOOKBACKS:
        sigs[f"reversal_{lb}d"] = reversal(close, lb)
    for fast, slow in MA_PAIRS:
        sigs[f"ma_{fast}_{slow}"] = ma_crossover(close, fast, slow)
    for i, name in enumerate(WEEKDAYS):
        sigs[f"long_{name}"] = day_of_week(close.index, i)
    return sigs
