"""Turning positions into daily strategy returns, with explicit trading costs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Strategy:
    """Daily gross returns and turnover for one strategy.

    `turnover` on day t is the size of the position change made at the previous close,
    so a flip from -1 to +1 counts as 2. Costs are charged per unit of turnover.
    """

    name: str
    gross: pd.Series
    turnover: pd.Series

    def net(self, cost_bps: float) -> pd.Series:
        return self.gross - self.turnover * cost_bps / 1e4


def from_positions(name: str, position: pd.Series, asset_returns: pd.Series) -> Strategy:
    """Hold `position` decided at the close of t over the return from t to t+1."""
    held = position.shift(1).fillna(0.0)
    gross = held * asset_returns.fillna(0.0)
    turnover = position.diff().abs().shift(1).fillna(position.abs().shift(1)).fillna(0.0)
    return Strategy(name, gross, turnover)


def holding_period(name: str, period_returns: pd.Series, trades_per_day: float = 2.0) -> Strategy:
    """A strategy that earns one part of each day, for example only the overnight return.

    Buying at one time and selling at another every day means two trades per day.
    """
    gross = period_returns.fillna(0.0)
    turnover = pd.Series(trades_per_day, index=gross.index)
    return Strategy(name, gross, turnover)


def buy_and_hold(asset_returns: pd.Series) -> Strategy:
    gross = asset_returns.fillna(0.0)
    return Strategy("buy_and_hold", gross, pd.Series(0.0, index=gross.index))
