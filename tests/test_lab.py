import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_lab import stats as st  # noqa: E402
from signal_lab.backtest import from_positions, holding_period  # noqa: E402
from signal_lab.data import open_equals_prev_close, open_prices_valid, returns  # noqa: E402
from signal_lab.signals import momentum, position_signals  # noqa: E402


def random_prices(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, n))
    return pd.DataFrame({"Open": open_, "High": close, "Low": close, "Close": close, "Volume": 1.0}, index=idx)


def test_signals_do_not_look_ahead():
    """Changing prices after day t must not change any position decided at day t."""
    df = random_prices()
    t = 400
    base = position_signals(df["Close"])
    shocked = df["Close"].copy()
    shocked.iloc[t + 1 :] *= 1.5
    after = position_signals(shocked)
    for name in base:
        pd.testing.assert_series_equal(base[name].iloc[: t + 1], after[name].iloc[: t + 1], check_names=False)


def test_positions_are_held_from_the_next_day():
    df = random_prices()
    r = returns(df)["cc"]
    pos = momentum(df["Close"], 5)
    s = from_positions("m", pos, r)
    expected = pos.shift(1).fillna(0) * r.fillna(0)
    pd.testing.assert_series_equal(s.gross, expected, check_names=False)


def test_turnover_and_costs():
    idx = pd.bdate_range("2020-01-01", periods=6)
    r = pd.Series(0.0, index=idx)
    flip = pd.Series([1, -1, 1, -1, 1, -1], index=idx, dtype=float)
    s = from_positions("flip", flip, r)
    assert s.turnover.tolist() == [0, 1, 2, 2, 2, 2]
    assert np.allclose(s.net(5).iloc[2:], -2 * 5 / 1e4)
    hold = holding_period("overnight", r)
    assert (hold.turnover == 2).all()


def test_hac_ttest_finds_a_real_edge_and_not_a_fake_one():
    rng = np.random.default_rng(1)
    edge = pd.Series(rng.normal(0.001, 0.01, 5000))
    none = pd.Series(rng.normal(0.0, 0.01, 5000))
    assert st.hac_ttest(edge)[1] < 0.001
    assert st.hac_ttest(none)[1] > 0.05


def test_benjamini_hochberg_is_less_strict_than_bonferroni():
    p = pd.Series([0.001, 0.004, 0.02, 0.03, 0.4, 0.8])
    adj = st.adjust_pvalues(p)
    assert adj["reject_bh"].sum() >= adj["reject_bonferroni"].sum()
    assert (adj["p_bonferroni"] >= adj["p"]).all()


def test_spa_is_studentized():
    """A noisy zero-mean model must not hide a precise, real edge."""
    rng = np.random.default_rng(2)
    n = 2000
    idx = pd.RangeIndex(n)
    good = rng.normal(0.0008, 0.005, n)
    noisy = rng.normal(0.0, 0.03, n)
    bench = pd.Series(0.0, index=idx)
    alone = st.spa_test(bench, pd.DataFrame({"good": good}), reps=500)
    both = st.spa_test(bench, pd.DataFrame({"good": good, "noisy": noisy}), reps=500)
    assert alone["consistent"] < 0.01
    assert both["consistent"] < 0.02
    assert both["best_model"] == "good"


def test_spa_does_not_reject_pure_noise():
    rng = np.random.default_rng(3)
    models = pd.DataFrame(rng.normal(0, 0.01, (1500, 10)))
    res = st.spa_test(pd.Series(0.0, index=models.index), models, reps=500)
    assert res["consistent"] > 0.05


def test_open_price_check_flags_copied_opens():
    df = random_prices()
    assert open_prices_valid(df)
    bad = df.copy()
    bad["Open"] = bad["Close"].shift(1)
    assert open_equals_prev_close(bad).dropna().min() > 0.9
    assert not open_prices_valid(bad)


def test_break_even_cost():
    gross = pd.Series([0.0002] * 10)
    turnover = pd.Series([1.0] * 10)
    assert st.break_even_cost_bps(gross, turnover) == pytest.approx(2.0)
