"""Statistics used to judge whether a signal's performance is real.

- Newey-West (HAC) t-tests, because daily strategy returns are autocorrelated and
  heteroskedastic, which makes the ordinary t-test overconfident.
- Stationary bootstrap confidence intervals for the Sharpe ratio, which keep the
  dependence between nearby days instead of assuming independent draws.
- Bonferroni and Benjamini-Hochberg corrections, because testing many signals on the
  same data makes some look significant by chance.
- Hansen's Superior Predictive Ability (SPA) test, which asks whether the best of all
  the signals beats a benchmark once the search over signals is accounted for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.bootstrap import SPA, StationaryBootstrap
from scipy import stats as sps
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.multitest import multipletests

TRADING_DAYS = 252


def ann_return(x: pd.Series) -> float:
    return float(x.mean() * TRADING_DAYS)


def ann_vol(x: pd.Series) -> float:
    return float(x.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(x) -> float:
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return float(np.sqrt(TRADING_DAYS) * x.mean() / sd) if sd > 0 else 0.0


def newey_west_lags(n: int) -> int:
    """The usual rule of thumb: floor(4 * (n / 100) ** (2 / 9))."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def hac_ttest(x: pd.Series) -> tuple[float, float]:
    """t-statistic and two-sided p-value for H0: mean daily return = 0, with Newey-West errors."""
    y = np.asarray(x, dtype=float)
    n = len(y)
    res = sm.OLS(y, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(n)})
    return float(res.tvalues[0]), float(res.pvalues[0])


def sharpe_ci(x: pd.Series, reps: int = 1000, block: int = 10, seed: int = 0) -> tuple[float, float]:
    """95% stationary bootstrap percentile interval for the annualised Sharpe ratio."""
    bs = StationaryBootstrap(block, np.asarray(x, dtype=float), seed=seed)
    ci = bs.conf_int(lambda a: np.array([sharpe(a)]), reps=reps, method="percentile")
    return float(ci[0, 0]), float(ci[1, 0])


def adjust_pvalues(p: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Bonferroni (controls the chance of any false positive) and Benjamini-Hochberg
    (controls the expected share of false positives among discoveries)."""
    p = p.astype(float)
    bonf = multipletests(p.values, alpha=alpha, method="bonferroni")
    bh = multipletests(p.values, alpha=alpha, method="fdr_bh")
    return pd.DataFrame(
        {
            "p": p.values,
            "p_bonferroni": bonf[1],
            "p_bh": bh[1],
            "reject_raw": p.values < alpha,
            "reject_bonferroni": bonf[0],
            "reject_bh": bh[0],
        },
        index=p.index,
    )


def spa_test(benchmark: pd.Series, models: pd.DataFrame, reps: int = 2000, block: int = 10, seed: int = 0) -> dict:
    """Hansen's (2005) studentized SPA test, written out directly.

    H0: no model has a higher expected daily return than the benchmark.
    d[t, k] = model k's return minus the benchmark's. The test statistic is the largest
    studentized mean, max_k sqrt(n) * mean(d_k) / omega_k, so a high-volatility signal
    cannot dominate just because its average moves around more. The null distribution
    comes from a stationary bootstrap that keeps nearby days together.

    Returns the lower, consistent (the one to report) and upper p-values.
    """
    d = models.to_numpy(dtype=float) - benchmark.to_numpy(dtype=float)[:, None]
    n, _ = d.shape
    dbar = d.mean(axis=0)
    bs = StationaryBootstrap(block, d, seed=seed)
    boot = np.array([data[0][0].mean(axis=0) for data in bs.bootstrap(reps)])
    omega = np.sqrt(n * boot.var(axis=0, ddof=1))
    omega = np.where(omega > 0, omega, np.inf)

    t_obs = max(0.0, float(np.max(np.sqrt(n) * dbar / omega)))
    threshold = -np.sqrt(omega**2 / n * 2 * np.log(np.log(n)))
    recentre = {
        "lower": np.maximum(dbar, 0.0),
        "consistent": np.where(dbar >= threshold, dbar, 0.0),
        "upper": dbar,
    }
    out = {}
    for name, g in recentre.items():
        z = np.sqrt(n) * (boot - g) / omega
        t_star = np.maximum(0.0, z.max(axis=1))
        out[name] = float((t_star > t_obs).mean())
    out["t_stat"] = t_obs
    out["best_model"] = str(models.columns[int(np.argmax(dbar / omega))])
    return out


def spa_test_arch(benchmark: pd.Series, models: pd.DataFrame, reps: int = 2000, block: int = 10, seed: int = 0) -> dict:
    """The `arch` package's SPA, kept for comparison. In arch 8.0.0 its p-values did not
    change when `studentize` was switched off, so it behaves like an unstudentized test."""
    spa = SPA(-benchmark.values, -models.values, block_size=block, reps=reps, seed=seed)
    spa.compute()
    pv = spa.pvalues
    return {"lower": float(pv["lower"]), "consistent": float(pv["consistent"]), "upper": float(pv["upper"])}


def stylised_facts(r: pd.Series, lags: int = 10) -> dict:
    """Fat tails and volatility clustering in daily returns."""
    r = r.dropna()
    jb = sps.jarque_bera(r)
    lb_r = acorr_ljungbox(r, lags=[lags], return_df=True)
    lb_r2 = acorr_ljungbox(r**2, lags=[lags], return_df=True)
    arch_lm = het_arch(r, nlags=lags)
    return {
        "n_days": int(len(r)),
        "ann_return": ann_return(r),
        "ann_vol": ann_vol(r),
        "skew": float(sps.skew(r)),
        "excess_kurtosis": float(sps.kurtosis(r)),
        "jarque_bera_p": float(jb.pvalue),
        f"ljung_box_returns_p_lag{lags}": float(lb_r["lb_pvalue"].iloc[0]),
        f"ljung_box_squared_p_lag{lags}": float(lb_r2["lb_pvalue"].iloc[0]),
        f"arch_lm_p_lag{lags}": float(arch_lm[1]),
        "days_beyond_4sd": int((np.abs((r - r.mean()) / r.std()) > 4).sum()),
        "expected_beyond_4sd_if_normal": float(len(r) * 2 * sps.norm.sf(4)),
    }


def acf(x: pd.Series, nlags: int = 20) -> np.ndarray:
    from statsmodels.tsa.stattools import acf as _acf

    return _acf(x.dropna(), nlags=nlags, fft=True)[1:]


def break_even_cost_bps(gross: pd.Series, turnover: pd.Series) -> float:
    """Cost per unit of turnover, in basis points, at which the average net return is zero."""
    t = float(turnover.mean())
    return float(gross.mean() / t * 1e4) if t > 0 else float("inf")


def rank_correlation(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    res = sps.spearmanr(a, b)
    return float(res.statistic), float(res.pvalue)
