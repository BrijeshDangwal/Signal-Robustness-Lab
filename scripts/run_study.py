"""Run the full study: data checks, stylised facts, signal tests in and out of sample,
multiple testing corrections, the SPA test, trading cost sensitivity and figures.

Usage: python scripts/run_study.py [--fast]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from signal_lab import stats as st  # noqa: E402
from signal_lab.backtest import buy_and_hold, from_positions, holding_period  # noqa: E402
from signal_lab.data import MARKETS, load_market, open_equals_prev_close, open_prices_valid, returns  # noqa: E402
from signal_lab.signals import position_signals  # noqa: E402

PERIODS = {"in_sample": ("2000-01-01", "2010-12-31"), "out_of_sample": ("2011-01-01", "2018-12-31")}
COSTS_BPS = [0, 1, 2, 5, 10]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def build(df: pd.DataFrame):
    r = returns(df)
    strategies = [from_positions(n, p, r["cc"]) for n, p in position_signals(df["Close"]).items()]
    open_ok = open_prices_valid(df)
    if open_ok:
        strategies.append(holding_period("overnight_only", r["overnight"]))
        strategies.append(holding_period("intraday_only", r["intraday"]))
    return strategies, buy_and_hold(r["cc"]), r, open_ok


def evaluate_period(strategies, bench, start, end, reps, seed):
    rows = []
    for s in strategies:
        g = s.gross.loc[start:end]
        t = s.turnover.loc[start:end]
        tstat, p = st.hac_ttest(g)
        lo, hi = st.sharpe_ci(g, reps=reps, seed=seed)
        row = {
            "signal": s.name,
            "ann_return": st.ann_return(g),
            "ann_vol": st.ann_vol(g),
            "sharpe": st.sharpe(g),
            "sharpe_ci_low": lo,
            "sharpe_ci_high": hi,
            "t_hac": tstat,
            "p": p,
            "turnover_per_day": float(t.mean()),
            "break_even_bps": st.break_even_cost_bps(g, t),
        }
        for c in COSTS_BPS[1:]:
            row[f"sharpe_net_{c}bps"] = st.sharpe(s.net(c).loc[start:end])
        rows.append(row)
    table = pd.DataFrame(rows).set_index("signal")
    adj = st.adjust_pvalues(table["p"])
    table = table.join(adj.drop(columns="p"))
    models = pd.DataFrame({s.name: s.gross.loc[start:end] for s in strategies})
    b = bench.gross.loc[start:end]
    cash = pd.Series(0.0, index=b.index)
    spa_reps = max(reps, 1000)
    spa = {
        "vs_buy_and_hold": st.spa_test(b, models, reps=spa_reps, seed=seed),
        "vs_cash": st.spa_test(cash, models, reps=spa_reps, seed=seed),
        "arch_package_unstudentized": {
            "vs_buy_and_hold": st.spa_test_arch(b, models, reps=spa_reps, seed=seed),
            "vs_cash": st.spa_test_arch(cash, models, reps=spa_reps, seed=seed),
        },
    }
    bench_stats = {"ann_return": st.ann_return(b), "ann_vol": st.ann_vol(b), "sharpe": st.sharpe(b)}
    return table, spa, bench_stats


def run_market(name: str, reps: int, seed: int) -> dict:
    df = load_market(name)
    strategies, bench, r, open_ok = build(df)
    tables, spas, benches = {}, {}, {}
    for period, (start, end) in PERIODS.items():
        tables[period], spas[period], benches[period] = evaluate_period(strategies, bench, start, end, reps, seed)

    is_t, oos_t = tables["in_sample"], tables["out_of_sample"]
    rho, rho_p = st.rank_correlation(is_t["sharpe"], oos_t["sharpe"])
    best = is_t["sharpe"].idxmax()
    oos_rank = int(oos_t["sharpe"].rank(ascending=False)[best])
    bh_is = is_t.index[is_t["reject_bh"]].tolist()
    survivors = [
        s for s in bh_is if oos_t.loc[s, "p"] < 0.05 and np.sign(oos_t.loc[s, "sharpe"]) == np.sign(is_t.loc[s, "sharpe"])
    ]

    combined = pd.concat({"in_sample": is_t, "out_of_sample": oos_t}, names=["period"]).reset_index()
    combined.to_csv(RESULTS / f"signals_{name}.csv", index=False, float_format="%.6g")

    best_strategy = next(s for s in strategies if s.name == best)
    return {
        "market": MARKETS[name],
        "open_prices_valid": open_ok,
        "open_equals_prev_close_by_year": {int(k): round(float(v), 3) for k, v in open_equals_prev_close(df).dropna().items()},
        "n_signals": len(strategies),
        "stylised_facts_1999_2018": st.stylised_facts(r["cc"]),
        "benchmark": benches,
        "spa": spas,
        "counts": {
            p: {
                "significant_raw": int(t["reject_raw"].sum()),
                "significant_bh": int(t["reject_bh"].sum()),
                "significant_bonferroni": int(t["reject_bonferroni"].sum()),
                "positive_sharpe": int((t["sharpe"] > 0).sum()),
            }
            for p, t in tables.items()
        },
        "selection": {
            "spearman_is_vs_oos_sharpe": rho,
            "spearman_p": rho_p,
            "best_in_sample_signal": best,
            "best_in_sample_sharpe": float(is_t.loc[best, "sharpe"]),
            "best_signal_oos_sharpe": float(oos_t.loc[best, "sharpe"]),
            "best_signal_oos_rank": oos_rank,
            "best_signal_oos_p": float(oos_t.loc[best, "p"]),
            "bh_significant_in_sample": bh_is,
            "survived_out_of_sample": survivors,
            "same_sign_share": float((np.sign(is_t["sharpe"]) == np.sign(oos_t["sharpe"])).mean()),
        },
        "_tables": tables,
        "_returns": r,
        "_best_strategy": best_strategy,
        "_bench": bench,
        "_df": df,
    }


# ---------------------------------------------------------------- figures


def fig_open_quality(res):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for out in res.values():
        s = pd.Series(out["open_equals_prev_close_by_year"])
        ax.plot(s.index, s.values * 100, marker="o", ms=3, label=out["market"])
    ax.set_ylabel("% of days open = previous close")
    ax.set_title("Data check: are the opening prices real?")
    ax.set_ylim(-3, 103)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "open_quality.png", dpi=160)
    plt.close(fig)


def fig_stylised(res):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    out = res["sp500"]
    r = out["_returns"]["cc"].dropna()
    z = (r - r.mean()) / r.std()
    grid = np.linspace(-8, 8, 400)
    axes[0].hist(z, bins=120, density=True, alpha=0.6, label="S&P 500 daily returns")
    axes[0].plot(grid, sps.norm.pdf(grid), lw=1.5, label="Normal")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-5, 1)
    axes[0].set_xlabel("standardised return")
    axes[0].set_title(f"Fat tails: excess kurtosis {out['stylised_facts_1999_2018']['excess_kurtosis']:.1f}")
    axes[0].legend(frameon=False, fontsize=8)
    lags = np.arange(1, 21)
    w = 0.4
    axes[1].bar(lags - w / 2, st.acf(r), width=w, label="returns")
    axes[1].bar(lags + w / 2, st.acf(r**2), width=w, label="squared returns")
    band = 1.96 / np.sqrt(len(r))
    axes[1].axhspan(-band, band, color="grey", alpha=0.2, label="95% band if independent")
    axes[1].set_xlabel("lag (days)")
    axes[1].set_title("Volatility clusters; direction barely does")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "stylised_facts.png", dpi=160)
    plt.close(fig)


def fig_is_vs_oos(res):
    fig, axes = plt.subplots(1, len(res), figsize=(10, 4.2), sharey=True)
    for ax, (_name, out) in zip(np.atleast_1d(axes), res.items(), strict=True):
        a = out["_tables"]["in_sample"]["sharpe"]
        b = out["_tables"]["out_of_sample"]["sharpe"].reindex(a.index)
        ax.axhline(0, color="grey", lw=0.8)
        ax.axvline(0, color="grey", lw=0.8)
        lim = [min(a.min(), b.min()) - 0.1, max(a.max(), b.max()) + 0.1]
        ax.plot(lim, lim, ls="--", color="grey", lw=0.8, label="same Sharpe in both periods")
        ax.scatter(a, b, s=22)
        best = out["selection"]["best_in_sample_signal"]
        ax.scatter([a[best]], [b[best]], s=60, facecolors="none", edgecolors="red", label=f"best in sample: {best}")
        ax.set_xlabel("Sharpe 2000 to 2010 (in sample)")
        ax.set_title(f"{out['market']}  (rank corr {out['selection']['spearman_is_vs_oos_sharpe']:.2f})")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    np.atleast_1d(axes)[0].set_ylabel("Sharpe 2011 to 2018 (out of sample)")
    fig.tight_layout()
    fig.savefig(FIGURES / "is_vs_oos_sharpe.png", dpi=160)
    plt.close(fig)


def fig_pvalues(res):
    fig, axes = plt.subplots(1, len(res), figsize=(10, 3.6), sharey=True)
    for ax, (_name, out) in zip(np.atleast_1d(axes), res.items(), strict=True):
        t = out["_tables"]["in_sample"].sort_values("p")
        m = len(t)
        k = np.arange(1, m + 1)
        ax.scatter(k, t["p"], s=18, label="raw p-value (sorted)")
        ax.plot(k, 0.05 * k / m, color="C1", label="Benjamini-Hochberg line")
        ax.axhline(0.05 / m, color="C3", ls="--", label="Bonferroni threshold")
        ax.axhline(0.05, color="grey", ls=":", label="0.05 with no correction")
        ax.set_yscale("log")
        ax.set_xlabel("signals, most significant first")
        ax.set_title(f"{out['market']}, 2000 to 2010")
    np.atleast_1d(axes)[0].set_ylabel("p-value (Newey-West t-test)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "pvalues_in_sample.png", dpi=160)
    plt.close(fig)


def fig_costs(res):
    fig, axes = plt.subplots(1, len(res), figsize=(10, 3.6), sharey=True)
    cols = ["sharpe"] + [f"sharpe_net_{c}bps" for c in COSTS_BPS[1:]]
    for ax, (_name, out) in zip(np.atleast_1d(axes), res.items(), strict=True):
        t = out["_tables"]["in_sample"]
        top = t[t["turnover_per_day"] > 0].sort_values("sharpe", ascending=False).head(5)
        for sig, row in top.iterrows():
            ax.plot(COSTS_BPS, row[cols].values, marker="o", ms=3, label=f"{sig} ({row['turnover_per_day']:.2f}/day)")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel("cost per unit of turnover (bps)")
        ax.set_title(f"{out['market']}: top 5 in sample")
        ax.legend(frameon=False, fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("annualised Sharpe, 2000 to 2010")
    fig.tight_layout()
    fig.savefig(FIGURES / "cost_sensitivity.png", dpi=160)
    plt.close(fig)


def fig_overnight(res):
    out = res["nasdaq"]
    if not out["open_prices_valid"]:
        return
    r = out["_returns"].loc["1999-02":]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    series = [
        ("overnight", "overnight only (close to next open)"),
        ("intraday", "intraday only (open to close)"),
        ("cc", "buy and hold"),
    ]
    for col, label in series:
        growth = (1 + r[col].fillna(0)).cumprod()
        ax.plot(growth.index, growth, lw=1.1, label=label)
    ax.set_yscale("log")
    ax.set_ylabel("growth of $1 (log scale)")
    ax.set_title("NASDAQ: where the index return was earned, before trading costs")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "overnight_vs_intraday.png", dpi=160)
    plt.close(fig)


def fig_rolling(res):
    fig, axes = plt.subplots(len(res), 1, figsize=(9, 5.2), sharex=True)
    for ax, (_name, out) in zip(np.atleast_1d(axes), res.items(), strict=True):
        s = out["_best_strategy"]
        roll = s.gross.loc["2000":].rolling(252).apply(st.sharpe, raw=True)
        bench = out["_bench"].gross.loc["2000":].rolling(252).apply(st.sharpe, raw=True)
        ax.plot(roll.index, roll, lw=1, label=f"best in-sample signal: {s.name}")
        ax.plot(bench.index, bench, lw=1, alpha=0.7, label="buy and hold")
        ax.axvline(pd.Timestamp("2011-01-01"), color="k", ls="--", lw=0.8)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_ylabel("1 year Sharpe")
        ax.set_title(out["market"], fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Rolling Sharpe; the dashed line is where out of sample starts", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "rolling_sharpe.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="fewer bootstrap draws, for CI")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None, help="write results and figures here instead of the repo")
    args = ap.parse_args()
    reps = 200 if args.fast else 2000
    global RESULTS, FIGURES
    if args.out_dir is not None:
        RESULTS, FIGURES = args.out_dir / "results", args.out_dir / "figures"
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    res = {m: run_market(m, reps, args.seed) for m in MARKETS}
    for fig in (fig_open_quality, fig_stylised, fig_is_vs_oos, fig_pvalues, fig_costs, fig_overnight, fig_rolling):
        fig(res)

    summary = {m: {k: v for k, v in out.items() if not k.startswith("_")} for m, out in res.items()}
    summary["settings"] = {"periods": PERIODS, "costs_bps": COSTS_BPS, "bootstrap_reps": reps, "seed": args.seed}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    for m, out in summary.items():
        if m == "settings":
            continue
        print(f"\n== {out['market']} ({out['n_signals']} signals, open prices valid: {out['open_prices_valid']})")
        print(" counts:", out["counts"])
        print(" SPA:", out["spa"])
        print(" selection:", {k: v for k, v in out["selection"].items()})


if __name__ == "__main__":
    main()
