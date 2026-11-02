# Walkthrough: every decision, and how to explain it

This file explains the project in the order I built it, so I can defend every choice in an interview.

## 1. The question

Trading desks see many candidate signals. Most of them look good in a backtest for one of two
reasons: they really work, or the researcher tried enough ideas that one looked good by chance.
The project measures how often simple, well known signals survive when you account for that.

## 2. Data, and the check that came first

* Daily OHLC for the S&P 500 and NASDAQ Composite, 1999 to 2018, bundled with the `arch` package
  (originally from Yahoo Finance). That gives about 5,000 days per index, and anyone can rerun it offline.
* Before any test, I checked the opening prices. In the S&P 500 file, the open equals the previous
  close on 96% of days from 1999 to 2005. An index made of 500 stocks essentially never opens
  exactly at yesterday's close, so the vendor filled the gap. An "overnight return" computed from
  that would be zero on most days, a fake effect. The code flags any year where more than 5% of
  opens are copies, and drops the overnight and intraday signals for that index.

**Likely question: how would you have noticed if you had not checked?** The S&P 500 overnight
strategy would have shown almost no risk before 2006 and then a sudden change. That kind of break
in the data is a red flag to investigate before believing a result.

## 3. Timing and look-ahead

* Every position is decided at the close of day t, using prices up to day t, and earns the return
  from close t to close t+1.
* The day-of-week signal uses the next trading day's weekday. That is known in advance from the
  calendar, so it is not look-ahead.
* A unit test changes all prices after day t and checks that no position up to day t changes.

## 4. Signals

Twenty simple rules, chosen because they are classic, not because they worked:
momentum over 5, 10, 21, 63, 126 and 252 days; reversal over 1, 2 and 5 days; moving average
crossovers (5/20, 10/50, 20/100, 50/200); long on one weekday only; and holding only overnight or only
intraday. Having many signals is deliberate: it creates the multiple testing problem the project studies.

## 5. Periods

* 1999 is warm up, so every signal (including 252 day momentum and the 200 day average) has full history.
* 2000 to 2010 is "in sample", where you would pick a signal.
* 2011 to 2018 is "out of sample", looked at once.

The two periods are very different regimes: two crashes in the first, a long calm rally in the second.

## 6. The statistics, one by one

**Newey-West t-test.** The ordinary t-test assumes each day is independent with the same variance.
Daily returns break both assumptions: volatility clusters and there is some autocorrelation. Newey-West
standard errors correct for both. I use the standard lag rule floor(4 (n/100)^(2/9)), about 9 days here.

**Stationary bootstrap for Sharpe intervals.** Resampling single days would destroy volatility
clusters and make the interval too narrow. The stationary bootstrap resamples blocks of random length
(mean 10 days), so calm and turbulent stretches stay together.

**Multiple testing.**
* Bonferroni multiplies each p-value by the number of tests. It controls the chance of even one false
  positive, so it is strict.
* Benjamini-Hochberg controls the expected share of false positives among the signals you call
  significant. It is less strict and more common when screening many ideas.
* Both treat signals one at a time and ignore how correlated they are (5 and 10 day momentum overlap a lot).

**Hansen's SPA test.** This asks one question: is the best signal better than the benchmark once you
account for having searched over all of them? It bootstraps the distribution of the best studentized
performance under the null that none beats the benchmark, and it handles correlation between signals.
I report the "consistent" p-value. Signals that are clearly bad are not allowed to inflate the
null distribution.

**Why I wrote SPA myself.** The `arch` package's SPA gave identical p-values with studentization on
and off. Without studentization, the most volatile signals dominate the maximum, so a precise but
modest edge (like the overnight return) gets drowned out. I implemented the studentized statistic,
max over k of sqrt(n) * mean_k / omega_k with omega from the bootstrap, and wrote a test for exactly this case.

## 7. Costs

* Turnover is the size of each position change: short to long is 2 units, and the overnight-only
  strategy trades twice a day (buy at the close, sell at the open).
* Break-even cost = average gross daily return / average daily turnover, in basis points. It is the
  cost per unit traded at which the edge disappears.
* For context, liquid index ETFs and futures cost on the order of a basis point or less per trade
  in spread, before market impact. So an edge that breaks even at 1.75 bps is fragile, and one at
  0.5 bps is not tradable.

## 8. What the results say

1. **S&P 500, 1 day reversal.** Sharpe 1.07 in 2000 to 2010, t = 3.86, Bonferroni p = 0.002, and SPA
   p = 0.0015 against cash. From 2011 to 2018: Sharpe 0.08, p = 0.80. Short-term reversal in the index
   was strong in the volatile 2000s and faded in the calm 2010s. The tests correctly said the edge was
   real in that sample. They cannot say it will continue.
2. **False discoveries.** NASDAQ 2 day reversal: raw p = 0.006 in sample, but Benjamini-Hochberg p =
   0.115, and it lost money afterwards. S&P Tuesday: raw p = 0.014 out of sample, but BH p = 0.226.
3. **Overnight returns on NASDAQ.** Sharpe 0.89 from 2011 to 2018, significant after Bonferroni and SPA,
   and positive in both periods. This matches published work on overnight versus trading-hours returns.
   But two trades a day mean it breaks even at 1.75 bps per trade. Also, index opens can be stale, so
   I would confirm it with ETF or futures prices before believing the size.
4. **Stylised facts.** Excess kurtosis 8.3, 29 days beyond 4 standard deviations (0.3 expected if
   normal), and strong ARCH effects. These justify the choice of tests.

## 9. Questions I expect, and my answers

* **Why not just use the best in-sample signal?** Because here the best in-sample signal ranked 10th
  of 18 out of sample on the S&P 500, and 17th of 20 on NASDAQ.
* **Rank correlation between periods is 0.63 on the S&P 500. Does that mean signals persist?** Partly.
  Short-horizon momentum lost money in both periods, and slow trend rules made a little in both. So the
  broad shape persisted, but the specific winner did not.
* **What would you do next?** Use ETF or futures data with real spreads, rolling train and test windows,
  a market impact model, and intraday data to study execution: for example, how the time of day you
  trade changes the cost.
* **Isn't SPA overkill for 20 signals?** Bonferroni ignores that the signals are correlated, so it is
  too strict. SPA uses the correlation, so it is fairer for a family of related rules.
