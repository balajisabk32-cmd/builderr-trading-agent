# Experiment log — what was tested, and what the data said

Every idea below was implemented and measured, then **removed** from `agent.py`.
The shipped agent contains only mechanisms that earned their place. This file is
the record so none of it gets re-derived.

## How a change earned its way in

Three datasets, each independent of the others. A mechanism shipped only if it
improved the two I could not see in advance, without giving back days already
scored:

| gate | dataset | what it guards against |
|---|---|---|
| **A — live** | the scored Round 2 window, Jul 7 2026 onward | the evaluator replays from entry, so past days are *deterministic*, not an estimate |
| **B — regimes** | 9 market eras, 26 years (2000–2026) | path luck — a single backtest is one draw |
| **C — samples** | the 3 `sample_regimes.json.gz` windows | overfitting to my own data pull |

Gate A is used as **accounting**, never as evidence. A parameter that scores well
on 17 already-visible days tells you nothing about the next 24, and selecting on
it is the definition of curve-fitting. Gates B and C are the evidence.

Fill mechanics in every harness replicate `live_runner.run_bot` exactly: bars
strictly before the decision day, `last_prices` = prior close, fills at the next
open ± slippage, buys clipped to cash on hand, marked to the close. Calibration
against the official scorer: **+0.54% vs the leaderboard's +0.52%, 11/11 trades,
17/17 days** — 0.02pt apart.

---

## Shipped

| change | live | samples | 26y mean CAGR | worst era |
|---|---|---|---|---|
| **Diversifier sleeve + group cap** | — | +0.96% (from −3.01%) | — | — |
| **Equity circuit breaker** | — | vol-spike −11.08% → −4.99% | +15.99% → +17.16% (2y) | — |
| **Regime filter 100 → 120 days** | +0.54% (unchanged) | +0.96% → **+2.24%** | +4.55% → **+4.91%** | −5.24% → **−4.32%** |

Baseline the rejections below are measured against: **live +0.54%, samples
+2.24%, 26y mean CAGR +4.91%, worst era −4.32%.**

---

## Rejected

### 1. Proportional exposure scaling — the near-miss
Scale gross linearly with QQQ's distance above its regime SMA instead of a binary
in/out.

| variant | live | samples | 26y mean | worst era |
|---|---|---|---|---|
| PROP_MIN 0.80 | +0.13% | +1.96% | **+6.25%** | −3.78% |
| PROP_MIN 0.90 | +0.13% | +2.08% | **+6.37%** | −3.88% |

**Worth +1.35pt/year over 26 years — the largest genuine edge found.** Rejected on
timing alone: QQQ sits ~1.5% above its 120-day average, so it cuts exposure *now*,
costing 0.41pt of already-scored days to buy ~0.14pt over a 3-week remainder.
**This ships in a longer round.** Revisit first for Round 3.

### 2. Sector-participation breadth gate
De-risk on the fraction of 9 sector ETFs holding their own 50-day SMA.

| | live | samples | 26y mean | worst era |
|---|---|---|---|---|
| breadth gate | +0.54% | +2.84% | **+3.12%** | **−2.94%** |
| breadth + proportional | +0.13% | +3.11% | +4.37% | −3.78% |

Falsified against its own stated criterion — it *did* improve worst-era drawdown
(−4.32% → −2.94%) but cost **1.79pt/year of CAGR**. It de-risks on sector
dispersion, which resolves upward more often than not. Combining it with #1
cancels #1's edge.

### 3. Rank hysteresis — inverted its own goal
Enter at rank ≤3, hold until rank falls past 5, to cut churn.

| | live | samples | 26y mean | trades |
|---|---|---|---|---|
| rank hysteresis | −0.66% | −1.90% | +3.27% | **11 → 25** |

Intended to *reduce* turnover; it more than doubled it. Giving held names slot
priority scrambles the momentum ordering that the group-cap passes consume, so
selection oscillates between two states. **Turnover control and rank-priority
interact badly in this architecture** — worth knowing before trying again.

### 4. Multi-window blended momentum
`0.20·Ret₂₀ + 0.50·Ret₆₀ + 0.30·Ret₁₂₀`.

live −0.21% · samples +1.78% · 26y +4.10% · worst era −5.61%. Worse on every
measure. Related: **vol-normalized ranking** (`momentum ÷ vol`, zero new
parameters) returned +12.55% vs +15.99% on 2y data with Sharpe 0.73 vs 0.79.

### 5. Momentum lookback length
| lookback | live | samples | 26y mean | regimes won |
|---|---|---|---|---|
| 40d | −0.75% | +3.40% | **+5.97%** | **6 / 9** |
| 90d (shipped) | **+0.54%** | +0.96% | +4.56% | 3 / 9 |

**40-day is the better strategy** — 6 of 9 eras, +1.41pt/year. Rejected because
its edge is *annual* and the remaining window is three weeks: +0.13pt expected
against −1.29pt certain. Its worst era is chop 2015–16 (−0.6%, 20.2% drawdown vs
90-day's +8.2%, 10.0%) — precisely the tape Round 2 is in, which independently
predicts its live-window loss. **Better bot for the top-3 re-run; wrong bot for
closing a gap in three weeks.**

### 6. Regime filter — the rejected ends
| length | live | samples | 26y mean | worst era |
|---|---|---|---|---|
| 20d | −1.36% | — | **+2.63%** | −6.21% |
| 80d | **+1.48%** | **−0.19%** | +4.49% | −1.38% |
| 150d | +0.54% | +0.99% | +4.91% | −4.20% |

80-day posts the best live number of anything tested and is a **luck artifact**:
neighbouring values swing 1.2pt (75d +1.30%, 85d +0.29%) and it fails gate C.
110/120/130/140 form a genuine plateau; 150 is a cliff. 120 chosen as the highest
value whose entire ±20% band (96–144) stays at-or-better than 100.

### 7. Universe breadth — the most dangerous result
| universe | live | 2y | samples |
|---|---|---|---|
| 11 tickers (shipped) | **+0.54%** | +17.78% (Sh 0.90) | **+0.96%** |
| ETFs only (31) | −3.53% | +22.13% (Sh 1.09) | −11.52% |
| all 81 | **−5.15%** | **+37.45% (Sh 1.55)** | −4.74% |

The 81-ticker version more than doubles the 2-year return at Sharpe 1.55. **It
would have dropped the live entry from rank 12 to below the benchmark.** Removing
single stocks didn't help — ETF-only was the worst on samples. Cause: 90-day
momentum has ~zero cross-sectional predictive power at a 3-week horizon (rank
correlation negative in 15/25 periods, mean **−0.07**), so a wider list adds
draws from a signal that doesn't predict, plus turnover (11 → 38 trades).

### 8. Deploying idle capital
~16% of the account sits in cash. Every way of spending it lost money.

| | peak gross | live | 2y | samples |
|---|---|---|---|---|
| shipped (3 positions) | 0.81× | **+0.54%** | **+17.78%** | **+0.96%** |
| 4 positions | 0.99× | −0.27% | +13.25% | −0.21% |
| 5 positions | 1.00× | +0.29% | +14.43% | −0.25% |

`MAX_GROSS` 0.90 → 1.00 moved peak gross only 0.81× → 0.84×; it never binds. The
real constraint is `MAX_WEIGHT 0.28 × 3`. The idle cash isn't drag — **it's the
absence of a 4th position worth owning.** Exceeding 1.0× is impossible anyway:
`live_runner.py:361` clips buys to cash, so only leveraged ETFs could do it, and
those are tech/broad-market — the opposite of the current rotation book.

### 9. Volatility target
| target | live | samples | 26y mean | worst era |
|---|---|---|---|---|
| 0.12 | +0.84% | +1.97% | +3.49% | −3.00% |
| 0.20 | +0.65% | +2.11% | +4.61% | −3.90% |
| **0.25 (shipped)** | +0.54% | **+2.24%** | **+4.91%** | −4.32% |
| 0.35 | +0.54% | +1.09%* | +4.81% | −7.54% |

Lower targets improve the live window *monotonically* — less exposure wins in a
falling tape. That is a **directional forecast, not an edge**: it costs up to
1.42pt/year across 26 years. Higher targets degrade the crises (dot-com −1.9% →
−3.3%, GFC −5.2% → −7.5%) because a higher target means less de-risking exactly
when volatility spikes. \*0.35 looked good on 2y data (+20.91% vs +17.78%) only
because that window contained no crisis — the clearest single-path trap found.

### 10. Defensive top-up in risk-on
Route idle cash to XLU/XLP when they pass a trend gate.

| cap | live Δ | selloff Δ | vol-spike Δ |
|---|---|---|---|
| ≤10% | +0.21 | −0.34 | **−1.52** |
| ≤20% | +0.51 | −0.70 | **−2.99** |
| ≤30% | +0.64 | −0.78 | **−4.12** |

Monotonically worse the more it's used. The sleeve is admitted on *trailing*
momentum — still positive for utilities entering Feb 2020 — so the book levers
0.77× → 0.93× immediately before a crash. **Idle cash is not dead weight; not
falling is what it's for.**

### 11. Neutral-band defensive carry
Hold trending XLP/XLU/XLV when QQQ is below its 50-day but above its regime SMA.

live −2.29% · samples +0.73% · 26y +3.46% · trades 32. Independently reproduces
#10: holding defensives instead of cash costs return in almost every era.

### 12. ATR trailing stop (chandelier exit)
| multiplier | live | samples | 26y mean | worst era |
|---|---|---|---|---|
| ×2.0 | −3.51% | +1.66% | +3.71% | −4.81% |
| ×2.5 | +0.54% | **+2.63%** | +4.20% | −4.92% |
| ×3.5 | +0.54% | +1.21% | +4.45% | −4.32% |

×2.5 improves samples but costs 0.71pt/year, and the response is non-monotonic in
its own multiplier — the 80-day spike signature again. Mechanically redundant: a
per-name stop sells into exactly the shakeouts the portfolio-level circuit breaker
is built to ride through. Keeping one.

### 13. Relative-momentum floor
Require a candidate's momentum to reach a fraction of the day's best, so a flat
gold print can't displace a semi running +40%.

Sum-of-returns: **−2.8% at 0.0, −4.3% at 0.15, −7.6% at 0.25, −0.5% at 0.40,
−1.6% at 0.60.** Non-monotonic with a cliff between 0.25 and 0.40, and the "good"
0.40 wins by cutting trades to 7 and sitting in cash. Textbook overfit.

### 14. Regime re-entry hysteresis
Require QQQ to reclaim its SMA by a buffer before going risk-on again. At 0.005 /
0.01 / 0.02 results were **worse in all three sample windows** — delaying
re-entry costs more than the whipsaw it avoids.

### 15. The five published starter strategies
Every one loses to the shipped agent on the live window, and four would rank
below the benchmark.

| strategy | live | 2y | 2y Sharpe | samples |
|---|---|---|---|---|
| **shipped** | **+0.54%** | +17.78% | 0.90 | **+0.96%** |
| Sector rotation | −2.95% | +21.11% | 0.86 | −25.38% |
| Play defense | −5.65% | +16.69% | 0.78 | −6.97% |
| **Ride the AI boom** | **−6.29%** | **+129.12%** | **1.72** | −1.29% |
| QQQ safety switch | −6.38% | +17.46% | 0.80 | −15.05% |

"Ride the AI boom" is the best backtest in this entire file — **+129% at Sharpe
1.72** — and currently −6.29%. It is also, essentially, this agent's original
specification. The AI/chip complex ran so hard for two years that any momentum
rule over those names looks superb historically. What moved the shipped agent off
it was the diversifier sleeve, which the ablation had made look like a mediocre
trade costing 1.9pt of calm-market return.

### 16. Ideas that cannot be built under the rules
Proposed repeatedly by strategy generators; blocked by the ruleset, not by data.

| idea | blocker |
|---|---|
| Market-neutral long/short | `AGENT_BRIEF.md:37` — **long only, no short-selling (v0)** |
| Factor rotation (MTUM/QUAL/USMV) | none of the three are in `universe.json` |
| Cash-equivalent sleeve (BIL/SHY) | absent from `universe.json`; holding real cash is the only option |
| Commodities sleeve (DBC) | absent from `universe.json` |
| Intraday/trailing stops, 1–3 day holds | `decide()` runs **once per day**, fills at the next open |
| 1.3–1.4× gross via 4–5 positions | no margin — `live_runner.py:361` clips buys to available cash |
| Tuning to the hidden admission windows | knowing them makes an entry **exhibition-only, no prize** |

---

## Two things that generalise

**A backtest that improves on one dataset and degrades on two independent ones is
noise, no matter how large the improvement.** The 81-ticker universe (+37%,
Sharpe 1.55) and "Ride the AI boom" (+129%, Sharpe 1.72) are the two prettiest
numbers here, and both are currently losing money.

**Non-monotonic response to a parameter is disqualifying on its own.** The 80-day
regime filter, the 0.40 momentum floor, and the ×2.5 ATR stop each posted the best
score in their sweep while their immediate neighbours were poor. Every one was a
single lucky value, and a smooth plateau — 110/120/130/140 — is what a real effect
looks like instead.
