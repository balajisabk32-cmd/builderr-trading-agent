"""Strategy-level checks for agent.py.

No network, no private engine, no third-party packages. These are not the
official builderr evals; they catch contract, cap, sizing, and regime bugs
before submission.

Run:
    python strategy_selftest.py
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import agent


UNIVERSE = (
    "NVDA", "AMD", "AVGO", "SMH", "XLK", "QQQ",   # AI/semi core
    "GLD", "TLT", "XLE", "XLV", "XLRE",            # diversifier sleeve
    "XLU", "XLP",                                  # risk-off / defensive sleeve
    "SPY",                                         # bystander (never targeted)
)

# Long enough to feed the 100-day regime SMA plus the 90-day momentum lookback.
HISTORY = 260


# --------------------------------------------------------------------------
# Fixtures -- fully deterministic, no RNG, so failures are always reproducible.
# --------------------------------------------------------------------------
def rets(n: int, drift: float, noise: float = 0.0) -> list[float]:
    """n daily returns of `drift`, with a deterministic +/-noise zig-zag.

    The alternating noise gives the series a realized volatility of roughly
    `noise` per day without any randomness -- so the vol-sizing paths are
    exercised with numbers that are stable across runs.
    """
    return [drift + (noise if i % 2 == 0 else -noise) for i in range(n)]


def bars(returns: list[float], start: float = 100.0) -> list[dict]:
    out, px, d = [], start, date(2024, 1, 1)
    for r in returns:
        px *= 1.0 + r
        out.append({
            "ts": d.isoformat(),
            "open": px, "high": px * 1.01, "low": px * 0.99, "close": px,
            "volume": 1_000_000,
        })
        d += timedelta(days=1)
    return out


def market(kind: str) -> dict[str, list[dict]]:
    # Bystanders drift DOWN so they fail the trend gate and cannot wander into
    # the book; each fixture then opts specific names back in.
    flat = bars(rets(HISTORY, -0.0004))
    data = {t: list(flat) for t in UNIVERSE}

    if kind == "semis_only":
        # Every core name trending hard, every diversifier falling. The group
        # cap must NOT starve the book here -- backfill has to fill all 3 slots.
        data["QQQ"] = bars(rets(HISTORY, 0.0015, 0.004))
        data["NVDA"] = bars(rets(HISTORY, 0.0040, 0.025))
        data["SMH"] = bars(rets(HISTORY, 0.0030, 0.004))
        data["XLK"] = bars(rets(HISTORY, 0.0025, 0.003))
        return data

    if kind == "broad":
        # Core AND diversifiers trending -> the book must span themes.
        data["QQQ"] = bars(rets(HISTORY, 0.0015, 0.004))
        data["NVDA"] = bars(rets(HISTORY, 0.0040, 0.025))
        data["SMH"] = bars(rets(HISTORY, 0.0035, 0.004))
        data["XLK"] = bars(rets(HISTORY, 0.0030, 0.003))
        data["GLD"] = bars(rets(HISTORY, 0.0020, 0.002))
        data["TLT"] = bars(rets(HISTORY, 0.0018, 0.003))
        return data

    if kind == "diversifier_leads":
        # Gold is the strongest thing on the board; it must win a slot outright.
        data["QQQ"] = bars(rets(HISTORY, 0.0012, 0.004))
        data["GLD"] = bars(rets(HISTORY, 0.0050, 0.003))
        data["NVDA"] = bars(rets(HISTORY, 0.0020, 0.025))
        data["SMH"] = bars(rets(HISTORY, 0.0015, 0.004))
        return data

    if kind == "risk_on":
        # QQQ in a clean uptrend -> risk-on regime.
        data["QQQ"] = bars(rets(HISTORY, 0.0015, 0.004))
        # Leaders, ranked by 90d momentum. NVDA is strongest but by far the
        # jumpiest -- inverse vol must underweight it relative to the ETFs.
        data["NVDA"] = bars(rets(HISTORY, 0.0040, 0.025))
        data["SMH"] = bars(rets(HISTORY, 0.0030, 0.004))
        data["XLK"] = bars(rets(HISTORY, 0.0025, 0.003))
        data["AMD"] = bars(rets(HISTORY, 0.0003, 0.004))   # too weak to make top 3
        # AVGO: strong 90-day number, but it has broken its 50-day trend and
        # must therefore be excluded regardless of how good the momentum looks.
        data["AVGO"] = bars(rets(HISTORY - 20, 0.005, 0.003) + rets(20, -0.010, 0.003))
        return data

    if kind == "risk_off":
        # QQQ rolls over hard enough to close below its 100-day average.
        data["QQQ"] = bars(rets(160, 0.0020, 0.004) + rets(HISTORY - 160, -0.0060, 0.005))
        data["NVDA"] = bars(rets(160, 0.0040, 0.020) + rets(HISTORY - 160, -0.0080, 0.020))
        data["XLU"] = bars(rets(HISTORY, 0.0012, 0.002))   # defensives still trending
        data["XLP"] = bars(rets(HISTORY, 0.0010, 0.002))
        return data

    if kind == "risk_off_no_hedge":
        # Everything falls, including the defensives -> the correct book is cash.
        data["QQQ"] = bars(rets(160, 0.0020, 0.004) + rets(HISTORY - 160, -0.0060, 0.005))
        data["XLU"] = bars(rets(HISTORY, -0.0010, 0.002))
        data["XLP"] = bars(rets(HISTORY, -0.0012, 0.002))
        return data

    if kind == "high_vol":
        # Risk-on trend, but every leader is extremely loud -> vol targeting
        # must shrink the invested fraction.
        data["QQQ"] = bars(rets(HISTORY, 0.0015, 0.030))
        data["NVDA"] = bars(rets(HISTORY, 0.0040, 0.045))
        data["SMH"] = bars(rets(HISTORY, 0.0035, 0.040))
        data["XLK"] = bars(rets(HISTORY, 0.0030, 0.038))
        return data

    if kind == "short":
        return {t: bars(rets(40, 0.001)) for t in UNIVERSE}

    raise ValueError(kind)


def portfolio(cash: float = 100_000.0, positions: dict[str, float] | None = None,
              prices: dict[str, float] | None = None) -> dict:
    return {
        "cash": cash,
        "positions": [{"ticker": t, "quantity": q, "avg_cost": 100.0}
                      for t, q in (positions or {}).items()],
        "last_prices": dict(prices or {}),
    }


def last_prices(m: dict[str, list[dict]]) -> dict[str, float]:
    return {t: b[-1]["close"] for t, b in m.items()}


def reset_agent_state() -> None:
    agent._last_rebalance_bar_date = None
    agent._last_targets = {}
    agent._last_regime_risk_on = None
    agent._equity_history.clear()


def beta_gross(weights: dict[str, float]) -> float:
    return sum(w * agent.BETA_MULTIPLE.get(t, 1.0) for t, w in weights.items())


# --------------------------------------------------------------------------
# Contract & edge cases
# --------------------------------------------------------------------------
def test_empty_data_returns_no_orders() -> None:
    reset_agent_state()
    assert agent.decide({}, portfolio(), 100_000) == []


def test_insufficient_history_holds_rather_than_trading() -> None:
    """Too little history to read the regime -> hold, do NOT liquidate."""
    reset_agent_state()
    m = market("short")
    assert agent.target_weights(m) == {}
    assert agent.decide(m, portfolio(positions={"NVDA": 100}), 100_000) == []


def test_malformed_input_never_raises() -> None:
    """The engine gives us whatever the feed produced; we must never crash."""
    reset_agent_state()
    m = market("risk_on")
    junk_states = [
        {"cash": None, "positions": None, "last_prices": None},
        {"positions": [{"ticker": "NVDA"}, {}, None, "garbage"]},
        {"cash": float("nan"), "positions": [{"ticker": "NVDA", "quantity": "12"}]},
        {"positions": [{"ticker": "NVDA", "quantity": -5, "avg_cost": "x"}]},
        {},
    ]
    for ps in junk_states:
        assert isinstance(agent.decide(m, ps, 0.0), list)
    # Bar-level corruption too.
    broken = {"QQQ": [{"ts": "2024-01-01"}], "NVDA": None, "SMH": [], "XLK": "nonsense"}
    assert agent.decide(broken, portfolio(), 100_000) == []
    assert agent.closes([{"close": 0.0}]) == []
    assert agent.closes([{"close": "abc"}]) == []


def test_is_deterministic() -> None:
    """Same code + same data => same orders. The fairness suite gates on this."""
    m = market("risk_on")
    runs = []
    for _ in range(3):
        reset_agent_state()
        runs.append(agent.decide(m, portfolio(), 100_000.0))
    assert runs[0] == runs[1] == runs[2], runs
    assert runs[0], "expected the risk-on book to actually trade"


# --------------------------------------------------------------------------
# 1. Regime control
# --------------------------------------------------------------------------
def test_regime_detects_both_states() -> None:
    assert agent.regime_is_risk_on(market("risk_on")) is True
    assert agent.regime_is_risk_on(market("risk_off")) is False
    assert agent.regime_is_risk_on(market("short")) is None


def test_risk_off_holds_only_defensives() -> None:
    weights = agent.target_weights(market("risk_off"))
    assert weights, "defensives are trending, so the risk-off book should be invested"
    assert set(weights).issubset(set(agent.RISK_OFF_UNIVERSE)), weights
    assert not (set(weights) & set(agent.RISK_ON_UNIVERSE)), weights
    # Equal split, per the spec.
    assert abs(weights["XLU"] - weights["XLP"]) < 1e-6, weights


def test_risk_off_goes_to_cash_when_defensives_are_falling() -> None:
    """A defensive asset in its own downtrend is not a hedge."""
    assert agent.target_weights(market("risk_off_no_hedge")) == {}


def test_regime_flip_liquidates_the_risk_on_book_immediately() -> None:
    """The safety switch must not wait for the weekly rebalance."""
    reset_agent_state()
    on, off = market("risk_on"), market("risk_off")

    agent.decide(on, portfolio(), 100_000.0)                 # establish risk-on
    held = {"NVDA": 100.0, "SMH": 200.0, "XLK": 150.0}
    ps = portfolio(cash=1_000.0, positions=held, prices=last_prices(off))
    orders = agent.decide(off, ps, 1_000.0)

    sold = {o["ticker"] for o in orders if o["side"] == "sell"}
    assert held.keys() <= sold, (held.keys(), sold)
    assert not [o for o in orders if o["side"] == "buy" and o["ticker"] in agent.RISK_ON_UNIVERSE]


# --------------------------------------------------------------------------
# 2. Leader selection
# --------------------------------------------------------------------------
def test_selects_top_three_by_momentum() -> None:
    leaders = agent.select_leaders(market("semis_only"))
    assert len(leaders) == agent.TOP_N, leaders
    assert leaders == ["NVDA", "SMH", "XLK"], leaders   # strict momentum order


def test_group_cap_forces_a_diversified_book() -> None:
    """With diversifiers available, the book must not be one theme in triplicate."""
    leaders = agent.select_leaders(market("broad"))
    assert len(leaders) == agent.TOP_N, leaders
    groups = [agent.TICKER_GROUP.get(t, t) for t in leaders]
    assert len(set(groups)) == len(groups), (leaders, groups)
    assert groups.count("ai_semis") == 1, (leaders, groups)
    assert leaders[0] == "NVDA", leaders                # strongest still leads


def test_backfill_prevents_a_starved_book() -> None:
    """If only one theme is trending, fill all 3 slots from it rather than starve.

    A strict one-per-group cap would strand the book at a single position here,
    which the sample windows cannot reveal but a semis melt-up would.
    """
    leaders = agent.select_leaders(market("semis_only"))
    assert len(leaders) == agent.TOP_N, leaders
    assert all(agent.TICKER_GROUP[t] == "ai_semis" for t in leaders), leaders


def test_diversifier_can_outrank_the_core() -> None:
    """The sleeve competes on the same rules; it is not a fixed allocation."""
    leaders = agent.select_leaders(market("diversifier_leads"))
    assert leaders[0] == "GLD", leaders


def test_broken_trend_is_excluded_even_with_strong_momentum() -> None:
    """AVGO has a strong 90-day return but is below its 50-day SMA -> out."""
    m = market("risk_on")
    series = agent.closes(m["AVGO"])
    assert agent.momentum(series, agent.MOMENTUM_DAYS) > 0.0, "fixture should be strong"
    assert series[-1] < agent.sma(series, agent.TREND_SMA_DAYS), "fixture should be broken"
    assert "AVGO" not in agent.select_leaders(m)


def test_weak_names_are_not_selected() -> None:
    assert "AMD" not in agent.select_leaders(market("risk_on"))


# --------------------------------------------------------------------------
# 3. Position sizing
# --------------------------------------------------------------------------
def test_inverse_vol_tilt_survives_the_caps() -> None:
    """REGRESSION: capping must not flatten the book back to equal weight.

    Redistributing clipped weight onto the peers silently converts inverse-vol
    sizing into equal-dollar sizing. NVDA is the jumpiest leader by a wide
    margin here, so it must end up materially SMALLER than the calm ETFs.
    """
    weights = agent.target_weights(market("semis_only"))
    assert set(weights) == {"NVDA", "SMH", "XLK"}, weights
    assert weights["NVDA"] < weights["SMH"] * 0.75, weights
    assert weights["NVDA"] < weights["XLK"] * 0.75, weights
    assert len(set(round(w, 4) for w in weights.values())) > 1, weights


def test_vol_targeting_shrinks_the_book_when_vol_spikes() -> None:
    calm = sum(agent.target_weights(market("risk_on")).values())
    loud = sum(agent.target_weights(market("high_vol")).values())
    assert loud < calm, (loud, calm)
    assert loud >= agent.MIN_GROSS * 0.9, loud


def test_defensive_trend_gate_excludes_a_falling_hedge() -> None:
    """Positive 3-month momentum is too slow on its own to spot a rolling hedge."""
    m = market("risk_off")
    # XLU rose for months, then broke down: still positive over 63 days, but
    # now below its 50-day SMA. That is not a hedge any more.
    # +0.4%/day for months, then 20 sessions of -0.5%: still +7% over 63 days,
    # but ~4.5% below its 50-day SMA.
    m["XLU"] = bars(rets(HISTORY - 20, 0.0040, 0.002) + rets(20, -0.0050, 0.002))
    series = agent.closes(m["XLU"])
    assert agent.momentum(series, agent.DEFENSIVE_MOM_DAYS) > 0.0, "fixture must stay positive"
    assert series[-1] < agent.sma(series, agent.TREND_SMA_DAYS), "fixture must be broken"
    assert "XLU" not in agent.defensive_weights(m), agent.defensive_weights(m)


def test_risk_on_book_never_holds_the_defensive_sleeve() -> None:
    """Idle cash stays cash in risk-on. Routing it to XLU/XLP was measured to
    cost 1.5-4.1 points in the vol-spike window (see experiments.md #10)."""
    m = market("broad")
    m["XLU"] = bars(rets(HISTORY, 0.0012, 0.002))   # trending, so it WOULD qualify
    m["XLP"] = bars(rets(HISTORY, 0.0010, 0.002))
    weights = agent.target_weights(m)
    assert weights, "risk-on book should not be empty"
    assert not (set(weights) & set(agent.RISK_OFF_UNIVERSE)), weights
    assert sum(weights.values()) <= agent.MAX_GROSS + 1e-9, weights


def test_caps_hold_in_every_regime() -> None:
    for kind in ("risk_on", "semis_only", "broad", "diversifier_leads",
                 "risk_off", "risk_off_no_hedge", "high_vol"):
        weights = agent.target_weights(market(kind))
        assert all(w <= agent.MAX_WEIGHT + 1e-9 for w in weights.values()), (kind, weights)
        assert all(w < 0.30 for w in weights.values()), (kind, weights)
        assert beta_gross(weights) <= agent.MAX_BETA_GROSS + 1e-9, (kind, weights)
        assert sum(weights.values()) <= agent.MAX_GROSS + 1e-9, (kind, weights)


def test_zero_vol_series_does_not_divide_by_zero() -> None:
    """A perfectly flat series has zero stdev; the floor must absorb it."""
    flat = bars([0.0] * 200)
    m = {t: list(flat) for t in UNIVERSE}
    m["QQQ"] = bars(rets(260, 0.001))
    w = agent.inverse_vol_weights(m, ["NVDA", "SMH", "XLK"], 0.9)
    assert w and all(v > 0 for v in w.values()), w
    assert abs(sum(w.values()) - 0.9) < 1e-9, w


# --------------------------------------------------------------------------
# 4. Execution
# --------------------------------------------------------------------------
def test_orders_are_well_formed_bounded_and_fast() -> None:
    reset_agent_state()
    m = market("risk_on")
    start = time.perf_counter()
    orders = agent.decide(m, portfolio(prices=last_prices(m)), 100_000.0)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, elapsed                       # engine budget is 5s
    assert 0 < len(orders) <= 50, orders
    for o in orders:
        assert set(o) == {"ticker", "side", "quantity"}, o
        assert o["side"] in ("buy", "sell"), o
        assert float(o["quantity"]) > 0, o
        assert o["ticker"] in UNIVERSE, o


def test_sells_are_emitted_before_buys() -> None:
    """The engine walks the list in order and clips buys to cash on hand."""
    reset_agent_state()
    m = market("risk_on")
    ps = portfolio(cash=0.0, positions={"XLV": 300.0, "SPY": 100.0},
                   prices=last_prices(m))
    orders = agent.decide(m, ps, 0.0)
    sides = [o["side"] for o in orders]
    assert "sell" in sides and "buy" in sides, orders
    assert sides.index("buy") > max(i for i, s in enumerate(sides) if s == "sell"), sides


def test_buys_respect_available_cash() -> None:
    """Never ask for more than cash plus (haircut) expected sale proceeds."""
    reset_agent_state()
    m = market("risk_on")
    prices = last_prices(m)
    cash = 25_000.0
    orders = agent.decide(m, portfolio(cash=cash, prices=prices), cash)
    spend = sum(float(o["quantity"]) * prices[o["ticker"]]
                for o in orders if o["side"] == "buy")
    assert spend <= cash * 1.001, (spend, cash)


def test_rebalance_is_throttled_between_scheduled_dates() -> None:
    """A second call on the same bar date must not churn the book again."""
    reset_agent_state()
    m = market("risk_on")
    prices = last_prices(m)
    first = agent.decide(m, portfolio(prices=prices), 100_000.0)
    assert first

    filled = {}
    cash = 100_000.0
    for o in first:
        if o["side"] == "buy":
            filled[o["ticker"]] = filled.get(o["ticker"], 0.0) + float(o["quantity"])
            cash -= float(o["quantity"]) * prices[o["ticker"]]
    assert agent.decide(m, portfolio(cash=cash, positions=filled, prices=prices), cash) == []


def test_stale_holdings_are_liquidated() -> None:
    reset_agent_state()
    m = market("risk_on")
    ps = portfolio(cash=50_000.0, positions={"XLV": 200.0}, prices=last_prices(m))
    orders = agent.decide(m, ps, 50_000.0)
    assert any(o["ticker"] == "XLV" and o["side"] == "sell" for o in orders), orders


def test_tiny_stale_position_is_not_sold() -> None:
    """Dust is not worth a trade slot or the slippage."""
    orders = agent.orders_to_rebalance(
        targets={"SPY": 0.20},
        positions={"XYZ": {"quantity": 0.5, "avg_cost": 100.0}},
        total_equity=100_000.0,
        prices={"XYZ": 100.0, "SPY": 500.0},
        cash_available=0.0,
    )
    assert not any(o["ticker"] == "XYZ" for o in orders), orders


def test_unpriceable_position_is_left_alone() -> None:
    """No price => no size => no order, rather than a garbage quantity."""
    orders = agent.orders_to_rebalance(
        targets={},
        positions={"GHOST": {"quantity": 100.0, "avg_cost": 50.0}},
        total_equity=100_000.0,
        prices={},
        cash_available=0.0,
    )
    assert orders == []


def test_order_count_stays_under_the_daily_cap() -> None:
    positions = {f"T{i}": {"quantity": 100.0, "avg_cost": 10.0} for i in range(80)}
    prices = {f"T{i}": 50.0 for i in range(80)}
    orders = agent.orders_to_rebalance({}, positions, 1_000_000.0, prices, 0.0)
    assert len(orders) <= agent.MAX_ORDERS_PER_DAY <= 50, len(orders)


def run() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} strategy checks passed.")


if __name__ == "__main__":
    run()
