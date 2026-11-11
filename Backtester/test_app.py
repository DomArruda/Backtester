"""Runnable self-checks: python test_app.py (no network needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app import sample_diversified, sample_sets
from engine import run_pipeline
from forecasting import (
    BASELINE_NAME,
    RULE_NAMES,
    SIMPLE_MODELS,
    aggregate_predictions,
    buy_hold_net_returns,
    fit_model,
    forecast_one,
    forecast_stats,
    performance_stats,
    positions,
    strategy_net_returns,
)
from portfolio import (
    STRATEGY_NAMES,
    average_pairwise_correlation,
    build_all,
)


def synthetic_prices(periods=420, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=periods)
    returns = pd.DataFrame(
        {
            "AAA": rng.normal(0.0004, 0.012, periods),
            "BBB": rng.normal(0.0002, 0.018, periods),
            "CCC": rng.normal(0.0003, 0.010, periods),
        },
        index=dates,
    )
    return 100.0 * (1.0 + returns).cumprod()


def check_portfolios():
    prices = synthetic_prices()
    returns = prices.pct_change().dropna()
    portfolios = build_all(returns, 0.0)
    assert list(portfolios) == STRATEGY_NAMES
    for name, weights in portfolios.items():
        assert abs(weights.sum() - 1.0) < 1e-9, f"{name} weights sum to {weights.sum()}"
        assert np.isfinite(weights).all() and (weights >= -1e-12).all(), name
    print(f"portfolios: all {len(STRATEGY_NAMES)} strategies build long-only weights summing to 1")


def check_baseline_and_options():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-01-01", periods=120)
    series = pd.Series(rng.normal(0.001, 0.01, 120), index=dates)
    pred = forecast_one(series, series.index[:80], series.index[80:], BASELINE_NAME)
    expected = series.shift(1).loc[series.index[80:]]
    pd.testing.assert_series_equal(pred, expected, check_names=False)
    print("baseline: guesses tomorrow's return equals yesterday's return")

    X = pd.DataFrame(rng.normal(size=(80, 5)))
    y = pd.Series(rng.normal(0.0, 0.01, 80))
    pipe = fit_model("XGBoost", X, y, {"n_estimators": 5})
    assert pipe.named_steps["model"].n_estimators == 5
    assert np.isfinite(pipe.predict(X.iloc[:3])).all()
    print("advanced options: XGBoost honored the custom tree count")


def check_strategy_alignment_and_costs():
    dates = pd.bdate_range("2024-01-01", periods=6)
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.03], index=dates)
    predicted = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01, -0.01], index=dates)
    previous = pd.Series([0.0, 0.01, -0.02, 0.03, -0.01, 0.02], index=dates)

    net, pos, turnover = strategy_net_returns(
        returns, predicted, "Forecast up / down", previous, cost_bps=0.0
    )
    assert pos.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert abs((1.0 + net).prod() - (1.01 * 1.0 * 1.03 * 1.0 * 1.02 * 1.0)) < 1e-12

    net_cost, _, turnover_cost = strategy_net_returns(
        returns, predicted, "Forecast up / down", previous, cost_bps=10.0
    )
    assert turnover_cost.tolist() == [1.0] * 6
    expected = (1.01 * 0.999) * (1.0 * 0.999) * (1.03 * 0.999) * (1.0 * 0.999) * (1.02 * 0.999) * (1.0 * 0.999)
    assert abs((1.0 + net_cost).prod() - expected) < 1e-12

    bh = buy_hold_net_returns(returns, 10.0)
    bh_expected = 0.999 * 1.01 * 0.98 * 1.03 * 0.99 * 1.02 * 0.97
    assert abs((1.0 + bh).prod() - bh_expected) < 1e-12
    print("strategies: positions earn the matching day's return; costs hit position changes")


def check_no_leakage():
    prices = synthetic_prices()
    kwargs = dict(
        train_share=0.7,
        portfolio_strategies=["Equal-Weight"],
        selection_metric="train",
        models=["Linear Regression"],
        rules=["Forecast up / down"],
        band=0.005,
        cost_bps=0.0,
        capital=10000.0,
        rf_annual=4.0,
    )
    base = run_pipeline(prices, **kwargs)
    base_pred = base["portfolio_predictions"]["Linear Regression"].copy()

    changed_last = prices.copy()
    changed_last.iloc[-1] *= 1.5
    last_pred = run_pipeline(changed_last, **kwargs)["portfolio_predictions"]["Linear Regression"]
    pd.testing.assert_series_equal(base_pred, last_pred)

    middle = len(prices) - 20
    changed_middle = prices.copy()
    changed_middle.iloc[middle] *= 1.5
    middle_pred = run_pipeline(changed_middle, **kwargs)["portfolio_predictions"]["Linear Regression"]
    changed_date = prices.index[middle]
    pd.testing.assert_series_equal(base_pred.loc[:changed_date], middle_pred.loc[:changed_date])
    assert not np.allclose(base_pred.loc[changed_date:], middle_pred.loc[changed_date:])
    print("no leakage: changing a final or mid-test price only affects predictions after it")


def check_pipeline():
    prices = synthetic_prices()
    result = run_pipeline(
        prices,
        train_share=0.7,
        portfolio_strategies=["Equal-Weight", "Minimum Variance", "Maximum Sharpe Ratio"],
        selection_metric="train",
        models=SIMPLE_MODELS,
        rules=RULE_NAMES,
        band=0.005,
        cost_bps=10.0,
        capital=10000.0,
        rf_annual=4.0,
    )
    assert result["best_portfolio"] in result["portfolios"]
    assert result["train_size"] + result["test_size"] == len(result["returns"])
    assert result["portfolio_forecast_table"].shape[0] == len(SIMPLE_MODELS)
    assert set(result["portfolio_forecast_table"]["Model"]) == set(SIMPLE_MODELS)
    assert list(result["portfolio_forecast_table"].columns) == [
        "Model", "MAE (%)", "Up-day accuracy (%)", "Down-day accuracy (%)"
    ]
    full_port = pd.concat(
        [result["train_portfolio_returns"], result["test_portfolio_returns"]]
    )
    pd.testing.assert_series_equal(
        result["portfolio_predictions"][BASELINE_NAME],
        full_port.shift(1).loc[result["test_portfolio_returns"].index],
        check_names=False,
    )
    combos = result["combo_table"]
    assert len(combos) == len(SIMPLE_MODELS) * len(RULE_NAMES)
    assert set(result["strategy_combos"]) == {
        "Equal-Weight", "Minimum Variance", "Maximum Sharpe Ratio"
    }
    for strategy, combo_table in result["strategy_combos"].items():
        assert len(combo_table) == len(SIMPLE_MODELS) * len(RULE_NAMES), strategy
    assert combos["Rank"].tolist() == list(range(1, len(combos) + 1))
    assert combos["Net return (%)"].is_monotonic_decreasing
    assert set(combos["Rule"]) == set(RULE_NAMES)
    assert set(result["combo_curves"]) == set(combos["Combination"])
    assert len(result["baseline_net"]) == result["test_size"]
    for combo, net in result["combo_curves"].items():
        assert len(net) == result["test_size"], combo
        assert np.isfinite(net).all(), combo
    print("pipeline: portfolios -> forecasts -> ranked model/rule backtest")


def check_diversified_sample():
    sp500 = pd.DataFrame({
        "Ticker": [f"S{i:02d}" for i in range(30)],
        "Sector": ["A"] * 6 + ["B"] * 6 + ["C"] * 6 + ["D"] * 6 + ["E"] * 6,
    })
    picks = sample_diversified(sp500, 10)
    counts = sp500[sp500["Ticker"].isin(picks)]["Sector"].value_counts()
    assert len(picks) == 10 and counts.max() == 2, (len(picks), counts.to_dict())
    picks7 = sample_diversified(sp500, 7)
    counts7 = sp500[sp500["Ticker"].isin(picks7)]["Sector"].value_counts()
    assert len(picks7) == 7 and counts7.max() == 2 and counts7.min() == 1
    print("diversify: picks spread across 5 sectors, roughly n/5 each")


def check_avg_correlation():
    dates = pd.bdate_range("2024-01-01", periods=100)
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=100), index=dates)
    returns = pd.DataFrame({"A": x, "B": x, "C": -x}, index=dates)
    equal = pd.Series({"A": 0.5, "B": 0.5, "C": 0.0})
    assert abs(average_pairwise_correlation(returns, equal) - 1.0) < 1e-12
    solo = average_pairwise_correlation(returns, pd.Series({"A": 1.0, "B": 0.0, "C": 0.0}))
    assert np.isnan(solo)
    # Position sizes matter: pairs weighted by w_i * w_j, not counted equally.
    tilted = pd.Series({"A": 0.5, "B": 0.25, "C": 0.25})
    weighted = average_pairwise_correlation(returns, tilted)
    assert abs(weighted - (-0.2)) < 1e-12, weighted
    assert average_pairwise_correlation(returns) < 0.6
    print("correlation: weighted by position size; ignored zero-weight names")


def check_sample_sets():
    sp500 = pd.DataFrame({
        "Ticker": [f"S{i:02d}" for i in range(60)],
        "Sector": ["A"] * 12 + ["B"] * 12 + ["C"] * 12 + ["D"] * 12 + ["E"] * 12,
    })
    sets = sample_sets(sp500, 5, "Random from the S&P 500", 3)
    assert len(sets) == 3 and len({frozenset(s) for s in sets}) == 3
    assert all(len(s) == 5 for s in sets)
    print("multi-set: n distinct random stock sets sampled")


def check_rules_and_positions():
    idx = pd.bdate_range("2024-01-01", periods=4)
    predicted = pd.Series([0.01, -0.01, 0.02, -0.02], index=idx)
    previous = pd.Series([-0.01, 0.01, -0.01, 0.01], index=idx)

    assert positions("Forecast up / down", predicted, previous).tolist() == [1.0, 0.0, 1.0, 0.0]
    assert positions("Forecast long / short", predicted, previous).tolist() == [1.0, -1.0, 1.0, -1.0]

    strong = positions("Strong forecast only", predicted, previous, band=0.005).ffill().fillna(0.0)
    assert strong.tolist() == [1.0, 0.0, 1.0, 0.0]
    mid = pd.Series([0.001, -0.001, 0.001, -0.001], index=idx)
    hold = positions("Strong forecast only", mid, previous, band=0.005).ffill().fillna(0.0)
    assert hold.tolist() == [0.0, 0.0, 0.0, 0.0]

    dips = positions("Buy dips when forecast up", predicted, previous).ffill().fillna(0.0)
    assert dips.tolist() == [1.0, 0.0, 1.0, 0.0]
    no_dip = positions(
        "Buy dips when forecast up",
        pd.Series([0.01, 0.01], index=idx[:2]),
        pd.Series([0.02, 0.02], index=idx[:2]),
    ).ffill().fillna(0.0)
    assert no_dip.tolist() == [0.0, 0.0]

    # The baseline guesses tomorrow = yesterday, so "buy dips" (needs yesterday
    # down AND a positive forecast) can never fire: the rule earns exactly 0%.
    net, pos, _ = strategy_net_returns(previous, previous, "Buy dips when forecast up", previous)
    assert pos.eq(0.0).all() and net.eq(0.0).all()
    print("rules: every rule produces the documented positions")


def check_performance_stats():
    dates = pd.bdate_range("2024-01-01", periods=4)
    net = pd.Series([0.01, -0.01, 0.01, -0.01], index=dates)
    stats = performance_stats(net, rf_daily=0.0)
    equity = (1.0 + net).cumprod()
    assert abs(stats["Net return (%)"] - (equity.iloc[-1] - 1.0) * 100) < 1e-12
    expected_dd = float((equity / equity.cummax() - 1.0).min()) * 100
    assert abs(stats["Max drawdown (%)"] - expected_dd) < 1e-12
    assert abs(stats["Sharpe"]) < 1e-9  # mean excess return is exactly zero
    assert all(pd.isna(v) for v in performance_stats(pd.Series(dtype=float)).values())
    print("stats: return, drawdown and Sharpe match hand-computed values")


def check_forecast_stats():
    stats = forecast_stats([0.01, -0.02, 0.03], [0.02, 0.01, 0.03])
    assert abs(stats["MAE (%)"] - (0.01 + 0.03 + 0.0) / 3 * 100) < 1e-12
    assert stats["Up-day accuracy (%)"] == 100.0
    assert stats["Down-day accuracy (%)"] == 0.0
    assert all(pd.isna(v) for v in forecast_stats([], []).values())
    print("forecast quality: MAE and up/down accuracy match hand-computed values")


def check_forecast_bounds():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2024-01-01", periods=300)
    train = rng.normal(0.0003, 0.01, 200)
    series = pd.Series(np.concatenate([train, np.full(100, 0.02)]), index=dates)
    pred = forecast_one(series, dates[:200], dates[200:], "Neural Network")
    bound = max(5.0 * float(np.std(train)), 0.01)
    assert np.isfinite(pred).all() and (pred.abs() <= bound + 1e-12).all()
    print("forecast bounds: no model predicts a daily return beyond 5x training volatility")


def check_engine_guards():
    base = dict(
        portfolio_strategies=["Equal-Weight"],
        selection_metric="train",
        rules=["Forecast up / down"],
        band=0.005,
        cost_bps=0.0,
        capital=1000.0,
        rf_annual=4.0,
    )
    cases = [
        (synthetic_prices(periods=40), {**base, "train_share": 0.7, "models": ["Linear Regression"]}),
        (synthetic_prices(periods=60), {**base, "train_share": 0.5, "models": ["Linear Regression"]}),
        (synthetic_prices(), {**base, "train_share": 0.7, "models": ["Does Not Exist"]}),
    ]
    for frame, kwargs in cases:
        try:
            run_pipeline(frame, **kwargs)
            raise AssertionError("run_pipeline accepted invalid inputs")
        except ValueError:
            pass
    print("guards: too little history, a short train window and unknown models are rejected")


def check_build_subset():
    returns = synthetic_prices().pct_change().dropna()
    subset = build_all(returns, 0.0, selected=["Equal-Weight"])
    assert list(subset) == ["Equal-Weight"]
    print("build_all: selected= builds only the requested strategies")


def check_aggregate_predictions():
    idx = pd.bdate_range("2024-01-01", periods=3)
    preds = {"M": pd.DataFrame(
        {"AAA": [0.01, 0.0, -0.01], "BBB": [0.02, 0.0, 0.0]}, index=idx
    )}
    weights = pd.Series({"AAA": 0.3, "BBB": 0.7})
    out = aggregate_predictions(preds, weights)["M"]
    assert np.allclose(out.to_numpy(), [0.017, 0.0, -0.003])
    print("aggregate: portfolio guess is the weighted sum of stock guesses")


if __name__ == "__main__":
    check_portfolios()
    check_baseline_and_options()
    check_rules_and_positions()
    check_performance_stats()
    check_forecast_stats()
    check_forecast_bounds()
    check_strategy_alignment_and_costs()
    check_no_leakage()
    check_engine_guards()
    check_build_subset()
    check_aggregate_predictions()
    check_pipeline()
    check_diversified_sample()
    check_avg_correlation()
    check_sample_sets()
    print("all checks passed")
