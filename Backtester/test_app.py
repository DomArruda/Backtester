"""Runnable self-checks: python test_app.py (no network needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app import sample_diversified
from engine import run_pipeline
from forecasting import (
    BASELINE_NAME,
    RULE_NAMES,
    SIMPLE_MODELS,
    buy_hold_net_returns,
    fit_model,
    forecast_one,
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


if __name__ == "__main__":
    check_portfolios()
    check_baseline_and_options()
    check_strategy_alignment_and_costs()
    check_no_leakage()
    check_pipeline()
    check_diversified_sample()
    check_avg_correlation()
    print("all checks passed")
