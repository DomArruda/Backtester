"""Pure pipeline: portfolio construction -> ML forecasts -> strategy backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting import (
    TRADING_DAYS,
    aggregate_predictions,
    evaluate_rules,
    forecast_portfolio,
    forecast_stats,
    performance_stats,
)
from portfolio import average_pairwise_correlation, build_all


def run_pipeline(
    prices: pd.DataFrame,
    train_share: float,
    portfolio_strategies: list[str],
    selection_metric: str,
    models: list[str],
    rules: list[str],
    band: float,
    cost_bps: float,
    capital: float,
    rf_annual: float,
    model_options: dict | None = None,
) -> dict:
    """Run the whole no-future-data backtest.

    prices: aligned adjusted-close DataFrame (one column per ticker).
    selection_metric: "train" (default, no test leakage) or "test".
    """
    prices = prices.dropna(how="any").sort_index()
    if len(prices) < 60:
        raise ValueError("need at least 60 aligned price rows")

    returns = (
        prices.pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(returns) < 50:
        raise ValueError("need at least 50 return rows after cleaning")

    n = int(len(returns) * train_share)
    if n < 30 or len(returns) - n < 5:
        raise ValueError(
            "need at least 30 training returns and 5 test returns; adjust the training share"
        )
    train_returns = returns.iloc[:n]
    test_returns = returns.iloc[n:]
    rf_daily = rf_annual / 100.0 / TRADING_DAYS

    portfolios, strategy_errors = {}, {}
    for name in portfolio_strategies:
        try:
            portfolios.update(
                build_all(train_returns, rf_daily, selected=[name])
            )
        except Exception as exc:
            strategy_errors[name] = str(exc)
    if not portfolios:
        raise ValueError(
            "no portfolio strategy could be built: "
            + " ".join(f"{k}: {v}" for k, v in strategy_errors.items())
        )

    portfolio_rows, portfolio_curves = [], {}
    for name, weights in portfolios.items():
        train_port = (train_returns * weights).sum(axis=1)
        test_port = (test_returns * weights).sum(axis=1)
        train_stats = performance_stats(train_port, rf_daily)
        test_stats = performance_stats(test_port, rf_daily)
        portfolio_rows.append({
            "Strategy": name,
            "Train return (%)": train_stats["Net return (%)"],
            "Test return (%)": test_stats["Net return (%)"],
            "Test Sharpe": test_stats["Sharpe"],
            "Test max drawdown (%)": test_stats["Max drawdown (%)"],
            "Train avg stock correlation": average_pairwise_correlation(train_returns, weights),
            "Test avg stock correlation": average_pairwise_correlation(test_returns, weights),
        })
        portfolio_curves[name] = test_port

    portfolio_table = pd.DataFrame(portfolio_rows)
    if selection_metric == "test":
        best_name = portfolio_table.sort_values(
            ["Test return (%)", "Test Sharpe"], ascending=[False, False], na_position="last"
        ).iloc[0]["Strategy"]
    else:
        best_name = portfolio_table.sort_values(
            ["Train return (%)", "Test Sharpe"], ascending=[False, False], na_position="last"
        ).iloc[0]["Strategy"]
    best_name = str(best_name)

    weights = portfolios[best_name]
    train_port = (train_returns * weights).sum(axis=1)
    test_port = (test_returns * weights).sum(axis=1)

    # Fit once per (ticker, forecaster) — fits are weight-independent — then
    # evaluate every optimization's split against the same predictions.
    predictions, model_errors = forecast_portfolio(
        returns, pd.Series(1.0, index=returns.columns), train_returns.index,
        test_returns.index, models, model_options=model_options,
    )
    strategy_combos = {}
    for name, split_weights in portfolios.items():
        split_port = (test_returns * split_weights).sum(axis=1)
        split_prev = pd.concat([
            (train_returns * split_weights).sum(axis=1), split_port
        ]).shift(1).loc[test_returns.index]
        split_aggregated = aggregate_predictions(predictions, split_weights)
        split_table, _, _, _, _ = evaluate_rules(
            split_port, split_aggregated, split_prev, rules,
            band=band, cost_bps=cost_bps, rf_daily=rf_daily,
        )
        strategy_combos[name] = split_table

    portfolio_predictions = aggregate_predictions(predictions, weights)

    portfolio_forecast_table = pd.DataFrame([
        {"Model": model, **forecast_stats(test_port, pred)}
        for model, pred in portfolio_predictions.items()
    ]).sort_values("MAE (%)", na_position="last").reset_index(drop=True)

    full_port = pd.concat([train_port, test_port])
    previous_returns = full_port.shift(1).loc[test_returns.index]

    combo_table, combo_curves, combo_positions, baseline_net, baseline_stats = evaluate_rules(
        test_port, portfolio_predictions, previous_returns, rules,
        band=band, cost_bps=cost_bps, rf_daily=rf_daily,
    )

    return {
        "prices": prices,
        "returns": returns,
        "train_returns": train_returns,
        "test_returns": test_returns,
        "portfolios": portfolios,
        "portfolio_table": portfolio_table,
        "portfolio_curves": portfolio_curves,
        "best_portfolio": best_name,
        "weights": weights,
        "train_portfolio_returns": train_port,
        "test_portfolio_returns": test_port,
        "predictions": predictions,
        "portfolio_predictions": portfolio_predictions,
        "model_errors": model_errors,
        "strategy_errors": strategy_errors,
        "portfolio_forecast_table": portfolio_forecast_table,
        "combo_table": combo_table,
        "combo_curves": combo_curves,
        "combo_positions": combo_positions,
        "strategy_combos": strategy_combos,
        "baseline_net": baseline_net,
        "baseline_stats": baseline_stats,
        "capital": capital,
        "cost_bps": cost_bps,
        "band": band,
        "model_options": model_options or {},
        "selection_metric": selection_metric,
        "models": list(models),
        "rules": list(rules),
        "portfolio_strategies": list(portfolios),
        "train_start": train_returns.index[0],
        "train_end": train_returns.index[-1],
        "test_start": test_returns.index[0],
        "test_end": test_returns.index[-1],
        "train_size": len(train_returns),
        "test_size": len(test_returns),
    }
