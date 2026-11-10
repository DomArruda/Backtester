"""Next-step return forecasts and forecast-driven trading rules."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

TRADING_DAYS = 252.0
LAGS = 5

BASELINE_NAME = "Yesterday's return (baseline)"

SIMPLE_MODELS = [
    "Linear Regression",
    BASELINE_NAME,
    "XGBoost",
    "Neural Network",
]

MODEL_NAMES = SIMPLE_MODELS + ["Ridge Regression", "Lasso Regression"]

# Advanced-tuning options: param -> (min, max, step, default).
ADVANCED_PARAMS = {
    "XGBoost": {
        "n_estimators": (50, 400, 50, 200),
        "max_depth": (2, 6, 1, 3),
        "learning_rate": (0.01, 0.30, 0.01, 0.05),
    },
    "Neural Network": {
        "hidden": (8, 64, 8, 32),
        "alpha": (0.0, 0.01, 0.0001, 0.0001),
    },
    "Ridge Regression": {"alpha": (0.01, 100.0, 0.01, 1.0)},
    "Lasso Regression": {"alpha": (0.0001, 1.0, 0.0001, 0.01)},
}


def make_estimator(name, options=None):
    options = options or {}
    if name == "Linear Regression":
        return LinearRegression()
    if name == BASELINE_NAME:
        raise ValueError("the baseline does not need an estimator")
    if name == "Ridge Regression":
        return Ridge(alpha=float(options.get("alpha", 1.0)), max_iter=10000)
    if name == "Lasso Regression":
        return Lasso(alpha=float(options.get("alpha", 0.01)), max_iter=10000)
    if name == "XGBoost":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
            subsample=0.8,
            n_estimators=int(options.get("n_estimators", 200)),
            max_depth=int(options.get("max_depth", 3)),
            learning_rate=float(options.get("learning_rate", 0.05)),
        )
    if name == "Neural Network":
        # Daily returns are ~1e-3, far below adam's default operating scale, so
        # the target is standardized (TransformedTargetRegressor) and training
        # stops early on a validation split to avoid overfitting a few hundred rows.
        return TransformedTargetRegressor(
            regressor=MLPRegressor(
                hidden_layer_sizes=(int(options.get("hidden", 32)),),
                alpha=float(options.get("alpha", 0.0001)),
                max_iter=500,
                early_stopping=True,
                random_state=42,
            ),
            transformer=StandardScaler(),
        )
    raise ValueError(f"Unknown model: {name}")


def fit_model(name, X, y, options=None):
    pipe = Pipeline([("scale", StandardScaler()), ("model", make_estimator(name, options))])
    pipe.fit(X, y)
    return pipe


def make_features(returns, lags=LAGS):
    return pd.DataFrame(
        {f"lag_{i}": returns.shift(i) for i in range(1, lags + 1)},
        index=returns.index,
    )


def forecast_one(series, train_index, test_index, model_name, options=None):
    """Fit one model on training rows and predict every test row from prior returns."""
    X_all = make_features(series)
    X_train = X_all.loc[train_index].dropna()
    y_train = series.loc[X_train.index]
    X_test = X_all.loc[test_index].dropna()
    if len(X_train) < 10:
        raise ValueError("need at least 10 training rows after building lags")
    if X_test.empty:
        raise ValueError("no test rows after building lags")
    if model_name == BASELINE_NAME:
        return X_test["lag_1"].rename(model_name)
    pipe = fit_model(model_name, X_train, y_train, options)
    # No model should forecast a daily return beyond anything plausible for the asset.
    bound = max(5.0 * float(np.std(y_train)), 0.01)
    return pd.Series(
        np.clip(pipe.predict(X_test), -bound, bound), index=X_test.index, name=model_name
    )


def forecast_portfolio(returns, weights, train_index, test_index, models,
                       model_options=None):
    """Forecast each positively weighted asset, one model at a time."""
    model_options = model_options or {}
    active = [t for t in weights.index if weights[t] > 1e-12]
    if not active:
        raise ValueError("the selected portfolio has no positive weights")
    predictions, errors = {}, {}
    for model in models:
        try:
            preds = pd.DataFrame(index=test_index, columns=active, dtype=float)
            for ticker in active:
                pred = forecast_one(
                    returns[ticker], train_index, test_index, model,
                    model_options.get(model),
                )
                preds.loc[pred.index, ticker] = pred
            if preds.isna().any().any():
                raise ValueError("missing test predictions")
            predictions[model] = preds
        except Exception as exc:
            errors[model] = str(exc)
    if not predictions:
        raise ValueError(
            "no model could be fitted: " + " ".join(f"{k}: {v}" for k, v in errors.items())
        )
    return predictions, errors


def aggregate_predictions(predictions, weights):
    return {
        model: (preds * weights.reindex(preds.columns)).sum(axis=1)
        for model, preds in predictions.items()
    }


def forecast_stats(actual, predicted):
    """Guess quality: mean absolute error plus how often the up/down sign was right."""
    aligned = pd.concat(
        [
            pd.Series(actual, dtype=float).rename("actual"),
            pd.Series(predicted, dtype=float).rename("predicted"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        return {
            "MAE (%)": np.nan,
            "Up-day accuracy (%)": np.nan,
            "Down-day accuracy (%)": np.nan,
        }
    err = aligned["predicted"].to_numpy() - aligned["actual"].to_numpy()
    a = aligned["actual"].to_numpy()
    p = aligned["predicted"].to_numpy()
    up, down = a > 0, a < 0
    return {
        "MAE (%)": float(np.mean(np.abs(err)) * 100.0),
        "Up-day accuracy (%)": float(np.mean(p[up] > 0) * 100.0) if up.any() else np.nan,
        "Down-day accuracy (%)": float(np.mean(p[down] < 0) * 100.0) if down.any() else np.nan,
    }


RULE_NAMES = [
    "Forecast up / down",
    "Strong forecast only",
    "Forecast long / short",
    "Buy dips when forecast up",
]

RULE_HELP = {
    "Forecast up / down": "Invest when the predicted next return is positive; cash otherwise.",
    "Strong forecast only": (
        "Invest only when the forecast exceeds the band; cash below the negative band; "
        "otherwise hold the previous position."
    ),
    "Forecast long / short": "Invest in positive forecasts and bet against negative forecasts.",
    "Buy dips when forecast up": (
        "Buy after a down day when the forecast is positive; cash when the forecast is "
        "negative; otherwise hold."
    ),
}


def positions(rule, predicted, previous_returns, band=0.005):
    predicted = pd.Series(predicted, dtype=float)
    previous_returns = pd.Series(previous_returns, dtype=float).reindex(predicted.index)
    if rule == "Forecast up / down":
        return (predicted > 0).astype(float)
    if rule == "Strong forecast only":
        out = pd.Series(np.nan, index=predicted.index)
        out[predicted > band] = 1.0
        out[predicted < -band] = 0.0
        return out
    if rule == "Forecast long / short":
        return pd.Series(np.where(predicted > 0, 1.0, -1.0), index=predicted.index)
    if rule == "Buy dips when forecast up":
        out = pd.Series(np.nan, index=predicted.index)
        out[predicted <= 0] = 0.0
        out[(predicted > 0) & (previous_returns < 0)] = 1.0
        return out
    raise ValueError(f"Unknown rule: {rule}")


def buy_hold_net_returns(test_returns, cost_bps=0.0):
    net = pd.Series(test_returns, dtype=float).copy()
    if len(net):
        cost = cost_bps / 10000.0
        net.iloc[0] = (1.0 - cost) * (1.0 + net.iloc[0]) - 1.0
    return net


def strategy_net_returns(test_returns, predicted, rule, previous_returns,
                         band=0.005, cost_bps=0.0):
    test_returns = pd.Series(test_returns, dtype=float)
    pos = positions(rule, predicted, previous_returns, band).ffill().fillna(0.0)
    turnover = pos.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(pos.iloc[0])
    cost = cost_bps / 10000.0
    factors = ((1.0 - cost * turnover) * (1.0 + pos * test_returns)).clip(lower=0.0)
    return factors - 1.0, pos, turnover


def performance_stats(net_returns, rf_daily=0.0):
    net = pd.Series(net_returns, dtype=float).dropna()
    if net.empty:
        return {"Net return (%)": np.nan, "Sharpe": np.nan, "Max drawdown (%)": np.nan}
    equity = (1.0 + net).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    excess = net - rf_daily
    sd = excess.std(ddof=1)
    sharpe = (
        float(np.sqrt(TRADING_DAYS) * excess.mean() / sd)
        if len(net) > 1 and sd > 1e-12 else np.nan
    )
    mdd = float((equity / equity.cummax() - 1.0).min())
    return {
        "Net return (%)": total * 100.0,
        "Sharpe": sharpe,
        "Max drawdown (%)": mdd * 100.0,
    }


def evaluate_rules(test_returns, predicted_by_model, previous_returns, rules,
                   band=0.005, cost_bps=0.0, rf_daily=0.0):
    baseline_net = buy_hold_net_returns(test_returns, cost_bps)
    baseline_stats = performance_stats(baseline_net, rf_daily)
    rows, curves, positions_out = [], {}, {}
    for model, predicted in predicted_by_model.items():
        for rule in rules:
            net, pos, _ = strategy_net_returns(
                test_returns, predicted, rule, previous_returns, band, cost_bps
            )
            stats = performance_stats(net, rf_daily)
            combo = f"{model} | {rule}"
            rows.append({
                "Combination": combo,
                "Model": model,
                "Rule": rule,
                **stats,
                "Vs buy & hold (pp)": stats["Net return (%)"] - baseline_stats["Net return (%)"],
            })
            curves[combo] = net
            positions_out[combo] = pos
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            ["Net return (%)", "Sharpe"], ascending=[False, False], na_position="last"
        ).reset_index(drop=True)
        table.insert(0, "Rank", np.arange(1, len(table) + 1))
    return table, curves, positions_out, baseline_net, baseline_stats
