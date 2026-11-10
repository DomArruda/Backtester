"""Portfolio construction strategies.

Each constructor takes a returns DataFrame (one column per asset) and returns a
weight Series indexed by asset, summing to 1 with long-only weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

TRADING_DAYS = 252.0


def as_frame(returns):
    return returns if isinstance(returns, pd.DataFrame) else returns.to_frame()


def align_rf(rf, idx):
    if isinstance(rf, pd.Series):
        return rf.reindex(idx).ffill().bfill().fillna(0.0)
    return pd.Series(float(rf), index=idx)


SUM_TO_ONE = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)


def _long_only_bounds(n):
    return tuple((0, 1) for _ in range(n))


def _equal_start(n):
    return np.full(n, 1.0 / n)


def _normalize(w, columns):
    w = pd.Series(np.asarray(w, dtype=float), index=columns).clip(lower=0)
    if not np.isfinite(w).all() or w.sum() <= 0:
        return pd.Series(1.0 / len(columns), index=columns)
    return w / w.sum()


def equal_weight_portfolio(returns):
    R = as_frame(returns)
    return pd.Series(1.0 / R.shape[1], index=R.columns)


def minimum_variance_portfolio(returns):
    R = as_frame(returns)
    n = R.shape[1]

    def obj(w):
        return (R * w).sum(axis=1).std(ddof=1) * np.sqrt(TRADING_DAYS)

    res = minimize(obj, _equal_start(n), method="SLSQP",
                   bounds=_long_only_bounds(n), constraints=SUM_TO_ONE)
    return pd.Series(res.x, index=R.columns)


def maximum_sharpe_ratio_portfolio(returns, risk_free):
    R = as_frame(returns)
    n = R.shape[1]
    rf = align_rf(risk_free, R.index)

    def neg_sharpe(w):
        ex = (R * w).sum(axis=1) - rf
        s = ex.std(ddof=1)
        return -(ex.mean() / s) * np.sqrt(TRADING_DAYS) if s and s > 0 else 1e6

    res = minimize(neg_sharpe, _equal_start(n), method="SLSQP",
                   bounds=_long_only_bounds(n), constraints=SUM_TO_ONE)
    return pd.Series(res.x, index=R.columns)


def _ivp_weights(cov_slice):
    ivp = 1.0 / np.diag(cov_slice)
    return ivp / ivp.sum()


def _cluster_variance(cov, idx):
    sub = cov[np.ix_(idx, idx)]
    w = _ivp_weights(sub)
    return float(w @ sub @ w)


def _recursive_bisection(cov, sorted_idx):
    weights = pd.Series(1.0, index=sorted_idx)
    clusters = [sorted_idx]
    while clusters:
        clusters = [
            half
            for cl in clusters
            for half in (cl[: len(cl) // 2], cl[len(cl) // 2:])
            if len(cl) > 1
        ]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
    return weights


def hierarchical_risk_parity_portfolio(returns, linkage_method="single"):
    R = as_frame(returns)
    n = R.shape[1]
    if n == 1:
        return pd.Series([1.0], index=R.columns)
    if n == 2:
        cov2 = R.cov().values
        return _normalize(_ivp_weights(cov2), R.columns)

    cov = R.cov().values
    corr = R.corr().values
    d = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    np.fill_diagonal(d, 0.0)
    diff = d[:, None, :] - d[None, :, :]
    d_tilde = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(d_tilde, 0.0)
    condensed = squareform(d_tilde, checks=False)
    link = linkage(condensed, method=linkage_method)
    sorted_idx = list(leaves_list(link))
    raw = _recursive_bisection(cov, sorted_idx)
    w = pd.Series(raw.values, index=R.columns[sorted_idx])
    return _normalize(w.reindex(R.columns), R.columns)


def average_pairwise_correlation(returns, weights=None, threshold=1e-12):
    """How much the portfolio's money moves together: the weighted average
    pairwise correlation among held stocks, weighted by position sizes."""
    R = as_frame(returns)
    if weights is not None:
        active = [c for c in R.columns if float(weights.get(c, 0.0)) > threshold]
        if len(active) < 2:
            return np.nan
        R = R[active]
        w = np.array([float(weights[c]) for c in active])
        w = w / w.sum()
    else:
        w = np.full(R.shape[1], 1.0 / R.shape[1])
    corr = R.corr().values
    pair_w = np.outer(w, w)
    upper = np.triu_indices(len(w), k=1)
    c_vals, w_vals = corr[upper], pair_w[upper]
    ok = np.isfinite(c_vals)
    if not ok.any():
        return np.nan
    return float(np.sum(c_vals[ok] * w_vals[ok]) / np.sum(w_vals[ok]))


STRATEGY_NAMES = [
    "Equal-Weight",
    "Minimum Variance",
    "Maximum Sharpe Ratio",
    "Hierarchical Risk Parity",
]

# Original algorithm names, as built in PortfolioOpt/PortfolioOpt/strategies.py.
STRATEGY_ALGOS = {
    "Equal-Weight": "Equal-Weight (1/N) Portfolio",
    "Minimum Variance": "Global Minimum Variance Portfolio (GMVP)",
    "Maximum Sharpe Ratio": "Maximum Sharpe Ratio Portfolio (tangency portfolio)",
    "Hierarchical Risk Parity": "Hierarchical Risk Parity (HRP)",
}

_BUILDERS = {
    "Equal-Weight": lambda r, rf: equal_weight_portfolio(r),
    "Minimum Variance": lambda r, rf: minimum_variance_portfolio(r),
    "Maximum Sharpe Ratio": lambda r, rf: maximum_sharpe_ratio_portfolio(r, rf),
    "Hierarchical Risk Parity": lambda r, rf: hierarchical_risk_parity_portfolio(r),
}


def build_all(returns, risk_free, selected=None):
    """Construct the chosen strategies and return name -> weight Series."""
    cols = as_frame(returns).columns
    chosen = STRATEGY_NAMES if selected is None else [s for s in STRATEGY_NAMES if s in selected]
    portfolios = {
        name: _BUILDERS[name](returns, risk_free) for name in chosen
    }
    return {
        k: _normalize(w, cols).reindex(cols).fillna(0.0)
        for k, w in portfolios.items()
    }
