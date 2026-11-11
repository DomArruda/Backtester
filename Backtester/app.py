"""Streamlit app: portfolio optimization + next-step forecasts + rule backtests."""
from __future__ import annotations

import io
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from engine import run_pipeline
from forecasting import (
    ADVANCED_PARAMS,
    MODEL_NAMES,
    RULE_HELP,
    RULE_NAMES,
    SIMPLE_MODELS,
)
from portfolio import STRATEGY_ALGOS, STRATEGY_NAMES

warnings.filterwarnings("ignore")

TRAIN_SHARE = 0.70          # 70% of history teaches, the last 30% grades
RF_ANNUAL = 4.0             # annual risk-free rate (%)
BAND = 0.005                # strong-signal band for one of the trading rules

# Resolve relative to this file: Streamlit Cloud runs from the repo root.
CAPYBARA = str(Path(__file__).resolve().parent / "capybara.png")

TRAIN_SHARE_LABEL = "70%"

PARAM_LABELS = {
    "n_estimators": "Trees",
    "max_depth": "Tree depth",
    "learning_rate": "Learning rate",
    "hidden": "Network size",
    "alpha": "Regularization (alpha)",
}

METRIC_HELP = {
    "Return": "Total percentage growth (or loss) over the period shown.",
    "Sharpe": "Average return per unit of risk — how smooth the ride was. "
              "Higher is better; above 1 is considered good.",
    "MAE": "Mean absolute error — on average, how far the daily guess was from "
           "the real return, in percentage points. Lower is better. (MAPE, the "
           "relative version, is unsuitable here: daily returns near zero make "
           "relative errors explode.)",
    "Drawdown": "The biggest peak-to-trough drop in value — the worst losing stretch.",
    "Up": "Of the days the portfolio actually rose, how often the model "
          "predicted a rise. The trading rules act on this call.",
    "Down": "Of the days the portfolio actually fell, how often the model "
            "predicted a fall. The trading rules act on this call.",
    "Correlation": "How much the portfolio's money moves together — the average "
                   "correlation between held stocks, weighted by position size "
                   "(from -1 to +1). Low correlation is what makes diversification "
                   "work: if it stays low in the test column too, the spreading of "
                   "risk actually happened; if it jumps up, the stocks started "
                   "moving together when it mattered.",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_closes(tickers, start, end):
    raw = yf.download(
        list(tickers),
        start=start,
        end=end + timedelta(days=1),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(1):
            close = raw.xs("Close", level=1, axis=1)
        elif "Close" in raw.columns.get_level_values(0):
            close = raw.xs("Close", level=0, axis=1)
        else:
            return pd.DataFrame()
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close = close.apply(pd.to_numeric, errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan)
    close.columns = [str(c) for c in close.columns]
    return close


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500():
    """Current S&P 500 constituents from Wikipedia (yfinance-style tickers)."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    tickers = (
        table["Symbol"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    )
    return pd.DataFrame({
        "Ticker": tickers.to_numpy(),
        "Company": table["Security"].astype(str).str.strip().to_numpy(),
        "Sector": table["GICS Sector"].astype(str).str.strip().replace("nan", "Unknown"),
    })


@st.cache_data(ttl=3600, show_spinner=False)
def run_cached_backtest(prices, tickers, **params):
    """Full pipeline, cached per (prices, tickers, settings) so repeated runs
    with the same setup — e.g. a whole class using the same inputs — are instant."""
    result = run_pipeline(prices=prices, **params)
    result["missing"] = [t for t in tickers if t not in prices.columns]
    return result


def line_chart(frame, y_label):
    fig = go.Figure()
    for column in frame.columns:
        fig.add_trace(
            go.Scatter(x=frame.index, y=frame[column], mode="lines", name=str(column))
        )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode="x unified",
        legend_title_text="",
        height=480,
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def equity_frame(series_map, capital):
    frame = pd.DataFrame(
        {name: (1.0 + series).cumprod() for name, series in series_map.items()}
    ) * capital
    first = frame.index[0] - pd.offsets.BDay(1)
    frame.loc[first] = capital
    return frame.sort_index()


def display_table(table, rename, helps, formats=None, highlight=None):
    """Subset, rename, add tooltips, and optionally tint the winner rows green."""
    formats = formats or {}
    out = table.rename(columns=rename)[list(rename.values())]
    config = {}
    for key, name in rename.items():
        if pd.api.types.is_numeric_dtype(out[name]):
            config[name] = st.column_config.NumberColumn(
                help=helps.get(key), format=formats.get(name, "%.2f")
            )
    if highlight is not None:
        flags = dict(enumerate(bool(f) for f in highlight))

        def tint_row(row):
            on = flags.get(row.name, False)
            return ["background-color: rgba(73, 190, 125, 0.18)" if on else None] * len(row.index)

        out = out.reset_index(drop=True).style.apply(tint_row, axis=1)
    return st.dataframe(out, hide_index=True, width="stretch", column_config=config)


def show_portfolios(result):
    st.subheader("Portfolio strategies")
    st.caption(
        "Each strategy is a different way of splitting your money across the "
        "chosen stocks. The splits are decided using only the training window."
    )
    st.caption(
        "**Stocks in this set:** " + ", ".join(result["prices"].columns)
    )
    table = result["portfolio_table"].sort_values(
        "Test return (%)", ascending=False, na_position="last"
    ).reset_index(drop=True)
    table["Algorithm"] = table["Strategy"].map(STRATEGY_ALGOS).fillna(table["Strategy"])
    best = result["best_portfolio"]
    row = table.loc[table["Strategy"] == best].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best optimization", best)
    c2.metric(
        "Turned $" f"{result['capital']:,.0f} into",
        f"${result['capital'] * (1.0 + row['Test return (%)'] / 100.0):,.0f}",
    )
    bh_growth = result["baseline_stats"]["Net return (%)"]
    c3.metric("Test growth", f"{row['Test return (%)']:+.2f}%",
              delta=f"{row['Test return (%)'] - bh_growth:+.2f} pp vs buy & hold",
              help=METRIC_HELP["Return"])
    c4.metric("Worst losing stretch", f"{row['Test max drawdown (%)']:.2f}%",
              help=METRIC_HELP["Drawdown"])
    st.caption(
        f"Winning algorithm: **{STRATEGY_ALGOS.get(best, best)}** — chosen by its "
        "test-window result (a peek at the exam, though the splits themselves "
        "used training data only), so treat it as a hint, not a promise. The "
        "default path is simply buying and holding these same stocks — the "
        "'vs buy & hold' arrow shows whether going off that path paid off."
    )

    st.markdown(f"**All {len(table)} strategies, compared on this stock set**")
    display_table(
        table,
        {
            "Strategy": "Strategy",
            "Algorithm": "Algorithm (original name)",
            "Test return (%)": "Growth (%)",
            "Test Sharpe": "Sharpe ratio",
            "Test max drawdown (%)": "Worst losing stretch (%)",
            "Train avg stock correlation": "Avg stock correlation (train)",
            "Test avg stock correlation": "Avg stock correlation (test)",
        },
        {"Sharpe": METRIC_HELP["Sharpe"], "Drawdown": METRIC_HELP["Drawdown"],
         "Return": METRIC_HELP["Return"], "Correlation": METRIC_HELP["Correlation"]},
        highlight=(table["Strategy"] == best).to_numpy(),
    )
    st.caption(
        "The green row is the winner — the numbers above describe only that "
        "row, and every strategy below is compared on this same stock set. "
        "The correlation columns show whether diversification actually "
        "happened: low average correlation means the stocks moved mostly on "
        "their own, and a test number that stays low means the spreading of "
        "risk held up outside the training window."
    )

    with st.expander("How each optimization works"):
        st.markdown(
            "**Equal-Weight (1/N)** — Put the same share of money in every "
            "stock. It makes no estimates at all, so nothing can fool it — "
            "research famously shows this simple split is hard to beat.\n\n"
            "**Minimum Variance (GMVP)** — Search for the split whose combined "
            "daily moves wiggle the least, using how volatile each stock is "
            "and how the stocks move together. It never looks at which stock "
            "*earns* more — only which combination is calmest.\n\n"
            "**Maximum Sharpe Ratio (tangency portfolio)** — Search for the "
            "split with the best historical return per unit of risk. Because "
            "it averages past returns, it trusts history more — which is also "
            "how it gets fooled when the future changes.\n\n"
            "**Hierarchical Risk Parity (HRP)** — Group stocks that move alike "
            "into families, split the money between families by their risk, "
            "then within each family by each stock's own risk. It avoids "
            "inverting the covariance matrix, so it stays stable even when "
            "correlations shift suddenly.\n\n"
            "All four decide their split using the training window only, and "
            "all are long-only: every stock gets 0% or more, and the shares "
            "sum to 100%. The default path — buying and holding the set — "
            "uses no optimization at all; everything here is an attempt to "
            "beat that by splitting differently."
        )

    with st.expander("Advanced: how average stock correlation is calculated"):
        st.markdown(
            "**The idea in one sentence:** it is an average of every pair of "
            "stocks' correlations, where each pair counts as much as the money "
            "in it — big pairs matter, tiny pairs barely do.\n\n"
            "**The recipe (three steps):**\n"
            "1. Look up the correlation between each pair of stocks' daily "
            "moves (a number from -1 to +1).\n"
            "2. Multiply it by (money in stock A) × (money in stock B) — the "
            "two positions, as fractions of the portfolio.\n"
            "3. Add those up, then divide by the total of the multipliers so "
            "the answer stays between -1 and +1."
        )
        st.markdown(
            "**Worked example** — a portfolio of 50% AAPL, 30% MSFT, 20% JNJ "
            "with made-up correlations:\n\n"
            "| Pair | Positions multiplied | Correlation | Contribution |\n"
            "| --- | --- | --- | --- |\n"
            "| AAPL–MSFT | 0.50 × 0.30 = 0.15 | 0.80 | 0.15 × 0.80 = 0.120 |\n"
            "| AAPL–JNJ | 0.50 × 0.20 = 0.10 | 0.30 | 0.10 × 0.30 = 0.030 |\n"
            "| MSFT–JNJ | 0.30 × 0.20 = 0.06 | 0.20 | 0.06 × 0.20 = 0.012 |\n\n"
            "Average = 0.162 ÷ 0.31 ≈ **0.52** — closer to 0.80 because the two "
            "big positions are the highly correlated pair.\n\n"
            "With only two stocks (say 60%/40%), the multiplier is 0.24 and the "
            "division cancels it, so the answer is simply that one correlation. "
            "The train and test columns above are this same recipe applied to "
            "each window's returns."
        )
        st.latex(r"\bar{\rho} \;=\; \frac{\sum_{i<j} w_i\, w_j\, \rho_{ij}}{\sum_{i<j} w_i\, w_j}")

    frame = equity_frame(result["portfolio_curves"], result["capital"])
    fig = line_chart(frame, "Portfolio value")
    fig.update_traces(line=dict(width=3), selector=dict(name=best))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption(
        f"Growth of ${result['capital']:,.0f} on the test window. "
        "The selected strategy is drawn thicker."
    )

    st.markdown(f"**How the selected strategy splits the money — {best}**")
    weights = result["weights"].sort_values(ascending=False)
    wdf = pd.DataFrame({"Ticker": weights.index, "Weight (%)": weights.to_numpy() * 100.0})
    left, right = st.columns([1, 2])
    left.dataframe(
        wdf,
        hide_index=True,
        width="stretch",
        column_config={"Weight (%)": st.column_config.NumberColumn(format="%.2f")},
    )
    bar = go.Figure(go.Bar(x=wdf["Ticker"], y=wdf["Weight (%)"]))
    bar.update_layout(
        height=320,
        template="plotly_white",
        yaxis_title="Share of the portfolio (%)",
        xaxis_title="",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    right.plotly_chart(bar, width="stretch", config={"displaylogo": False})


def show_forecasts_and_rules(result):
    st.subheader("Were the guesses good, and did trading on them pay off?")
    st.caption(
        "Each model guesses the next day's return from the last five days of "
        "returns. The rules then turn each guess into a move: invest or sit "
        "in cash. Costs apply whenever the position changes; borrow costs and "
        "taxes are ignored."
    )
    st.caption(
        "**Stocks in this set:** " + ", ".join(result["prices"].columns)
    )

    display_table(
        result["portfolio_forecast_table"],
        {
            "Model": "Model",
            "MAE (%)": "Mean absolute error (%)",
            "Up-day accuracy (%)": "Up days called right (%)",
            "Down-day accuracy (%)": "Down days called right (%)",
        },
        {"MAE": METRIC_HELP["MAE"], "Up": METRIC_HELP["Up"], "Down": METRIC_HELP["Down"]},
    )
    st.caption(
        "The trading rules act on the up/down call, not the size of the guess — "
        "so the two accuracy columns matter more than MAE for strategy results. "
        "Small MAE differences (say, 0.81 vs 0.83) are noise, and a model can be "
        "good at up days while being poor at down days. We do not use MAPE "
        "(the percentage version of MAE) because daily returns are often close "
        "to zero, which makes relative errors blow up."
    )

    if result["model_options"]:
        with st.expander("Advanced parameters used"):
            for model, opts in result["model_options"].items():
                st.markdown(
                    f"**{model}:** "
                    + ", ".join(f"{PARAM_LABELS.get(k, k)} = {v}" for k, v in opts.items())
                )

    st.subheader("Trading on the guesses")
    st.caption(
        "Everything below is directly verifiable: every line and number is the "
        "real outcome of trading on the test window — no guessed returns are "
        "plotted, only what each rule would actually have earned, minus "
        "trading costs. The default path is buy & hold — buy the set, do "
        "nothing; every optimization and rule is an attempt to do better by "
        "going off that path."
    )
    table = result["combo_table"]
    if table.empty:
        st.info("No model/rule combinations were produced.")
        return
    best = table.iloc[0]
    bh = result["baseline_stats"]

    net_returns = table["Net return (%)"].dropna()
    bh_rank = int((net_returns > bh["Net return (%)"]).sum()) + 1
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best trading strategy", f"{best['Model']} · {best['Rule']}")
    c2.metric("Its result", f"{best['Net return (%)']:+.2f}%",
              delta=f"{best['Vs buy & hold (pp)']:+.2f} pp vs buy & hold",
              help=METRIC_HELP["Return"])
    c3.metric("Buy & hold's rank", f"#{bh_rank} of {len(net_returns) + 1}",
              help="Where simply buying and holding would place among all tested "
                   "model + rule combinations (plus itself). Rank #1 means no "
                   "rule beat just holding.")
    c4.metric("Its worst losing stretch", f"{best['Max drawdown (%)']:.2f}%",
              help=METRIC_HELP["Drawdown"])
    st.caption(
        f"For comparison, simply buying and holding: {bh['Net return (%)']:+.2f}% "
        f"growth, {bh['Max drawdown (%)']:.2f}% worst losing stretch."
    )

    top = table.head(5)
    frame = {"Buy & hold": result["baseline_net"]}
    for combo in top["Combination"]:
        frame[combo] = result["combo_curves"][combo]
    fig = line_chart(equity_frame(frame, result["capital"]), "Portfolio value")
    fig.update_traces(line=dict(width=3), selector=dict(name=top["Combination"].iloc[0]))
    fig.update_traces(
        line=dict(color="#9AA0A6", width=1.5, dash="dash"),
        selector=dict(name="Buy & hold"),
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption(
        "The five best model + rule combinations against buy & hold (dashed "
        "gray); bold line = the single best. Only the top five are drawn so "
        "the picture stays readable — every combination lives in the panel below."
    )

    with st.expander("How the trading rules work"):
        st.markdown(
            "Every rule converts a model's guess into a position for the day: "
            "invest (long) or sit in cash — no shorting. The position for a "
            "day comes from that morning's guess — built only from returns "
            "before that day — and earns that day's real return. Trading "
            "costs apply whenever the position changes.\n\n"
            "**Forecast up / down** — Invest when the guess is positive; cash "
            "when negative. The simplest trend-follower.\n\n"
            "**Strong forecast only** — Invest only when the guess is "
            "confidently positive (beyond the 0.5% band); cash when "
            "confidently negative; in between, keep yesterday's position.\n\n"
            "**Buy dips when forecast up** — Buy only after a down day with a "
            "positive guess (the dip gets bought); cash when the guess is "
            "negative; otherwise hold.\n\n"
            "Same guess, different rule, different money — that is why the "
            "chart above shows big gaps between models, and why the ranking "
            "table below matters."
        )

    with st.expander("All model + rule combinations"):
        display_table(
            table,
            {
                "Rank": "Rank",
                "Model": "Model",
                "Rule": "Rule",
                "Net return (%)": "Result (%)",
                "Sharpe": "Sharpe ratio",
                "Max drawdown (%)": "Worst losing stretch (%)",
                "Vs buy & hold (pp)": "Difference vs buy & hold (pp)",
            },
            {"Sharpe": METRIC_HELP["Sharpe"], "Drawdown": METRIC_HELP["Drawdown"],
             "Return": METRIC_HELP["Return"]},
            formats={"Rank": "%d"},
        )
        st.caption(
            "Reading tips: the naive baseline guesses 'tomorrow = yesterday', so "
            "the buy-dips rule — which needs yesterday down AND a positive "
            "forecast — can never fire for it. Its 0.00% result is by logic, not "
            "failure. And the best of many combinations is partly luck, so treat "
            "the winner as a hint."
        )
        st.download_button(
            "Download results CSV",
            table.to_csv(index=False),
            file_name="strategy_results.csv",
            mime="text/csv",
        )


def sample_uniform(sp500, count):
    return tuple(sorted(sp500["Ticker"].drop_duplicates().sample(n=int(count))))


def sample_diversified(sp500, count):
    """Spread the picks across 5 randomly chosen sectors, roughly n/5 each."""
    count = int(count)
    sectors = sp500["Sector"].dropna()
    n_sectors = min(5, sectors.nunique())
    per, extra = divmod(count, n_sectors)
    picks = []
    for i, sector in enumerate(pd.Series(sectors.unique()).sample(n_sectors)):
        pool = sp500.loc[sectors.index[sectors == sector], "Ticker"].drop_duplicates()
        k = min(per + (1 if i < extra else 0), len(pool))
        if k:
            picks += pool.sample(n=k).tolist()
    return tuple(sorted(picks))


def sample_sp500(sp500, count, mode):
    if mode == "Diversified from the S&P 500":
        return sample_diversified(sp500, count)
    return sample_uniform(sp500, count)


def sample_sets(sp500, count, mode, n):
    """n distinct random stock sets, each of `count` stocks."""
    seen, sets, tries = set(), [], 0
    while len(sets) < n and tries < n * 20:
        tries += 1
        candidate = sample_sp500(sp500, count, mode)
        key = frozenset(candidate)
        if key not in seen:
            seen.add(key)
            sets.append(candidate)
    return sets


def sidebar_tickers():
    """Ticker choice: typed, random S&P 500 samples, or sector-diversified."""
    st.subheader("1. Your stocks")
    mode = st.radio(
        "Choose stocks by",
        ["Type tickers", "Random from the S&P 500", "Diversified from the S&P 500"],
    )
    if mode == "Type tickers":
        text = st.text_input(
            "Tickers (comma separated)", "AAPL, MSFT, GOOGL, AMZN", key="typed_tickers"
        )
        tickers = tuple(dict.fromkeys(t.strip().upper() for t in text.split(",") if t.strip()))
        return tickers, [tickers]

    try:
        sp500 = fetch_sp500()
    except Exception as exc:
        st.warning(f"Could not load the S&P 500 list: {exc}. Type tickers instead.")
        return (), [()]
    count = st.number_input("How many stocks", 2, 50, 5)
    n_sets = st.number_input("Random sets to compare", 1, 5, 1)
    if st.button("Shuffle selection"):
        st.session_state["random_sample"] = {
            "mode": mode, "count": count, "n_sets": n_sets,
            "sets": sample_sets(sp500, count, mode, n_sets),
        }
    state = st.session_state.get("random_sample")
    if (state is None or state["mode"] != mode or state["count"] != count
            or state["n_sets"] != n_sets):
        state = {
            "mode": mode, "count": count, "n_sets": n_sets,
            "sets": sample_sets(sp500, count, mode, n_sets),
        }
        st.session_state["random_sample"] = state
    note = ", spread across 5 sectors" if mode.startswith("Diversified") else ""
    for i, sample in enumerate(state["sets"], 1):
        label = f"Set {i}{note}" if len(state["sets"]) > 1 else f"Picked for you{note}"
        st.caption(f"{label}: " + ", ".join(sample))
    return state["sets"][0], state["sets"]


def compact_tickers(tickers, keep=4):
    if len(tickers) <= keep + 1:
        return ", ".join(tickers)
    return f"{len(tickers)} stocks: {', '.join(tickers[:keep])}, …"


def run_signature(stock_sets, start, end, strategies, models, capital, cost_pct, model_options):
    return repr((stock_sets, str(start), str(end), tuple(strategies), tuple(models),
                 capital, float(cost_pct), model_options))


def show_set_comparison(results, labels):
    """Rank the random stock sets by their own backtest results."""
    st.markdown(
        "This panel answers two questions: **which random set won?** (the "
        "metrics) and **which strategy does best in general?** (the averages)."
    )
    st.selectbox(
        "Results shown in both tabs below — best performer by default",
        labels,
        key="set_view",
    )
    rows = []
    per_strategy = {}
    per_rule = {}
    per_pair = {}
    per_opt_rule = {}
    per_model_mae = {}
    maes_all = []
    for i, (tickers, res) in enumerate(results, 1):
        table = res["portfolio_table"]
        row = table.loc[table["Strategy"] == res["best_portfolio"]].iloc[0]
        maes_all.append(float(np.nanmean(
            res["portfolio_forecast_table"]["MAE (%)"].to_numpy(dtype=float)
        )))
        best_combo = res["combo_table"].iloc[0]
        rows.append({
            "Set": f"Set {i}",
            "Tickers": compact_tickers(tickers),
            "Strategy": res["best_portfolio"],
            "Growth (%)": row["Test return (%)"],
            "Sharpe": row["Test Sharpe"],
            "Worst losing stretch (%)": row["Test max drawdown (%)"],
            "Best rule result (%)": best_combo["Net return (%)"],
            "Best rule": f"{best_combo['Model']} · {best_combo['Rule']}",
        })
        for _, srow in table.iterrows():
            stats = per_strategy.setdefault(srow["Strategy"], {"growth": [], "sharpe": []})
            stats["growth"].append(srow["Test return (%)"])
            stats["sharpe"].append(srow["Test Sharpe"])
        for rule, grp in res["combo_table"].groupby("Rule"):
            stats = per_rule.setdefault(rule, {"net": [], "sharpe": []})
            stats["net"].append(float(grp["Net return (%)"].mean()))
            stats["sharpe"].append(float(grp["Sharpe"].mean()))
        bhs = res["baseline_stats"]
        stats = per_rule.setdefault("Buy & hold (default)", {"net": [], "sharpe": []})
        stats["net"].append(float(bhs["Net return (%)"]))
        stats["sharpe"].append(float(bhs["Sharpe"]))
        for _, crow in res["combo_table"].iterrows():
            key = (crow["Model"], crow["Rule"])
            stats = per_pair.setdefault(key, {"net": [], "sharpe": []})
            stats["net"].append(float(crow["Net return (%)"]))
            stats["sharpe"].append(float(crow["Sharpe"]))
        for strat, combo_table in res.get("strategy_combos", {}).items():
            for rule, grp in combo_table.groupby("Rule"):
                key = (strat, rule)
                stats = per_opt_rule.setdefault(key, {"net": [], "sharpe": []})
                stats["net"].append(float(grp["Net return (%)"].mean()))
                stats["sharpe"].append(float(grp["Sharpe"].mean()))
        for _, mrow in res["portfolio_forecast_table"].iterrows():
            per_model_mae.setdefault(mrow["Model"], []).append(float(mrow["MAE (%)"]))

    avg_growth = float(np.nanmean([r["Growth (%)"] for r in rows]))
    def avg_of(name, key):
        vals = np.asarray(per_strategy[name][key], dtype=float)
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
    def rule_avg(name, key):
        vals = np.asarray(per_rule[name][key], dtype=float)
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
    def mae_avg(model):
        vals = np.asarray(per_model_mae[model], dtype=float)
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
    def pair_avg(model, rule, key):
        vals = np.asarray(per_pair[(model, rule)][key], dtype=float)
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
    def opt_rule_avg(strat, rule, key):
        vals = np.asarray(per_opt_rule[(strat, rule)][key], dtype=float)
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
    best_avg_name = max(per_strategy, key=lambda n: avg_of(n, "growth"))
    best_rule_overall = max(per_rule, key=lambda r: rule_avg(r, "net"))
    best_forecaster = min(per_model_mae, key=mae_avg)
    st.markdown("**On average, across all random sets**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Best optimization strategy", best_avg_name,
              delta=f"{avg_of(best_avg_name, 'growth'):+.2f}% avg growth",
              help="Portfolio-split strategy with the best average growth over "
                   "every set's test window.")
    c2.metric("Most accurate forecaster", best_forecaster,
              delta=f"{mae_avg(best_forecaster):.2f}% avg miss (MAE)",
              delta_color="inverse",
              help="Model with the lowest average mean absolute error across "
                   "every set — lower is better.")
    c3.metric("Most dominant trading rule", best_rule_overall,
              delta=f"{rule_avg(best_rule_overall, 'net'):+.2f}% avg result",
              help="The rule whose average over every model and every set is "
                   "best. If this reads 'Buy & hold (default)', no "
                   "forecast-driven rule beat just holding.")

    with st.expander("Every random set — details"):
        display_table(
            pd.DataFrame(rows),
            {
                "Set": "Set",
                "Tickers": "Tickers",
                "Strategy": "Strategy",
                "Growth (%)": "Growth (%)",
                "Sharpe": "Sharpe ratio",
                "Worst losing stretch (%)": "Worst losing stretch (%)",
                "Best rule result (%)": "Best rule result (%)",
                "Best rule": "Winning combination (model · rule)",
            },
            {"Sharpe": METRIC_HELP["Sharpe"], "Drawdown": METRIC_HELP["Drawdown"],
             "Return": METRIC_HELP["Return"]},
            highlight=[label == st.session_state["set_view"].split(" —")[0]
                       for label in [r["Set"] for r in rows]],
        )
    with st.expander("Strategy averages — all sets"):
        strategy_table = pd.DataFrame([
            {
                "Strategy": name,
                "Sets": len(s["growth"]),
                "Avg growth (%)": avg_of(name, "growth"),
                "Avg Sharpe": avg_of(name, "sharpe"),
            }
            for name, s in per_strategy.items()
        ]).sort_values(
            "Avg growth (%)", ascending=False, na_position="last"
        ).reset_index(drop=True)
        display_table(
            strategy_table,
            {
                "Strategy": "Strategy",
                "Sets": "Sets tested",
                "Avg growth (%)": "Average growth (%)",
                "Avg Sharpe": "Average Sharpe",
            },
            {"Return": METRIC_HELP["Return"], "Sharpe": METRIC_HELP["Sharpe"]},
            formats={"Sets tested": "%d"},
            highlight=[i == 0 for i in range(len(strategy_table))],
        )
    if per_opt_rule:
        with st.expander("Optimization × rule averages — all sets"):
            st.caption(
                "Every portfolio split (optimization) was traded with every rule "
                "under the same forecasts, averaged over all sets. This shows how "
                "the split and the rule interact — a rule that looks best on one "
                "split may not look best on another."
            )
            opt_rule_table = pd.DataFrame([
                {
                    "Optimization": strat,
                    "Rule": rule,
                    "Sets": len(s["net"]),
                    "Avg result (%)": opt_rule_avg(strat, rule, "net"),
                    "Avg Sharpe": opt_rule_avg(strat, rule, "sharpe"),
                }
                for (strat, rule), s in per_opt_rule.items()
            ]).sort_values(
                "Avg result (%)", ascending=False, na_position="last"
            ).reset_index(drop=True)
            display_table(
                opt_rule_table,
                {
                    "Optimization": "Optimization",
                    "Rule": "Rule",
                    "Sets": "Sets tested",
                    "Avg result (%)": "Average result (%)",
                    "Avg Sharpe": "Average Sharpe",
                },
                {"Return": METRIC_HELP["Return"], "Sharpe": METRIC_HELP["Sharpe"]},
                formats={"Sets tested": "%d"},
                highlight=[i == 0 for i in range(len(opt_rule_table))],
            )
    with st.expander("Trading rules & forecasters — all sets"):
        st.caption(
            "Each row averages one model + rule combination over every set's "
            "test window. Model-agnostic rule rows are the averages of the "
            "matched pairs underneath. Buy & hold (default) is included as the "
            "no-forecast benchmark — it is not a forecast-driven rule."
        )
        rule_table = pd.DataFrame([
            {
                "Rule": rule,
                "Sets": len(s["net"]),
                "Avg result (%)": rule_avg(rule, "net"),
                "Avg Sharpe": rule_avg(rule, "sharpe"),
            }
            for rule, s in per_rule.items()
        ]).sort_values(
            "Avg result (%)", ascending=False, na_position="last"
        ).reset_index(drop=True)
        st.markdown("**Model-agnostic (each rule averaged over every model)**")
        display_table(
            rule_table,
            {
                "Rule": "Rule",
                "Sets": "Sets tested",
                "Avg result (%)": "Average result (%)",
                "Avg Sharpe": "Average Sharpe",
            },
            {"Return": METRIC_HELP["Return"], "Sharpe": METRIC_HELP["Sharpe"]},
            formats={"Sets tested": "%d"},
            highlight=[i == 0 for i in range(len(rule_table))],
        )
        pair_table = pd.DataFrame([
            {
                "Forecaster": model,
                "Rule": rule,
                "Sets": len(s["net"]),
                "Avg result (%)": pair_avg(model, rule, "net"),
                "Avg Sharpe": pair_avg(model, rule, "sharpe"),
            }
            for (model, rule), s in per_pair.items()
        ]).sort_values(
            "Avg result (%)", ascending=False, na_position="last"
        ).reset_index(drop=True)
        st.markdown("**Forecaster × rule (each combination averaged over every set)**")
        display_table(
            pair_table,
            {
                "Forecaster": "Forecaster",
                "Rule": "Rule",
                "Sets": "Sets tested",
                "Avg result (%)": "Average result (%)",
                "Avg Sharpe": "Average Sharpe",
            },
            {"Return": METRIC_HELP["Return"], "Sharpe": METRIC_HELP["Sharpe"]},
            formats={"Sets tested": "%d"},
            highlight=[i == 0 for i in range(len(pair_table))],
        )
    st.caption(
        "The averages smooth out single-set luck; compare diversified vs "
        "plain-random by running once per sidebar mode. **Bias note:** the "
        "headline metrics use each set's winning split (the default path) — the "
        "Optimization × rule panel shows how every split ranks every rule, so "
        "check whether the ranking survives before trusting it. Fair warning: "
        "picking the best of several random sets is heavy peeking — expect the "
        "winner's edge to shrink on fresh data."
    )


def main():
    st.set_page_config(page_title="Portfolio + Forecast Backtester",
                       page_icon=CAPYBARA, layout="wide")
    st.title("Portfolio + Forecast Backtester")
    st.caption(
        "Pick stocks, see how different portfolio splits would have done, test "
        "whether simple forecasts could have beaten buy & hold."
    )

    with st.sidebar:
        tickers, stock_sets = sidebar_tickers()

        start = st.date_input("History starts", date.today() - timedelta(days=4 * 365))
        end = st.date_input("History ends", date.today())

        st.subheader("2. Portfolio strategy")
        strategies = st.multiselect(
            "Ways to split the money", STRATEGY_NAMES, default=STRATEGY_NAMES
        )

        st.subheader("3. Forecast models")
        advanced = st.checkbox("Advanced: show every model and tune parameters", value=False)
        if advanced:
            models = st.multiselect("Models", MODEL_NAMES, default=SIMPLE_MODELS)
            model_options = {}
            for model in models:
                spec = ADVANCED_PARAMS.get(model)
                if not spec:
                    continue
                st.markdown(f"**{model}**")
                model_options[model] = {
                    param: st.number_input(
                        PARAM_LABELS[param], lo, hi, default, step,
                        key=f"adv|{model}|{param}",
                    )
                    for param, (lo, hi, step, default) in spec.items()
                }
        else:
            models = st.multiselect("Models", SIMPLE_MODELS, default=SIMPLE_MODELS)
            model_options = {}

        st.subheader("4. Money and costs")
        capital = st.number_input("Starting capital", 100.0, 10_000_000.0, 10_000.0, 100.0)
        cost_pct = st.number_input("Cost per trade (%)", 0.0, 1.0, 0.10, 0.01)
        run = st.button("Run backtest", type="primary", width="stretch")

    with st.expander("Browse the S&P 500"):
        st.caption("These are the stocks you can pick in the sidebar, grouped with their sector.")
        try:
            st.dataframe(fetch_sp500(), hide_index=True, width="stretch", height=420)
        except Exception as exc:
            st.error(f"Could not load the S&P 500 list from Wikipedia: {exc}")

    if run:
        if not tickers or not strategies or not models:
            st.error("Choose at least one ticker, portfolio strategy and model.")
            return
        if start >= end:
            st.error("The history start date must be before the end date.")
            return
        try:
            with st.spinner(
                "Downloading prices, building portfolios and running the backtest..."
            ):
                multi = len(stock_sets) > 1
                results = []
                progress = (
                    st.progress(0.0, text="Backtesting random sets...") if multi else None
                )
                for i, stock_set in enumerate(stock_sets, 1):
                    label = f"Set {i} of {len(stock_sets)} ({', '.join(stock_set)})"
                    if progress:
                        progress.progress((i - 1) / len(stock_sets), text=f"Backtesting {label}...")
                    try:
                        prices = fetch_closes(stock_set, start, end)
                        prices = prices.dropna(axis=1, how="all").dropna(how="any")
                        if prices.empty or len(prices) < 60:
                            raise ValueError(
                                "not enough overlapping price history; try more tickers "
                                "or a wider date range"
                            )
                        res = run_cached_backtest(
                            prices,
                            stock_set,
                            train_share=TRAIN_SHARE,
                            portfolio_strategies=strategies,
                            selection_metric="test",
                            models=models,
                            rules=RULE_NAMES,
                            band=BAND,
                            cost_bps=cost_pct * 100.0,
                            capital=capital,
                            rf_annual=RF_ANNUAL,
                            model_options=model_options,
                        )
                        results.append((stock_set, res))
                        if progress:
                            growth = float(
                                res["portfolio_table"]
                                .loc[res["portfolio_table"]["Strategy"] == res["best_portfolio"],
                                     "Test return (%)"]
                                .iloc[0]
                            )
                            st.toast(
                                f"{label} finished: {growth:+.1f}% growth", icon="✅"
                            )
                    except Exception as exc:
                        if multi:
                            st.toast(f"{label} skipped: {exc}", icon="⚠️")
                            st.warning(f"Set {i} ({', '.join(stock_set)}) skipped: {exc}")
                        else:
                            raise
                if progress:
                    progress.empty()
                if not results:
                    raise ValueError("no stock set could be backtested")

                def best_combo_return(pair):
                    combo_table = pair[1]["combo_table"]
                    if combo_table.empty:
                        return float("-inf")
                    return float(combo_table.iloc[0]["Net return (%)"])

                results.sort(key=best_combo_return, reverse=True)
                tickers, result = results[0]
                st.session_state["set_results"] = results if multi else None
                st.session_state["result"] = result
                st.session_state["run_signature"] = run_signature(
                    stock_sets, start, end, strategies, models, capital, cost_pct, model_options
                )
        except Exception as exc:
            st.error(f"Could not complete the backtest: {exc}")
            return

    if "result" not in st.session_state:
        image_col, text_col = st.columns([1, 3], gap="medium")
        with image_col:
            st.image(CAPYBARA, width=150)
        with text_col:
            st.markdown("### Pick stocks, split your money, test the forecasts")
            st.markdown(
                "1. Choose stocks in the sidebar — type them, sample the S&P 500, "
                "or go sector-diversified.\n"
                "2. Hit **Run backtest**.\n"
                "3. Compare portfolio strategies, guess quality and trading rules — "
                "then judge for yourself whether the clever stuff beat buy & hold."
            )
        return

    set_results = st.session_state.get("set_results")
    result = st.session_state["result"]
    labels = None
    if st.session_state.get("run_signature") != run_signature(
        stock_sets, start, end, strategies, models, capital, cost_pct, model_options
    ):
        st.info(
            "ℹ️ The sidebar has changed since this run — press **Run backtest** "
            "to refresh the results below."
        )
    if set_results:
        labels = [
            f"Set {i}: {', '.join(tickers)}" + (" — best performer" if i == 1 else "")
            for i, (tickers, _) in enumerate(set_results, 1)
        ]
        if st.session_state.get("set_view") not in labels:
            st.session_state["set_view"] = labels[0]
        result = set_results[labels.index(st.session_state["set_view"])][1]

    if result.get("missing"):
        st.warning(
            f"No data for: {', '.join(result['missing'])}. Results use the other tickers."
        )
    for name, message in result.get("strategy_errors", {}).items():
        st.warning(f"Portfolio strategy skipped — {name}: {message}")
    for name, message in result.get("model_errors", {}).items():
        st.warning(f"Model skipped — {name}: {message}")

    st.caption(
        f"The models learned from {result['train_start']:%d %b %Y} to "
        f"{result['train_end']:%d %b %Y} ({result['train_size']} trading days) and "
        f"were graded on {result['test_start']:%d %b %Y} to {result['test_end']:%d %b %Y} "
        f"({result['test_size']} trading days) — data they had never seen."
    )

    if set_results:
        showing = str(st.session_state.get("set_view", labels[0])).split(":")[0]
        with st.expander(
            f"Compare all {len(set_results)} random sets — tabs are showing: {showing}"
        ):
            show_set_comparison(set_results, labels)

    tabs = st.tabs(["1. Portfolio strategies", "2. Forecasts & trading rules"])
    with tabs[0]:
        show_portfolios(result)
    with tabs[1]:
        show_forecasts_and_rules(result)

    with st.expander("How this stays honest (no peeking at the answers)"):
        st.markdown(
            "- Every decision — portfolio splits and model fitting — used only the "
            f"first {TRAIN_SHARE_LABEL} of the history (the training window).\n"
            "- Each forecast for a day only uses returns from before that day.\n"
            "- The last part of the history (the test window) is kept as an exam: "
            "the models never saw it while learning.\n"
            "- The highlighted portfolio strategy and the winning model + rule are "
            "both picked by their test-window result. That is an in-sample choice "
            "on the test window, so treat it as a hint, not a promise — validate "
            "the winner on a new period before trusting it.\n\n"
            "The trading rules:\n\n"
            + "\n".join(f"- **{name}** — {RULE_HELP[name]}" for name in RULE_NAMES)
        )


if __name__ == "__main__":
    main()
