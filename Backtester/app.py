"""Streamlit app: portfolio optimization + next-step forecasts + rule backtests."""
from __future__ import annotations

import io
import warnings
from datetime import date, timedelta

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


def display_table(table, rename, helps, formats=None):
    """Subset, rename and add tooltips so the table reads in plain English."""
    formats = formats or {}
    out = table.rename(columns=rename)[list(rename.values())]
    config = {}
    for key, name in rename.items():
        if pd.api.types.is_numeric_dtype(out[name]):
            config[name] = st.column_config.NumberColumn(
                help=helps.get(key), format=formats.get(name, "%.2f")
            )
    return st.dataframe(out, hide_index=True, width="stretch", column_config=config)


def show_portfolios(result):
    st.subheader("Portfolio strategies")
    st.caption(
        "Each strategy is a different way of splitting your money across the "
        "chosen stocks. The splits are decided using only the training window."
    )
    table = result["portfolio_table"].sort_values(
        "Test return (%)", ascending=False, na_position="last"
    ).reset_index(drop=True)
    table["Algorithm"] = table["Strategy"].map(STRATEGY_ALGOS).fillna(table["Strategy"])
    best = result["best_portfolio"]
    row = table.loc[table["Strategy"] == best].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected strategy", best)
    c2.metric(
        "Turned $" f"{result['capital']:,.0f} into",
        f"${result['capital'] * (1.0 + row['Test return (%)'] / 100.0):,.0f}",
    )
    c3.metric("Growth on test window", f"{row['Test return (%)']:+.2f}%",
              help=METRIC_HELP["Return"])
    c4.metric("Worst losing stretch", f"{row['Test max drawdown (%)']:.2f}%",
              help=METRIC_HELP["Drawdown"])
    st.caption(
        f"Selected algorithm: **{STRATEGY_ALGOS.get(best, best)}**. The app picked "
        "this strategy because it performed best on the test window (the exam it "
        "had never seen). Its splits were still decided using only the training "
        "window, but the choice itself looks at the exam — treat it as a hint, "
        "not a promise."
    )

    st.markdown("**All tested strategies**")
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
    )
    st.caption(
        "The correlation columns show whether diversification actually happened: "
        "low average correlation means the stocks moved mostly on their own, and "
        "a test number that stays low means the spreading of risk held up outside "
        "the training window."
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
        "returns. The rules then turn each guess into a move: invest, do "
        "nothing, or bet against. Costs apply whenever the position changes; "
        "borrow costs and taxes are ignored."
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
        "trading costs."
    )
    frame = {"Buy & hold": result["baseline_net"]}
    for model in result["portfolio_predictions"]:
        curve = result["combo_curves"].get(f"{model} | Forecast up / down")
        if curve is not None:
            frame[f"{model} — invest when it says up"] = curve
    fig = line_chart(equity_frame(frame, result["capital"]), "Portfolio value")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption(
        "Each line is the growth of your money if you invested whenever that "
        "model said 'up' and sat in cash otherwise. A model only beats the gray "
        "buy & hold line if its up/down calls were good enough to overcome the "
        "days it sat out. The table below tries the other rules too."
    )
    table = result["combo_table"]
    if table.empty:
        st.info("No model/rule combinations were produced.")
        return
    best = table.iloc[0]
    bh = result["baseline_stats"]
    st.markdown(f"**Best combination: follow {best['Model']} with the rule “{best['Rule']}”**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Result", f"{best['Net return (%)']:+.2f}%", help=METRIC_HELP["Return"])
    c2.metric("Beat buy & hold by", f"{best['Vs buy & hold (pp)']:+.2f} pp")
    c3.metric("Sharpe ratio", "n/a" if pd.isna(best["Sharpe"]) else f"{best['Sharpe']:.2f}",
              help=METRIC_HELP["Sharpe"])
    c4.metric("Worst losing stretch", f"{best['Max drawdown (%)']:.2f}%",
              help=METRIC_HELP["Drawdown"])
    st.caption(
        f"For comparison, simply buying and holding: {bh['Net return (%)']:+.2f}% "
        f"growth, {bh['Max drawdown (%)']:.2f}% worst losing stretch."
    )

    st.markdown("**All model + rule combinations — best result first**")
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
        "Reading tips: the naive baseline guesses 'tomorrow = yesterday', so the "
        "buy-dips rule — which needs yesterday down AND a positive forecast — can "
        "never fire for it. Its 0.00% result is by logic, not failure. And the "
        "best of many combinations is partly luck, so treat the winner as a hint."
    )
    st.download_button(
        "Download results CSV",
        table.to_csv(index=False),
        file_name="strategy_results.csv",
        mime="text/csv",
    )

    combo = st.selectbox(
        "Combination to chart", table["Combination"].tolist(), key="combo_chart"
    )
    frame = equity_frame(
        {combo: result["combo_curves"][combo], "Buy & hold": result["baseline_net"]},
        result["capital"],
    )
    fig = line_chart(frame, "Portfolio value")
    fig.update_traces(line=dict(width=3), selector=dict(name=combo))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption(
        f"Growth of ${result['capital']:,.0f} on the test window: the chosen "
        "combination versus simply buying and holding."
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


def sidebar_tickers():
    """Ticker choice: typed, random S&P 500 sample, or sector-diversified."""
    st.subheader("1. Your stocks")
    mode = st.radio(
        "Choose stocks by",
        ["Type tickers", "Random from the S&P 500", "Diversified from the S&P 500"],
    )
    if mode == "Type tickers":
        text = st.text_input(
            "Tickers (comma separated)", "AAPL, MSFT, GOOGL, AMZN", key="typed_tickers"
        )
        return tuple(dict.fromkeys(t.strip().upper() for t in text.split(",") if t.strip()))

    try:
        sp500 = fetch_sp500()
    except Exception as exc:
        st.warning(f"Could not load the S&P 500 list: {exc}. Type tickers instead.")
        return ()
    count = st.number_input("How many stocks", 2, 50, 5)
    if st.button("Shuffle selection"):
        st.session_state["random_sample"] = {
            "mode": mode, "count": count, "tickers": sample_sp500(sp500, count, mode),
        }
    state = st.session_state.get("random_sample")
    if state is None or state["mode"] != mode or state["count"] != count:
        state = {"mode": mode, "count": count, "tickers": sample_sp500(sp500, count, mode)}
        st.session_state["random_sample"] = state
    note = ", spread across 5 sectors" if mode.startswith("Diversified") else ""
    st.caption(f"Picked for you{note}: " + ", ".join(state["tickers"]))
    return tuple(state["tickers"])


def main():
    st.set_page_config(page_title="Portfolio + Forecast Backtester",
                       page_icon="capybara.png", layout="wide")
    st.title("Portfolio + Forecast Backtester")
    st.caption(
        "Pick stocks, see how different portfolio splits would have done, test "
        "whether simple forecasts could have beaten buy & hold."
    )

    with st.sidebar:
        tickers = sidebar_tickers()

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
                prices = fetch_closes(tickers, start, end)
                prices = prices.dropna(axis=1, how="all").dropna(how="any")
                if prices.empty or len(prices) < 60:
                    raise ValueError(
                        "not enough overlapping price history; try more tickers or a wider date range"
                    )
                result = run_cached_backtest(
                    prices,
                    tickers,
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
                st.session_state["result"] = result
        except Exception as exc:
            st.error(f"Could not complete the backtest: {exc}")
            return

    if "result" not in st.session_state:
        st.info("Set up the backtest in the sidebar, then click **Run backtest**.")
        return

    result = st.session_state["result"]
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
