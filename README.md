# Portfolio + Forecast Backtester

A Streamlit app built for intro-to-finance classrooms: pick stocks, compare
classic portfolio-optimization strategies, and test whether simple ML forecasts
could have beaten buy & hold — all with strict train/test separation.

## What it does

1. **Portfolio strategies** — compares Equal-Weight (1/N), Global Minimum
   Variance (GMVP), Maximum Sharpe Ratio (tangency portfolio) and Hierarchical
   Risk Parity (HRP) on a held-out test window, then applies the winner's
   weights everywhere else. Includes the average correlation between held
   stocks (train vs test) so students can see whether diversification actually
   held up.
2. **Forecast accuracy** — models guess each stock's next-day return from the
   last five days of returns: Linear Regression, yesterday's-return baseline,
   XGBoost and a Neural Network (Ridge/Lasso appear under "Advanced"). Scores
   report mean absolute error (MAE).
3. **Trading rules** — turns forecasts into invest/hold/bet-against rules
   (with trading costs) and ranks every model + rule combination against
   buy & hold.

Stocks can be typed, sampled randomly from the S&P 500, or diversified across
5 random sectors (~n/5 each). Downloads, the S&P 500 list and full pipeline
runs are cached so repeated runs with the same setup are instant.

## Run it

```bash
cd Backtester
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Linux/macOS: .venv/bin/pip
.venv\Scripts\python test_app.py                # no-network self-checks
.venv\Scripts\streamlit run app.py
```

## Layout

| File | Role |
| --- | --- |
| `app.py` | Streamlit UI (sidebar inputs, tabs, charts, S&P 500 picker) |
| `engine.py` | Orchestrates the no-future-data pipeline |
| `forecasting.py` | Models, baseline, prediction clipping, trading rules |
| `portfolio.py` | The four portfolio-optimization algorithms |
| `test_app.py` | Runnable self-checks (`python test_app.py`) |

## Honesty notes (worth teaching)

- Portfolio weights and every model are fitted on the training window only;
  each forecast for a day uses only returns from before that day.
- The highlighted portfolio and winning model + rule are chosen by their
  test-window results — an in-sample choice on the test window, so treat the
  winner as a hint, not a promise.
- Costs, short-borrow, margin interest and taxes are simplified or excluded.

The sibling folders `StonkForecaster/` and `PortfolioOptimization/` are
independent projects with their own git history and are intentionally ignored
by this repository.
