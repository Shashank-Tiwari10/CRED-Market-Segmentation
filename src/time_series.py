"""
Stage 4 - Time-series forecasting of monthly spend per segment.

For every series (6 clusters + overall market) we fit a non-seasonal ARIMA
and a seasonal SARIMA (m = 12) via an AIC-selected grid search, evaluate on
the last 3 months, then refit on the full history to forecast 6 months ahead.

Models are fitted on log(spend) so that seasonality and growth are
multiplicative, matching how the panel was generated.  With only 21 training
points the selection uses the small-sample corrected AICc, stationary
initialisation (so every candidate's likelihood spans the whole sample) and
a cap on the total number of ARMA terms; the 3-month holdout MAPE is the
independent check on the selected models.
"""
from __future__ import annotations

import itertools
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

import config
from src.utils import get_logger, mae, mape, rmse

log = get_logger("time_series")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

Order = Tuple[int, int, int]
SeasonalOrder = Tuple[int, int, int, int]


# --------------------------------------------------------------------------- #
# Series construction
# --------------------------------------------------------------------------- #
def build_cluster_series(clustered: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the holder x month panel into one monthly spend series per
    cluster plus an ``Overall`` column.  Index is a monthly DatetimeIndex.
    """
    merged = panel.merge(clustered[["holder_id", "cluster"]], on="holder_id", how="inner")
    merged["timestamp_month"] = pd.to_datetime(merged["timestamp_month"])
    piv = merged.pivot_table(index="timestamp_month", columns="cluster", values="monthly_spend", aggfunc="sum")
    piv.columns = [f"cluster_{c}" for c in piv.columns]
    piv["Overall"] = piv.sum(axis=1)
    piv.index = pd.DatetimeIndex(piv.index, freq="MS")
    log.info("Built %d monthly series x %d months", piv.shape[1], piv.shape[0])
    return piv


# --------------------------------------------------------------------------- #
# Model fitting
# --------------------------------------------------------------------------- #
def _fit_sarimax(y: pd.Series, order: Order, seasonal_order: SeasonalOrder = (0, 0, 0, 0)):
    """
    Fit a (S)ARIMA on the log scale (multiplicative seasonality / growth).

    ``trend="c"`` is always included: for an undifferenced model it is the
    level, for a differenced model it becomes the drift term that carries the
    ~15% YoY growth of Indian card spend into the forecast.
    """
    model = SARIMAX(
        np.log(y), order=order, seasonal_order=seasonal_order, trend="c",
        enforce_stationarity=config.TS_ENFORCE_STATIONARITY,
        enforce_invertibility=config.TS_ENFORCE_STATIONARITY,
    )
    return model.fit(disp=False, maxiter=300)


def _criterion(fit) -> float:
    """Information criterion used for model selection (see config.TS_SELECTION_CRITERION)."""
    crit = config.TS_SELECTION_CRITERION.lower()
    if crit == "aicc":
        return float(fit.aicc)
    if crit == "bic":
        return float(fit.bic)
    return float(fit.aic)


def _grid_search(
    y: pd.Series, orders: List[Order], seasonal_orders: List[SeasonalOrder]
) -> Tuple[Optional[object], Order, SeasonalOrder, float]:
    """Return the (fit, order, seasonal_order, criterion) with the lowest information criterion."""
    best = (None, (0, 0, 0), (0, 0, 0, 0), np.inf)
    for order, sorder in itertools.product(orders, seasonal_orders):
        if order == (0, 0, 0) and sorder[:3] == (0, 0, 0):
            continue
        if order[0] + order[2] + sorder[0] + sorder[2] > config.TS_MAX_ARMA_TERMS:
            continue
        try:
            fit = _fit_sarimax(y, order, sorder)
            aic = _criterion(fit)
            if np.isfinite(aic) and aic < best[3]:
                best = (fit, order, sorder, aic)
        except Exception as exc:  # pragma: no cover - numerical edge cases
            log.debug("order %s x %s failed: %s", order, sorder, exc)
    return best


def fit_arima(y: pd.Series, grid: Optional[Dict[str, List[int]]] = None) -> Dict[str, object]:
    """AIC-selected non-seasonal ARIMA over p in {0,1,2}, d in {0,1}, q in {0,1,2}."""
    g = grid or config.ARIMA_GRID
    orders = [(p, d, q) for p in g["p"] for d in g["d"] for q in g["q"]]
    fit, order, _, aic = _grid_search(y, orders, [(0, 0, 0, 0)])
    log.debug("ARIMA best order=%s AIC=%.1f", order, aic)
    return {"fit": fit, "order": order, "seasonal_order": (0, 0, 0, 0), "aic": aic}


def fit_sarima(
    y: pd.Series, grid: Optional[Dict[str, List[int]]] = None, seasonal_grid: Optional[Dict[str, object]] = None
) -> Dict[str, object]:
    """AIC-selected SARIMA: same (p,d,q) grid x (P,D,Q) in {0,1}^3 with m = 12."""
    g = grid or config.ARIMA_GRID
    sg = seasonal_grid or config.SARIMA_SEASONAL_GRID
    orders = [(p, d, q) for p in g["p"] for d in g["d"] for q in g["q"]]
    sorders = [(P, D, Q, sg["m"]) for P in sg["P"] for D in sg["D"] for Q in sg["Q"]]
    fit, order, sorder, aic = _grid_search(y, orders, sorders)
    log.debug("SARIMA best order=%s seasonal=%s AIC=%.1f", order, sorder, aic)
    return {"fit": fit, "order": order, "seasonal_order": sorder, "aic": aic}


def _forecast(fit, steps: int, index: pd.DatetimeIndex) -> Tuple[pd.Series, pd.DataFrame]:
    """Forecast on the log scale and back-transform to rupees."""
    fc = fit.get_forecast(steps=steps)
    mean = pd.Series(np.exp(np.asarray(fc.predicted_mean)), index=index)
    ci = pd.DataFrame(np.exp(np.asarray(fc.conf_int(alpha=0.05))), index=index, columns=["lower", "upper"])
    return mean, ci


def evaluate_series(
    y: pd.Series, holdout: int = config.TS_HOLDOUT_MONTHS, horizon: int = config.TS_FORECAST_HORIZON
) -> Dict[str, object]:
    """
    Train on all but the last ``holdout`` months, score both models on the
    holdout, then refit on the full series for a ``horizon``-month forecast.
    """
    train, test = y.iloc[:-holdout], y.iloc[-holdout:]
    future_idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    out: Dict[str, object] = {"holdout_start": test.index[0]}
    for name, fitter in (("arima", fit_arima), ("sarima", fit_sarima)):
        spec = fitter(train)
        pred, _ = _forecast(spec["fit"], holdout, test.index)
        full = _fit_sarimax(y, spec["order"], spec["seasonal_order"])
        future, future_ci = _forecast(full, horizon, future_idx)
        out[name] = {
            "order": spec["order"], "seasonal_order": spec["seasonal_order"], "aic": spec["aic"],
            "holdout_pred": pred, "future": future, "future_ci": future_ci,
            "mape": mape(test.values, pred.values), "rmse": rmse(test.values, pred.values),
            "mae": mae(test.values, pred.values),
            # residuals on the log scale, skipping the diffuse burn-in observations
            "residuals": pd.Series(np.asarray(full.resid), index=y.index).iloc[int(full.loglikelihood_burn):],
        }
    return out


def evaluate_forecasts(series_df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, object]], pd.DataFrame]:
    """Run ``evaluate_series`` for every column; return per-series results and a tidy metrics table."""
    results: Dict[str, Dict[str, object]] = {}
    rows = []
    for col in series_df.columns:
        res = evaluate_series(series_df[col])
        results[col] = res
        for model in ("arima", "sarima"):
            r = res[model]
            rows.append(
                {
                    "series": col, "model": model.upper(), "order": str(r["order"]),
                    "seasonal_order": str(r["seasonal_order"]), "aic": round(r["aic"], 1),
                    "mape": r["mape"], "rmse": r["rmse"], "mae": r["mae"],
                }
            )
        log.info("%-10s ARIMA%s MAPE=%.1f%% | SARIMA%s%s MAPE=%.1f%%", col,
                 res["arima"]["order"], res["arima"]["mape"] * 100,
                 res["sarima"]["order"], res["sarima"]["seasonal_order"], res["sarima"]["mape"] * 100)
    metrics = pd.DataFrame(rows)
    return results, metrics


def summarise(metrics: pd.DataFrame) -> Dict[str, float]:
    """Headline numbers: overall-market MAPE for both models plus cluster averages."""
    overall = metrics[metrics["series"] == "Overall"].set_index("model")
    clusters = metrics[metrics["series"] != "Overall"].groupby("model")["mape"].mean()
    return {
        "sarima_mape_overall": float(overall.loc["SARIMA", "mape"]),
        "arima_mape_overall": float(overall.loc["ARIMA", "mape"]),
        "sarima_mape_cluster_avg": float(clusters["SARIMA"]),
        "arima_mape_cluster_avg": float(clusters["ARIMA"]),
        "sarima_advantage_pp": float((overall.loc["ARIMA", "mape"] - overall.loc["SARIMA", "mape"]) * 100),
    }


def run_time_series(clustered: pd.DataFrame, panel: pd.DataFrame, save: bool = True) -> Dict[str, object]:
    """Full Stage-4 driver."""
    series_df = build_cluster_series(clustered, panel)
    results, metrics = evaluate_forecasts(series_df)
    summary = summarise(metrics)
    log.info("Overall market: SARIMA MAPE %.1f%% vs ARIMA %.1f%% (SARIMA better by %.1f pp)",
             summary["sarima_mape_overall"] * 100, summary["arima_mape_overall"] * 100, summary["sarima_advantage_pp"])
    if save:
        series_df.to_csv(config.PROCESSED_DIR / "monthly_series_by_cluster.csv")
        metrics.to_csv(config.REPORTS_DIR / "time_series_metrics.csv", index=False)
        fc = pd.DataFrame({f"{c}_sarima_forecast": results[c]["sarima"]["future"] for c in series_df.columns})
        fc.to_csv(config.REPORTS_DIR / "sarima_6m_forecasts.csv")
    return {"series": series_df, "results": results, "metrics": metrics, "summary": summary}
