"""
Stage 6 - Monte Carlo sensitivity analysis.

1. Dedup sensitivity (1,000 runs): jitter the cascading-filter thresholds by
   +/-5% and record the resulting addressable-holder count.
2. Cluster stability (100 runs): re-run k-Means on 80% subsamples and
   measure Jaccard agreement with the full-data clustering.
3. Propensity stability (100 runs): bootstrap the training set, refit XGBoost
   with the tuned hyper-parameters and record the test AUC.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

import config
from src.data_generator import TOP_50_CITY_NAMES
from src.preprocessing import aggregate_to_holders
from src.segmentation import compute_jaccard
from src.utils import get_logger

log = get_logger("monte_carlo")


# --------------------------------------------------------------------------- #
# 1. Dedup sensitivity
# --------------------------------------------------------------------------- #
def run_dedup_sensitivity(
    clean_cards: pd.DataFrame,
    n_runs: int = config.MC_DEDUP_RUNS,
    seed: int = config.SEED,
    jitter: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    Vary the three numeric thresholds uniformly within +/- jitter and count the
    holders surviving the cascade.  Counts are scaled to the national market
    using the baseline (un-jittered) result as the 38M anchor.
    """
    rng = np.random.default_rng(seed)
    jt = {**config.MC_THRESHOLD_JITTER, **(jitter or {})}
    base = config.FILTER_THRESHOLDS
    geo = clean_cards.loc[clean_cards["city"].isin(TOP_50_CITY_NAMES)]
    holders = aggregate_to_holders(geo)
    score = holders["credit_score"].to_numpy()
    spend = holders["avg_monthly_spend"].to_numpy()
    repay = holders["avg_monthly_repayment_pct"].to_numpy()

    def count(th_score: float, th_spend: float, th_repay: float) -> int:
        return int(((score >= th_score) & (spend >= th_spend) & (repay >= th_repay)).sum())

    baseline = count(base["min_credit_score"], base["min_monthly_spend"], base["min_repayment_pct"])
    scale = config.REAL_MARKET_HOLDERS / baseline

    draws = np.column_stack(
        [
            rng.uniform(base["min_credit_score"] - jt["min_credit_score"], base["min_credit_score"] + jt["min_credit_score"], n_runs),
            rng.uniform(base["min_monthly_spend"] - jt["min_monthly_spend"], base["min_monthly_spend"] + jt["min_monthly_spend"], n_runs),
            rng.uniform(base["min_repayment_pct"] - jt["min_repayment_pct"], base["min_repayment_pct"] + jt["min_repayment_pct"], n_runs),
        ]
    )
    counts = np.array([count(*row) for row in draws])
    counts_m = counts * scale / 1e6
    ci = np.percentile(counts_m, [2.5, 97.5])
    summary = {
        "runs": n_runs, "baseline_sample_holders": baseline, "scale_factor": scale,
        "mean_M": float(counts_m.mean()), "std_M": float(counts_m.std(ddof=1)), "median_M": float(np.median(counts_m)),
        "ci95_low_M": float(ci[0]), "ci95_high_M": float(ci[1]), "min_M": float(counts_m.min()), "max_M": float(counts_m.max()),
    }
    log.info("Dedup MC (%d runs): mean %.2fM | 95%% CI %.2fM - %.2fM | std %.2fM",
             n_runs, summary["mean_M"], ci[0], ci[1], summary["std_M"])
    runs = pd.DataFrame(draws, columns=["min_credit_score", "min_monthly_spend", "min_repayment_pct"])
    runs["sample_holders"] = counts
    runs["market_holders_M"] = counts_m
    return {"summary": summary, "counts_m": counts_m, "runs": runs}


# --------------------------------------------------------------------------- #
# 2. Cluster stability
# --------------------------------------------------------------------------- #
def run_cluster_stability(
    X: np.ndarray,
    full_labels: np.ndarray,
    n_runs: int = config.MC_CLUSTER_RUNS,
    fraction: float = config.MC_BOOTSTRAP_FRACTION,
    k: int = config.TARGET_K,
    seed: int = config.SEED,
) -> Dict[str, object]:
    """Bootstrap 80% of rows, re-cluster, and compare with the full-data labels via Jaccard."""
    rng = np.random.default_rng(seed)
    values = []
    for i in range(n_runs):
        idx = rng.choice(len(X), size=int(len(X) * fraction), replace=False)
        labels = KMeans(n_clusters=k, n_init=3, random_state=int(rng.integers(0, 2**31 - 1))).fit_predict(X[idx])
        jac, _, _ = compute_jaccard(full_labels[idx], labels)
        values.append(jac)
        if (i + 1) % 25 == 0:
            log.debug("cluster stability run %d/%d: running mean Jaccard %.3f", i + 1, n_runs, np.mean(values))
    values = np.array(values)
    summary = {"runs": n_runs, "mean": float(values.mean()), "std": float(values.std(ddof=1)),
               "min": float(values.min()), "max": float(values.max())}
    log.info("Cluster stability (%d runs): Jaccard mean %.3f | std %.3f | min %.3f | max %.3f (target >= %.2f)",
             n_runs, summary["mean"], summary["std"], summary["min"], summary["max"], config.MIN_JACCARD)
    return {"summary": summary, "values": values}


# --------------------------------------------------------------------------- #
# 3. Propensity model stability
# --------------------------------------------------------------------------- #
def run_auc_stability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_params: Dict[str, object],
    n_runs: int = config.MC_AUC_RUNS,
    fraction: float = config.MC_BOOTSTRAP_FRACTION,
    seed: int = config.SEED,
) -> Dict[str, object]:
    """Resample 80% of the training rows with replacement, refit XGBoost, record test AUC."""
    rng = np.random.default_rng(seed)
    values = []
    for i in range(n_runs):
        idx = rng.choice(len(X_train), size=int(len(X_train) * fraction), replace=True)
        model = xgb.XGBClassifier(
            **best_params, objective="binary:logistic", tree_method="hist",
            random_state=int(rng.integers(0, 2**31 - 1)), n_jobs=-1, verbosity=0,
        )
        model.fit(X_train.iloc[idx], y_train.iloc[idx], verbose=False)
        values.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
        if (i + 1) % 25 == 0:
            log.debug("AUC stability run %d/%d: running mean %.4f", i + 1, n_runs, np.mean(values))
    values = np.array(values)
    ci = np.percentile(values, [2.5, 97.5])
    summary = {"runs": n_runs, "mean": float(values.mean()), "std": float(values.std(ddof=1)),
               "ci95_low": float(ci[0]), "ci95_high": float(ci[1])}
    log.info("AUC stability (%d runs): mean %.4f | std %.4f | 95%% CI %.4f - %.4f",
             n_runs, summary["mean"], summary["std"], ci[0], ci[1])
    return {"summary": summary, "values": values}


def run_monte_carlo(
    clean_cards: pd.DataFrame, X: np.ndarray, labels: np.ndarray, prop: Dict[str, object],
    dedup_runs: int = config.MC_DEDUP_RUNS, cluster_runs: int = config.MC_CLUSTER_RUNS, auc_runs: int = config.MC_AUC_RUNS,
    save: bool = True,
) -> Dict[str, object]:
    """Full Stage-6 driver."""
    dedup = run_dedup_sensitivity(clean_cards, n_runs=dedup_runs)
    cluster = run_cluster_stability(X, labels, n_runs=cluster_runs)
    aucs = run_auc_stability(prop["X_train"], prop["y_train"], prop["X_test"], prop["y_test"], prop["best_params"], n_runs=auc_runs)
    if save:
        dedup["runs"].to_csv(config.REPORTS_DIR / "mc_dedup_runs.csv", index=False)
        pd.DataFrame({"jaccard": cluster["values"]}).to_csv(config.REPORTS_DIR / "mc_cluster_stability_runs.csv", index=False)
        pd.DataFrame({"test_auc": aucs["values"]}).to_csv(config.REPORTS_DIR / "mc_auc_runs.csv", index=False)
        pd.DataFrame(
            [{"analysis": "dedup_holder_count", **dedup["summary"]},
             {"analysis": "cluster_stability_jaccard", **cluster["summary"]},
             {"analysis": "propensity_auc", **aucs["summary"]}]
        ).to_csv(config.REPORTS_DIR / "monte_carlo_summary.csv", index=False)
    return {"dedup": dedup, "cluster": cluster, "auc": aucs}
