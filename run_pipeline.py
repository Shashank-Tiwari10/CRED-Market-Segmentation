"""
CRED Market Entry and Segmentation Diagnostic - Master Pipeline
Run: python run_pipeline.py [--quick] [--skip-generation] [--trials N]

Stages
------
1. Synthetic data generation      (src/data_generator.py)
2. Cleaning, dedup, features      (src/preprocessing.py)
3. k-Means + hierarchical         (src/segmentation.py)
4. ARIMA / SARIMA forecasting     (src/time_series.py)
5. XGBoost propensity + SHAP      (src/propensity.py)
6. Monte Carlo sensitivity        (src/monte_carlo.py)
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Dict

import pandas as pd

import config
from src import visualization as viz
from src.data_generator import generate_credit_card_data, load_raw
from src.monte_carlo import run_auc_stability, run_cluster_stability, run_dedup_sensitivity
from src.preprocessing import cascading_filter, clean_data, engineer_features
from src.propensity import (
    beachhead_sizing,
    create_labels,
    decile_analysis,
    evaluate_model,
    make_feature_matrix,
    shap_analysis,
    split_data,
    train_baseline,
    train_xgboost_optuna,
)
from src.segmentation import (
    cluster_profiles,
    compute_jaccard,
    label_clusters,
    run_hierarchical,
    run_kmeans,
)
from src.time_series import build_cluster_series, evaluate_forecasts, summarise
from src.utils import fmt_millions, pass_fail, save_json, setup_logging, timer
from src.visualization import generate_all_plots

log = logging.getLogger("cred.pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the CRED segmentation pipeline end-to-end.")
    p.add_argument("--quick", action="store_true", help="smoke-test mode: fewer rows, trials and MC runs")
    p.add_argument("--skip-generation", action="store_true", help="reuse data/raw if it exists")
    p.add_argument("--trials", type=int, default=None, help="override Optuna trial count")
    p.add_argument("--debug", action="store_true", help="DEBUG-level logging")
    return p.parse_args()


def main() -> Dict[str, object]:
    args = parse_args()
    setup_logging(logging.DEBUG if args.debug else logging.INFO, log_file=config.OUTPUT_DIR / "pipeline.log")
    t0 = time.perf_counter()
    n_cards = 40_000 if args.quick else config.N_CARDS
    trials = args.trials or (10 if args.quick else config.OPTUNA_TRIALS)
    mc_dedup = 200 if args.quick else config.MC_DEDUP_RUNS
    mc_cluster = 10 if args.quick else config.MC_CLUSTER_RUNS
    mc_auc = 10 if args.quick else config.MC_AUC_RUNS
    timings: Dict[str, float] = {}

    log.info("=" * 60)
    log.info("CRED MARKET ENTRY & SEGMENTATION DIAGNOSTIC - PIPELINE START")
    log.info("sample = %s cards (represents %sM cards / %sM holders)", f"{n_cards:,}",
             config.REAL_MARKET_CARDS // 10**6, config.REAL_MARKET_HOLDERS // 10**6)
    log.info("=" * 60)

    # ------------------------------------------------------------------ 1
    with timer("Stage 1 - synthetic data generation", log) as t:
        if args.skip_generation and config.RAW_CARDS_FILE.exists() and config.RAW_PANEL_FILE.exists():
            cards, panel = load_raw()
            log.info("Loaded existing raw data (%s cards)", f"{len(cards):,}")
        else:
            cards, panel = generate_credit_card_data(n_cards=n_cards)
    timings["1_generation"] = t["seconds"]

    # ------------------------------------------------------------------ 2
    with timer("Stage 2 - cleaning, cascading dedup, feature engineering", log) as t:
        clean, purity_report = clean_data(cards)
        holders, filter_log = cascading_filter(clean)
        features = engineer_features(holders, panel)
        holders.to_csv(config.HOLDERS_FILE, index=False)
        features.to_csv(config.FEATURES_FILE, index=False)
        filter_log.to_csv(config.REPORTS_DIR / "dedup_filter_log.csv", index=False)
        pd.DataFrame([{k: v for k, v in purity_report.items() if k != "imputed_columns"}]).to_csv(
            config.REPORTS_DIR / "data_purity.csv", index=False)
    timings["2_preprocessing"] = t["seconds"]

    # ------------------------------------------------------------------ 3
    with timer("Stage 3 - k-Means segmentation + hierarchical validation", log) as t:
        clustered, km, X, sweep, silhouette = run_kmeans(features)
        names = label_clusters(clustered)
        clustered["cluster_name"] = clustered["cluster"].map(names)
        profiles = cluster_profiles(clustered)
        profiles.insert(0, "cluster_name", [names[c] for c in profiles.index])
        hier_idx, hier_labels, Z = run_hierarchical(X)
        jaccard, jmat, _ = compute_jaccard(clustered["cluster"].to_numpy()[hier_idx], hier_labels)
        best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
        log.info("k-Means vs Ward weighted Jaccard = %.3f (target >= %.2f)", jaccard, config.MIN_JACCARD)
        clustered.to_csv(config.CLUSTERED_FILE, index=False)
        profiles.to_csv(config.REPORTS_DIR / "cluster_profiles.csv")
        sweep.to_csv(config.REPORTS_DIR / "kmeans_k_sweep.csv", index=False)
        jmat.to_csv(config.REPORTS_DIR / "jaccard_stability.csv")
        labels = clustered["cluster"].to_numpy()
        viz.plot_elbow(sweep)
        viz.plot_silhouette(X, labels, silhouette)
        viz.plot_dendrogram(Z)
        viz.plot_pca_scatter(X, labels, names)
        viz.plot_tsne_scatter(X, labels, names)
        viz.plot_radar(profiles, names)
        viz.plot_cluster_heatmap(profiles, names)
        viz.plot_cluster_distribution(clustered, names)
    timings["3_segmentation"] = t["seconds"]

    # ------------------------------------------------------------------ 4
    with timer("Stage 4 - ARIMA / SARIMA forecasting", log) as t:
        series_df = build_cluster_series(clustered, panel)
        ts_results, ts_metrics = evaluate_forecasts(series_df)
        ts_summary = summarise(ts_metrics)
        series_df.to_csv(config.PROCESSED_DIR / "monthly_series_by_cluster.csv")
        ts_metrics.to_csv(config.REPORTS_DIR / "time_series_metrics.csv", index=False)
        pd.DataFrame({f"{c}_sarima_forecast": ts_results[c]["sarima"]["future"] for c in series_df.columns}).to_csv(
            config.REPORTS_DIR / "sarima_6m_forecasts.csv")
        viz.plot_overall_forecast(series_df["Overall"], ts_results["Overall"])
        viz.plot_per_cluster_forecast(series_df, ts_results, names)
        viz.plot_arima_vs_sarima(ts_metrics)
        viz.plot_residuals(ts_results["Overall"]["sarima"]["residuals"])
        viz.plot_seasonal_decomposition(series_df["Overall"])
    timings["4_time_series"] = t["seconds"]

    # ------------------------------------------------------------------ 5
    with timer(f"Stage 5 - XGBoost propensity model (Optuna {trials} trials) + SHAP", log) as t:
        y, label_info = create_labels(features)
        Xp = make_feature_matrix(features)
        X_train, X_test, y_train, y_test = split_data(Xp, y)
        baseline = train_baseline(X_train, y_train)
        base_metrics = evaluate_model(baseline, X_test, y_test)
        log.info("Baseline logistic regression AUC = %.4f", base_metrics["roc_auc"])
        model, study, best_params = train_xgboost_optuna(X_train, y_train, n_trials=trials)
        prop_metrics = evaluate_model(model, X_test, y_test)
        log.info("XGBoost test AUC = %.4f | PR-AUC = %.4f | F1 = %.3f (target AUC >= %.2f)",
                 prop_metrics["roc_auc"], prop_metrics["pr_auc"], prop_metrics["f1"], config.MIN_AUC)
        proba = model.predict_proba(X_test)[:, 1]
        base_proba = baseline.predict_proba(X_test)[:, 1]
        deciles = decile_analysis(y_test.to_numpy(), proba)
        beachhead = beachhead_sizing(deciles, n_holders=len(features))
        explainer, shap_values, X_shap, top3, shap_imp = shap_analysis(model, X_test)

        import joblib

        joblib.dump(model, config.MODELS_DIR / "xgboost_propensity.pkl")
        joblib.dump(study, config.MODELS_DIR / "optuna_study.pkl")
        joblib.dump(explainer, config.MODELS_DIR / "shap_explainer.pkl")
        joblib.dump(baseline, config.MODELS_DIR / "logistic_baseline.pkl")
        joblib.dump(km, config.MODELS_DIR / "kmeans_k6.pkl")
        deciles.to_csv(config.REPORTS_DIR / "decile_analysis.csv", index=False)
        beachhead.to_csv(config.REPORTS_DIR / "beachhead_sizing.csv", index=False)
        pd.DataFrame([{"model": "LogisticRegression", **base_metrics}, {"model": "XGBoost", **prop_metrics}]).to_csv(
            config.REPORTS_DIR / "propensity_metrics.csv", index=False)
        shap_imp.rename("mean_abs_shap").to_csv(config.REPORTS_DIR / "shap_feature_importance.csv")
        save_json({**best_params, **label_info}, config.REPORTS_DIR / "xgboost_best_params.json")
        scored = clustered[["holder_id", "cluster", "cluster_name"]].copy()
        scored["cred_adopter"] = y.values
        scored["propensity_score"] = model.predict_proba(Xp)[:, 1]
        scored.to_csv(config.PROCESSED_DIR / "holder_propensity_scores.csv", index=False)

        viz.plot_roc(y_test.to_numpy(), {"XGBoost": proba, "Logistic baseline": base_proba})
        viz.plot_pr(y_test.to_numpy(), {"XGBoost": proba, "Logistic baseline": base_proba})
        viz.plot_confusion(y_test.to_numpy(), (proba >= prop_metrics["threshold"]).astype(int), prop_metrics["threshold"])
        viz.plot_shap_summary(shap_values, X_shap)
        viz.plot_shap_importance(shap_values, X_shap)
        viz.plot_shap_dependence(shap_values, X_shap, top3)
        viz.plot_decile_lift(deciles)
        viz.plot_xgb_importance(model, list(Xp.columns))
        viz.plot_optuna_history(study)
    timings["5_propensity"] = t["seconds"]

    # ------------------------------------------------------------------ 6
    with timer(f"Stage 6 - Monte Carlo sensitivity ({mc_dedup}/{mc_cluster}/{mc_auc} runs)", log) as t:
        dedup = run_dedup_sensitivity(clean, n_runs=mc_dedup)
        stability = run_cluster_stability(X, labels, n_runs=mc_cluster)
        aucs = run_auc_stability(X_train, y_train, X_test, y_test, best_params, n_runs=mc_auc)
        dedup["runs"].to_csv(config.REPORTS_DIR / "mc_dedup_runs.csv", index=False)
        pd.DataFrame({"jaccard": stability["values"]}).to_csv(config.REPORTS_DIR / "mc_cluster_stability_runs.csv", index=False)
        pd.DataFrame({"test_auc": aucs["values"]}).to_csv(config.REPORTS_DIR / "mc_auc_runs.csv", index=False)
        pd.DataFrame([
            {"analysis": "dedup_holder_count", **dedup["summary"]},
            {"analysis": "cluster_stability_jaccard", **stability["summary"]},
            {"analysis": "propensity_auc", **aucs["summary"]},
        ]).to_csv(config.REPORTS_DIR / "monte_carlo_summary.csv", index=False)
        viz.plot_mc_holder_dist(dedup["counts_m"], (dedup["summary"]["ci95_low_M"], dedup["summary"]["ci95_high_M"]))
        viz.plot_mc_jaccard_dist(stability["values"])
        viz.plot_mc_auc_dist(aucs["values"])
        viz.plot_mc_convergence(dedup["counts_m"])
    timings["6_monte_carlo"] = t["seconds"]

    # ------------------------------------------------------------------ summary
    metrics = {
        "sample_cards": int(len(cards)),
        "sample_holders_after_dedup": int(len(features)),
        "purity": float(purity_report["purity"]),
        "best_k": best_k,
        "silhouette": float(silhouette),
        "jaccard": float(jaccard),
        "sarima_mape": ts_summary["sarima_mape_overall"],
        "arima_mape": ts_summary["arima_mape_overall"],
        "sarima_mape_cluster_avg": ts_summary["sarima_mape_cluster_avg"],
        "arima_mape_cluster_avg": ts_summary["arima_mape_cluster_avg"],
        "sarima_advantage_pp": ts_summary["sarima_advantage_pp"],
        "auc": prop_metrics["roc_auc"],
        "pr_auc": prop_metrics["pr_auc"],
        "baseline_auc": base_metrics["roc_auc"],
        "positive_rate": label_info["final_positive_rate"],
        "top_decile_lift": float(deciles.loc[0, "lift"]),
        "top_decile_capture": float(deciles.loc[0, "capture"]),
        "beachhead_core_M": float(beachhead.loc[0, "market_holders_M"]),
        "beachhead_total_M": float(beachhead.loc[2, "market_holders_M"]),
        "mc_holder_ci_low_M": dedup["summary"]["ci95_low_M"],
        "mc_holder_ci_high_M": dedup["summary"]["ci95_high_M"],
        "mc_holder_mean_M": dedup["summary"]["mean_M"],
        "mc_jaccard_mean": stability["summary"]["mean"],
        "mc_auc_mean": aucs["summary"]["mean"],
        "mc_auc_ci_low": aucs["summary"]["ci95_low"],
        "mc_auc_ci_high": aucs["summary"]["ci95_high"],
        "cluster_names": names,
        "timings_seconds": timings,
        "total_seconds": time.perf_counter() - t0,
        "quick_mode": bool(args.quick),
    }
    missing = generate_all_plots(metrics)
    metrics["missing_figures"] = missing
    save_json(metrics, config.REPORTS_DIR / "final_metrics.json")
    print_summary(metrics)
    return metrics


def print_summary(m: Dict[str, object]) -> None:
    lines = [
        "=" * 60,
        "CRED PIPELINE - FINAL METRICS SUMMARY",
        "=" * 60,
        f"Data Purity:              {m['purity'] * 100:5.1f}% (target: >=92%)    {pass_fail(m['purity'] >= config.MIN_PURITY)}",
        f"Optimal Clusters:         k={m['best_k']}  (target: 6)         {pass_fail(m['best_k'] == config.TARGET_K)}",
        f"Silhouette Score:         {m['silhouette']:.2f} (target: ~0.42)      {pass_fail(abs(m['silhouette'] - config.TARGET_SILHOUETTE) <= 0.03)}",
        f"Jaccard Stability:        {m['jaccard']:.2f} (target: >=0.75)      {pass_fail(m['jaccard'] >= config.MIN_JACCARD)}",
        f"SARIMA MAPE:              {m['sarima_mape'] * 100:4.1f}% (target: <12%)      {pass_fail(m['sarima_mape'] < config.MAX_MAPE)}",
        f"ARIMA MAPE:               {m['arima_mape'] * 100:4.1f}% (target: <12%)      {pass_fail(m['arima_mape'] < config.MAX_MAPE)}",
        f"XGBoost AUC:              {m['auc']:.2f} (target: >=0.80)      {pass_fail(m['auc'] >= config.MIN_AUC)}",
        f"Top Decile Lift:          {m['top_decile_lift']:.1f}x                       [INFO]",
        f"Beachhead Size:           {m['beachhead_core_M']:.1f}M core / {m['beachhead_total_M']:.1f}M total  [INFO]",
        f"MC Holder 95% CI:         {m['mc_holder_ci_low_M']:.1f}M - {m['mc_holder_ci_high_M']:.1f}M              [INFO]",
        f"MC Jaccard Mean:          {m['mc_jaccard_mean']:.2f}                       [INFO]",
        f"MC AUC Mean:              {m['mc_auc_mean']:.3f} (95% CI {m['mc_auc_ci_low']:.3f}-{m['mc_auc_ci_high']:.3f})  [INFO]",
        f"Total runtime:            {m['total_seconds'] / 60:.1f} min",
        "=" * 60,
    ]
    text = "\n".join(lines)
    print(text)
    (config.REPORTS_DIR / "pipeline_summary.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
