"""
Generate the six analysis notebooks programmatically so they always stay in
sync with the ``src`` modules.  Run once:  python notebooks/build_notebooks.py
Then execute them:  jupyter nbconvert --execute --inplace notebooks/0*.ipynb

Each notebook re-uses the artefacts written by ``run_pipeline.py`` where the
computation is heavy (Optuna, Monte Carlo) and recomputes the light parts
inline so a reader can follow the logic cell by cell.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = '''import sys, warnings, logging
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
import config
from src.utils import setup_logging
setup_logging(logging.INFO)
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 40)
FIG = config.FIGURES_DIR'''


def nb(title: str, cells: list) -> nbf.NotebookNode:
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\nCRED Market Entry and Segmentation Diagnostic - Shashank Tiwari, IIT Kharagpur")]
    n.cells.append(nbf.v4.new_code_cell(SETUP))
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return n


NOTEBOOKS = {
    "01_data_generation.ipynb": nb("01 - Synthetic Data Generation", [
        ("md", "## Why synthetic?\nNo public row-level Indian credit-card data exists. We synthesise a 200K-card sample whose "
               "marginals follow RBI / CIBIL / NPCI headline patterns (119M cards, ~3.1 cards per holder, 65/35 gender skew, "
               "Tier-1 concentration) and embed **six latent behavioural archetypes** so that segmentation has a recoverable structure."),
        ("code", "from src.data_generator import generate_credit_card_data, TOP_50_CITIES\n"
                 "cards, panel = generate_credit_card_data(n_cards=config.N_CARDS, save=False)\n"
                 "cards.head()"),
        ("md", "### Archetype design\nEvery holder is drawn from one of six archetypes; behavioural traits (repayment, UPI credit, "
               "reward redemption, digital share, EMI, delinquency) are holder-level and each card jitters around them."),
        ("code", "pd.DataFrame(config.ARCHETYPES).T[['share','income_mult','score_shift','repay','upi','reward','emi_prob','missed_prob','premium_prob','cards_lambda']]"),
        ("code", "fig, axes = plt.subplots(2, 3, figsize=(16, 8))\n"
                 "num = pd.to_numeric\n"
                 "num(cards['age'], errors='coerce').plot.hist(bins=40, ax=axes[0,0], title='Age')\n"
                 "num(cards['annual_income_lakhs'], errors='coerce').clip(upper=150).plot.hist(bins=40, ax=axes[0,1], title='Annual income (lakhs)')\n"
                 "cards['credit_score'].plot.hist(bins=40, ax=axes[0,2], title='Credit score')\n"
                 "cards['card_issuer'].value_counts().plot.bar(ax=axes[1,0], title='Issuer')\n"
                 "cards['city_tier'].value_counts().plot.bar(ax=axes[1,1], title='City tier')\n"
                 "np.log10(cards['avg_monthly_spend']).plot.hist(bins=40, ax=axes[1,2], title='log10 monthly spend per card')\n"
                 "plt.tight_layout()"),
        ("md", "### Seasonality in the 24-month panel\nDiwali (Oct-Nov) 1.35-1.45x, December 1.2x, January 0.8x, March 1.1x, plus ~15% YoY growth and a mildly persistent macro shock."),
        ("code", "s = panel.groupby('timestamp_month')['monthly_spend'].sum() / 1e6\n"
                 "ax = s.plot(figsize=(12, 4), marker='o', title='Aggregate monthly spend (INR million, sample)')\n"
                 "ax.set_ylabel('INR M')"),
        ("code", "print(f\"cards={len(cards):,} holders={cards['holder_id'].nunique():,} avg cards/holder={len(cards)/cards['holder_id'].nunique():.2f}\")\n"
                 "print('null share by column (injected noise):'); cards.isna().mean()[lambda s: s > 0].round(4)"),
    ]),
    "02_eda_and_cleaning.ipynb": nb("02 - EDA, Cleaning and the Cascading Market-Sizing Filter", [
        ("code", "from src.data_generator import load_raw\nfrom src.preprocessing import clean_data, cascading_filter, engineer_features\n"
                 "cards, panel = load_raw()\nclean, report = clean_data(cards)\nreport"),
        ("md", "## Data purity\n`purity = valid_rows / total_rows` must be >= 92%. Rows with out-of-range bureau scores, "
               "implausible incomes or more than three nulls are dropped; the remaining sparse nulls are median/mode imputed."),
        ("md", "## Cascading filter: 119M cards -> 38M holders\n1. Geography (top-50 cities)  2. Bureau score >= 600  3. Spend >= INR 2,000  4. Repayment >= 15%"),
        ("code", "holders, filter_log = cascading_filter(clean)\nfilter_log"),
        ("code", "ax = filter_log.set_index('stage')['holders_scaled_to_market_M'].plot.bar(figsize=(10,4), title='Holders remaining after each filter (scaled to market, M)')\n"
                 "ax.set_ylabel('M holders')"),
        ("md", "## Feature engineering (12 holder-level features)"),
        ("code", "features = engineer_features(holders, panel)\nfeatures[config.ENGINEERED_FEATURES].describe().T"),
        ("code", "import seaborn as sns\nplt.figure(figsize=(11, 8))\n"
                 "sns.heatmap(features[config.ENGINEERED_FEATURES].corr(), cmap='RdBu_r', center=0, annot=True, fmt='.2f', annot_kws={'size': 7})\n"
                 "plt.title('Feature correlation');"),
    ]),
    "03_segmentation.ipynb": nb("03 - ML Segmentation (k-Means + Ward validation)", [
        ("code", "from src.segmentation import run_kmeans, cluster_profiles, label_clusters, run_hierarchical, compute_jaccard\n"
                 "features = pd.read_csv(config.FEATURES_FILE)\n"
                 "clustered, km, X, sweep, sil = run_kmeans(features)\nsweep.round(3)"),
        ("md", "## Choosing k\nInertia keeps falling with k (as it always does); the silhouette peaks at **k = 6**, matching the six designed archetypes."),
        ("code", "display(Image(FIG / 'elbow_plot.png', width=800)); display(Image(FIG / 'silhouette_k6.png', width=700))"),
        ("code", "names = label_clusters(clustered)\nprofiles = cluster_profiles(clustered)\nprofiles.insert(0, 'cluster_name', [names[c] for c in profiles.index])\nprofiles"),
        ("code", "display(Image(FIG / 'cluster_heatmap.png', width=1000)); display(Image(FIG / 'cluster_radar.png', width=700))"),
        ("md", "## Hierarchical validation\nWard linkage on an 8,000-row sample, cut at k = 6, compared with k-Means via best-match Jaccard (Hungarian assignment)."),
        ("code", "idx, hl, Z = run_hierarchical(X)\njac, jmat, mapping = compute_jaccard(clustered['cluster'].to_numpy()[idx], hl)\nprint(f'weighted Jaccard = {jac:.3f}')\njmat"),
        ("code", "display(Image(FIG / 'dendrogram_ward.png', width=900)); display(Image(FIG / 'cluster_scatter_pca.png', width=700)); display(Image(FIG / 'cluster_scatter_tsne.png', width=700))"),
        ("md", "### Sanity check against the hidden archetypes\nThe latent archetype is never used for fitting - this crosstab only shows how well an unsupervised method recovers it."),
        ("code", "pd.crosstab(clustered['cluster_name'] if 'cluster_name' in clustered else clustered['cluster'].map(names), clustered['latent_archetype'])"),
    ]),
    "04_time_series.ipynb": nb("04 - Time-Series Forecasting (ARIMA vs SARIMA)", [
        ("code", "from src.time_series import build_cluster_series, evaluate_forecasts, summarise\n"
                 "clustered = pd.read_csv(config.CLUSTERED_FILE)\npanel = pd.read_csv(config.RAW_PANEL_FILE)\n"
                 "series = build_cluster_series(clustered, panel)\n(series / 1e6).round(2).tail(6)"),
        ("code", "display(Image(FIG / 'ts_seasonality_decomposition.png', width=900))"),
        ("md", "## Model selection\nGrid search over (p,d,q) in {0,1,2}x{0,1}x{0,1,2} and, for SARIMA, (P,D,Q,12) in {0,1}^3. "
               "Models are fit on log(spend) with a drift term; selection uses the small-sample corrected AICc. Holdout = last 3 months."),
        ("code", "results, metrics = evaluate_forecasts(series)\nmetrics.round(4)"),
        ("code", "summarise(metrics)"),
        ("code", "display(Image(FIG / 'ts_overall_forecast.png', width=1000)); display(Image(FIG / 'ts_per_cluster_forecast.png', width=1100))"),
        ("code", "display(Image(FIG / 'ts_arima_vs_sarima.png', width=900)); display(Image(FIG / 'ts_residuals.png', width=1100))"),
    ]),
    "05_propensity_model.ipynb": nb("05 - Propensity Modelling (XGBoost + Optuna + SHAP)", [
        ("code", "import joblib\nfrom src.propensity import create_labels, make_feature_matrix, split_data, evaluate_model, decile_analysis, beachhead_sizing\n"
                 "features = pd.read_csv(config.FEATURES_FILE)\ny, info = create_labels(features)\nX = make_feature_matrix(features)\n"
                 "X_train, X_test, y_train, y_test = split_data(X, y)\ninfo"),
        ("md", "## Synthetic adoption label\nRule: credit_score >= threshold AND monthly spend > INR 15k AND Tier-1/2 AND age 25-45, then 5% of labels are flipped. "
               "The threshold auto-adjusts so the positive rate lands in 10-15%."),
        ("md", "## Load the Optuna-tuned model\nThe 100-trial search runs inside `run_pipeline.py`; here we reload the artefact and re-score."),
        ("code", "model = joblib.load(config.MODELS_DIR / 'xgboost_propensity.pkl')\nstudy = joblib.load(config.MODELS_DIR / 'optuna_study.pkl')\n"
                 "baseline = joblib.load(config.MODELS_DIR / 'logistic_baseline.pkl')\n"
                 "pd.DataFrame([{'model': 'Logistic', **evaluate_model(baseline, X_test, y_test)}, {'model': 'XGBoost', **evaluate_model(model, X_test, y_test)}])"),
        ("code", "print('best params:'); pd.Series(study.best_params)"),
        ("code", "display(Image(FIG / 'roc_curve.png', width=550)); display(Image(FIG / 'pr_curve.png', width=550)); display(Image(FIG / 'optuna_optimization_history.png', width=800))"),
        ("md", "## Explainability"),
        ("code", "display(Image(FIG / 'shap_summary.png', width=800)); display(Image(FIG / 'shap_importance.png', width=700)); display(Image(FIG / 'shap_dependence_top3.png', width=1100))"),
        ("md", "## Decile lift and beachhead sizing"),
        ("code", "proba = model.predict_proba(X_test)[:, 1]\ndeciles = decile_analysis(y_test.to_numpy(), proba)\ndeciles.round(4)"),
        ("code", "beachhead_sizing(deciles, n_holders=len(features))"),
        ("code", "display(Image(FIG / 'decile_lift.png', width=900))"),
    ]),
    "06_monte_carlo.ipynb": nb("06 - Monte Carlo Sensitivity Analysis", [
        ("md", "Three robustness checks: (1) 1,000 runs jittering the dedup thresholds by +/-5%, (2) 100 bootstrap re-clusterings, (3) 100 bootstrap XGBoost refits. "
               "The full runs execute inside `run_pipeline.py`; this notebook reloads the run logs and re-runs a small demo."),
        ("code", "mc = pd.read_csv(config.REPORTS_DIR / 'monte_carlo_summary.csv')\nmc"),
        ("code", "dedup = pd.read_csv(config.REPORTS_DIR / 'mc_dedup_runs.csv')\ndedup.describe().round(3)"),
        ("code", "display(Image(FIG / 'mc_holder_count_dist.png', width=800)); display(Image(FIG / 'mc_convergence.png', width=800))"),
        ("code", "display(Image(FIG / 'mc_jaccard_dist.png', width=800)); display(Image(FIG / 'mc_auc_dist.png', width=800))"),
        ("md", "### Small live demo (50 dedup runs)"),
        ("code", "from src.data_generator import load_raw\nfrom src.preprocessing import clean_data\nfrom src.monte_carlo import run_dedup_sensitivity\n"
                 "cards, _ = load_raw(panel=False)\nclean, _ = clean_data(cards)\ndemo = run_dedup_sensitivity(clean, n_runs=50)\ndemo['summary']"),
    ]),
}


def main() -> None:
    for name, notebook in NOTEBOOKS.items():
        nbf.write(notebook, HERE / name)
        print("wrote", name)


if __name__ == "__main__":
    main()
