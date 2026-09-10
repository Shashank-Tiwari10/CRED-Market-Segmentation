"""
All plotting functions.  Every function saves a PNG into ``outputs/figures``
and returns the path so callers can log it.  Matplotlib runs headless (Agg).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.metrics import silhouette_samples  # noqa: E402
from statsmodels.graphics.tsaplots import plot_acf  # noqa: E402
from statsmodels.tsa.seasonal import seasonal_decompose  # noqa: E402

import config  # noqa: E402
from src.utils import get_logger  # noqa: E402

log = get_logger("visualization")
sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)
FIG = config.FIGURES_DIR


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.debug("saved %s", path.name)
    return path


def _cluster_colors(labels: Sequence[int]) -> List[str]:
    pal = sns.color_palette("tab10", n_colors=max(10, int(np.max(labels)) + 1))
    return [pal[int(lb)] for lb in labels]


# =========================================================================== #
# Stage 3 - Segmentation
# =========================================================================== #
def plot_elbow(sweep: pd.DataFrame, target_k: int = config.TARGET_K) -> Path:
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(sweep["k"], sweep["inertia"], "o-", color=config.PALETTE[0], label="Inertia (WCSS)")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Inertia", color=config.PALETTE[0])
    ax2 = ax1.twinx()
    ax2.plot(sweep["k"], sweep["silhouette"], "s--", color=config.PALETTE[1], label="Silhouette")
    ax2.set_ylabel("Silhouette score", color=config.PALETTE[1])
    ax2.grid(False)
    ax1.axvline(target_k, color="grey", ls=":", lw=1.5)
    sil_k = float(sweep.loc[sweep["k"] == target_k, "silhouette"].iloc[0])
    ax1.annotate(f"k = {target_k}\nsilhouette = {sil_k:.3f}", xy=(target_k, sweep["inertia"].max() * 0.9),
                 xytext=(target_k + 0.4, sweep["inertia"].max() * 0.92), fontsize=10)
    ax1.set_title("k-Means elbow plot with silhouette overlay")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    return _save(fig, "elbow_plot.png")


def plot_silhouette(X: np.ndarray, labels: np.ndarray, sil_avg: float, k: int = config.TARGET_K,
                    sample: int = 6000, seed: int = config.SEED) -> Path:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Xs, ls = X[idx], labels[idx]
    vals = silhouette_samples(Xs, ls)
    fig, ax = plt.subplots(figsize=(9, 6))
    y_lower = 10
    pal = sns.color_palette("tab10", 10)
    for c in range(k):
        v = np.sort(vals[ls == c])
        y_upper = y_lower + len(v)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, v, facecolor=pal[c], alpha=0.8)
        ax.text(-0.08, y_lower + 0.5 * len(v), str(c))
        y_lower = y_upper + 10
    ax.axvline(sil_avg, color="red", ls="--", label=f"mean silhouette = {sil_avg:.3f}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_yticks([])
    ax.set_title(f"Silhouette plot for k = {k}")
    ax.legend(loc="lower right")
    return _save(fig, f"silhouette_k{k}.png")


def plot_dendrogram(Z: np.ndarray, k: int = config.TARGET_K) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, truncate_mode="lastp", p=40, leaf_rotation=90, leaf_font_size=8,
               show_contracted=True, ax=ax, color_threshold=Z[-(k - 1), 2])
    ax.axhline(Z[-(k - 1), 2], color="red", ls="--", lw=1, label=f"cut at k = {k}")
    ax.set_title("Hierarchical clustering dendrogram (Ward linkage, truncated)")
    ax.set_ylabel("Ward distance")
    ax.legend()
    return _save(fig, "dendrogram_ward.png")


def _scatter(ax: plt.Axes, emb: np.ndarray, labels: np.ndarray, names: Dict[int, str]) -> None:
    pal = sns.color_palette("tab10", 10)
    for c in sorted(np.unique(labels)):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.5, color=pal[int(c)], label=f"{c}: {names.get(int(c), c)}")
    ax.legend(markerscale=3, fontsize=8, loc="best")


def plot_pca_scatter(X: np.ndarray, labels: np.ndarray, names: Dict[int, str],
                     sample: int = 15000, seed: int = config.SEED) -> Path:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    pca = PCA(n_components=2, random_state=seed).fit(X)
    emb = pca.transform(X[idx])
    fig, ax = plt.subplots(figsize=(10, 7))
    _scatter(ax, emb, labels[idx], names)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    ax.set_title("k-Means clusters - PCA projection")
    return _save(fig, "cluster_scatter_pca.png")


def plot_tsne_scatter(X: np.ndarray, labels: np.ndarray, names: Dict[int, str],
                      sample: int = config.TSNE_SAMPLE, seed: int = config.SEED) -> Path:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    emb = TSNE(n_components=2, random_state=seed, perplexity=40, init="pca").fit_transform(X[idx])
    fig, ax = plt.subplots(figsize=(10, 7))
    _scatter(ax, emb, labels[idx], names)
    ax.set_title(f"k-Means clusters - t-SNE projection (n = {len(idx):,})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    return _save(fig, "cluster_scatter_tsne.png")


def plot_radar(profiles: pd.DataFrame, names: Dict[int, str]) -> Path:
    feats = config.ENGINEERED_FEATURES
    data = profiles[feats].copy()
    norm = (data - data.min()) / (data.max() - data.min() + 1e-12)
    angles = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10, 9), subplot_kw=dict(polar=True))
    pal = sns.color_palette("tab10", 10)
    for c, row in norm.iterrows():
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(angles, vals, color=pal[int(c)], lw=2, label=f"{c}: {names.get(int(c), c)}")
        ax.fill(angles, vals, color=pal[int(c)], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace("_", "\n") for f in feats], fontsize=8)
    ax.set_yticklabels([])
    ax.set_title("Cluster radar - min-max normalised feature means", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    return _save(fig, "cluster_radar.png")


def plot_cluster_heatmap(profiles: pd.DataFrame, names: Dict[int, str]) -> Path:
    feats = config.ENGINEERED_FEATURES
    data = profiles[feats].copy()
    z = (data - data.mean()) / (data.std() + 1e-12)
    z.index = [f"{c}: {names.get(int(c), c)}" for c in z.index]
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(z, cmap="RdBu_r", center=0, annot=data.round(2), fmt="", cbar_kws={"label": "z-score across clusters"}, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Cluster x feature means (colour = z-score, text = raw mean)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    return _save(fig, "cluster_heatmap.png")


def plot_cluster_distribution(clustered: pd.DataFrame, names: Dict[int, str]) -> Path:
    counts = clustered["cluster"].value_counts().sort_index()
    scaled = counts / counts.sum() * config.REAL_MARKET_HOLDERS / 1e6
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar([f"{c}\n{names.get(int(c), c)}" for c in counts.index], counts.values,
                  color=sns.color_palette("tab10", 10)[: len(counts)])
    for b, s, n in zip(bars, scaled.values, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{n:,}\n({s:.1f}M)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Holders in sample")
    ax.set_title("Cluster sizes (sample count and scaled to 38M market)")
    plt.setp(ax.get_xticklabels(), fontsize=8)
    return _save(fig, "cluster_distribution.png")


# =========================================================================== #
# Stage 4 - Time series
# =========================================================================== #
def plot_overall_forecast(series: pd.Series, result: Dict[str, object]) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(series.index, series.values / 1e6, "k-o", lw=2, ms=4, label="Actual")
    for name, color in (("arima", config.PALETTE[1]), ("sarima", config.PALETTE[0])):
        r = result[name]
        ax.plot(r["holdout_pred"].index, r["holdout_pred"].values / 1e6, "--", color=color, lw=2,
                label=f"{name.upper()} holdout (MAPE {r['mape'] * 100:.1f}%)")
        ax.plot(r["future"].index, r["future"].values / 1e6, ":", color=color, lw=2, label=f"{name.upper()} 6-month forecast")
        if "future_ci" in r:
            ci = r["future_ci"]
            ax.fill_between(ci.index, ci.iloc[:, 0] / 1e6, ci.iloc[:, 1] / 1e6, color=color, alpha=0.12)
    ax.axvspan(result["holdout_start"], series.index[-1], color="grey", alpha=0.12, label="Holdout (last 3 months)")
    ax.set_ylabel("Monthly spend (INR million, sample)")
    ax.set_title("Overall market - actual vs ARIMA / SARIMA")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, "ts_overall_forecast.png")


def plot_per_cluster_forecast(series_df: pd.DataFrame, results: Dict[str, Dict[str, object]],
                              names: Dict[int, str]) -> Path:
    cols = [c for c in series_df.columns if c != "Overall"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for ax, col in zip(axes.ravel(), cols):
        s, r = series_df[col], results[col]
        ax.plot(s.index, s.values / 1e6, "k-", lw=1.5, label="Actual")
        ax.plot(r["sarima"]["holdout_pred"].index, r["sarima"]["holdout_pred"].values / 1e6, "--", color=config.PALETTE[0], label="SARIMA")
        ax.plot(r["arima"]["holdout_pred"].index, r["arima"]["holdout_pred"].values / 1e6, "--", color=config.PALETTE[1], label="ARIMA")
        ax.plot(r["sarima"]["future"].index, r["sarima"]["future"].values / 1e6, ":", color=config.PALETTE[0])
        cid = int(str(col).split("_")[-1]) if str(col).startswith("cluster_") else None
        title = f"{col}: {names.get(cid, '')}" if cid is not None else str(col)
        ax.set_title(f"{title}\nSARIMA MAPE {r['sarima']['mape'] * 100:.1f}% | ARIMA {r['arima']['mape'] * 100:.1f}%", fontsize=9)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Per-cluster monthly spend forecasts (INR million, sample)", y=1.02)
    return _save(fig, "ts_per_cluster_forecast.png")


def plot_arima_vs_sarima(metrics: pd.DataFrame) -> Path:
    piv = metrics.pivot(index="series", columns="model", values="mape") * 100
    fig, ax = plt.subplots(figsize=(11, 5.5))
    piv[["ARIMA", "SARIMA"]].plot.bar(ax=ax, color=[config.PALETTE[1], config.PALETTE[0]])
    ax.axhline(config.MAX_MAPE * 100, color="red", ls="--", label="12% target ceiling")
    ax.set_ylabel("Holdout MAPE (%)")
    ax.set_title("ARIMA vs SARIMA - holdout MAPE per series")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, "ts_arima_vs_sarima.png")


def plot_residuals(resid: pd.Series, title: str = "SARIMA residual diagnostics (overall market)") -> Path:
    resid = resid.dropna()
    fig = plt.figure(figsize=(15, 4.5))
    gs = GridSpec(1, 3, figure=fig)
    ax0 = fig.add_subplot(gs[0])
    plot_acf(resid, ax=ax0, lags=min(10, len(resid) // 2 - 1))
    ax0.set_title("Residual ACF")
    ax1 = fig.add_subplot(gs[1])
    sns.histplot(resid, kde=True, ax=ax1, color=config.PALETTE[0])
    ax1.set_title("Residual histogram")
    ax2 = fig.add_subplot(gs[2])
    stats.probplot(resid, dist="norm", plot=ax2)
    ax2.set_title("Residual Q-Q plot")
    fig.suptitle(title)
    return _save(fig, "ts_residuals.png")


def plot_seasonal_decomposition(series: pd.Series) -> Path:
    dec = seasonal_decompose(series, model="multiplicative", period=12)
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, comp, name in zip(axes, (dec.observed, dec.trend, dec.seasonal, dec.resid), ("Observed", "Trend", "Seasonal", "Residual")):
        ax.plot(comp.index, comp.values, color=config.PALETTE[0])
        ax.set_ylabel(name)
    axes[0].set_title("Seasonal decomposition - overall market monthly spend (multiplicative, period = 12)")
    return _save(fig, "ts_seasonality_decomposition.png")


# =========================================================================== #
# Stage 5 - Propensity
# =========================================================================== #
def plot_roc(y_true: np.ndarray, scores: Dict[str, np.ndarray]) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for (name, s), color in zip(scores.items(), config.PALETTE):
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve - CRED adoption propensity")
    ax.legend(loc="lower right")
    return _save(fig, "roc_curve.png")


def plot_pr(y_true: np.ndarray, scores: Dict[str, np.ndarray]) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for (name, s), color in zip(scores.items(), config.PALETTE):
        p, r, _ = precision_recall_curve(y_true, s)
        ax.plot(r, p, lw=2, color=color, label=f"{name} (PR-AUC = {auc(r, p):.3f})")
    ax.axhline(y_true.mean(), color="grey", ls=":", label=f"base rate = {y_true.mean():.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="upper right")
    return _save(fig, "pr_curve.png")


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
    ax.set_title(f"Confusion matrix (threshold = {threshold:.2f})")
    return _save(fig, "confusion_matrix.png")


def plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame) -> Path:
    import shap

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False, max_display=len(X.columns))
    plt.title("SHAP summary (beeswarm)")
    return _save(fig, "shap_summary.png")


def plot_shap_importance(shap_values: np.ndarray, X: pd.DataFrame) -> Path:
    imp = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    imp.plot.barh(ax=ax, color=config.PALETTE[0])
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("SHAP feature importance")
    return _save(fig, "shap_importance.png")


def plot_shap_dependence(shap_values: np.ndarray, X: pd.DataFrame, top3: List[str]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, f in zip(axes, top3):
        j = list(X.columns).index(f)
        ax.scatter(X[f], shap_values[:, j], s=8, alpha=0.5, color=config.PALETTE[0])
        ax.axhline(0, color="grey", lw=1, ls=":")
        ax.set_xlabel(f)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"Dependence: {f}", fontsize=10)
    fig.suptitle("SHAP dependence plots - top 3 features")
    return _save(fig, "shap_dependence_top3.png")


def plot_decile_lift(deciles: pd.DataFrame) -> Path:
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.bar(deciles["decile"], deciles["adoption_rate"] * 100, color=config.PALETTE[0], alpha=0.85, label="Adoption rate (%)")
    ax1.axhline(deciles["overall_rate"].iloc[0] * 100, color="grey", ls=":", label="Overall rate")
    ax1.set_xlabel("Propensity decile (1 = highest score)")
    ax1.set_ylabel("Actual adoption rate (%)")
    ax2 = ax1.twinx()
    ax2.plot(deciles["decile"], deciles["cumulative_capture"] * 100, "o-", color=config.PALETTE[3], label="Cumulative % of adopters captured")
    ax2.set_ylabel("Cumulative capture (%)")
    ax2.grid(False)
    for d, l in zip(deciles["decile"], deciles["lift"]):
        ax1.text(d, 1, f"lift {l:.1f}x", ha="center", fontsize=8, color="white" if l > 1 else "black")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    ax1.set_title("Decile lift chart - test set")
    return _save(fig, "decile_lift.png")


def plot_xgb_importance(model, feature_names: List[str]) -> Path:
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    imp.plot.barh(ax=ax, color=config.PALETTE[2])
    ax.set_xlabel("Gain-based importance")
    ax.set_title("XGBoost native feature importance")
    return _save(fig, "feature_importance_xgb.png")


def plot_optuna_history(study) -> Path:
    trials = [t for t in study.trials if t.value is not None]
    vals = np.array([t.value for t in trials])
    best = np.maximum.accumulate(vals)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(range(len(vals)), vals, s=14, color=config.PALETTE[0], alpha=0.6, label="Trial AUC")
    ax.plot(range(len(best)), best, color=config.PALETTE[3], lw=2, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Validation ROC-AUC")
    ax.set_title(f"Optuna optimisation history ({len(vals)} trials, best = {best[-1]:.4f})")
    ax.legend()
    return _save(fig, "optuna_optimization_history.png")


# =========================================================================== #
# Stage 6 - Monte Carlo
# =========================================================================== #
def plot_mc_holder_dist(counts_m: np.ndarray, ci: Sequence[float]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.histplot(counts_m, bins=40, kde=True, color=config.PALETTE[0], ax=ax)
    ax.axvline(np.mean(counts_m), color="black", ls="-", label=f"mean = {np.mean(counts_m):.2f}M")
    ax.axvline(ci[0], color="red", ls="--", label=f"95% CI: {ci[0]:.2f}M - {ci[1]:.2f}M")
    ax.axvline(ci[1], color="red", ls="--")
    ax.set_xlabel("Addressable unique holders (millions, scaled)")
    ax.set_title(f"Monte Carlo dedup sensitivity - {len(counts_m):,} runs")
    ax.legend()
    return _save(fig, "mc_holder_count_dist.png")


def plot_mc_jaccard_dist(values: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.histplot(values, bins=25, kde=True, color=config.PALETTE[2], ax=ax)
    ax.axvline(np.mean(values), color="black", label=f"mean = {np.mean(values):.3f}")
    ax.axvline(config.MIN_JACCARD, color="red", ls="--", label=f"target >= {config.MIN_JACCARD}")
    ax.set_xlabel("Jaccard vs full-data clustering")
    ax.set_title(f"Cluster stability - {len(values)} bootstrap runs")
    ax.legend()
    return _save(fig, "mc_jaccard_dist.png")


def plot_mc_auc_dist(values: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.histplot(values, bins=25, kde=True, color=config.PALETTE[4], ax=ax)
    lo, hi = np.percentile(values, [2.5, 97.5])
    ax.axvline(np.mean(values), color="black", label=f"mean = {np.mean(values):.4f}")
    ax.axvline(lo, color="red", ls="--", label=f"95% CI: {lo:.4f} - {hi:.4f}")
    ax.axvline(hi, color="red", ls="--")
    ax.set_xlabel("Test ROC-AUC")
    ax.set_title(f"Propensity model stability - {len(values)} bootstrap runs")
    ax.legend()
    return _save(fig, "mc_auc_dist.png")


def plot_mc_convergence(counts_m: np.ndarray) -> Path:
    running = np.cumsum(counts_m) / np.arange(1, len(counts_m) + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(running, color=config.PALETTE[0], lw=2)
    ax.axhline(running[-1], color="grey", ls=":", label=f"final mean = {running[-1]:.2f}M")
    ax.set_xlabel("Simulation run")
    ax.set_ylabel("Running mean of holder count (M)")
    ax.set_title("Monte Carlo convergence of addressable-holder estimate")
    ax.legend()
    return _save(fig, "mc_convergence.png")


# =========================================================================== #
# Pipeline summary dashboard
# =========================================================================== #
def plot_pipeline_summary(metrics: Dict[str, object]) -> Path:
    """One-page dashboard of the headline metrics against their targets."""
    rows = [
        ("Data purity", metrics["purity"] * 100, config.MIN_PURITY * 100, "%", ">="),
        ("Silhouette (k=6)", metrics["silhouette"], config.TARGET_SILHOUETTE, "", "~"),
        ("Jaccard k-Means vs Ward", metrics["jaccard"], config.MIN_JACCARD, "", ">="),
        ("SARIMA MAPE", metrics["sarima_mape"] * 100, config.MAX_MAPE * 100, "%", "<"),
        ("ARIMA MAPE", metrics["arima_mape"] * 100, config.MAX_MAPE * 100, "%", "<"),
        ("XGBoost AUC", metrics["auc"], config.MIN_AUC, "", ">="),
        ("MC Jaccard mean", metrics["mc_jaccard_mean"], config.MIN_JACCARD, "", ">="),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [r[0] for r in rows]
    actual = [r[1] for r in rows]
    target = [r[2] for r in rows]
    y = np.arange(len(rows))
    ax.barh(y - 0.2, actual, height=0.4, color=config.PALETTE[0], label="Achieved")
    ax.barh(y + 0.2, target, height=0.4, color="lightgrey", label="Target")
    for i, r in enumerate(rows):
        ax.text(max(r[1], r[2]) * 1.02, i, f"{r[1]:.2f}{r[3]}  (target {r[4]} {r[2]:.2f}{r[3]})", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(actual), max(target)) * 1.6)
    ax.set_title("CRED pipeline - headline metrics vs targets")
    ax.legend(loc="lower right")
    return _save(fig, "pipeline_summary.png")


EXPECTED_FIGURES = [
    "elbow_plot.png", "silhouette_k6.png", "dendrogram_ward.png", "cluster_scatter_pca.png",
    "cluster_scatter_tsne.png", "cluster_radar.png", "cluster_heatmap.png", "cluster_distribution.png",
    "ts_overall_forecast.png", "ts_per_cluster_forecast.png", "ts_arima_vs_sarima.png", "ts_residuals.png",
    "ts_seasonality_decomposition.png", "roc_curve.png", "pr_curve.png", "confusion_matrix.png",
    "shap_summary.png", "shap_importance.png", "shap_dependence_top3.png", "decile_lift.png",
    "feature_importance_xgb.png", "optuna_optimization_history.png", "mc_holder_count_dist.png",
    "mc_jaccard_dist.png", "mc_auc_dist.png", "mc_convergence.png", "pipeline_summary.png",
]


def generate_all_plots(metrics: Dict[str, object]) -> List[str]:
    """
    Final plotting step: draw the summary dashboard and verify that every
    expected figure exists.  Returns the list of missing figures (empty = OK).
    """
    plot_pipeline_summary(metrics)
    missing = [f for f in EXPECTED_FIGURES if not (FIG / f).exists()]
    if missing:
        log.warning("Missing figures: %s", missing)
    else:
        log.info("All %d expected figures present in %s", len(EXPECTED_FIGURES), FIG)
    return missing
