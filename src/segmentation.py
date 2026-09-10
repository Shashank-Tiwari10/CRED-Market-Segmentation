"""
Stage 3 - ML segmentation: k-Means (k = 2..12), hierarchical validation
(Ward linkage) and Jaccard-based agreement between the two.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

import config
from src.utils import get_logger

log = get_logger("segmentation")

CLUSTER_NAMES = {
    # Filled after profiling: cluster id -> business label (see label_clusters)
}


LOG_FEATURES = ["total_card_limit", "income_to_limit_ratio"]


def scale_features(df: pd.DataFrame, features: Optional[List[str]] = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardise the engineered features (zero mean, unit variance).

    The two heavy-tailed monetary ratios (``total_card_limit`` and
    ``income_to_limit_ratio``) are log1p-transformed first so that a handful of
    ultra-high-limit holders do not dominate Euclidean distances.
    """
    feats = features or config.ENGINEERED_FEATURES
    mat = df[feats].astype(float).copy()
    for c in LOG_FEATURES:
        if c in mat:
            mat[c] = np.log1p(mat[c].clip(lower=0))
    scaler = StandardScaler()
    X = scaler.fit_transform(mat.to_numpy(dtype=float))
    return X, scaler


def kmeans_sweep(
    X: np.ndarray,
    k_range: List[int] = config.K_RANGE,
    seed: int = config.SEED,
    sample_size: int = config.SILHOUETTE_SAMPLE,
) -> pd.DataFrame:
    """Fit k-Means for every k and collect inertia + three validity indices."""
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(sample_size, len(X)), random_state=seed)
        records.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": sil,
                "calinski_harabasz": calinski_harabasz_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
            }
        )
        log.debug("k=%d inertia=%.0f silhouette=%.3f CH=%.0f DB=%.3f",
                  k, km.inertia_, sil, records[-1]["calinski_harabasz"], records[-1]["davies_bouldin"])
    return pd.DataFrame(records)


def run_kmeans(
    df: pd.DataFrame,
    k: int = config.TARGET_K,
    seed: int = config.SEED,
    sweep: bool = True,
) -> Tuple[pd.DataFrame, KMeans, np.ndarray, pd.DataFrame, float]:
    """
    Scale, sweep k, fit the final k-Means model and attach cluster labels.

    Returns
    -------
    (clustered_df, kmeans_model, X_scaled, sweep_results, silhouette_at_k)
    """
    X, scaler = scale_features(df)
    results = kmeans_sweep(X, seed=seed) if sweep else pd.DataFrame()
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
    labels = km.labels_
    sil = float(silhouette_score(X, labels, sample_size=min(config.SILHOUETTE_SAMPLE, len(X)), random_state=seed))
    out = df.copy()
    out["cluster"] = labels
    if sweep:
        best_k = int(results.loc[results["silhouette"].idxmax(), "k"])
        log.info("k sweep complete - silhouette-optimal k = %d; using k = %d (silhouette = %.3f)", best_k, k, sil)
    else:
        log.info("k-Means k=%d silhouette=%.3f", k, sil)
    return out, km, X, results, sil


def cluster_profiles(clustered: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    """Mean of every feature per cluster, plus size and share."""
    feats = features or config.ENGINEERED_FEATURES
    extra = [c for c in ("age", "annual_income_lakhs", "credit_score", "num_cards", "avg_monthly_spend") if c in clustered]
    prof = clustered.groupby("cluster")[feats + extra].mean()
    prof.insert(0, "size", clustered["cluster"].value_counts().sort_index())
    prof.insert(1, "share_pct", (prof["size"] / len(clustered) * 100).round(2))
    prof.insert(2, "scaled_to_market_M", (prof["size"] / len(clustered) * config.REAL_MARKET_HOLDERS / 1e6).round(2))
    return prof.round(4)


def label_clusters(clustered: pd.DataFrame) -> Dict[int, str]:
    """
    Map each k-Means cluster to the latent archetype that dominates it (for
    interpretation only - the archetype column is never used for fitting).
    """
    if "latent_archetype" not in clustered:
        return {c: f"Segment {c}" for c in sorted(clustered["cluster"].unique())}
    ct = pd.crosstab(clustered["cluster"], clustered["latent_archetype"])
    mapping: Dict[int, str] = {}
    used: set = set()
    # greedy best-match so two clusters do not share a name
    for _ in range(len(ct)):
        best = None
        for c in ct.index:
            if c in mapping:
                continue
            for a in ct.columns:
                if a in used:
                    continue
                v = ct.loc[c, a] / ct.loc[c].sum()
                if best is None or v > best[2]:
                    best = (c, a, v)
        if best is None:
            break
        mapping[int(best[0])] = best[1]
        used.add(best[1])
    for c in ct.index:
        mapping.setdefault(int(c), f"Segment {c}")
    return mapping


def run_hierarchical(
    X: np.ndarray,
    k: int = config.TARGET_K,
    sample_size: int = config.HIERARCHICAL_SAMPLE,
    seed: int = config.SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Agglomerative clustering with Ward linkage on a stratified random sample
    (full-pairwise linkage is O(n^2) memory, so we validate on a subsample).

    Returns
    -------
    (sample_idx, hier_labels, linkage_matrix)
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    Z = linkage(X[idx], method="ward")
    labels = fcluster(Z, t=k, criterion="maxclust") - 1
    log.info("Ward hierarchical clustering on %s rows -> %d clusters", f"{len(idx):,}", len(np.unique(labels)))
    return idx, labels, Z


def compute_jaccard(labels_a: np.ndarray, labels_b: np.ndarray) -> Tuple[float, pd.DataFrame, Dict[int, int]]:
    """
    Best-match Jaccard agreement between two labelings of the same points.

    For each cluster in ``labels_a`` we compute the Jaccard index against every
    cluster in ``labels_b``; the Hungarian algorithm finds the one-to-one mapping
    that maximises total Jaccard.  The overall score is the size-weighted mean
    of the matched per-cluster values.

    Returns
    -------
    (weighted_jaccard, jaccard_matrix, mapping a->b)
    """
    a_ids, b_ids = np.unique(labels_a), np.unique(labels_b)
    mat = np.zeros((len(a_ids), len(b_ids)))
    for i, a in enumerate(a_ids):
        sa = labels_a == a
        for j, b in enumerate(b_ids):
            sb = labels_b == b
            inter = np.logical_and(sa, sb).sum()
            union = np.logical_or(sa, sb).sum()
            mat[i, j] = inter / union if union else 0.0
    ri, ci = linear_sum_assignment(-mat)
    mapping = {int(a_ids[r]): int(b_ids[c]) for r, c in zip(ri, ci)}
    weights = np.array([(labels_a == a_ids[r]).sum() for r in ri], dtype=float)
    weighted = float(np.sum(mat[ri, ci] * weights) / weights.sum())
    jm = pd.DataFrame(mat, index=[f"kmeans_{a}" for a in a_ids], columns=[f"hier_{b}" for b in b_ids])
    jm["best_match_jaccard"] = [mat[r, ci[list(ri).index(r)]] if r in ri else np.nan for r in range(len(a_ids))]
    return weighted, jm.round(4), mapping


def run_segmentation(features: pd.DataFrame, save: bool = True) -> Dict[str, object]:
    """Full Stage-3 driver. Returns a dict with models, labels, metrics and tables."""
    clustered, km, X, sweep, sil = run_kmeans(features)
    profiles = cluster_profiles(clustered)
    names = label_clusters(clustered)
    clustered["cluster_name"] = clustered["cluster"].map(names)
    profiles.insert(0, "cluster_name", [names[c] for c in profiles.index])

    idx, hier_labels, Z = run_hierarchical(X)
    jaccard, jmat, mapping = compute_jaccard(clustered["cluster"].to_numpy()[idx], hier_labels)
    log.info("k-Means vs Ward weighted Jaccard = %.3f (target >= %.2f)", jaccard, config.MIN_JACCARD)

    best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"]) if len(sweep) else config.TARGET_K
    results = {
        "clustered": clustered, "kmeans": km, "X": X, "sweep": sweep, "silhouette": sil,
        "best_k": best_k, "profiles": profiles, "cluster_names": names,
        "hier_idx": idx, "hier_labels": hier_labels, "linkage": Z,
        "jaccard": jaccard, "jaccard_matrix": jmat, "jaccard_mapping": mapping,
    }
    if save:
        clustered.to_csv(config.CLUSTERED_FILE, index=False)
        profiles.to_csv(config.REPORTS_DIR / "cluster_profiles.csv")
        sweep.to_csv(config.REPORTS_DIR / "kmeans_k_sweep.csv", index=False)
        jmat.to_csv(config.REPORTS_DIR / "jaccard_stability.csv")
    return results
