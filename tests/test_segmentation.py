"""Tests for Stage 3: scaling, k-Means, hierarchical clustering, Jaccard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.segmentation import (
    cluster_profiles,
    compute_jaccard,
    kmeans_sweep,
    label_clusters,
    run_hierarchical,
    run_kmeans,
    scale_features,
)


@pytest.fixture(scope="module")
def blobs() -> pd.DataFrame:
    """Six well-separated synthetic archetypes in the 12-feature space."""
    rng = np.random.default_rng(0)
    n_per = 300
    centres = rng.uniform(-4, 4, size=(6, len(config.ENGINEERED_FEATURES)))
    rows = []
    for c in range(6):
        pts = centres[c] + rng.normal(0, 0.6, size=(n_per, len(config.ENGINEERED_FEATURES)))
        df = pd.DataFrame(np.abs(pts), columns=config.ENGINEERED_FEATURES)
        df["latent_archetype"] = f"arch_{c}"
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["holder_id"] = [f"H{i}" for i in range(len(out))]
    return out


def test_scale_features_standardised(blobs):
    X, scaler = scale_features(blobs)
    assert X.shape == (len(blobs), len(config.ENGINEERED_FEATURES))
    assert np.allclose(X.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(X.std(axis=0), 1, atol=1e-6)


def test_kmeans_sweep_columns(blobs):
    X, _ = scale_features(blobs)
    sweep = kmeans_sweep(X, k_range=[2, 3, 6], sample_size=500)
    assert list(sweep["k"]) == [2, 3, 6]
    for col in ("inertia", "silhouette", "calinski_harabasz", "davies_bouldin"):
        assert col in sweep.columns
    assert sweep["inertia"].is_monotonic_decreasing


def test_run_kmeans_recovers_six_clusters(blobs):
    clustered, km, X, sweep, sil = run_kmeans(blobs, k=6, sweep=False)
    assert clustered["cluster"].nunique() == 6
    assert sil > 0.3
    ct = pd.crosstab(clustered["cluster"], clustered["latent_archetype"])
    # each cluster should be dominated by one archetype
    assert (ct.max(axis=1) / ct.sum(axis=1) > 0.9).all()


def test_cluster_profiles_and_labels(blobs):
    clustered, *_ = run_kmeans(blobs, k=6, sweep=False)
    prof = cluster_profiles(clustered)
    assert len(prof) == 6
    assert prof["size"].sum() == len(clustered)
    assert np.isclose(prof["share_pct"].sum(), 100, atol=0.1)
    names = label_clusters(clustered)
    assert len(set(names.values())) == 6


def test_hierarchical_and_jaccard(blobs):
    clustered, _, X, _, _ = run_kmeans(blobs, k=6, sweep=False)
    idx, hl, Z = run_hierarchical(X, k=6, sample_size=900)
    assert len(idx) == 900 and len(hl) == 900
    assert Z.shape == (899, 4)
    jac, jm, mapping = compute_jaccard(clustered["cluster"].to_numpy()[idx], hl)
    assert 0.9 <= jac <= 1.0
    assert jm.shape[0] == 6
    assert len(mapping) == 6


def test_jaccard_identity_and_permutation():
    a = np.array([0, 0, 1, 1, 2, 2, 2])
    assert compute_jaccard(a, a)[0] == 1.0
    perm = np.array([2, 2, 0, 0, 1, 1, 1])  # relabelled but identical partition
    assert compute_jaccard(a, perm)[0] == 1.0
    b = np.array([0, 0, 0, 1, 2, 2, 2])
    assert compute_jaccard(a, b)[0] < 1.0
