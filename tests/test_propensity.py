"""Tests for Stage 5: labels, models, deciles, beachhead sizing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.propensity import (
    beachhead_sizing,
    create_labels,
    decile_analysis,
    evaluate_model,
    make_feature_matrix,
    split_data,
    train_baseline,
    train_xgboost_optuna,
)


@pytest.fixture(scope="module")
def holders() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 3000
    df = pd.DataFrame({
        "holder_id": [f"H{i}" for i in range(n)],
        "age": rng.integers(21, 70, n),
        "credit_score": rng.integers(600, 900, n),
        "avg_monthly_spend": rng.lognormal(9.5, 0.8, n),
        "city_tier": rng.choice(["Tier-1", "Tier-2", "Tier-3"], n, p=[0.45, 0.35, 0.2]),
        "annual_income_lakhs": rng.lognormal(3, 0.6, n),
        "num_cards": rng.integers(1, 6, n),
        "city_tier_code": 0,
        "gender_code": rng.integers(0, 2, n),
    })
    df["city_tier_code"] = df["city_tier"].map({"Tier-1": 1, "Tier-2": 2, "Tier-3": 3})
    for c in config.ENGINEERED_FEATURES:
        df[c] = rng.random(n)
    return df


def test_labels_rate_in_target_band(holders):
    y, info = create_labels(holders, target_rate=(0.05, 0.40))
    assert set(y.unique()) <= {0, 1}
    assert 0.05 <= info["rule_positive_rate"] <= 0.40
    assert info["n_flipped"] > 0
    assert len(y) == len(holders)


def test_labels_reproducible(holders):
    y1, _ = create_labels(holders, seed=5, target_rate=(0.05, 0.40))
    y2, _ = create_labels(holders, seed=5, target_rate=(0.05, 0.40))
    assert (y1 == y2).all()


def test_split_stratified(holders):
    y, _ = create_labels(holders, target_rate=(0.05, 0.40))
    X = make_feature_matrix(holders)
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    assert len(X_te) == int(round(len(X) * config.TEST_SIZE))
    assert abs(y_tr.mean() - y_te.mean()) < 0.03


def test_models_beat_random(holders):
    y, _ = create_labels(holders, target_rate=(0.05, 0.40))
    X = make_feature_matrix(holders)
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    base = train_baseline(X_tr, y_tr)
    m_base = evaluate_model(base, X_te, y_te)
    model, study, params = train_xgboost_optuna(X_tr, y_tr, n_trials=3)
    m_xgb = evaluate_model(model, X_te, y_te)
    assert len(study.trials) == 3
    assert set(params) >= {"max_depth", "learning_rate", "n_estimators"}
    assert m_base["roc_auc"] > 0.6
    assert m_xgb["roc_auc"] > 0.7
    for k in ("precision", "recall", "f1", "pr_auc", "threshold"):
        assert 0 <= m_xgb[k] <= 1


def test_decile_analysis_structure():
    rng = np.random.default_rng(0)
    scores = rng.random(2000)
    y = (rng.random(2000) < scores).astype(int)  # calibrated: higher score -> more positives
    d = decile_analysis(y, scores)
    assert list(d["decile"]) == list(range(1, 11))
    assert d["count"].sum() == 2000
    assert np.isclose(d["capture"].sum(), 1.0)
    assert d["cumulative_capture"].is_monotonic_increasing
    assert d.loc[0, "lift"] > 1.0
    assert d.loc[0, "adoption_rate"] > d.loc[9, "adoption_rate"]


def test_beachhead_sizing_scales_to_market():
    rng = np.random.default_rng(0)
    scores = rng.random(2000)
    y = (rng.random(2000) < scores).astype(int)
    d = decile_analysis(y, scores)
    bh = beachhead_sizing(d, n_holders=2000)
    assert len(bh) == 4
    assert np.isclose(bh.loc[0, "market_holders_M"], 3.8, atol=0.05)
    assert np.isclose(bh.loc[2, "market_holders_M"], 11.4, atol=0.05)
    assert bh.loc[0, "lift"] >= bh.loc[2, "lift"]
