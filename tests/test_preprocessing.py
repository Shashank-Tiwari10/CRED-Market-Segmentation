"""Tests for Stage 1-2: generation, cleaning, cascading filter, features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.data_generator import TOP_50_CITY_NAMES, generate_credit_card_data
from src.preprocessing import aggregate_to_holders, cascading_filter, clean_data, engineer_features


@pytest.fixture(scope="module")
def raw():
    cards, panel = generate_credit_card_data(n_cards=6000, seed=7, save=False)
    return cards, panel


@pytest.fixture(scope="module")
def cleaned(raw):
    cards, _ = raw
    return clean_data(cards)


def test_generator_shape_and_holders(raw):
    cards, panel = raw
    assert len(cards) == 6000
    assert cards["card_id"].is_unique
    n_holders = cards["holder_id"].nunique()
    assert 1500 <= n_holders <= 2800  # ~3.1 cards per holder
    assert panel["timestamp_month"].nunique() == config.N_MONTHS
    assert set(panel["holder_id"]) == set(cards["holder_id"])


def test_generator_reproducible():
    a, _ = generate_credit_card_data(n_cards=1500, seed=3, save=False)
    b, _ = generate_credit_card_data(n_cards=1500, seed=3, save=False)
    pd.testing.assert_frame_equal(a, b)


def test_holder_demographics_consistent(raw):
    cards, _ = raw
    clean, _ = clean_data(cards)
    # after imputation a holder should still have one dominant value per demographic
    for col in ("gender", "city_tier", "state"):
        nunique = clean.groupby("holder_id")[col].nunique()
        assert (nunique == 1).mean() > 0.95


def test_noise_injected_and_cleaned(raw, cleaned):
    cards, _ = raw
    clean, report = cleaned
    assert cards.isna().any().any(), "generator should inject nulls"
    assert not clean[["age", "annual_income_lakhs", "credit_score"]].isna().any().any()
    assert clean["credit_score"].between(300, 900).all()
    assert (clean["annual_income_lakhs"] >= 1).all()
    assert report["purity"] >= config.MIN_PURITY
    assert report["valid_rows"] == len(clean)


def test_cascading_filter_monotone(cleaned):
    clean, _ = cleaned
    holders, log_df = cascading_filter(clean, verbose=False)
    counts = log_df["holders"].tolist()
    assert counts == sorted(counts, reverse=True)
    assert holders["holder_id"].is_unique
    assert holders["city"].isin(TOP_50_CITY_NAMES).all()
    assert (holders["credit_score"] >= config.FILTER_THRESHOLDS["min_credit_score"]).all()
    assert (holders["avg_monthly_spend"] >= config.FILTER_THRESHOLDS["min_monthly_spend"]).all()
    assert (holders["avg_monthly_repayment_pct"] >= config.FILTER_THRESHOLDS["min_repayment_pct"]).all()


def test_cascading_filter_custom_thresholds(cleaned):
    clean, _ = cleaned
    loose, _ = cascading_filter(clean, thresholds={"min_credit_score": 500}, verbose=False)
    strict, _ = cascading_filter(clean, thresholds={"min_credit_score": 700}, verbose=False)
    assert len(loose) > len(strict)


def test_aggregate_to_holders_sums(cleaned):
    clean, _ = cleaned
    agg = aggregate_to_holders(clean)
    hid = agg["holder_id"].iloc[0]
    expected = clean.loc[clean["holder_id"] == hid, "card_limit_lakhs"].sum()
    assert np.isclose(agg.loc[agg["holder_id"] == hid, "total_card_limit"].iloc[0], expected)


def test_engineered_features_present_and_bounded(raw, cleaned):
    _, panel = raw
    clean, _ = cleaned
    holders, _ = cascading_filter(clean, verbose=False)
    feats = engineer_features(holders, panel)
    for c in config.ENGINEERED_FEATURES:
        assert c in feats.columns
        assert not feats[c].isna().any()
    assert feats["repayment_discipline"].between(0, 1).all()
    assert feats["digital_savviness"].between(0, 1).all()
    assert set(feats["premium_card_flag"].unique()) <= {0, 1}
    assert set(feats["emi_propensity"].unique()) <= {0, 1}
    assert (feats["spend_volatility"] > 0).all()
    assert (feats["utilization_ratio"] >= 0).all()
