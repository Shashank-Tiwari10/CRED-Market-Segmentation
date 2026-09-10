"""
Stage 2 - Cleaning, cascading market-sizing filter (deduplication) and
holder-level feature engineering.

The cascading filter is the signature technique of the project: a sequential
set of business filters that collapses 119M card records into ~38M unique,
addressable holders.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from src.data_generator import TOP_50_CITY_NAMES
from src.utils import get_logger

log = get_logger("preprocessing")

NUMERIC_COLS = [
    "age", "annual_income_lakhs", "credit_score", "card_limit_lakhs", "avg_monthly_spend",
    "avg_monthly_repayment_pct", "months_active", "num_cards", "upi_credit_usage",
    "missed_payments_12m", "reward_redemption_rate", "digital_transaction_pct",
]
CATEGORICAL_COLS = ["gender", "city_tier", "city", "state", "card_type", "card_issuer"]


# --------------------------------------------------------------------------- #
# 2A. Cleaning
# --------------------------------------------------------------------------- #
def clean_data(
    cards: pd.DataFrame,
    impute_threshold: float = config.NULL_IMPUTE_THRESHOLD,
    max_nulls_per_row: int = config.MAX_NULLS_PER_ROW,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Clean the raw card table.

    Steps: coerce numeric types, drop invalid credit scores (<300 or >900),
    drop implausible incomes (<1 lakh), drop rows with too many nulls, impute
    the remainder (median for numeric, mode for categorical).

    Returns
    -------
    (clean_df, report) where report contains ``purity`` = valid_rows / total_rows.
    """
    df = cards.copy()
    total = len(df)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "has_emi_active" in df.columns:
        df["has_emi_active"] = df["has_emi_active"].astype(str).str.lower().isin(["true", "1"])

    report: Dict[str, float] = {"total_rows": total}

    bad_score = (df["credit_score"] < 300) | (df["credit_score"] > 900)
    report["dropped_invalid_credit_score"] = int(bad_score.sum())
    df = df.loc[~bad_score]

    bad_income = df["annual_income_lakhs"] < 1
    report["dropped_invalid_income"] = int(bad_income.sum())
    df = df.loc[~bad_income]

    null_counts = df.isna().sum(axis=1)
    too_many = null_counts > max_nulls_per_row
    report["dropped_excess_nulls"] = int(too_many.sum())
    df = df.loc[~too_many].copy()

    # Impute columns with a small null share; drop rows for columns above threshold
    null_share = df.isna().mean()
    imputed: List[str] = []
    for col, share in null_share.items():
        if share == 0:
            continue
        if share < impute_threshold:
            if col in NUMERIC_COLS:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            imputed.append(col)
        else:
            df = df.loc[df[col].notna()]
    report["imputed_columns"] = imputed

    # Restore integer dtypes after imputation
    for col in ("age", "credit_score", "months_active", "num_cards", "missed_payments_12m"):
        df[col] = df[col].round().astype(int)

    valid = len(df)
    purity = valid / total
    report.update({"valid_rows": valid, "purity": purity})
    log.info("Cleaning: %s -> %s rows | purity = %.2f%% (target >= %.0f%%)",
             f"{total:,}", f"{valid:,}", purity * 100, config.MIN_PURITY * 100)
    log.info("  dropped: score=%d income=%d nulls=%d | imputed: %s",
             report["dropped_invalid_credit_score"], report["dropped_invalid_income"],
             report["dropped_excess_nulls"], ", ".join(imputed) or "none")
    return df.reset_index(drop=True), report


# --------------------------------------------------------------------------- #
# 2B. Cascading market-sizing filter
# --------------------------------------------------------------------------- #
def aggregate_to_holders(cards: pd.DataFrame) -> pd.DataFrame:
    """Collapse card rows into one row per holder (demographics + card aggregates)."""
    agg = cards.groupby("holder_id", sort=True).agg(
        holder_name=("holder_name", "first"),
        age=("age", "first"),
        gender=("gender", "first"),
        city_tier=("city_tier", "first"),
        city=("city", "first"),
        state=("state", "first"),
        annual_income_lakhs=("annual_income_lakhs", "first"),
        credit_score=("credit_score", "first"),
        num_cards=("card_id", "count"),
        total_card_limit=("card_limit_lakhs", "sum"),
        avg_monthly_spend=("avg_monthly_spend", "sum"),
        avg_monthly_repayment_pct=("avg_monthly_repayment_pct", "mean"),
        digital_transaction_pct=("digital_transaction_pct", "mean"),
        card_diversity=("card_issuer", "nunique"),
        premium_card_flag=("card_type", lambda s: int((s == "Premium").any())),
        emi_propensity=("has_emi_active", lambda s: int(bool(s.any()))),
        missed_payment_flag=("missed_payments_12m", lambda s: int((s > 0).any())),
        upi_credit_adoption=("upi_credit_usage", "max"),
        reward_engagement=("reward_redemption_rate", "mean"),
        months_active_max=("months_active", "max"),
        latent_archetype=("latent_archetype", "first"),
    )
    return agg.reset_index()


def cascading_filter(
    cards: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sequential market-sizing filter that deduplicates cards -> holders.

    1. Geography: keep cards in the top-50 Indian cities
    2. Bureau score: drop holders with credit_score < 600
    3. Spend tier: drop holders with avg_monthly_spend < INR 2,000
    4. Repayment: drop holders with avg repayment < 15%

    Returns
    -------
    (holders_df, filter_log) - the holder-level table and a per-stage log.
    """
    th = {**config.FILTER_THRESHOLDS, **(thresholds or {})}
    start_cards = len(cards)
    start_holders = cards["holder_id"].nunique()
    rows: List[Dict[str, object]] = [
        {"stage": "0. Raw (clean)", "cards": start_cards, "holders": start_holders,
         "removed_pct": 0.0, "cumulative_removed_pct": 0.0}
    ]

    geo = cards.loc[cards["city"].isin(TOP_50_CITY_NAMES)]
    holders = aggregate_to_holders(geo)
    _log_stage(rows, "1. Geography (top-50 cities)", len(geo), len(holders), start_holders)

    prev = len(holders)
    holders = holders.loc[holders["credit_score"] >= th["min_credit_score"]]
    _log_stage(rows, f"2. Bureau score >= {th['min_credit_score']:.0f}", None, len(holders), start_holders, prev)

    prev = len(holders)
    holders = holders.loc[holders["avg_monthly_spend"] >= th["min_monthly_spend"]]
    _log_stage(rows, f"3. Spend >= INR {th['min_monthly_spend']:,.0f}", None, len(holders), start_holders, prev)

    prev = len(holders)
    holders = holders.loc[holders["avg_monthly_repayment_pct"] >= th["min_repayment_pct"]]
    _log_stage(rows, f"4. Repayment >= {th['min_repayment_pct']:.3f}", None, len(holders), start_holders, prev)

    filter_log = pd.DataFrame(rows)
    filter_log["holders_scaled_to_market_M"] = (
        filter_log["holders"] / filter_log["holders"].iloc[-1] * config.REAL_MARKET_HOLDERS / 1e6
    ).round(2)
    if verbose:
        for _, r in filter_log.iterrows():
            log.info("  %-32s holders=%7s  removed=%5.1f%%  cumulative=%5.1f%%",
                     r["stage"], f"{int(r['holders']):,}", r["removed_pct"], r["cumulative_removed_pct"])
        log.info("Dedup complete: %s cards -> %s holders (represents 119M -> 38M)",
                 f"{start_cards:,}", f"{len(holders):,}")
    return holders.reset_index(drop=True), filter_log


def _log_stage(rows: List[Dict[str, object]], name: str, n_cards: Optional[int],
               n_holders: int, start_holders: int, prev_holders: Optional[int] = None) -> None:
    prev = prev_holders if prev_holders is not None else start_holders
    rows.append(
        {
            "stage": name,
            "cards": n_cards if n_cards is not None else np.nan,
            "holders": n_holders,
            "removed_pct": round((1 - n_holders / prev) * 100, 2),
            "cumulative_removed_pct": round((1 - n_holders / start_holders) * 100, 2),
        }
    )


# --------------------------------------------------------------------------- #
# 2C. Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(
    holders: pd.DataFrame,
    panel: pd.DataFrame,
    blur: float = config.ARCHETYPE_BLUR,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """
    Build the 12 holder-level engineered features.

    ``spend_volatility`` comes from the 24-month panel (std / mean of monthly
    spend).  A small Gaussian blur (``blur`` x per-archetype std) is added to the
    continuous features so cluster boundaries are realistic rather than razor sharp.
    """
    df = holders.copy()
    vol = (
        panel.groupby("holder_id")["monthly_spend"]
        .agg(lambda s: s.std() / s.mean() if s.mean() > 0 else 0.0)
        .rename("spend_volatility")
    )
    df = df.merge(vol, left_on="holder_id", right_index=True, how="left")
    df["spend_volatility"] = df["spend_volatility"].fillna(df["spend_volatility"].median())

    df["utilization_ratio"] = df["avg_monthly_spend"] / (df["total_card_limit"] * 100_000)
    df["repayment_discipline"] = df["avg_monthly_repayment_pct"]
    df["digital_savviness"] = df["digital_transaction_pct"]
    df["income_to_limit_ratio"] = df["annual_income_lakhs"] / df["total_card_limit"]

    if blur > 0:
        rng = np.random.default_rng(seed)
        continuous = [
            "total_card_limit", "utilization_ratio", "repayment_discipline", "digital_savviness",
            "spend_volatility", "income_to_limit_ratio", "upi_credit_adoption", "reward_engagement",
        ]
        group_std = df.groupby("latent_archetype")[continuous].transform("std")
        noise = rng.normal(0, 1, (len(df), len(continuous))) * group_std.to_numpy() * blur
        df[continuous] = df[continuous].to_numpy() + noise
        df["total_card_limit"] = df["total_card_limit"].clip(lower=0.1)
        df["utilization_ratio"] = df["utilization_ratio"].clip(lower=0.0)
        df["income_to_limit_ratio"] = df["income_to_limit_ratio"].clip(lower=0.1)
        for c in ("repayment_discipline", "digital_savviness", "upi_credit_adoption", "reward_engagement"):
            df[c] = df[c].clip(0, 1)
        df["spend_volatility"] = df["spend_volatility"].clip(lower=0.01)

    df["city_tier_code"] = df["city_tier"].map({"Tier-1": 1, "Tier-2": 2, "Tier-3": 3}).astype(int)
    df["gender_code"] = (df["gender"] == "Female").astype(int)

    keep = [
        "holder_id", "age", "gender", "city_tier", "city", "state", "annual_income_lakhs",
        "credit_score", "num_cards", "avg_monthly_spend", "city_tier_code", "gender_code",
        *config.ENGINEERED_FEATURES, "latent_archetype",
    ]
    out = df[keep].copy()
    for c in config.ENGINEERED_FEATURES:
        out[c] = out[c].astype(float).round(6)
    log.info("Engineered %d features for %s holders", len(config.ENGINEERED_FEATURES), f"{len(out):,}")
    return out


def run_preprocessing(cards: pd.DataFrame, panel: pd.DataFrame, save: bool = True):
    """Convenience wrapper: clean -> cascade -> features. Returns (features, purity_report, filter_log)."""
    clean, report = clean_data(cards)
    holders, filter_log = cascading_filter(clean)
    features = engineer_features(holders, panel)
    if save:
        holders.to_csv(config.HOLDERS_FILE, index=False)
        features.to_csv(config.FEATURES_FILE, index=False)
        filter_log.to_csv(config.REPORTS_DIR / "dedup_filter_log.csv", index=False)
        pd.DataFrame([{k: v for k, v in report.items() if k != "imputed_columns"}]).to_csv(
            config.REPORTS_DIR / "data_purity.csv", index=False
        )
    return features, report, filter_log
