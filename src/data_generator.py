"""
Stage 1 - Synthetic data generation.

Builds a card-level dataset that mirrors the Indian credit-card market
(119M cards / 38M holders, RBI FY24-25) at a representative sample size,
plus a 24-month spend panel for time-series work.

Six latent archetypes are baked into the holder population so that the
downstream k-Means segmentation has a recoverable structure (k = 6).
Behavioural traits are drawn once per HOLDER and each of the holder's cards
jitters around those traits, mirroring how a person's cards look alike in
real bureau data.  Roughly 5% of rows receive intentional nulls / invalid
values so that the cleaning stage has something meaningful to do.
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from faker.providers.person.en_IN import Provider as IndianNames

import config
from src.utils import get_logger, set_seed

log = get_logger("data_generator")

# --------------------------------------------------------------------------- #
# Reference geography: top-50 Indian cities (city, state, tier)
# --------------------------------------------------------------------------- #
TOP_50_CITIES: List[Tuple[str, str, str]] = [
    ("Mumbai", "Maharashtra", "Tier-1"), ("Delhi", "Delhi", "Tier-1"),
    ("Bengaluru", "Karnataka", "Tier-1"), ("Hyderabad", "Telangana", "Tier-1"),
    ("Chennai", "Tamil Nadu", "Tier-1"), ("Kolkata", "West Bengal", "Tier-1"),
    ("Pune", "Maharashtra", "Tier-1"), ("Ahmedabad", "Gujarat", "Tier-1"),
    ("Surat", "Gujarat", "Tier-2"), ("Jaipur", "Rajasthan", "Tier-2"),
    ("Lucknow", "Uttar Pradesh", "Tier-2"), ("Kanpur", "Uttar Pradesh", "Tier-2"),
    ("Nagpur", "Maharashtra", "Tier-2"), ("Indore", "Madhya Pradesh", "Tier-2"),
    ("Thane", "Maharashtra", "Tier-2"), ("Bhopal", "Madhya Pradesh", "Tier-2"),
    ("Visakhapatnam", "Andhra Pradesh", "Tier-2"), ("Patna", "Bihar", "Tier-2"),
    ("Vadodara", "Gujarat", "Tier-2"), ("Ghaziabad", "Uttar Pradesh", "Tier-2"),
    ("Ludhiana", "Punjab", "Tier-2"), ("Agra", "Uttar Pradesh", "Tier-2"),
    ("Nashik", "Maharashtra", "Tier-2"), ("Faridabad", "Haryana", "Tier-2"),
    ("Meerut", "Uttar Pradesh", "Tier-2"), ("Rajkot", "Gujarat", "Tier-2"),
    ("Varanasi", "Uttar Pradesh", "Tier-2"), ("Srinagar", "Jammu & Kashmir", "Tier-2"),
    ("Aurangabad", "Maharashtra", "Tier-2"), ("Dhanbad", "Jharkhand", "Tier-2"),
    ("Amritsar", "Punjab", "Tier-2"), ("Navi Mumbai", "Maharashtra", "Tier-2"),
    ("Allahabad", "Uttar Pradesh", "Tier-2"), ("Ranchi", "Jharkhand", "Tier-2"),
    ("Gurugram", "Haryana", "Tier-2"), ("Noida", "Uttar Pradesh", "Tier-2"),
    ("Coimbatore", "Tamil Nadu", "Tier-2"), ("Kochi", "Kerala", "Tier-2"),
    ("Chandigarh", "Chandigarh", "Tier-2"), ("Mysuru", "Karnataka", "Tier-2"),
    ("Jabalpur", "Madhya Pradesh", "Tier-3"), ("Gwalior", "Madhya Pradesh", "Tier-3"),
    ("Vijayawada", "Andhra Pradesh", "Tier-3"), ("Jodhpur", "Rajasthan", "Tier-3"),
    ("Madurai", "Tamil Nadu", "Tier-3"), ("Raipur", "Chhattisgarh", "Tier-3"),
    ("Kota", "Rajasthan", "Tier-3"), ("Guwahati", "Assam", "Tier-3"),
    ("Thiruvananthapuram", "Kerala", "Tier-3"), ("Dehradun", "Uttarakhand", "Tier-3"),
]

# Smaller towns that fall outside the top-50 (removed by the geography filter)
OTHER_CITIES: List[Tuple[str, str, str]] = [
    ("Bareilly", "Uttar Pradesh", "Tier-3"), ("Aligarh", "Uttar Pradesh", "Tier-3"),
    ("Moradabad", "Uttar Pradesh", "Tier-3"), ("Tiruchirappalli", "Tamil Nadu", "Tier-3"),
    ("Bhubaneswar", "Odisha", "Tier-3"), ("Salem", "Tamil Nadu", "Tier-3"),
    ("Warangal", "Telangana", "Tier-3"), ("Guntur", "Andhra Pradesh", "Tier-3"),
    ("Bhiwandi", "Maharashtra", "Tier-3"), ("Saharanpur", "Uttar Pradesh", "Tier-3"),
    ("Gorakhpur", "Uttar Pradesh", "Tier-3"), ("Bikaner", "Rajasthan", "Tier-3"),
    ("Amravati", "Maharashtra", "Tier-3"), ("Cuttack", "Odisha", "Tier-3"),
    ("Firozabad", "Uttar Pradesh", "Tier-3"), ("Kollam", "Kerala", "Tier-2"),
    ("Durgapur", "West Bengal", "Tier-2"), ("Ajmer", "Rajasthan", "Tier-3"),
    ("Siliguri", "West Bengal", "Tier-2"), ("Jamshedpur", "Jharkhand", "Tier-2"),
    ("Udaipur", "Rajasthan", "Tier-3"), ("Mangaluru", "Karnataka", "Tier-2"),
    ("Hubballi", "Karnataka", "Tier-3"), ("Belagavi", "Karnataka", "Tier-3"),
    ("Tirupati", "Andhra Pradesh", "Tier-3"), ("Shimla", "Himachal Pradesh", "Tier-3"),
]

TOP_50_CITY_NAMES = {c[0] for c in TOP_50_CITIES}
ARCHETYPE_NAMES = list(config.ARCHETYPES.keys())


def _cities_by_tier(cities: List[Tuple[str, str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    out: Dict[str, List[Tuple[str, str]]] = {"Tier-1": [], "Tier-2": [], "Tier-3": []}
    for city, state, tier in cities:
        out[tier].append((city, state))
    return out


def _sample_names(rng: np.random.Generator, n: int) -> np.ndarray:
    """Vectorised Indian names from the Faker en_IN provider word lists."""
    first = np.array(list(IndianNames.first_names))
    last = np.array(list(IndianNames.last_names))
    return np.char.add(np.char.add(rng.choice(first, n), " "), rng.choice(last, n))


def _month_index() -> pd.DatetimeIndex:
    return pd.date_range(config.TS_START, periods=config.N_MONTHS, freq="MS")


def _beta_from_mean(rng: np.random.Generator, mean: np.ndarray, kappa: float) -> np.ndarray:
    """Beta draws parameterised by mean and concentration (a + b = kappa)."""
    mean = np.clip(mean, 0.02, 0.98)
    return rng.beta(mean * kappa, (1 - mean) * kappa)


# --------------------------------------------------------------------------- #
# Holder level
# --------------------------------------------------------------------------- #
def _generate_holders(rng: np.random.Generator, n_cards: int) -> pd.DataFrame:
    """
    Create the holder population.  Demographics AND behavioural traits are
    drawn once per holder from the archetype distributions; the card table
    later jitters around these holder-level traits.
    """
    n_holders_target = int(round(n_cards / config.AVG_CARDS_PER_HOLDER))
    shares = np.array([config.ARCHETYPES[a]["share"] for a in ARCHETYPE_NAMES])
    shares = shares / shares.sum()
    archetype_idx = rng.choice(len(ARCHETYPE_NAMES), size=n_holders_target, p=shares)

    def par(key: str) -> np.ndarray:
        return np.array([config.ARCHETYPES[ARCHETYPE_NAMES[i]][key] for i in archetype_idx], dtype=float)

    num_cards = np.minimum(rng.poisson(par("cards_lambda")) + 1, 8)
    # Trim so the total equals n_cards exactly
    cum = np.cumsum(num_cards)
    cut = int(np.searchsorted(cum, n_cards, side="left"))
    num_cards = num_cards[: cut + 1].copy()
    archetype_idx = archetype_idx[: cut + 1]
    num_cards[-1] = n_cards - int(num_cards[:-1].sum())
    if num_cards[-1] <= 0:
        num_cards, archetype_idx = num_cards[:-1], archetype_idx[:-1]
    n_holders = len(num_cards)
    kappa = config.BETA_CONCENTRATION

    age = np.clip(rng.normal(35 + par("age_shift"), 10), 21, 70).round().astype(int)
    gender = rng.choice(list(config.GENDER_PROBS), size=n_holders, p=list(config.GENDER_PROBS.values()))
    income = np.clip(rng.lognormal(3.0, config.INCOME_SIGMA_WITHIN, n_holders) * par("income_mult"), 3, 200)
    credit_score = np.clip(rng.normal(720 + par("score_shift"), 60), 300, 900).round().astype(int)

    # City tier with archetype tilt toward / away from Tier-1
    base = np.array(list(config.CITY_TIER_PROBS.values()))
    tier1_boost = par("tier1_boost")
    tier_probs = np.tile(base, (n_holders, 1))
    tier_probs[:, 0] += tier1_boost
    tier_probs[:, 1] -= tier1_boost * 0.4
    tier_probs[:, 2] -= tier1_boost * 0.6
    tier_probs = np.clip(tier_probs, 0.02, None)
    tier_probs /= tier_probs.sum(axis=1, keepdims=True)
    u = rng.random(n_holders)
    tier_idx = (u[:, None] > np.cumsum(tier_probs, axis=1)).sum(axis=1)
    tiers = np.array(list(config.CITY_TIER_PROBS))[np.minimum(tier_idx, 2)]

    # City / state: most holders in top-50 cities, a slice outside
    top, other = _cities_by_tier(TOP_50_CITIES), _cities_by_tier(OTHER_CITIES)
    outside = rng.random(n_holders) < config.NON_TOP50_CITY_SHARE
    city = np.empty(n_holders, dtype=object)
    state = np.empty(n_holders, dtype=object)
    for tier in ("Tier-1", "Tier-2", "Tier-3"):
        mask_in = (tiers == tier) & ~outside
        pool = top[tier]
        pick = rng.integers(0, len(pool), mask_in.sum())
        city[mask_in] = [pool[i][0] for i in pick]
        state[mask_in] = [pool[i][1] for i in pick]
        mask_out = (tiers == tier) & outside
        pool_o = other[tier] if other[tier] else other["Tier-3"]
        pick = rng.integers(0, len(pool_o), mask_out.sum())
        city[mask_out] = [pool_o[i][0] for i in pick]
        state[mask_out] = [pool_o[i][1] for i in pick]
        if not other[tier]:
            tiers[mask_out] = "Tier-3"

    holders = pd.DataFrame(
        {
            "holder_id": [f"H{idx:07d}" for idx in range(n_holders)],
            "holder_name": _sample_names(rng, n_holders),
            "age": age,
            "gender": gender,
            "city_tier": tiers,
            "city": city,
            "state": state,
            "annual_income_lakhs": income.round(2),
            "credit_score": credit_score,
            "num_cards": num_cards,
            "latent_archetype": np.array(ARCHETYPE_NAMES)[archetype_idx],
            # ---- holder-level behavioural traits (hidden; drive card rows) ----
            "_spend_ratio": _beta_from_mean(rng, par("spend_ratio"), kappa),
            "_repay": _beta_from_mean(rng, par("repay"), kappa),
            "_digital": _beta_from_mean(rng, par("digital"), kappa),
            "_upi": _beta_from_mean(rng, par("upi"), kappa),
            "_reward": _beta_from_mean(rng, par("reward"), kappa),
            "_emi": rng.random(n_holders) < par("emi_prob"),
            "_missed": rng.random(n_holders) < par("missed_prob"),
            "_premium": rng.random(n_holders) < par("premium_prob"),
            "_limit_mult": par("limit_mult"),
        }
    )
    return holders


# --------------------------------------------------------------------------- #
# Card level
# --------------------------------------------------------------------------- #
def _generate_cards(rng: np.random.Generator, holders: pd.DataFrame) -> pd.DataFrame:
    """Expand holders into card rows; card attributes jitter around holder traits."""
    rep = holders.loc[holders.index.repeat(holders["num_cards"])].reset_index(drop=True)
    n_cards = len(rep)
    card_pos = rep.groupby("holder_id").cumcount().to_numpy()  # 0 = first card

    def jitter(trait: np.ndarray, sd: float) -> np.ndarray:
        return np.clip(trait + rng.normal(0, sd, n_cards), 0.0, 1.0)

    # card_type: premium holders get >= 1 Premium card, others mostly Rewards/Basic
    prem_holder = rep["_premium"].to_numpy()
    u = rng.random(n_cards)
    rewards_share = config.CARD_TYPE_PROBS["Rewards"] / (
        config.CARD_TYPE_PROBS["Rewards"] + config.CARD_TYPE_PROBS["Basic"]
    )
    p_prem = np.where(prem_holder, 0.55, 0.03)
    card_type = np.where(
        (u < p_prem) | (prem_holder & (card_pos == 0)),
        "Premium",
        np.where(rng.random(n_cards) < rewards_share, "Rewards", "Basic"),
    )
    issuer = rng.choice(
        list(config.CARD_ISSUER_PROBS), size=n_cards, p=list(config.CARD_ISSUER_PROBS.values())
    )

    # Limit correlated with income: monthly income x U(0.5, 3.0) (lakhs)
    monthly_income_lakhs = rep["annual_income_lakhs"].to_numpy() / 12.0
    limit_lakhs = monthly_income_lakhs * rng.uniform(0.5, 3.0, n_cards) * rep["_limit_mult"].to_numpy()
    limit_lakhs = np.clip(limit_lakhs, 0.2, 100)

    # Spend correlated with limit: limit x spend_ratio x 1e5 / 12 (spend_ratio ~ Beta)
    spend_ratio = jitter(rep["_spend_ratio"].to_numpy(), 0.05)
    spend = np.clip(limit_lakhs * spend_ratio * 100_000 / 12, 200, None)

    repayment_pct = np.clip(jitter(rep["_repay"].to_numpy(), 0.05), 0.05, 1.0)
    months_active = rng.integers(6, 121, n_cards)
    upi = jitter(rep["_upi"].to_numpy(), 0.06)
    digital = jitter(rep["_digital"].to_numpy(), 0.05)
    reward = jitter(rep["_reward"].to_numpy(), 0.06)

    emi_holder = rep["_emi"].to_numpy()
    has_emi = emi_holder & ((rng.random(n_cards) < 0.6) | (card_pos == 0))
    missed_holder = rep["_missed"].to_numpy()
    missed = np.where(missed_holder, rng.poisson(1.2, n_cards) + (card_pos == 0).astype(int), 0)

    cards = pd.DataFrame(
        {
            "card_id": [str(uuid.UUID(int=int(x))) for x in rng.integers(0, 2**63, n_cards, dtype=np.int64)],
            "holder_id": rep["holder_id"].to_numpy(),
            "holder_name": rep["holder_name"].to_numpy(),
            "age": rep["age"].to_numpy(),
            "gender": rep["gender"].to_numpy(),
            "city_tier": rep["city_tier"].to_numpy(),
            "city": rep["city"].to_numpy(),
            "state": rep["state"].to_numpy(),
            "annual_income_lakhs": rep["annual_income_lakhs"].to_numpy(),
            "credit_score": rep["credit_score"].to_numpy(),
            "card_type": card_type,
            "card_issuer": issuer,
            "card_limit_lakhs": limit_lakhs.round(3),
            "avg_monthly_spend": spend.round(2),
            "avg_monthly_repayment_pct": repayment_pct.round(4),
            "months_active": months_active,
            "num_cards": rep["num_cards"].to_numpy(),
            "upi_credit_usage": upi.round(4),
            "has_emi_active": has_emi,
            "missed_payments_12m": missed,
            "reward_redemption_rate": reward.round(4),
            "digital_transaction_pct": digital.round(4),
            "latent_archetype": rep["latent_archetype"].to_numpy(),
        }
    )
    return cards


# --------------------------------------------------------------------------- #
# Monthly panel (time series)
# --------------------------------------------------------------------------- #
def _macro_shock_path(n_months: int, seed: int = config.SEED) -> np.ndarray:
    """
    Common market-wide monthly shock shared by every holder.

    Drawn from a dedicated RNG stream (``seed + 1``) so the macro path is
    identical regardless of the sample size, and modelled as a mildly
    persistent AR(1) process in log space (real macro shocks - fuel prices,
    rate changes, festive-season timing - do not reset every month).
    """
    rng = np.random.default_rng(seed + 1)
    eps = rng.normal(0, config.MARKET_SHOCK_SIGMA, n_months)
    rho = config.MARKET_SHOCK_PERSISTENCE
    s = np.zeros(n_months)
    for t in range(n_months):
        s[t] = (rho * s[t - 1] if t else 0.0) + eps[t]
    return np.exp(s)


def _generate_monthly_panel(rng: np.random.Generator, cards: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """
    Build a holder x month spend panel (24 months, Jul-2022 .. Jun-2024).

    monthly_spend = base x seasonality x trend x market_shock x holder_noise
    monthly_repayment = monthly_spend x repayment_pct x (1 + noise)
    """
    base = (
        cards.groupby("holder_id", sort=True)
        .agg(
            base_spend=("avg_monthly_spend", "sum"),
            repay=("avg_monthly_repayment_pct", "mean"),
            archetype=("latent_archetype", "first"),
        )
        .reset_index()
    )
    months = _month_index()
    n_h, n_m = len(base), len(months)

    seasonal = np.array([config.SEASONALITY[m.month] for m in months])
    festive = np.isin([m.month for m in months], [10, 11, 12])
    trend = (1 + config.MONTHLY_TREND) ** np.arange(n_m)
    market_shock = _macro_shock_path(n_m, seed=seed)

    vol = base["archetype"].map(lambda a: config.ARCHETYPES[a]["volatility"]).to_numpy()
    fboost = base["archetype"].map(lambda a: config.ARCHETYPES[a]["festive_boost"]).to_numpy()

    season_matrix = np.tile(seasonal, (n_h, 1))
    season_matrix[:, festive] = 1 + (season_matrix[:, festive] - 1) * fboost[:, None]
    holder_noise = np.exp(
        rng.normal(0, 1, (n_h, n_m)) * (config.HOLDER_NOISE_SIGMA * (vol / 0.12))[:, None]
    )

    spend = (
        base["base_spend"].to_numpy()[:, None]
        * season_matrix * trend[None, :] * market_shock[None, :] * holder_noise
    )
    repay = spend * base["repay"].to_numpy()[:, None] * (1 + rng.normal(0, 0.05, (n_h, n_m)))
    repay = np.clip(repay, 0, spend)

    panel = pd.DataFrame(
        {
            "holder_id": np.repeat(base["holder_id"].to_numpy(), n_m),
            "timestamp_month": np.tile(months.strftime("%Y-%m-%d").to_numpy(), n_h),
            "monthly_spend": spend.round(2).ravel(),
            "monthly_repayment": repay.round(2).ravel(),
        }
    )
    return panel


# --------------------------------------------------------------------------- #
# Noise injection
# --------------------------------------------------------------------------- #
def _inject_noise(rng: np.random.Generator, cards: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """
    Corrupt ~``fraction`` of rows so the cleaning stage is meaningful:
      * 50% of noisy rows: 1-2 random nulls (imputable)
      * 15%: 4+ nulls (dropped by the >3-nulls rule)
      * 20%: out-of-range credit score
      * 15%: implausible income (< 1 lakh)
    """
    cards = cards.copy()
    n = len(cards)
    noisy = rng.choice(n, size=int(n * fraction), replace=False)
    rng.shuffle(noisy)
    a, b, c = int(len(noisy) * 0.50), int(len(noisy) * 0.65), int(len(noisy) * 0.85)
    light_null, heavy_null, bad_score, bad_income = noisy[:a], noisy[a:b], noisy[b:c], noisy[c:]

    nullable = [
        "age", "gender", "city", "annual_income_lakhs", "card_type", "card_issuer",
        "months_active", "upi_credit_usage", "reward_redemption_rate", "digital_transaction_pct",
    ]
    for col_name in nullable:
        cards[col_name] = cards[col_name].astype(object)

    for idx in light_null:
        for col_name in rng.choice(nullable, size=rng.integers(1, 3), replace=False):
            cards.at[idx, col_name] = np.nan
    for idx in heavy_null:
        for col_name in rng.choice(nullable, size=rng.integers(4, 7), replace=False):
            cards.at[idx, col_name] = np.nan

    cards.loc[bad_score, "credit_score"] = rng.choice([250, 275, 920, 950, 999], size=len(bad_score))
    cards.loc[bad_income, "annual_income_lakhs"] = rng.uniform(0.1, 0.9, size=len(bad_income)).round(2)
    return cards


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_credit_card_data(
    n_cards: int = config.N_CARDS,
    seed: int = config.SEED,
    noise_fraction: float = config.NOISE_FRACTION,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate the card-level dataset and the monthly spend panel.

    Parameters
    ----------
    n_cards : number of card rows in the sample (scaled to 119M in the write-up)
    seed : RNG seed for reproducibility
    noise_fraction : share of card rows corrupted with nulls / invalid values
    save : write CSVs to ``data/raw`` when True

    Returns
    -------
    (cards, panel) DataFrames
    """
    rng = set_seed(seed)
    log.info("Generating %s synthetic card records (scaled to %sM real cards)",
             f"{n_cards:,}", config.REAL_MARKET_CARDS // 1_000_000)

    holders = _generate_holders(rng, n_cards)
    cards = _generate_cards(rng, holders)
    panel = _generate_monthly_panel(rng, cards, seed=seed)
    cards = _inject_noise(rng, cards, noise_fraction)

    _print_summary(cards, panel)
    if save:
        cards.to_csv(config.RAW_CARDS_FILE, index=False)
        panel.to_csv(config.RAW_PANEL_FILE, index=False, compression="gzip")
        _write_reference_table()
        log.info("Saved %s and %s", config.RAW_CARDS_FILE.name, config.RAW_PANEL_FILE.name)
    return cards, panel


def _print_summary(cards: pd.DataFrame, panel: pd.DataFrame) -> None:
    income = pd.to_numeric(cards["annual_income_lakhs"], errors="coerce")
    n_holders = cards["holder_id"].nunique()
    log.info("Total cards: %s | unique holders: %s | avg cards/holder: %.2f",
             f"{len(cards):,}", f"{n_holders:,}", len(cards) / n_holders)
    log.info("Income (lakhs) - mean %.1f | median %.1f | p90 %.1f | max %.1f",
             income.mean(), income.median(), income.quantile(0.9), income.max())
    log.info("Archetype mix: %s", cards.drop_duplicates("holder_id")["latent_archetype"]
             .value_counts(normalize=True).round(3).to_dict())
    log.info("Panel: %s rows across %d months", f"{len(panel):,}", panel["timestamp_month"].nunique())


def _write_reference_table() -> None:
    """Static reference figures (public RBI / NPCI / TransUnion CIBIL headline numbers)."""
    ref = pd.DataFrame(
        [
            ("credit_cards_in_force", 119_000_000, "RBI FY24-25 (approx)"),
            ("unique_cardholders_est", 38_000_000, "119M / 3.1 cards per holder"),
            ("avg_cards_per_holder", 3.1, "CIBIL multi-card estimate"),
            ("upi_credit_yoy_growth_pct", 180, "NPCI credit-on-UPI growth"),
            ("cred_valuation_inr_cr", 2735, "Company disclosures (approx)"),
            ("sample_cards", config.N_CARDS, "This project"),
            ("card_scale_factor", config.CARD_SCALE_FACTOR, "119M / sample"),
        ],
        columns=["metric", "value", "source_note"],
    )
    ref.to_csv(config.RBI_REFERENCE_FILE, index=False)


def load_raw(panel: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load previously generated raw files."""
    cards = pd.read_csv(config.RAW_CARDS_FILE)
    pnl = pd.read_csv(config.RAW_PANEL_FILE) if panel else None
    return cards, pnl


if __name__ == "__main__":  # pragma: no cover
    from src.utils import setup_logging

    setup_logging()
    generate_credit_card_data()
