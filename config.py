"""
Central configuration for the CRED Market Entry and Segmentation Diagnostic.

All paths are relative to the project root so the repository is portable.
All tunable parameters (sample sizes, archetype mixes, model settings) live here
so the synthetic data can be re-tuned without touching pipeline code.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

for _d in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_CARDS_FILE = RAW_DIR / "credit_cards_119M_sample.csv"
RAW_PANEL_FILE = RAW_DIR / "monthly_spend_panel.csv.gz"
HOLDERS_FILE = PROCESSED_DIR / "holders_38M_sample.csv"
FEATURES_FILE = PROCESSED_DIR / "features_engineered.csv"
CLUSTERED_FILE = PROCESSED_DIR / "clustered_holders.csv"
RBI_REFERENCE_FILE = EXTERNAL_DIR / "rbi_market_reference.csv"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Market scaling: the sample represents the full Indian market
# --------------------------------------------------------------------------- #
REAL_MARKET_CARDS = 119_000_000       # RBI FY24-25 credit cards in force
REAL_MARKET_HOLDERS = 38_000_000      # 119M / 3.1 avg cards per holder
AVG_CARDS_PER_HOLDER = 3.1
N_CARDS = 200_000                     # representative sample size
CARD_SCALE_FACTOR = REAL_MARKET_CARDS / N_CARDS

# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #
NOISE_FRACTION = 0.05                 # share of rows given nulls / invalid values
NON_TOP50_CITY_SHARE = 0.08           # holders living outside the top-50 cities
N_MONTHS = 24
TS_START = "2022-07-01"               # Jul-2022 .. Jun-2024
MONTHLY_TREND = 0.02                  # ~27% YoY growth in card spend (RBI FY24)
MARKET_SHOCK_SIGMA = 0.05             # common month-level shock innovation (spec: 5% noise)
MARKET_SHOCK_PERSISTENCE = 0.5        # AR(1) coefficient of the macro shock (0 = white noise)
HOLDER_NOISE_SIGMA = 0.12             # idiosyncratic month-level noise per holder

# Calendar-month seasonality multipliers (index = month number)
SEASONALITY = {
    1: 0.80,   # Jan post-festive dip
    2: 0.97,
    3: 1.10,   # Financial year-end
    4: 0.98,
    5: 1.00,
    6: 1.00,
    7: 1.00,
    8: 1.02,
    9: 1.03,
    10: 1.35,  # Diwali / festive
    11: 1.45,  # Diwali / festive
    12: 1.20,  # Year-end
}

# Six latent archetypes that the segmentation should recover.
# Behavioural traits are drawn at HOLDER level from Beta(mean, concentration)
# and card-level attributes jitter around the holder trait, so that a holder's
# cards look alike (as they do in real bureau data).  Values are the archetype
# trait means; "*_prob" entries are holder-level Bernoulli probabilities.
BETA_CONCENTRATION = 150.0            # higher = tighter archetypes
INCOME_SIGMA_WITHIN = 0.30            # log-normal sigma within an archetype (marginal ~0.8)
ARCHETYPES = {
    "Premium High-Spenders": {
        "share": 0.15, "income_mult": 2.6, "score_shift": 55, "age_shift": 6,
        "spend_ratio": 0.52, "repay": 0.90, "digital": 0.80, "upi": 0.08,
        "reward": 0.30, "emi_prob": 0.05, "missed_prob": 0.03, "premium_prob": 0.97,
        "cards_lambda": 2.0, "tier1_boost": 0.25, "limit_mult": 1.5,
        "volatility": 0.08, "festive_boost": 1.10,
    },
    "Reward Chasers": {
        "share": 0.20, "income_mult": 0.95, "score_shift": 15, "age_shift": 0,
        "spend_ratio": 0.35, "repay": 0.78, "digital": 0.72, "upi": 0.25,
        "reward": 0.90, "emi_prob": 0.08, "missed_prob": 0.04, "premium_prob": 0.25,
        "cards_lambda": 5.5, "tier1_boost": 0.05, "limit_mult": 1.0,
        "volatility": 0.16, "festive_boost": 1.20,
    },
    "EMI-Dependent": {
        "share": 0.15, "income_mult": 0.80, "score_shift": -15, "age_shift": 2,
        "spend_ratio": 0.38, "repay": 0.60, "digital": 0.50, "upi": 0.15,
        "reward": 0.20, "emi_prob": 0.97, "missed_prob": 0.25, "premium_prob": 0.03,
        "cards_lambda": 1.3, "tier1_boost": -0.05, "limit_mult": 0.95,
        "volatility": 0.10, "festive_boost": 1.05,
    },
    "Conservative Savers": {
        "share": 0.20, "income_mult": 1.0, "score_shift": 35, "age_shift": 8,
        "spend_ratio": 0.10, "repay": 0.96, "digital": 0.42, "upi": 0.06,
        "reward": 0.12, "emi_prob": 0.03, "missed_prob": 0.02, "premium_prob": 0.03,
        "cards_lambda": 0.1, "tier1_boost": -0.05, "limit_mult": 0.7,
        "volatility": 0.04, "festive_boost": 0.95,
    },
    "Digital-First": {
        "share": 0.15, "income_mult": 0.85, "score_shift": 0, "age_shift": -8,
        "spend_ratio": 0.25, "repay": 0.72, "digital": 0.94, "upi": 0.78,
        "reward": 0.50, "emi_prob": 0.10, "missed_prob": 0.08, "premium_prob": 0.05,
        "cards_lambda": 1.7, "tier1_boost": 0.15, "limit_mult": 0.9,
        "volatility": 0.26, "festive_boost": 1.10,
    },
    "Revolvers": {
        "share": 0.15, "income_mult": 0.75, "score_shift": -95, "age_shift": 0,
        "spend_ratio": 0.65, "repay": 0.22, "digital": 0.55, "upi": 0.35,
        "reward": 0.20, "emi_prob": 0.12, "missed_prob": 0.97, "premium_prob": 0.03,
        "cards_lambda": 2.2, "tier1_boost": -0.05, "limit_mult": 1.0,
        "volatility": 0.34, "festive_boost": 1.15,
    },
}
ARCHETYPE_BLUR = 0.30                 # Gaussian blur (fraction of cluster std) on engineered features

CARD_TYPE_PROBS = {"Premium": 0.20, "Rewards": 0.45, "Basic": 0.35}
CARD_ISSUER_PROBS = {
    "HDFC": 0.28, "SBI": 0.22, "ICICI": 0.18, "Axis": 0.12, "Kotak": 0.08, "Others": 0.12,
}
CITY_TIER_PROBS = {"Tier-1": 0.45, "Tier-2": 0.35, "Tier-3": 0.20}
GENDER_PROBS = {"Male": 0.65, "Female": 0.35}

# --------------------------------------------------------------------------- #
# Cleaning & cascading market-sizing filter
# --------------------------------------------------------------------------- #
MIN_PURITY = 0.92
NULL_IMPUTE_THRESHOLD = 0.05          # impute columns with < 5% nulls
MAX_NULLS_PER_ROW = 3                 # drop rows with more nulls than this
FILTER_THRESHOLDS = {
    "min_credit_score": 600,
    "min_monthly_spend": 2000.0,
    "min_repayment_pct": 0.15,
}

ENGINEERED_FEATURES = [
    "total_card_limit", "utilization_ratio", "repayment_discipline",
    "digital_savviness", "card_diversity", "premium_card_flag",
    "emi_propensity", "spend_volatility", "income_to_limit_ratio",
    "missed_payment_flag", "upi_credit_adoption", "reward_engagement",
]

# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
K_RANGE = list(range(2, 13))
TARGET_K = 6
TARGET_SILHOUETTE = 0.42
MIN_JACCARD = 0.75
SILHOUETTE_SAMPLE = 10_000            # rows used for silhouette scoring
HIERARCHICAL_SAMPLE = 8_000           # Ward linkage is O(n^2) memory
TSNE_SAMPLE = 4_000

# --------------------------------------------------------------------------- #
# Time series
# --------------------------------------------------------------------------- #
TS_HOLDOUT_MONTHS = 3
TS_FORECAST_HORIZON = 6
MAX_MAPE = 0.12
ARIMA_GRID = {"p": [0, 1, 2], "d": [0, 1], "q": [0, 1, 2]}
TS_SELECTION_CRITERION = "aicc"       # "aic" | "aicc" | "bic" - AICc is the small-sample (n = 21) safe choice
TS_ENFORCE_STATIONARITY = True        # stationary initialisation so likelihoods cover the whole sample
TS_MAX_ARMA_TERMS = 3                 # cap p+q+P+Q so 21 points cannot support an over-parameterised fit
SARIMA_SEASONAL_GRID = {"P": [0, 1], "D": [0, 1], "Q": [0, 1], "m": 12}

# --------------------------------------------------------------------------- #
# Propensity model
# --------------------------------------------------------------------------- #
LABEL_RULES = {
    "min_credit_score": 750,
    "min_monthly_spend": 15_000.0,
    "city_tiers": ["Tier-1", "Tier-2"],
    "age_range": (25, 45),
    "label_noise": 0.05,
}
TARGET_POSITIVE_RATE = (0.10, 0.15)
TEST_SIZE = 0.2
OPTUNA_TRIALS = 100
MIN_AUC = 0.80
PROPENSITY_FEATURES = ENGINEERED_FEATURES + [
    "age", "annual_income_lakhs", "credit_score", "num_cards",
    "avg_monthly_spend", "city_tier_code", "gender_code",
]
SHAP_SAMPLE = 2_000

# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #
MC_DEDUP_RUNS = 1_000
MC_CLUSTER_RUNS = 100
MC_AUC_RUNS = 100
MC_BOOTSTRAP_FRACTION = 0.8
MC_THRESHOLD_JITTER = {
    "min_credit_score": 30,           # 600 +/- 30
    "min_monthly_spend": 100.0,       # 2000 +/- 100
    "min_repayment_pct": 0.0075,      # 0.15 +/- 0.0075
}

# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
FIG_DPI = 130
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
