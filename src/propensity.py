"""
Stage 5 - Propensity modelling: which holders are most likely to adopt a
premium credit-card platform such as CRED?

Pipeline: rule-based synthetic label -> stratified split -> logistic-regression
baseline -> Optuna-tuned XGBoost -> SHAP explainability -> decile / lift
analysis -> beachhead sizing.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.utils import get_logger, holder_scale_factor

log = get_logger("propensity")
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def create_labels(
    df: pd.DataFrame,
    rules: Optional[Dict[str, object]] = None,
    seed: int = config.SEED,
    target_rate: Tuple[float, float] = config.TARGET_POSITIVE_RATE,
) -> Tuple[pd.Series, Dict[str, object]]:
    """
    Synthetic CRED-adoption label from CRED's known user profile:
    credit_score >= 750, monthly spend > INR 15k, Tier-1/2 city, age 25-45,
    with 5% of labels flipped so the problem is non-trivial.

    The credit-score threshold is auto-adjusted (in steps of 10) until the
    positive rate lands inside ``target_rate``.
    """
    r = {**config.LABEL_RULES, **(rules or {})}
    rng = np.random.default_rng(seed)
    threshold = int(r["min_credit_score"])
    lo, hi = target_rate
    p_noise = float(r["label_noise"])
    for _ in range(20):
        base = (
            (df["credit_score"] >= threshold)
            & (df["avg_monthly_spend"] > r["min_monthly_spend"])
            & (df["city_tier"].isin(r["city_tiers"]))
            & (df["age"].between(*r["age_range"]))
        ).astype(int)
        rate = base.mean()
        # expected positive rate AFTER flipping 5% of labels
        expected_final = rate * (1 - p_noise) + (1 - rate) * p_noise
        if lo <= expected_final <= hi:
            break
        threshold += 10 if expected_final > hi else -10
    noise = rng.random(len(base)) < r["label_noise"]
    label = base.to_numpy().copy()
    label[noise] = 1 - label[noise]
    y = pd.Series(label, index=df.index, name="cred_adopter")
    info = {"credit_score_threshold": threshold, "rule_positive_rate": float(rate),
            "final_positive_rate": float(y.mean()), "n_flipped": int(noise.sum())}
    log.info("Labels: credit_score >= %d | rule rate %.1f%% | after 5%% noise %.1f%% (%s positives)",
             threshold, rate * 100, y.mean() * 100, f"{int(y.sum()):,}")
    return y, info


def make_feature_matrix(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    feats = features or config.PROPENSITY_FEATURES
    return df[feats].astype(float)


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = config.TEST_SIZE, seed: int = config.SEED):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def train_baseline(X_train: pd.DataFrame, y_train: pd.Series, seed: int = config.SEED):
    """Logistic-regression baseline on standardised features."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    clf.fit(X_train, y_train)
    return clf


def _xgb(params: Dict[str, object], seed: int = config.SEED) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **params, objective="binary:logistic", eval_metric="auc", tree_method="hist",
        random_state=seed, n_jobs=-1, verbosity=0,
    )


def train_xgboost_optuna(
    X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = config.OPTUNA_TRIALS, seed: int = config.SEED,
) -> Tuple[xgb.XGBClassifier, optuna.Study, Dict[str, object]]:
    """
    Optuna (TPE) search over the spec'd hyper-parameter space, scored by
    validation ROC-AUC on a stratified 80/20 split of the training data.
    The best parameters are then refit on the full training set.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        }
        model = _xgb(params, seed)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed), study_name="xgb_propensity")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    log.info("Optuna: %d trials, best validation AUC = %.4f", len(study.trials), study.best_value)
    log.debug("Best params: %s", best)
    model = _xgb(best, seed)
    model.fit(X_train, y_train, verbose=False)
    return model, study, best


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """ROC-AUC, PR-AUC, and precision / recall / F1 at the F1-optimal threshold."""
    proba = model.predict_proba(X_test)[:, 1]
    prec, rec, thr = precision_recall_curve(y_test, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    best_i = int(np.argmax(f1))
    threshold = float(thr[best_i])
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "threshold": threshold,
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def shap_analysis(model: xgb.XGBClassifier, X: pd.DataFrame, sample: int = config.SHAP_SAMPLE, seed: int = config.SEED):
    """TreeExplainer SHAP values on a random test sample. Returns (explainer, shap_values, X_sample, top3)."""
    import shap

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Xs = X.iloc[idx].reset_index(drop=True)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(Xs)
    if isinstance(values, list):  # older shap API returns [neg, pos]
        values = values[-1]
    values = np.asarray(values)
    importance = pd.Series(np.abs(values).mean(axis=0), index=Xs.columns).sort_values(ascending=False)
    top3 = importance.index[:3].tolist()
    log.info("SHAP top features: %s", ", ".join(f"{f} ({importance[f]:.3f})" for f in importance.index[:5]))
    return explainer, values, Xs, top3, importance


# --------------------------------------------------------------------------- #
# Decile analysis & beachhead sizing
# --------------------------------------------------------------------------- #
def decile_analysis(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Sort by score, cut into 10 deciles (1 = highest), report rate, lift, capture."""
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(scores)})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["decile"] = (np.arange(len(df)) * 10 // len(df)) + 1
    overall = df["y"].mean()
    total_pos = df["y"].sum()
    g = df.groupby("decile").agg(count=("y", "size"), adopters=("y", "sum"), adoption_rate=("y", "mean"),
                                 min_score=("score", "min"), mean_score=("score", "mean")).reset_index()
    g["lift"] = g["adoption_rate"] / overall
    g["capture"] = g["adopters"] / total_pos
    g["cumulative_capture"] = g["capture"].cumsum()
    g["cumulative_lift"] = g["cumulative_capture"] / (g["decile"] / 10)
    g["overall_rate"] = overall
    top = g.iloc[0]
    log.info("Top decile: adoption %.1f%% (lift %.2fx) captures %.1f%% of adopters",
             top["adoption_rate"] * 100, top["lift"], top["capture"] * 100)
    return g


def beachhead_sizing(deciles: pd.DataFrame, n_holders: int) -> pd.DataFrame:
    """
    Translate deciles into market cohorts.  Core = top decile (10% of 38M =
    3.8M).  Adjacent = deciles 2-3, which still convert above the base rate
    in the model.  Total beachhead = core + adjacent (8-12M).
    """
    scale = holder_scale_factor(n_holders)
    per_decile_holders = n_holders / 10
    rows = []
    cohorts = [("Core (decile 1)", [1]), ("Adjacent (deciles 2-3)", [2, 3]), ("Beachhead total (deciles 1-3)", [1, 2, 3]),
               ("Remaining market (deciles 4-10)", list(range(4, 11)))]
    for name, ds in cohorts:
        sub = deciles[deciles["decile"].isin(ds)]
        rows.append(
            {
                "cohort": name,
                "deciles": "-".join(map(str, ds)) if len(ds) > 1 else str(ds[0]),
                "share_of_market_pct": len(ds) * 10,
                "sample_holders": int(per_decile_holders * len(ds)),
                "market_holders_M": round(per_decile_holders * len(ds) * scale / 1e6, 2),
                "adoption_rate": round(float(sub["adopters"].sum() / sub["count"].sum()), 4),
                "lift": round(float(sub["adopters"].sum() / sub["count"].sum() / deciles["overall_rate"].iloc[0]), 3),
                "share_of_adopters_pct": round(float(sub["capture"].sum() * 100), 1),
            }
        )
    out = pd.DataFrame(rows)
    log.info("Beachhead: core %.1fM, with adjacent cohorts %.1fM holders",
             out.loc[0, "market_holders_M"], out.loc[2, "market_holders_M"])
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_propensity(features: pd.DataFrame, n_trials: int = config.OPTUNA_TRIALS, save: bool = True) -> Dict[str, object]:
    """Full Stage-5 driver."""
    y, label_info = create_labels(features)
    X = make_feature_matrix(features)
    X_train, X_test, y_train, y_test = split_data(X, y)

    baseline = train_baseline(X_train, y_train)
    base_metrics = evaluate_model(baseline, X_test, y_test)
    log.info("Baseline logistic regression AUC = %.4f", base_metrics["roc_auc"])

    model, study, best_params = train_xgboost_optuna(X_train, y_train, n_trials=n_trials)
    metrics = evaluate_model(model, X_test, y_test)
    log.info("XGBoost test AUC = %.4f | PR-AUC = %.4f | F1 = %.3f @ thr %.2f (target AUC >= %.2f)",
             metrics["roc_auc"], metrics["pr_auc"], metrics["f1"], metrics["threshold"], config.MIN_AUC)

    proba = model.predict_proba(X_test)[:, 1]
    deciles = decile_analysis(y_test.to_numpy(), proba)
    beachhead = beachhead_sizing(deciles, n_holders=len(features))
    explainer, shap_values, X_shap, top3, shap_importance = shap_analysis(model, X_test)

    if save:
        joblib.dump(model, config.MODELS_DIR / "xgboost_propensity.pkl")
        joblib.dump(study, config.MODELS_DIR / "optuna_study.pkl")
        joblib.dump(explainer, config.MODELS_DIR / "shap_explainer.pkl")
        joblib.dump(baseline, config.MODELS_DIR / "logistic_baseline.pkl")
        deciles.to_csv(config.REPORTS_DIR / "decile_analysis.csv", index=False)
        beachhead.to_csv(config.REPORTS_DIR / "beachhead_sizing.csv", index=False)
        pd.DataFrame([{"model": "LogisticRegression", **base_metrics}, {"model": "XGBoost", **metrics}]).to_csv(
            config.REPORTS_DIR / "propensity_metrics.csv", index=False)
        shap_importance.rename("mean_abs_shap").to_csv(config.REPORTS_DIR / "shap_feature_importance.csv")
        pd.Series(best_params).to_json(config.REPORTS_DIR / "xgboost_best_params.json", indent=2)
        scored = features[["holder_id"]].copy()
        scored["cred_adopter"] = y.values
        scored["propensity_score"] = model.predict_proba(X)[:, 1]
        scored.to_csv(config.PROCESSED_DIR / "holder_propensity_scores.csv", index=False)

    return {
        "y": y, "label_info": label_info, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "baseline": baseline, "baseline_metrics": base_metrics, "model": model, "study": study, "best_params": best_params,
        "metrics": metrics, "proba": proba, "deciles": deciles, "beachhead": beachhead,
        "explainer": explainer, "shap_values": shap_values, "X_shap": X_shap, "top3": top3,
    }
