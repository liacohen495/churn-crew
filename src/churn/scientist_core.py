"""Crew 2 (Data Scientist) — deterministic work: features, training, evaluation, scoring."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    CLEAN_DATA, CONTRACT, EVAL_REPORT, FEATURES, HIGH_RISK, ID_COL, MEDIUM_RISK, METRICS,
    MODEL, MODEL_CARD, SCIENTIST_DIR, SCORED, SEED, TARGET, TEST_SIZE,
)
from .logging_utils import get_logger
from .validation import check_contract, check_required_features, load_contract, require

log = get_logger("scientist")

FEATURE_LIST = SCIENTIST_DIR / "feature_list.json"
BINARY_COLS = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

# Features the modeling step cannot run without (checked by a gate before training)
CORE_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges", "num_services", "avg_monthly_spend",
    "charge_increase", "is_new_customer", "has_auto_payment",
    "Contract_Month-to-month", "Contract_One year", "Contract_Two year",
    "InternetService_Fiber optic", "TechSupport_No", "OnlineSecurity_No",
]


# --------------------------------------------------------------------------- validate inputs
def validate_inputs() -> dict:
    """Gate: the clean dataset must match the dataset contract."""
    df = pd.read_csv(CLEAN_DATA)
    contract = load_contract(CONTRACT)
    require("contract_vs_clean_data", check_contract(df, contract))
    return {"status": "passed", "rows": len(df), "columns_checked": len(contract["columns"])}


# --------------------------------------------------------------------------- features
def build_features() -> dict:
    df = pd.read_csv(CLEAN_DATA)
    contract = load_contract(CONTRACT)
    require("contract_vs_clean_data", check_contract(df, contract))

    f = pd.DataFrame({ID_COL: df[ID_COL]})
    f["tenure"] = df["tenure"]
    f["MonthlyCharges"] = df["MonthlyCharges"]
    f["TotalCharges"] = df["TotalCharges"]

    # Engineered features
    f["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    f["avg_monthly_spend"] = (df["TotalCharges"] / df["tenure"].clip(lower=1)).round(2)
    f["charge_increase"] = (df["MonthlyCharges"] - f["avg_monthly_spend"]).round(2)
    f["is_new_customer"] = (df["tenure"] <= 12).astype(int)
    f["has_auto_payment"] = df["PaymentMethod"].str.contains("automatic").astype(int)

    for c in BINARY_COLS:
        f[c] = (df[c] == "Yes").astype(int)
    f["gender_Male"] = (df["gender"] == "Male").astype(int)

    multi = [c for c in contract["required_model_inputs"]
             if c not in BINARY_COLS + ["gender", "tenure", "MonthlyCharges", "TotalCharges"]]
    dummies = pd.get_dummies(df[multi], prefix_sep="_", dtype=int)
    f = pd.concat([f, dummies], axis=1)
    f[TARGET] = (df[TARGET] == "Yes").astype(int)

    feature_cols = [c for c in f.columns if c not in (ID_COL, TARGET)]
    f.to_csv(FEATURES, index=False)
    FEATURE_LIST.write_text(json.dumps({"features": feature_cols}, indent=2))
    log.info("Saved %s (%s features)", FEATURES.name, len(feature_cols))
    return {"output": FEATURES.name, "n_features": len(feature_cols),
            "engineered": ["num_services", "avg_monthly_spend", "charge_increase",
                           "is_new_customer", "has_auto_payment"]}


# --------------------------------------------------------------------------- train + compare
def _metrics(y, proba, thr=0.5) -> dict:
    pred = (proba >= thr).astype(int)
    order = np.argsort(-proba)
    top = order[: max(1, int(len(y) * 0.10))]
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4) if len(set(proba)) > 1 else 0.5,
        # Share of all churners found by calling the riskiest 10% of customers
        "top10_capture": round(y.iloc[top].sum() / y.sum(), 4),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def train_and_compare() -> dict:
    f = pd.read_csv(FEATURES)
    listed = json.loads(FEATURE_LIST.read_text())["features"]
    require("required_features", check_required_features(f, sorted(set(listed + CORE_FEATURES))))

    X, y = f[listed], f[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    variants = {
        "baseline_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5,
                                       random_state=SEED)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=5, class_weight="balanced",
            random_state=SEED, n_jobs=-1),
    }
    results = {}
    for name, m in variants.items():
        m.fit(X_tr, y_tr)
        results[name] = _metrics(y_te, m.predict_proba(X_te)[:, 1])
        log.info("%-20s AUC=%.3f recall=%.3f acc=%.3f", name,
                 results[name]["roc_auc"], results[name]["recall"], results[name]["accuracy"])

    candidates = {k: v for k, v in results.items() if k != "baseline_majority"}
    best = max(candidates, key=lambda k: (candidates[k]["roc_auc"], k))
    log.info("Selected model: %s", best)

    bundle = {"model": variants[best], "model_name": best, "features": listed,
              "explainer": variants["logistic_regression"], "seed": SEED}
    joblib.dump(bundle, MODEL)

    summary = {
        "selected_model": best,
        "selection_rule": "highest ROC-AUC on the held-out test set",
        "train_rows": len(X_tr), "test_rows": len(X_te),
        "test_churn_rate": round(y_te.mean(), 4),
        "decision_threshold": 0.5,
        "models": results,
        "top_drivers": _global_drivers(variants["logistic_regression"], listed),
    }
    METRICS.write_text(json.dumps(summary, indent=2))
    _write_eval_report(summary)
    _score_all(f, bundle, set(X_te.index))
    return {k: summary[k] for k in ("selected_model", "selection_rule")} | {
        "metrics": {k: {m: v[m] for m in ("roc_auc", "recall", "accuracy", "top10_capture")}
                    for k, v in results.items()}}


# --------------------------------------------------------------------------- explanations
_LABELS = {
    "Contract_Month-to-month": "Monthly contract",
    "Contract_One year": "One-year contract",
    "Contract_Two year": "Two-year contract",
    "InternetService_Fiber optic": "Fiber optic",
    "InternetService_DSL": "DSL internet",
    "InternetService_No": "No internet service",
    "PaymentMethod_Electronic check": "Pays by electronic check",
    "PaymentMethod_Mailed check": "Pays by mailed check",
    "TechSupport_No": "No tech support",
    "TechSupport_Yes": "Has tech support",
    "OnlineSecurity_No": "No online security",
    "OnlineSecurity_Yes": "Has online security",
    "OnlineBackup_No": "No online backup",
    "DeviceProtection_No": "No device protection",
    "StreamingTV_Yes": "Streams TV",
    "StreamingMovies_Yes": "Streams movies",
    "MultipleLines_Yes": "Multiple lines",
    "PaperlessBilling": "Paperless billing",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Has partner",
    "Dependents": "Has dependents",
    "is_new_customer": "New customer",
    "has_auto_payment": "Automatic payment",
}
_NUMERIC_LABELS = {
    "tenure": ("New customer", "Long tenure"),
    "MonthlyCharges": ("Low bill", "High bill"),
    "TotalCharges": ("Low lifetime spend", "High lifetime spend"),
    "num_services": ("Few services", "Many services"),
    "avg_monthly_spend": ("Low average spend", "High average spend"),
    "charge_increase": ("Bill went down", "Bill went up recently"),
}


# Only these phrases are shown, so every reason reads as a plain business fact.
_RISK_REASONS = {
    "New customer", "Monthly contract", "Fiber optic", "No tech support", "No online security",
    "No online backup", "No device protection", "Pays by electronic check", "Paperless billing",
    "Senior citizen", "High bill", "Bill went up recently", "Streams TV", "Streams movies",
    "Multiple lines",
}
_SAFE_REASONS = {
    "Long tenure", "Two-year contract", "One-year contract", "DSL internet", "No internet service",
    "Has tech support", "Has online security", "Automatic payment", "Has partner",
    "Has dependents", "Low bill",
}


def _label(feat: str, scaled: float) -> str | None:
    if feat in _NUMERIC_LABELS:
        return _NUMERIC_LABELS[feat][1 if scaled > 0 else 0]
    return _LABELS.get(feat, feat.replace("_", ": "))


def _global_drivers(lr: Pipeline, feats: list[str], k: int = 8) -> dict:
    coef = pd.Series(lr.named_steps["clf"].coef_[0], index=feats)
    return {
        "increase_churn": coef.sort_values(ascending=False).head(k).round(3).to_dict(),
        "decrease_churn": coef.sort_values().head(k).round(3).to_dict(),
    }


def _explain(lr: Pipeline, X: pd.DataFrame, proba: np.ndarray, k: int = 3) -> list[str]:
    """Top reasons per customer = largest coefficient x scaled value from the LR model.
    Binary features only count when active (value 1), so reasons read naturally."""
    scaler, clf = lr.named_steps["scale"], lr.named_steps["clf"]
    Z = scaler.transform(X)
    contrib = Z * clf.coef_[0]
    feats = list(X.columns)
    binary = {c for c in feats if set(X[c].unique()) <= {0, 1}}
    reasons = []
    for i in range(len(X)):
        row = contrib[i]
        risky = proba[i] >= MEDIUM_RISK
        order = np.argsort(-row) if risky else np.argsort(row)
        picked = []
        for j in order:
            if (risky and row[j] <= 0) or (not risky and row[j] >= 0):
                break
            f = feats[j]
            if f in binary and X.iat[i, j] == 0:
                continue
            lab = _label(f, Z[i, j])
            allowed = _RISK_REASONS if risky else _SAFE_REASONS
            if lab in allowed and lab not in picked:
                picked.append(lab)
            if len(picked) == k:
                break
        reasons.append(", ".join(picked) if picked else "No strong signal")
    return reasons


def _score_all(f: pd.DataFrame, bundle: dict, test_idx: set) -> None:
    X = f[bundle["features"]]
    proba = bundle["model"].predict_proba(X)[:, 1]
    out = pd.DataFrame({
        ID_COL: f[ID_COL],
        "churn_probability": proba.round(4),
        "risk_level": np.where(proba >= HIGH_RISK, "High",
                               np.where(proba >= MEDIUM_RISK, "Medium", "Low")),
        "why": _explain(bundle["explainer"], X, proba),
        "actually_churned": f[TARGET].map({1: "Yes", 0: "No"}),
        "in_test_set": [i in test_idx for i in f.index],
    })
    clean = pd.read_csv(CLEAN_DATA, usecols=[ID_COL, "tenure", "Contract", "MonthlyCharges"])
    out = out.merge(clean, on=ID_COL, how="left").sort_values(
        "churn_probability", ascending=False)
    out.to_csv(SCORED, index=False)
    log.info("Saved %s (%s customers scored)", SCORED.name, len(out))


# --------------------------------------------------------------------------- reports
def _write_eval_report(s: dict) -> None:
    rows = "\n".join(
        f"| {n} | {m['roc_auc']:.3f} | {m['recall']:.3f} | {m['precision']:.3f} | "
        f"{m['f1']:.3f} | {m['accuracy']:.3f} | {m['top10_capture']:.1%} |"
        for n, m in s["models"].items())
    best = s["models"][s["selected_model"]]
    tn, fp, fn, tp = np.array(best["confusion_matrix"]).ravel()
    up = ", ".join(f"`{k}`" for k in list(s["top_drivers"]["increase_churn"])[:5])
    down = ", ".join(f"`{k}`" for k in list(s["top_drivers"]["decrease_churn"])[:5])
    md = f"""# Evaluation Report

**Selected model:** `{s['selected_model']}` ({s['selection_rule']}).
Train / test split: {s['train_rows']:,} / {s['test_rows']:,} rows, stratified, seed {s.get('seed', SEED)}.
Test-set churn rate: {s['test_churn_rate']:.1%}. Decision threshold: {s['decision_threshold']}.

## Model comparison (held-out test set)

| Model | ROC-AUC | Recall | Precision | F1 | Accuracy | Churners in top 10% |
|---|---|---|---|---|---|---|
{rows}

**Why not accuracy?** The majority-class baseline already scores
{s['models']['baseline_majority']['accuracy']:.1%} accuracy by predicting "no churn" for
everyone, and it catches zero churners. ROC-AUC and recall measure what the business needs:
finding the customers who are about to leave.

## Selected model — confusion matrix (threshold 0.5)

| | Predicted stay | Predicted churn |
|---|---|---|
| **Actually stayed** | {tn} | {fp} |
| **Actually churned** | {fn} | {tp} |

The model catches **{tp} of {tp + fn} churners ({best['recall']:.0%})**. The cost is {fp}
false alarms, customers who get a retention offer they didn't need.

## Strongest drivers (logistic regression coefficients)
- Push churn **up**: {up}
- Push churn **down**: {down}
"""
    EVAL_REPORT.write_text(md, encoding="utf-8")


def read_metrics() -> dict:
    return json.loads(METRICS.read_text())


def write_model_card_template() -> None:
    """Non-LLM model_card.md, used by run_pipeline.py (no API key needed)."""
    s = read_metrics()
    b = s["models"][s["selected_model"]]
    md = f"""# Model Card — Customer Churn Risk

## Model purpose
Rank telecom customers by their probability of leaving in the next month, so the retention
team can contact the riskiest customers first. Intended user: a retention / customer-success manager.
Not intended for pricing, credit or any decision that denies a customer service.

## Model
`{s['selected_model']}` chosen over logistic regression / random forest by ROC-AUC.
Per-customer reasons come from the logistic regression variant (coefficient × scaled value).

## Training data summary
Kaggle "Telco Customer Churn" (IBM sample), 7,043 customers, 20 attributes plus engineered features.
{s['train_rows']:,} rows used for training, {s['test_rows']:,} held out. Churn rate ≈ 26.5%.

## Metrics (held-out test set)
- ROC-AUC: **{b['roc_auc']:.3f}**
- Recall: **{b['recall']:.1%}** — share of churners the model flags
- Precision: {b['precision']:.1%}
- Accuracy: {b['accuracy']:.1%} (majority baseline: {s['models']['baseline_majority']['accuracy']:.1%})
- Calling the riskiest 10% of customers reaches **{b['top10_capture']:.0%}** of all churners

## Limitations
- Single snapshot from one (fictional) US telecom; patterns may not transfer to other markets or time periods.
- No time dimension: the model cannot tell when a customer will leave.
- Scores for training-set customers are optimistic; only the test set measures true performance.
- Reasons are associations, not causes.

## Ethical considerations
- `gender` and `SeniorCitizen` are in the data. They are used as inputs here; a production
  version should test whether removing them changes fairness or accuracy.
- Flagging a customer must lead to a better offer, never to worse service.
- Customer IDs are pseudonymous; no personal contact data is used.
"""
    MODEL_CARD.write_text(md, encoding="utf-8")
