"""Crew 1 (Data Analyst) — deterministic work.

The agents call these functions through tools, so every number in the
artifacts comes from pandas, never from the LLM.
"""
import base64
import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import (  # noqa: E402
    ANALYST_DIR, CLEAN_DATA, CONTRACT, DATA_RAW, EDA_REPORT, EDA_STATS, ID_COL, TARGET,
)
from .logging_utils import get_logger  # noqa: E402

log = get_logger("analyst")

EXPECTED_RAW_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CHARTS_DIR = ANALYST_DIR / "charts"


# --------------------------------------------------------------------------- 1. ingest + clean
def ingest_and_clean() -> dict:
    """Load the raw Kaggle CSV, validate its shape, clean it, save clean_data.csv."""
    if not DATA_RAW.exists():
        raise FileNotFoundError(f"Raw dataset not found at {DATA_RAW}")
    raw = pd.read_csv(DATA_RAW)
    log.info("Loaded raw data: %s rows x %s cols", *raw.shape)

    missing = [c for c in EXPECTED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw data is missing expected columns: {missing}")

    df = raw[EXPECTED_RAW_COLUMNS].copy()
    notes = []

    # Strip whitespace in text columns
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].str.strip()

    # TotalCharges arrives as text; blanks are brand-new customers (tenure 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blank = int(df["TotalCharges"].isna().sum())
    if n_blank:
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
        notes.append(f"{n_blank} blank TotalCharges (tenure = 0, new customers) filled with 0.")

    # SeniorCitizen 0/1 -> Yes/No so all yes/no fields look the same
    df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})

    n_dup = int(df.duplicated(subset=[ID_COL]).sum())
    if n_dup:
        df = df.drop_duplicates(subset=[ID_COL])
        notes.append(f"Dropped {n_dup} duplicate customer IDs.")

    df = df.sort_values(ID_COL).reset_index(drop=True)  # deterministic row order
    df.to_csv(CLEAN_DATA, index=False)
    log.info("Saved %s (%s rows)", CLEAN_DATA.name, len(df))

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "churn_rate": round((df[TARGET] == "Yes").mean(), 4),
        "cleaning_notes": notes or ["No cleaning issues found."],
        "output": str(CLEAN_DATA.relative_to(CLEAN_DATA.parents[2])),
    }


# --------------------------------------------------------------------------- 2. EDA
def _churn_rate_by(df: pd.DataFrame, col: str) -> dict:
    s = df.groupby(col)[TARGET].apply(lambda x: (x == "Yes").mean()).round(4)
    return s.sort_values(ascending=False).to_dict()


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _bar(rates: dict, title: str, fname: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    labels = list(rates.keys())
    vals = [v * 100 for v in rates.values()]
    ax.barh(labels[::-1], vals[::-1], color="#2563eb")
    ax.set_xlabel("Churn rate (%)")
    ax.set_title(title, loc="left", fontsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=9)
    fig.savefig(CHARTS_DIR / fname, dpi=110, bbox_inches="tight")
    return _fig_to_b64(fig)


def run_eda() -> dict:
    """Descriptive statistics + charts. Saves eda_stats.json and eda_report.html."""
    df = pd.read_csv(CLEAN_DATA)
    CHARTS_DIR.mkdir(exist_ok=True)

    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
    ).astype(str)

    churned = df[df[TARGET] == "Yes"]
    stayed = df[df[TARGET] == "No"]
    stats = {
        "rows": len(df),
        "overall_churn_rate": round((df[TARGET] == "Yes").mean(), 4),
        "churned_customers": len(churned),
        "monthly_revenue_lost_to_churn": round(churned["MonthlyCharges"].sum(), 2),
        "avg_monthly_charges": {
            "churned": round(churned["MonthlyCharges"].mean(), 2),
            "stayed": round(stayed["MonthlyCharges"].mean(), 2),
        },
        "avg_tenure_months": {
            "churned": round(churned["tenure"].mean(), 1),
            "stayed": round(stayed["tenure"].mean(), 1),
        },
        "churn_rate_by": {
            c: _churn_rate_by(df, c)
            for c in ["Contract", "tenure_group", "InternetService", "PaymentMethod",
                      "TechSupport", "OnlineSecurity", "SeniorCitizen", "PaperlessBilling"]
        },
    }
    with open(EDA_STATS, "w") as f:
        json.dump(stats, f, indent=2)

    charts = [
        ("Contract", "Churn rate by contract type", "contract.png"),
        ("tenure_group", "Churn rate by tenure", "tenure.png"),
        ("InternetService", "Churn rate by internet service", "internet.png"),
        ("PaymentMethod", "Churn rate by payment method", "payment.png"),
        ("TechSupport", "Churn rate by tech support", "tech_support.png"),
    ]
    imgs = [(t, _bar(stats["churn_rate_by"][c], t, f)) for c, t, f in charts]

    # Monthly charges distribution
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(stayed["MonthlyCharges"], bins=30, alpha=0.6, label="Stayed", color="#94a3b8")
    ax.hist(churned["MonthlyCharges"], bins=30, alpha=0.75, label="Churned", color="#dc2626")
    ax.set_xlabel("Monthly charges ($)")
    ax.set_title("Monthly charges: churned vs stayed", loc="left", fontsize=11)
    ax.legend(frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(CHARTS_DIR / "monthly_charges.png", dpi=110, bbox_inches="tight")
    imgs.append(("Monthly charges: churned vs stayed", _fig_to_b64(fig)))

    _write_eda_html(df, stats, imgs)
    log.info("Saved %s and %s", EDA_STATS.name, EDA_REPORT.name)
    return {"output": [str(EDA_STATS.name), str(EDA_REPORT.name)], "stats": stats}


def _write_eda_html(df: pd.DataFrame, stats: dict, imgs: list) -> None:
    desc = df[NUMERIC].describe().round(2).to_html(classes="t")
    cards = "".join(
        f'<figure><img src="data:image/png;base64,{b}" alt="{t}"><figcaption>{t}</figcaption></figure>'
        for t, b in imgs
    )
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>EDA Report — Telco Customer Churn</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px;color:#0f172a}}
h1{{margin-bottom:4px}} .muted{{color:#64748b}}
.kpis{{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}}
.kpi{{border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;min-width:180px}}
.kpi b{{display:block;font-size:24px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
figure{{margin:0;border:1px solid #e2e8f0;border-radius:10px;padding:10px}}
figure img{{width:100%}} figcaption{{font-size:13px;color:#64748b}}
table.t{{border-collapse:collapse;font-size:13px}} table.t td,table.t th{{border:1px solid #e2e8f0;padding:4px 8px}}
</style></head><body>
<h1>EDA Report — Telco Customer Churn</h1>
<p class="muted">Generated automatically by the Data Analyst Crew. Source: Kaggle "Telco Customer Churn" (IBM sample).</p>
<div class="kpis">
<div class="kpi">Customers<b>{stats['rows']:,}</b></div>
<div class="kpi">Churn rate<b>{stats['overall_churn_rate']*100:.1f}%</b></div>
<div class="kpi">Monthly revenue lost<b>${stats['monthly_revenue_lost_to_churn']:,.0f}</b></div>
<div class="kpi">Avg tenure (churned / stayed)<b>{stats['avg_tenure_months']['churned']} / {stats['avg_tenure_months']['stayed']} mo</b></div>
</div>
<h2>Numeric summary</h2>{desc}
<h2>Churn drivers</h2><div class="grid">{cards}</div>
</body></html>"""
    EDA_REPORT.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------- 3. dataset contract
def build_contract() -> dict:
    """Write dataset_contract.json: schema, allowed values, assumptions, constraints."""
    df = pd.read_csv(CLEAN_DATA)
    columns = {}
    for c in df.columns:
        spec = {"dtype": "number" if c in NUMERIC else "string", "nullable": False}
        if c in NUMERIC:
            spec["min"] = 0
            spec["max"] = float(df[c].max()) * 1.5  # room for new data, still catches garbage
        elif c != ID_COL:
            spec["allowed_values"] = sorted(df[c].dropna().unique().tolist())
        columns[c] = spec
    columns[ID_COL]["unique"] = True

    contract = {
        "dataset": "telco_customer_churn",
        "version": "1.0",
        "file": "artifacts/analyst/clean_data.csv",
        "primary_key": ID_COL,
        "target": {"column": TARGET, "positive_class": "Yes", "allowed_values": ["No", "Yes"]},
        "min_rows": 1000,
        "columns": columns,
        "required_model_inputs": [c for c in df.columns if c not in (ID_COL, TARGET)],
        "assumptions": [
            "Each row is one customer at a single point in time (snapshot, not a time series).",
            "Churn = 'Yes' means the customer left within the last month.",
            "Blank TotalCharges belong to customers with tenure 0 and were set to 0.",
            "SeniorCitizen was converted from 0/1 to No/Yes.",
        ],
        "constraints": [
            "No missing values in any column.",
            "customerID must be unique.",
            "tenure, MonthlyCharges and TotalCharges must be >= 0.",
            "Categorical columns may only contain the listed allowed_values.",
        ],
    }
    with open(CONTRACT, "w") as f:
        json.dump(contract, f, indent=2)
    log.info("Saved %s (%s columns)", CONTRACT.name, len(columns))
    return {"output": CONTRACT.name, "columns": len(columns), "target": TARGET}


def read_eda_stats() -> dict:
    with open(EDA_STATS) as f:
        return json.load(f)


# --------------------------------------------------------------------------- fallback insights
def write_insights_template() -> None:
    """Non-LLM insights.md, used by run_pipeline.py (no API key needed)."""
    s = read_eda_stats()
    by = s["churn_rate_by"]
    pct = lambda v: f"{v*100:.0f}%"  # noqa: E731
    md = f"""# Business Insights — Telco Customer Churn

**{pct(s['overall_churn_rate'])} of {s['rows']:,} customers churned**, taking
**${s['monthly_revenue_lost_to_churn']:,.0f} in monthly revenue** with them.

## Key findings
1. **Contract type is the strongest signal.** Month-to-month: {pct(by['Contract']['Month-to-month'])} churn,
   two-year contracts: {pct(by['Contract']['Two year'])}.
2. **New customers leave most.** First-year customers churn at {pct(by['tenure_group']['0-12 months'])};
   customers past 4 years at {pct(by['tenure_group']['49-72 months'])}.
3. **Fiber optic customers churn more** ({pct(by['InternetService']['Fiber optic'])}) than DSL
   ({pct(by['InternetService']['DSL'])}).
4. **Electronic-check payers churn at {pct(by['PaymentMethod']['Electronic check'])}**, the highest of any payment method.
5. **Support matters.** No tech support: {pct(by['TechSupport']['No'])} churn vs {pct(by['TechSupport']['Yes'])} with it.

## Recommendations
- Offer month-to-month customers an incentive to move to a 1-year contract.
- Build an onboarding programme for the first 12 months.
- Bundle tech support / online security for fiber customers.
- Encourage automatic payment methods.
"""
    from .config import INSIGHTS
    INSIGHTS.write_text(md, encoding="utf-8")
