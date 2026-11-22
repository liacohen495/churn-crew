"""Customer Retention Dashboard — one user (retention manager), one screen.

    streamlit run app/streamlit_app.py
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from churn import config as C  # noqa: E402

st.set_page_config(page_title="Customer Retention Dashboard", page_icon="📉", layout="wide")

RISK_ICON = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}


@st.cache_data
def load():
    scored = pd.read_csv(C.SCORED)
    metrics = json.loads(C.METRICS.read_text())
    status = json.loads(C.RUN_STATUS.read_text()) if C.RUN_STATUS.exists() else None
    return scored, metrics, status


def run_pipeline():
    with st.spinner("Running the pipeline…"):
        r = subprocess.run([sys.executable, str(ROOT / "run_pipeline.py")],
                           capture_output=True, text=True, cwd=ROOT)
    st.cache_data.clear()
    if r.returncode == 0:
        st.sidebar.success("Analysis updated")
    else:
        st.sidebar.error("Pipeline stopped at a validation gate — see logs/pipeline.log")


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Filters")
if not C.SCORED.exists():
    st.title("Customer Retention Dashboard")
    st.warning("No model output yet. Click **Run analysis** or run `python main.py`.")
    if st.sidebar.button("Run analysis", type="primary", width="stretch"):
        run_pipeline()
        st.rerun()
    st.stop()

scored, metrics, status = load()
risk = st.sidebar.selectbox("Risk level", ["All risk levels", "High", "Medium", "Low"])
contracts = st.sidebar.multiselect("Contract", sorted(scored["Contract"].unique()))
search = st.sidebar.text_input("Search customer ID")
test_only = st.sidebar.toggle("Only held-out test customers", value=False,
                              help="Scores for these customers were never seen in training.")
if st.sidebar.button("Run analysis", type="primary", width="stretch"):
    run_pipeline()
    st.rerun()

if status:
    icon = "✅" if status["status"] == "success" else "⚠️"
    st.sidebar.caption(f"{icon} Last CrewAI Flow run: **{status['status']}** "
                       f"({status['finished_at'][:16].replace('T', ' ')} UTC)")

# ------------------------------------------------------------------ header + KPIs
st.title("Customer Retention Dashboard")
best = metrics["models"][metrics["selected_model"]]
high = scored[scored["risk_level"] == "High"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("High-risk customers", f"{len(high):,}")
k2.metric("Churners caught (recall)", f"{best['recall']:.0%}",
          help="Share of customers who really left that the model flagged (test set).")
k3.metric("Monthly revenue at risk", f"${high['MonthlyCharges'].sum():,.0f}",
          help="Sum of monthly bills of high-risk customers.")
k4.metric("Total customers", f"{len(scored):,}")

tab1, tab2, tab3 = st.tabs(["Customer list", "Insights & charts", "Model comparison"])

# ------------------------------------------------------------------ tab 1
with tab1:
    df = scored.copy()
    if risk != "All risk levels":
        df = df[df["risk_level"] == risk]
    if contracts:
        df = df[df["Contract"].isin(contracts)]
    if search:
        df = df[df["customerID"].str.contains(search.strip(), case=False)]
    if test_only:
        df = df[df["in_test_set"]]

    view = pd.DataFrame({
        "Customer ID": df["customerID"],
        "Churn risk": df["risk_level"].map(RISK_ICON),
        "Probability": df["churn_probability"],
        "Why": df["why"],
        "Contract": df["Contract"],
        "Tenure (mo)": df["tenure"],
        "Monthly bill": df["MonthlyCharges"],
    })
    st.caption(f"{len(view):,} customers · sorted by churn probability")
    st.dataframe(
        view, hide_index=True, width="stretch", height=520,
        column_config={
            "Probability": st.column_config.ProgressColumn(
                "Probability", format="percent", min_value=0, max_value=1),
            "Monthly bill": st.column_config.NumberColumn(format="$%.2f"),
        },
    )
    st.download_button("Download list (CSV)", df.to_csv(index=False),
                       file_name="churn_risk_customers.csv", mime="text/csv")

# ------------------------------------------------------------------ tab 2
with tab2:
    left, right = st.columns([1.1, 1])
    with left:
        if C.INSIGHTS.exists():
            st.markdown(C.INSIGHTS.read_text(encoding="utf-8"))
    with right:
        charts = C.ANALYST_DIR / "charts"
        for name in ["contract.png", "tenure.png", "internet.png", "tech_support.png"]:
            if (charts / name).exists():
                st.image(str(charts / name), width="stretch")

# ------------------------------------------------------------------ tab 3
with tab3:
    rows = []
    for name, m in metrics["models"].items():
        rows.append({
            "Model": name.replace("_", " ").title() + (" ✅" if name == metrics["selected_model"] else ""),
            "ROC-AUC": m["roc_auc"], "Recall": m["recall"], "Precision": m["precision"],
            "Accuracy": m["accuracy"], "Churners in top 10%": m["top10_capture"],
        })
    comp = pd.DataFrame(rows)
    st.dataframe(comp, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in ["ROC-AUC", "Recall", "Precision", "Accuracy",
                                          "Churners in top 10%"]})
    st.bar_chart(comp.set_index("Model")[["ROC-AUC", "Recall", "Accuracy"]],
                 stack=False, height=300)
    st.info("The baseline reaches ~73% accuracy by predicting 'no churn' for everyone and catches "
            "no churners. That's why the model is chosen by ROC-AUC and recall.")
    with st.expander("Full evaluation report"):
        st.markdown(C.EVAL_REPORT.read_text(encoding="utf-8"))
    with st.expander("Model card"):
        st.markdown(C.MODEL_CARD.read_text(encoding="utf-8"))
