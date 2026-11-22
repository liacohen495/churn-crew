"""Customer Retention Dashboard — one user (retention manager), one screen.

    streamlit run app/streamlit_app.py
"""
import base64
import json
import subprocess
import sys
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from churn import config as C  # noqa: E402

st.set_page_config(page_title="Customer Retention Dashboard", page_icon="📉", layout="wide")

# ------------------------------------------------------------------ design tokens
INDIGO = "#4F46E5"
INK = "#111827"        # primary text   (17:1 on white)
INK_2 = "#374151"      # secondary text (10:1 on white)
MUTED = "#4B5563"      # captions       (7.5:1 on white, 7:1 on #F7F8FA)
BORDER = "#E5E7EB"
GRID = "#EEF0F4"
OTHER_BAR = "#858C99"  # non-selected model bars (3.4:1 on white)

RISK_LEVELS = ["High", "Medium", "Low"]
RISK_COLORS = ["red", "orange", "green"]  # muted theme colours set in .streamlit/config.toml

CSS = f"""
<style>
/* Hide Streamlit chrome: Deploy button, main menu, footer, top decoration */
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], #MainMenu,
[data-testid="stDecoration"], footer {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}

.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}

/* Header */
.app-header h1 {{ font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em;
                  color: {INK}; margin: 0 0 .25rem 0; padding: 0; line-height: 1.2; }}
.app-header p {{ font-size: 1.05rem; color: {MUTED}; margin: 0 0 1.4rem 0; }}

/* KPI cards */
.kpi-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px;
             margin-bottom: 1.4rem; }}
@media (max-width: 1000px) {{ .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 560px)  {{ .kpi-grid {{ grid-template-columns: 1fr; }} }}
.kpi {{ background: #fff; border: 1px solid {BORDER}; border-radius: 12px;
        box-shadow: 0 1px 2px rgba(16,24,40,.05); padding: 18px 20px; }}
.kpi-top {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; min-height: 42px; }}
.kpi-icon {{ width: 32px; height: 32px; border-radius: 8px; background: #EEF2FF; color: {INDIGO};
             display: inline-flex; align-items: center; justify-content: center; flex: none; }}
.kpi-label {{ font-size: .9rem; font-weight: 500; color: {INK_2}; }}
.kpi-value {{ font-size: 1.9rem; font-weight: 700; color: {INK}; letter-spacing: -0.02em;
              line-height: 1.15; font-variant-numeric: tabular-nums; }}
.kpi-caption {{ font-size: .85rem; color: {MUTED}; margin-top: 4px; line-height: 1.4; }}

/* White cards for st.container(key="card_…") */
[class*="st-key-card"] {{ background: #fff; border: 1px solid {BORDER}; border-radius: 12px;
                          box-shadow: 0 1px 2px rgba(16,24,40,.05); padding: 20px 22px; }}
.st-key-card_insights h1 {{ font-size: 1.35rem; font-weight: 700; line-height: 1.3; padding-top: 0; }}
.st-key-card_insights h2 {{ font-size: 1.05rem; font-weight: 600; padding-top: .6rem; }}
.st-key-card_insights p, .st-key-card_insights li {{ font-size: .95rem; line-height: 1.6; color: {INK_2}; }}
.card-title {{ font-size: 1rem; font-weight: 600; color: {INK}; margin: 0 0 .15rem 0; }}
.card-sub {{ font-size: .875rem; color: {MUTED}; margin: 0 0 .5rem 0; }}

/* Tabs */
[data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
button[data-baseweb="tab"] {{ padding: 8px 14px; border-radius: 8px 8px 0 0; }}
button[data-baseweb="tab"] p {{ font-size: .95rem; font-weight: 500; color: {INK_2}; }}
button[data-baseweb="tab"][aria-selected="true"] p {{ color: {INDIGO}; font-weight: 600; }}
[data-baseweb="tab-highlight"] {{ background-color: {INDIGO}; }}

/* Sidebar */
[data-testid="stSidebar"] h2 {{ font-size: 1.05rem; font-weight: 600; }}

/* Clear keyboard focus everywhere */
*:focus-visible, button:focus-visible, a:focus-visible, input:focus-visible,
[role="tab"]:focus-visible, [role="switch"]:focus-visible, [role="combobox"]:focus-visible {{
    outline: 2px solid {INDIGO} !important; outline-offset: 2px !important; }}
</style>
"""
st.html(CSS)

ICONS = {  # 18px stroke icons (Lucide-style), decorative
    "alert": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>'
             '<path d="M12 9v4"/><path d="M12 17h.01"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "dollar": '<path d="M12 2v20"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
             '<path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
}


def _icon_img(icon: str) -> str:
    # st.html strips inline <svg>, so the icon is embedded as a data-URI image instead.
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' "
           f"fill='none' stroke='{INDIGO}' stroke-width='2' stroke-linecap='round' "
           f"stroke-linejoin='round'>{ICONS[icon]}</svg>")
    return f'<img src="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}" alt="" width="18" height="18">'


def kpi_card(icon: str, label: str, value: str, caption: str) -> str:
    return (f'<div class="kpi"><div class="kpi-top"><span class="kpi-icon">{_icon_img(icon)}</span>'
            f'<span class="kpi-label">{escape(label)}</span></div>'
            f'<div class="kpi-value">{escape(value)}</div>'
            f'<div class="kpi-caption">{escape(caption)}</div></div>')


def card_heading(title: str, sub: str | None = None):
    st.html(f'<p class="card-title">{escape(title)}</p>'
            + (f'<p class="card-sub">{escape(sub)}</p>' if sub else ""))


def _base_chart(chart: alt.Chart) -> alt.Chart:
    return (chart.configure(font="Inter", background="#FFFFFF")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, labelFontSize=12, titleFontSize=12,
                            titleFontWeight=500, gridColor=GRID, domain=False, tickColor=GRID,
                            labelPadding=8)
            .configure_title(font="Inter", fontSize=14, fontWeight=600, color=INK, anchor="start"))


def rate_chart(rates: dict) -> alt.Chart:
    """Horizontal bars of churn rate per group; single series, direct value labels."""
    df = pd.DataFrame({"Group": list(rates.keys()), "Churn rate": list(rates.values())})
    base = alt.Chart(df).encode(
        y=alt.Y("Group:N", sort=None, title=None, axis=alt.Axis(labelLimit=180, ticks=False)),
        x=alt.X("Churn rate:Q", title="Churn rate", scale=alt.Scale(domain=[0, max(0.6, df["Churn rate"].max() * 1.15)]),
                axis=alt.Axis(format="%", tickCount=4)),
        tooltip=[alt.Tooltip("Group:N"), alt.Tooltip("Churn rate:Q", format=".1%")],
    )
    bars = base.mark_bar(color=INDIGO, cornerRadiusEnd=4, height={"band": 0.62})
    labels = base.mark_text(align="left", dx=6, color=INK_2, fontSize=12, fontWeight=500).encode(
        text=alt.Text("Churn rate:Q", format=".0%"))
    return _base_chart((bars + labels).properties(height=44 * len(df) + 30))


def metric_chart(comp: pd.DataFrame, metric: str) -> alt.Chart:
    """One metric across models; the selected model is indigo, others gray (also named in labels)."""
    df = comp[["Model", "selected", metric]].rename(columns={metric: "value"})
    base = alt.Chart(df).encode(
        y=alt.Y("Model:N", sort=None, title=None, axis=alt.Axis(labelLimit=200, ticks=False)),
        x=alt.X("value:Q", title=None, scale=alt.Scale(domain=[0, 1.12]),
                axis=alt.Axis(format=".1f", tickCount=3)),
        tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("value:Q", title=metric, format=".3f")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.6}).encode(
        color=alt.condition("datum.selected", alt.value(INDIGO), alt.value(OTHER_BAR)))
    labels = base.mark_text(align="left", dx=6, color=INK_2, fontSize=12, fontWeight=500).encode(
        text=alt.Text("value:Q", format=".2f"))
    return _base_chart((bars + labels).properties(height=170, title=metric))


@st.cache_data
def load():
    scored = pd.read_csv(C.SCORED)
    metrics = json.loads(C.METRICS.read_text())
    status = json.loads(C.RUN_STATUS.read_text()) if C.RUN_STATUS.exists() else None
    eda_path = C.ANALYST_DIR / "eda_stats.json"
    eda = json.loads(eda_path.read_text()) if eda_path.exists() else None
    return scored, metrics, status, eda


def run_pipeline():
    with st.spinner("Running the pipeline…"):
        r = subprocess.run([sys.executable, str(ROOT / "run_pipeline.py")],
                           capture_output=True, text=True, cwd=ROOT)
    st.cache_data.clear()
    if r.returncode == 0:
        st.sidebar.success("Analysis updated")
    else:
        st.sidebar.error("Pipeline stopped at a validation gate — see logs/pipeline.log")


def header():
    st.html('<div class="app-header"><h1>Customer Retention Dashboard</h1>'
            '<p>Which customers should the retention team call this week?</p></div>')


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Filters")
if not C.SCORED.exists():
    header()
    st.warning("No model output yet. Click **Run analysis** or run `python main.py`.")
    if st.sidebar.button("Run analysis", type="primary", width="stretch"):
        run_pipeline()
        st.rerun()
    st.stop()

scored, metrics, status, eda = load()
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
header()
best = metrics["models"][metrics["selected_model"]]
high = scored[scored["risk_level"] == "High"]

st.html('<div class="kpi-grid">'
        + kpi_card("alert", "High-risk customers", f"{len(high):,}",
                   f"{len(high) / len(scored):.0%} of all customers — call these first")
        + kpi_card("target", "Churners caught (recall)", f"{best['recall']:.0%}",
                   "Share of customers who really left that the model flagged (test set)")
        + kpi_card("dollar", "Monthly revenue at risk", f"${high['MonthlyCharges'].sum():,.0f}",
                   "Sum of monthly bills of high-risk customers")
        + kpi_card("users", "Total customers", f"{len(scored):,}", "Scored by the selected model")
        + '</div>')

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
        "Churn risk": df["risk_level"].map(lambda r: [r]),
        "Probability": df["churn_probability"],
        "Why": df["why"],
        "Contract": df["Contract"],
        "Tenure (mo)": df["tenure"],
        "Monthly bill": df["MonthlyCharges"],
    })
    with st.container(key="card_customers"):
        card_heading("Customers to contact",
                     f"{len(view):,} customers · sorted by churn probability")
        st.dataframe(
            view, hide_index=True, width="stretch", height=520,
            column_config={
                "Churn risk": st.column_config.MultiselectColumn(
                    "Churn risk", options=RISK_LEVELS, color=RISK_COLORS, width="small"),
                "Probability": st.column_config.ProgressColumn(
                    "Probability", format="percent", min_value=0, max_value=1, color=INDIGO),
                "Monthly bill": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
        st.download_button("Download list (CSV)", df.to_csv(index=False),
                           file_name="churn_risk_customers.csv", mime="text/csv",
                           icon=":material/download:")

# ------------------------------------------------------------------ tab 2
EDA_CHARTS = [  # (key in eda_stats["churn_rate_by"], title, fallback png)
    ("Contract", "Churn rate by contract type", "contract.png"),
    ("tenure_group", "Churn rate by tenure", "tenure.png"),
    ("InternetService", "Churn rate by internet service", "internet.png"),
    ("TechSupport", "Churn rate by tech support", "tech_support.png"),
]
with tab2:
    left, right = st.columns([1.1, 1], gap="medium")
    with left:
        if C.INSIGHTS.exists():
            with st.container(key="card_insights"):
                st.markdown(C.INSIGHTS.read_text(encoding="utf-8"))
    with right:
        charts = C.ANALYST_DIR / "charts"
        for i, (key, title, png) in enumerate(EDA_CHARTS):
            rates = (eda or {}).get("churn_rate_by", {}).get(key)
            if rates:
                with st.container(key=f"card_chart_{i}"):
                    card_heading(title)
                    st.altair_chart(rate_chart(rates), width="stretch", theme=None)
            elif (charts / png).exists():
                with st.container(key=f"card_chart_{i}"):
                    st.image(str(charts / png), width="stretch")

# ------------------------------------------------------------------ tab 3
with tab3:
    rows = []
    for name, m in metrics["models"].items():
        sel = name == metrics["selected_model"]
        rows.append({
            "Model": name.replace("_", " ").title() + (" (selected)" if sel else ""),
            "selected": sel,
            "ROC-AUC": m["roc_auc"], "Recall": m["recall"], "Precision": m["precision"],
            "Accuracy": m["accuracy"], "Churners in top 10%": m["top10_capture"],
        })
    comp = pd.DataFrame(rows)
    with st.container(key="card_models_table"):
        card_heading("Model comparison", "Held-out test set · selected model highlighted in indigo")
        st.dataframe(comp.drop(columns="selected"), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ["ROC-AUC", "Recall", "Precision", "Accuracy",
                                              "Churners in top 10%"]})
    with st.container(key="card_models_chart"):
        cols = st.columns(3, gap="medium")
        for col, metric in zip(cols, ["ROC-AUC", "Recall", "Accuracy"]):
            with col:
                st.altair_chart(metric_chart(comp, metric), width="stretch", theme=None)
    st.info("The baseline reaches ~73% accuracy by predicting 'no churn' for everyone and catches "
            "no churners. That's why the model is chosen by ROC-AUC and recall.",
            icon=":material/info:")
    with st.expander("Full evaluation report"):
        st.markdown(C.EVAL_REPORT.read_text(encoding="utf-8"))
    with st.expander("Model card"):
        st.markdown(C.MODEL_CARD.read_text(encoding="utf-8"))
