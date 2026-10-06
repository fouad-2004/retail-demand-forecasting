"""Historical dashboard. Run: python -m streamlit run retail_dashboard.py"""

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting import (
    build_weekly_sales,
    clean_transactions,
    score_forecasts,
    walk_forward_forecasts,
)

st.set_page_config(page_title="Retail Forecast Lab", page_icon="📈", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("RETAIL_DATA_PATH", ROOT / "data" / "Online Retail.xlsx"))
CUTOFF = pd.Timestamp("2011-10-03")
CALENDAR = pd.date_range("2010-12-12", "2011-12-04", freq="W-SUN")
COLORS = {"Actual": "#168C83", "LinearRegression": "#D97745", "MA6": "#6375BF"}
LABELS = {"Actual": "Recorded sales", "LinearRegression": "Linear regression", "MA6": "Six-week average"}


@st.cache_data(show_spinner="Reading and cleaning the transaction data…")
def load_data(path, modified_time):
    """The modification time invalidates the cache when the source file changes."""
    raw = pd.read_excel(path)
    required = {"InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice", "InvoiceDate"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="raise")
    if raw["InvoiceDate"].min() > pd.Timestamp("2010-12-06") or raw["InvoiceDate"].max() < pd.Timestamp("2011-12-05"):
        raise ValueError("Use the complete UCI Online Retail file covering December 2010–December 2011.")
    clean = clean_transactions(raw)
    counts = raw.resample("W-SUN", on="InvoiceDate").size().reindex(CALENDAR, fill_value=0)
    unknown = tuple(counts.index[counts == 0].strftime("%Y-%m-%d"))
    earlier = clean.loc[clean["InvoiceDate"] < CUTOFF].copy()
    earlier["ProductCode"] = earlier["StockCode"].astype(str)
    shortlist = (
        earlier.groupby("ProductCode")
        .agg(Description=("Description", "first"), Invoices=("InvoiceNo", "nunique"))
        .sort_values("Invoices", ascending=False)
        .head(10)
    )
    return raw, clean, unknown, shortlist


@st.cache_data(show_spinner="Running the fixed forecasting methods…")
def evaluate(weekly):
    training = weekly.loc[weekly.index < CUTOFF]
    validation = walk_forward_forecasts(training, training.index[-8:])
    test = walk_forward_forecasts(weekly, weekly.index[weekly.index >= CUTOFF])
    validation_scores = score_forecasts(validation)
    test_scores = score_forecasts(test)
    metrics = (
        validation_scores["MAE"].rename("Validation MAE").to_frame()
        .join(test_scores["MAE"].rename("Test MAE"))
        .join(validation_scores["WeeksScored"].rename("Validation weeks scored"))
        .join(test_scores["WeeksScored"].rename("Test weeks scored"))
    )
    return validation, test, metrics


def sales_chart(frame, title):
    chart = go.Figure()
    for column in frame.columns:
        chart.add_trace(go.Scatter(
            x=frame.index, y=frame[column], mode="lines+markers",
            name=LABELS.get(column, column),
            line={"color": COLORS.get(column, COLORS["Actual"]), "width": 2.5},
            marker={"size": 5}, connectgaps=False,
            hovertemplate="%{x|%d %b %Y}<br>%{y:,.1f} units<extra>%{fullData.name}</extra>",
        ))
    chart.update_layout(
        title=title, template="plotly_white", height=410,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        xaxis_title="Week ending Sunday", yaxis_title="Units sold",
        hovermode="x unified", legend={"orientation": "h", "y": 1.13},
    )
    return chart


st.title("Retail Forecast Lab")
st.caption("Historical weekly sales · transparent baselines · one-week-ahead evaluation")

if not DATA_PATH.is_file():
    st.info("Add Online Retail.xlsx to the data folder beside retail_dashboard.py, then refresh this page.")
    st.markdown("Dataset: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)")
    st.stop()

try:
    raw, cleaned, unknown_weeks, shortlist = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
except (ValueError, OSError, ImportError) as exc:
    st.error(str(exc))
    st.stop()

if shortlist.empty:
    st.error("No qualifying products were found in the selection period.")
    st.stop()

codes = shortlist.index.tolist()
default_index = codes.index("85123A") if "85123A" in codes else 0
st.sidebar.header("Explore a product")
product_code = st.sidebar.selectbox(
    "Product", codes, index=default_index,
    format_func=lambda code: f"{code} — {shortlist.loc[code, 'Description']}",
)
st.sidebar.caption("Shortlist ranked by invoice count before 3 October 2011.")
st.sidebar.divider()
st.sidebar.write("**Fixed methods**")
st.sidebar.write("Regression: three previous weeks")
st.sidebar.write("Baseline: six-week moving average")
st.sidebar.caption("Methods are updated weekly using earlier observations only.")

weekly = build_weekly_sales(cleaned, product_code, CALENDAR, unknown_weeks)
try:
    validation, test, metrics = evaluate(weekly)
except ValueError as exc:
    st.warning(f"Insufficient complete evaluation data: {exc}")
    st.stop()

st.subheader(str(shortlist.loc[product_code, "Description"]))
left, middle, right = st.columns(3)
left.metric("Recorded units · complete weeks", f"{weekly.sum():,.0f}")
middle.metric("Regression test MAE", f"{metrics.loc['LinearRegression', 'Test MAE']:,.2f}")
right.metric("Six-week average test MAE", f"{metrics.loc['MA6', 'Test MAE']:,.2f}")
st.caption("MAE is average absolute error in units. Lower is better. It is not an accuracy percentage.")

history_tab, evaluation_tab, invoices_tab = st.tabs(["Sales history", "Forecast evaluation", "Invoice breakdown"])

with history_tab:
    st.plotly_chart(sales_chart(weekly.rename("Actual").to_frame(), "Recorded weekly sales"), width="stretch")
    st.caption("The calendar excludes partial boundary weeks. Gaps mark weeks with no records anywhere in the source dataset; their cause is unknown.")
    st.dataframe(weekly.to_frame(), width="stretch")

with evaluation_tab:
    st.dataframe(metrics.rename(index=LABELS).round(2), width="stretch")
    period = st.radio("Evaluation period", ["Test · 9 weeks", "Validation · 8 weeks"], horizontal=True)
    predictions = test if period.startswith("Test") else validation
    st.plotly_chart(sales_chart(predictions, "One-week-ahead predictions"), width="stretch")
    detail = predictions.copy()
    detail["Regression absolute error"] = (detail["Actual"] - detail["LinearRegression"]).abs()
    detail["Baseline absolute error"] = (detail["Actual"] - detail["MA6"]).abs()
    st.dataframe(detail.round(2), width="stretch")
    st.download_button(
        "Download predictions CSV", detail.to_csv().encode("utf-8"),
        file_name=f"{product_code}_{'test' if period.startswith('Test') else 'validation'}_predictions.csv",
        mime="text/csv",
    )

with invoices_tab:
    st.write("Inspect how individual invoices contribute to a test week's sales.")
    ranked_weeks = test["Actual"].dropna().sort_values(ascending=False).index.tolist()
    chosen_week = st.selectbox(
        "Week ending", ranked_weeks,
        format_func=lambda date: f"{date:%d %b %Y} · {test.loc[date, 'Actual']:,.0f} units",
    )
    start = chosen_week - pd.Timedelta(days=6)
    end = chosen_week + pd.Timedelta(days=1)
    rows = cleaned.loc[
        (cleaned["StockCode"].astype(str) == product_code)
        & (cleaned["InvoiceDate"] >= start)
        & (cleaned["InvoiceDate"] < end)
    ]
    totals = rows.groupby("InvoiceNo")["Quantity"].sum().sort_values(ascending=False)
    if totals.empty:
        st.info("No qualifying invoices were recorded for this product during this week.")
    else:
        total = totals.sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Invoices", f"{len(totals):,}")
        c2.metric("Largest invoice share", f"{100 * totals.iloc[0] / total:.1f}%")
        c3.metric("Top three invoices' share", f"{100 * totals.head(3).sum() / total:.1f}%")
        invoice_frame = totals.rename("Product units").to_frame()
        invoice_frame["Share of weekly units (%)"] = 100 * invoice_frame["Product units"] / total
        st.dataframe(invoice_frame.round(2), width="stretch")
        st.caption("Concentration in a large invoice does not establish its cause or make it an invalid sale.")

with st.expander("How to interpret this experiment"):
    st.markdown("""
    - **Target:** recorded positive-quantity, positive-price, non-cancelled sales. Stock availability and unmet demand are unknown.
    - **Cleaning assumption:** retain one copy of each completely identical transaction row. Missing customer IDs are retained.
    - **Evaluation:** eight validation weeks and nine later test weeks; regression refits using earlier completed weeks. Regression predictions are clipped at zero.
    - **Limitations:** one year of history, small evaluation samples, and no promotion or inventory data. Sudden large orders can dominate weekly errors.
    - **Exploration:** the original experiment was developed on product 85123A. Other products reuse the fixed settings for exploration; they are not independent confirmation of decisions made after inspecting this historical test period.
    """)
    st.write(f"Source rows: {len(raw):,} · Cleaned rows: {len(cleaned):,} · Unknown calendar weeks: {len(unknown_weeks)}")
