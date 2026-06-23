import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import glob

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales Forecast",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { color: #1565C0; }
    h2 { color: #283593; }
    .metric-card {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
        border-left: 4px solid #1565C0;
    }
    .winner-badge {
        background: #E8F5E9;
        border-left: 4px solid #2E7D32;
        padding: 10px 16px;
        border-radius: 6px;
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Walmart_Spark.svg/200px-Walmart_Spark.svg.png", width=80)
st.sidebar.title("Walmart Sales\nForecasting")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🔍 EDA",
    "🤖 Model Comparison",
    "📈 Predictions",
    "🗂️ Raw Data",
])

PLOTS_DIR = "plots"

def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def show_plot(filename):
    path = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.info(f"Plot not found: {filename} — run main.py first.")

def all_plots():
    return sorted(glob.glob(os.path.join(PLOTS_DIR, "*.png")))

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
df_master = load_csv("retail_master_merged.csv")
df_comp   = load_csv("model_comparison.csv")
df_pred   = load_csv("predictions.csv")

data_ready = df_master is not None

# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────
if page == "📊 Overview":
    st.title("🛒 Walmart Store Sales Forecasting")
    st.markdown("End-to-end retail sales forecasting using **9 models** — AR, ML, and Deep Learning.")
    st.markdown("---")

    if not data_ready:
        st.error("⚠️ No data found. Run **main.py** first to generate outputs.")
        st.code("python main.py", language="bash")
        st.stop()

    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df_master):,}")
    with col2:
        n_stores = df_master["Store"].nunique() if "Store" in df_master.columns else "—"
        st.metric("Stores", n_stores)
    with col3:
        n_depts = df_master["Dept"].nunique() if "Dept" in df_master.columns else "—"
        st.metric("Departments", n_depts)
    with col4:
        if "Weekly_Sales" in df_master.columns:
            total = df_master["Weekly_Sales"].sum()
            st.metric("Total Sales", f"${total/1e9:.2f}B")

    st.markdown("---")

    # Winner
    if df_comp is not None:
        best = df_comp.sort_values("RMSE").iloc[0]
        st.markdown(f"""
        <div class="winner-badge">
        🏆 Best Model: &nbsp;<strong>{best['Model']}</strong>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        RMSE: <strong>${best['RMSE']:,.2f}</strong>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        MAE: <strong>${best['MAE']:,.2f}</strong>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        R²: <strong>{best['R2']:.4f}</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Models Used")
    model_info = {
        "Baseline — Lag 1 Week":       "Predicts next week = current week. Simplest possible baseline.",
        "Baseline — Rolling Mean 4W":  "Predicts next week = average of last 4 weeks.",
        "Linear Regression":           "ML regression on all engineered features.",
        "Random Forest":               "Ensemble of 200 trees on all features. Captures non-linearity.",
        "XGBoost":                     "Gradient boosted trees. Often best on tabular retail data.",
        "ARIMA(2,1,2)":               "Autoregressive model on single store-dept time series.",
        "SARIMAX (ARX)":              "ARIMA + seasonal component + external regressors (ARX).",
        "Prophet":                     "Facebook's trend + seasonality decomposition model.",
        "LSTM":                        "Deep learning sequence model. Learns temporal patterns.",
    }
    for name, desc in model_info.items():
        st.markdown(f"**{name}** — {desc}")

    st.markdown("---")
    st.subheader("Sales Trend")
    show_plot("eda_01_total_sales_trend.png")

# ─────────────────────────────────────────────
# PAGE: EDA
# ─────────────────────────────────────────────
elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")

    if not data_ready:
        st.error("Run main.py first."); st.stop()

    eda_plots = [
        ("eda_01_total_sales_trend.png",       "Total Weekly Sales Over Time"),
        ("eda_02_sales_by_store_type.png",     "Sales by Store Type"),
        ("eda_03_store_size_vs_sales.png",     "Store Size vs Sales"),
        ("eda_04_top_departments.png",         "Top 15 Departments"),
        ("eda_05_holiday_impact.png",          "Holiday Impact"),
        ("eda_06_markdown_vs_sales.png",       "Markdown vs Sales"),
        ("eda_07_monthly_seasonality.png",     "Monthly Seasonality"),
        ("eda_08_sales_distribution.png",      "Sales Distribution"),
        ("eda_09_correlation_heatmap.png",     "Correlation Heatmap"),
        ("eda_10_seasonality_heatmap.png",     "Seasonality Heatmap (Month × Week)"),
    ]

    for i in range(0, len(eda_plots), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j < len(eda_plots):
                fname, title = eda_plots[i+j]
                with col:
                    st.subheader(title)
                    show_plot(fname)

    st.markdown("---")
    st.subheader("Dataset Summary")
    st.dataframe(df_master.describe().T.round(2), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: MODEL COMPARISON
# ─────────────────────────────────────────────
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Performance Comparison")

    if df_comp is None:
        st.error("Run main.py first."); st.stop()

    comp = df_comp.sort_values("RMSE").reset_index(drop=True)
    comp.index += 1

    st.subheader("📋 Results Table")
    st.dataframe(
        comp.style
            .background_gradient(subset=["RMSE","MAE"], cmap="RdYlGn_r")
            .background_gradient(subset=["R2"], cmap="RdYlGn")
            .format({"RMSE":"{:,.2f}","MAE":"{:,.2f}","R2":"{:.4f}"}),
        use_container_width=True,
        height=380,
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RMSE Comparison")
        show_plot("comparison_rmse_all_models.png")
    with col2:
        st.subheader("MAE Comparison")
        show_plot("comparison_mae_all_models.png")

    st.subheader("R² Comparison")
    show_plot("comparison_r2_all_models.png")

    st.markdown("---")
    st.subheader("🌳 Feature Importances")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Random Forest**")
        show_plot("model_rf_feature_importance.png")
    with col2:
        st.markdown("**XGBoost**")
        show_plot("model_xgb_feature_importance.png")

    st.markdown("---")
    st.subheader("📉 Residual Analysis")
    col1, col2 = st.columns(2)
    with col1:
        show_plot("residuals_over_time.png")
    with col2:
        show_plot("residuals_distribution.png")

# ─────────────────────────────────────────────
# PAGE: PREDICTIONS
# ─────────────────────────────────────────────
elif page == "📈 Predictions":
    st.title("📈 Actual vs Predicted")

    pred_plots = sorted([f for f in os.listdir(PLOTS_DIR) if f.startswith("pred_")] if os.path.exists(PLOTS_DIR) else [])

    if not pred_plots:
        st.error("Run main.py first."); st.stop()

    # ML model predictions
    st.subheader("ML Models (All Store-Dept combinations)")
    ml_plots = [p for p in pred_plots if any(x in p for x in ["linear","random","xgboost"])]
    for p in ml_plots:
        name = p.replace("pred_","").replace("_"," ").replace(".png","").title()
        st.markdown(f"**{name}**")
        show_plot(p)

    st.markdown("---")

    # Time series
    st.subheader("Time Series Models (Single Store-Dept)")
    ts_plots = [p for p in pred_plots if any(x in p for x in ["arima","sarimax","prophet","lstm","all_ts"])]
    for p in ts_plots:
        name = p.replace("pred_","").replace("_"," ").replace(".png","").title()
        st.markdown(f"**{name}**")
        show_plot(p)

    st.markdown("---")

    # Interactive filter
    if df_pred is not None:
        st.subheader("🔎 Browse Predictions by Store & Department")
        stores = sorted(df_pred["Store"].unique())
        depts  = sorted(df_pred["Dept"].unique())

        col1, col2 = st.columns(2)
        sel_store = col1.selectbox("Store", stores)
        sel_dept  = col2.selectbox("Department", depts)

        filtered = df_pred[(df_pred["Store"]==sel_store) & (df_pred["Dept"]==sel_dept)].copy()
        filtered["Date"] = pd.to_datetime(filtered["Date"])

        if len(filtered) == 0:
            st.warning("No predictions for this combination.")
        else:
            fig, ax = plt.subplots(figsize=(12,4))
            ax.plot(filtered["Date"], filtered["Weekly_Sales"], label="Actual", color="black", linewidth=1.5)
            if "Pred_LR" in filtered.columns:
                ax.plot(filtered["Date"], filtered["Pred_LR"], label="Linear Regression",
                        color="#9C27B0", linestyle="--", linewidth=1.2)
            if "Pred_RF" in filtered.columns:
                ax.plot(filtered["Date"], filtered["Pred_RF"], label="Random Forest",
                        color="#1565C0", linestyle="--", linewidth=1.2)
            if "Pred_XGB" in filtered.columns:
                ax.plot(filtered["Date"], filtered["Pred_XGB"], label="XGBoost",
                        color="#E65100", linestyle="--", linewidth=1.2)
            ax.set_title(f"Store {sel_store} — Dept {sel_dept}")
            ax.set_xlabel("Date"); ax.set_ylabel("Weekly Sales ($)")
            ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.autofmt_xdate()
            st.pyplot(fig)
            plt.close()

            st.dataframe(filtered.set_index("Date").round(2), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: RAW DATA
# ─────────────────────────────────────────────
elif page == "🗂️ Raw Data":
    st.title("🗂️ Dataset Explorer")

    if not data_ready:
        st.error("Run main.py first."); st.stop()

    st.subheader("Merged Master Dataset")
    st.markdown(f"**Shape:** {df_master.shape[0]:,} rows × {df_master.shape[1]} columns")

    cols = st.multiselect("Select columns to view", df_master.columns.tolist(),
                          default=["Store","Dept","Date","Weekly_Sales","IsHoliday",
                                   "Temperature","Fuel_Price","CPI","Unemployment"])
    st.dataframe(df_master[cols].head(500), use_container_width=True)

    st.markdown("---")
    st.subheader("Download Files")
    for fname, label in [("retail_master_merged.csv","Merged Dataset"),
                          ("model_comparison.csv","Model Comparison"),
                          ("predictions.csv","Predictions")]:
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                st.download_button(f"⬇️ Download {label}", f, file_name=fname, mime="text/csv")
