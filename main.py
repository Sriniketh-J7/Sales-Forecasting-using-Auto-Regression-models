import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SALES_FILE   = "sales_data.csv"
FEAT_FILE    = "features_data.csv"
STORE_FILE   = "stores_data.csv"
PLOTS_DIR    = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_RATIO  = 0.8
RANDOM_STATE = 42
LSTM_WINDOW  = 12        # weeks lookback
LSTM_EPOCHS  = 30
FORECAST_STORE = None    # None = auto pick best store-dept
FORECAST_DEPT  = None

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))

def metrics(name, y_true, y_pred):
    return {"Model": name,
            "RMSE":  round(rmse(y_true, y_pred), 2),
            "MAE":   round(mae(y_true, y_pred), 2),
            "R2":    round(r2(y_true, y_pred), 4)}

def save_plot(name):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, name), dpi=150)
    plt.close()
    print(f"  saved: plots/{name}")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("\n=== 1. Loading Data ===")
for f in [SALES_FILE, FEAT_FILE, STORE_FILE]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing file: {f}  — place it in the same folder as main.py")

sales    = pd.read_csv(SALES_FILE)
features = pd.read_csv(FEAT_FILE)
stores   = pd.read_csv(STORE_FILE)

print(f"  sales:    {sales.shape}")
print(f"  features: {features.shape}")
print(f"  stores:   {stores.shape}")

# ─────────────────────────────────────────────
# 2. CLEAN & MERGE
# ─────────────────────────────────────────────
print("\n=== 2. Cleaning & Merging ===")

sales["Date"]    = pd.to_datetime(sales["Date"], dayfirst=False)
features["Date"] = pd.to_datetime(features["Date"], dayfirst=False)

# Normalize to week-ending Friday so keys match
sales["Date"]    = sales["Date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
features["Date"] = features["Date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()

# Drop negative Weekly_Sales (returns / data errors)
neg = (sales["Weekly_Sales"] < 0).sum()
if neg:
    print(f"  Dropping {neg} negative Weekly_Sales rows")
    sales = sales[sales["Weekly_Sales"] >= 0].copy()

# Deduplicate sales (sum duplicates)
sales = (sales.groupby(["Store", "Dept", "Date"], as_index=False)
               .agg({"Weekly_Sales": "sum", "IsHoliday": "max"}))

# Deduplicate features (average numerics)
feat_num = features.select_dtypes(include=np.number).columns.tolist()
feat_agg = {c: ("max" if c == "IsHoliday" else "mean") for c in features.columns
            if c not in ["Store", "Date"]}
features = features.groupby(["Store", "Date"], as_index=False).agg(feat_agg)

# Merge
df = sales.merge(features, on=["Store", "Date"], how="left", suffixes=("", "_feat"))
df = df.merge(stores, on="Store", how="left")

# Fill MarkDown NaN with 0 (no promotion = 0)
md_cols = [c for c in df.columns if c.lower().startswith("markdown")]
for c in md_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)

# Fill other numeric NaN with median
for c in df.select_dtypes(include=np.number).columns:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    df[c] = df[c].fillna(df[c].median())

# Categorical fill
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].fillna("Unknown")

# IQR winsorise Weekly_Sales per store-dept (remove outliers without dropping rows)
def iqr_clip(s, k=3.0):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return s.clip(q1 - k*iqr, q3 + k*iqr)

df["Weekly_Sales"] = df.groupby(["Store","Dept"])["Weekly_Sales"].transform(iqr_clip)

df = df.sort_values(["Store","Dept","Date"]).reset_index(drop=True)
df.to_csv("retail_master_merged.csv", index=False)
print(f"  Merged shape: {df.shape}  → saved retail_master_merged.csv")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n=== 3. Feature Engineering ===")

# Date features
df["Year"]         = df["Date"].dt.year
df["Month"]        = df["Date"].dt.month
df["Quarter"]      = df["Date"].dt.quarter
df["WeekOfYear"]   = df["Date"].dt.isocalendar().week.astype(int)
df["IsMonthEnd"]   = df["Date"].dt.is_month_end.astype(int)
df["IsQuarterEnd"] = df["Date"].dt.is_quarter_end.astype(int)
df["IsHoliday"]    = pd.to_numeric(df.get("IsHoliday", 0), errors="coerce").fillna(0).astype(int)

# Holiday proximity
df["Holiday_PrevWeek"] = df.groupby(["Store","Dept"])["IsHoliday"].shift(1).fillna(0).astype(int)
df["Holiday_NextWeek"] = df.groupby(["Store","Dept"])["IsHoliday"].shift(-1).fillna(0).astype(int)

# Markdown aggregate
df["MarkDown_Total"] = df[md_cols].sum(axis=1) if md_cols else 0.0
df["MarkDown_Any"]   = (df["MarkDown_Total"] > 0).astype(int)

# Sales density
df["Size"] = pd.to_numeric(df.get("Size", 1), errors="coerce").fillna(1)
df["Sales_per_sqft"] = df["Weekly_Sales"] / (df["Size"] + 1)

# Lag features
for lag in [1, 2, 4, 8, 13, 26, 52]:
    df[f"Sales_Lag_{lag}"] = df.groupby(["Store","Dept"])["Weekly_Sales"].shift(lag)

# Rolling stats
for win in [4, 8, 13, 26]:
    roll = df.groupby(["Store","Dept"])["Weekly_Sales"].transform(
        lambda s: s.shift(1).rolling(win, min_periods=1).mean())
    df[f"Sales_RollMean_{win}"] = roll
    roll_std = df.groupby(["Store","Dept"])["Weekly_Sales"].transform(
        lambda s: s.shift(1).rolling(win, min_periods=2).std())
    df[f"Sales_RollStd_{win}"] = roll_std

# Exponential weighted mean (captures recent trend better)
df["Sales_EWM_4"] = df.groupby(["Store","Dept"])["Weekly_Sales"].transform(
    lambda s: s.shift(1).ewm(span=4, min_periods=1).mean())
df["Sales_EWM_13"] = df.groupby(["Store","Dept"])["Weekly_Sales"].transform(
    lambda s: s.shift(1).ewm(span=13, min_periods=1).mean())

# Week-over-week change in external drivers
for col in ["Temperature","Fuel_Price","CPI","Unemployment"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"Delta_{col}"] = df.groupby("Store")[col].diff()

# Dept seasonal baseline (avg by week-of-year per dept)
seasonal = (df.groupby(["Dept","WeekOfYear"])["Weekly_Sales"]
              .mean().reset_index()
              .rename(columns={"Weekly_Sales":"Dept_Seasonal_Avg"}))
df = df.merge(seasonal, on=["Dept","WeekOfYear"], how="left")

# Store-level total & dept share
store_tot = (df.groupby(["Store","Date"])["Weekly_Sales"]
               .sum().reset_index()
               .rename(columns={"Weekly_Sales":"Store_Total_Sales"}))
df = df.merge(store_tot, on=["Store","Date"], how="left")
df["Dept_Share"] = df["Weekly_Sales"] / (df["Store_Total_Sales"] + 1)

# Store type one-hot
if "Type" in df.columns:
    df["Type"] = df["Type"].astype(str)
    type_dummies = pd.get_dummies(df["Type"], prefix="StoreType", drop_first=False)
    df = pd.concat([df.drop(columns=["Type"]), type_dummies], axis=1)

# Fill NaN from lags (no history at series start → 0)
lag_roll_cols = [c for c in df.columns if any(x in c for x in
    ["Lag","Roll","EWM","Delta","Dept_Seasonal","Store_Total","Dept_Share","Sales_per"])]
for c in lag_roll_cols:
    df[c] = df[c].replace([np.inf,-np.inf], np.nan).fillna(0)

df = df.sort_values(["Store","Dept","Date"]).reset_index(drop=True)
print(f"  Features after engineering: {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 4. EDA PLOTS
# ─────────────────────────────────────────────
print("\n=== 4. EDA Plots ===")

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10})

# 4a. Total weekly sales trend
weekly = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
plt.figure(figsize=(13,4))
plt.plot(weekly["Date"], weekly["Weekly_Sales"], color="#1f77b4", linewidth=1.2)
plt.title("Total Weekly Sales Over Time")
plt.xlabel("Date"); plt.ylabel("Total Weekly Sales ($)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.gcf().autofmt_xdate()
save_plot("eda_01_total_sales_trend.png")

# 4b. Sales by store type
st_cols = [c for c in df.columns if c.startswith("StoreType_")]
if st_cols:
    tmp = df[["Weekly_Sales"] + st_cols].copy()
    tmp["StoreType"] = tmp[st_cols].idxmax(axis=1).str.replace("StoreType_","",regex=False)
    by_type = tmp.groupby("StoreType")["Weekly_Sales"].mean().sort_values(ascending=False)
    plt.figure(figsize=(6,4))
    bars = plt.bar(by_type.index, by_type.values, color=["#1f77b4","#ff7f0e","#2ca02c"])
    plt.title("Average Weekly Sales by Store Type")
    plt.xlabel("Store Type"); plt.ylabel("Avg Weekly Sales ($)")
    for b in bars:
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+200,
                 f"${b.get_height():,.0f}", ha="center", va="bottom", fontsize=9)
    save_plot("eda_02_sales_by_store_type.png")

# 4c. Store size vs total sales
store_sum = df.groupby("Store").agg(Total_Sales=("Weekly_Sales","sum"),
                                     Size=("Size","first")).reset_index()
plt.figure(figsize=(7,5))
plt.scatter(store_sum["Size"], store_sum["Total_Sales"], alpha=0.7, color="#1f77b4")
plt.title("Store Size vs Total Sales")
plt.xlabel("Store Size (sq ft)"); plt.ylabel("Total Sales ($)")
save_plot("eda_03_store_size_vs_sales.png")

# 4d. Top 15 departments by total sales
dept_tot = df.groupby("Dept")["Weekly_Sales"].sum().sort_values(ascending=False).head(15)
plt.figure(figsize=(11,4))
plt.bar(dept_tot.index.astype(str), dept_tot.values, color="#2196F3")
plt.title("Top 15 Departments by Total Sales")
plt.xlabel("Department"); plt.ylabel("Total Sales ($)")
save_plot("eda_04_top_departments.png")

# 4e. Holiday vs non-holiday
hol = df.groupby("IsHoliday")["Weekly_Sales"].mean().reset_index()
hol["Label"] = hol["IsHoliday"].map({0:"Non-Holiday", 1:"Holiday"})
plt.figure(figsize=(5,4))
colors = ["#90CAF9","#F44336"]
bars = plt.bar(hol["Label"], hol["Weekly_Sales"], color=colors)
plt.title("Holiday vs Non-Holiday: Avg Weekly Sales")
plt.ylabel("Avg Weekly Sales ($)")
for b in bars:
    plt.text(b.get_x()+b.get_width()/2, b.get_height()+50,
             f"${b.get_height():,.0f}", ha="center", fontsize=10, fontweight="bold")
save_plot("eda_05_holiday_impact.png")

# 4f. Markdown vs sales
plt.figure(figsize=(7,5))
sub = df[df["MarkDown_Total"] > 0].sample(min(5000, len(df)), random_state=42)
plt.scatter(sub["MarkDown_Total"], sub["Weekly_Sales"], alpha=0.2, color="#E91E63", s=8)
plt.title("Total MarkDown vs Weekly Sales")
plt.xlabel("MarkDown Total ($)"); plt.ylabel("Weekly Sales ($)")
save_plot("eda_06_markdown_vs_sales.png")

# 4g. Monthly sales seasonality
df["MonthName"] = df["Date"].dt.strftime("%b")
month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
monthly = df.groupby("Month")["Weekly_Sales"].mean().reset_index()
monthly["MonthName"] = pd.to_datetime(monthly["Month"], format="%m").dt.strftime("%b")
monthly = monthly.sort_values("Month")
plt.figure(figsize=(10,4))
plt.plot(monthly["MonthName"], monthly["Weekly_Sales"], marker="o", color="#4CAF50", linewidth=2)
plt.fill_between(range(len(monthly)), monthly["Weekly_Sales"].values,
                 alpha=0.2, color="#4CAF50")
plt.xticks(range(len(monthly)), monthly["MonthName"])
plt.title("Average Weekly Sales by Month (Seasonality)")
plt.xlabel("Month"); plt.ylabel("Avg Weekly Sales ($)")
save_plot("eda_07_monthly_seasonality.png")

# 4h. Sales distribution
plt.figure(figsize=(8,4))
plt.hist(df["Weekly_Sales"], bins=80, color="#7B1FA2", edgecolor="white", linewidth=0.3)
plt.title("Weekly Sales Distribution")
plt.xlabel("Weekly Sales ($)"); plt.ylabel("Count")
save_plot("eda_08_sales_distribution.png")

# 4i. Correlation heatmap of key features
corr_cols = ["Weekly_Sales","Temperature","Fuel_Price","CPI","Unemployment",
             "MarkDown_Total","Size","IsHoliday","WeekOfYear","Month"]
corr_cols = [c for c in corr_cols if c in df.columns]
plt.figure(figsize=(9,7))
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink":0.8})
plt.title("Feature Correlation Heatmap")
save_plot("eda_09_correlation_heatmap.png")

# 4j. Seasonality heatmap (week × month)
pivot = df.groupby(["Month","WeekOfYear"])["Weekly_Sales"].mean().unstack("WeekOfYear").fillna(0)
plt.figure(figsize=(16,5))
sns.heatmap(pivot, cmap="YlOrRd", linewidths=0, cbar_kws={"label":"Avg Sales ($)"})
plt.title("Sales Seasonality: Month × Week of Year")
plt.xlabel("Week of Year"); plt.ylabel("Month")
save_plot("eda_10_seasonality_heatmap.png")

df.drop(columns=["MonthName"], inplace=True, errors="ignore")
print("  EDA complete — 10 plots saved")

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT (time-based)
# ─────────────────────────────────────────────
print("\n=== 5. Train/Test Split ===")

unique_dates = sorted(df["Date"].unique())
split_idx    = max(1, min(int(len(unique_dates) * TRAIN_RATIO), len(unique_dates)-1))
split_date   = unique_dates[split_idx - 1]

train_df = df[df["Date"] <= split_date].copy()
test_df  = df[df["Date"] >  split_date].copy()

print(f"  Split date : {pd.Timestamp(split_date).date()}")
print(f"  Train rows : {len(train_df):,}")
print(f"  Test rows  : {len(test_df):,}")

# ─────────────────────────────────────────────
# 6. ML FEATURES
# ─────────────────────────────────────────────
DROP_COLS = ["Weekly_Sales","Date","Store_Total_Sales","Sales_per_sqft","Dept_Share"]
feat_cols  = [c for c in df.columns if c not in DROP_COLS]

X_train = pd.get_dummies(train_df[feat_cols], drop_first=True)
X_test  = pd.get_dummies(test_df[feat_cols],  drop_first=True)
X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

X_train = X_train.replace([np.inf,-np.inf], np.nan).fillna(0)
X_test  = X_test.replace([np.inf,-np.inf],  np.nan).fillna(0)

y_train = train_df["Weekly_Sales"].astype(float).values
y_test  = test_df["Weekly_Sales"].astype(float).values

scaler        = StandardScaler()
X_train_sc    = scaler.fit_transform(X_train)
X_test_sc     = scaler.transform(X_test)

results = []

# ─────────────────────────────────────────────
# 7. BASELINE MODELS
# ─────────────────────────────────────────────
print("\n=== 6. Baseline Models ===")

# Baseline 1: Lag-1
pred_lag1 = test_df["Sales_Lag_1"].fillna(test_df["Weekly_Sales"].mean()).values
results.append(metrics("Baseline — Lag 1 Week", y_test, pred_lag1))
print("  Baseline Lag-1 done")

# Baseline 2: Rolling Mean 4 weeks
pred_roll4 = test_df["Sales_RollMean_4"].fillna(test_df["Weekly_Sales"].mean()).values
results.append(metrics("Baseline — Rolling Mean 4W", y_test, pred_roll4))
print("  Baseline Rolling Mean done")

# ─────────────────────────────────────────────
# 8. LINEAR REGRESSION
# ─────────────────────────────────────────────
print("\n=== 7. Linear Regression ===")
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
pred_lr = lr.predict(X_test_sc).clip(min=0)
results.append(metrics("Linear Regression", y_test, pred_lr))
print(f"  RMSE: {rmse(y_test, pred_lr):,.2f}")

# ─────────────────────────────────────────────
# 9. RANDOM FOREST
# ─────────────────────────────────────────────
print("\n=== 8. Random Forest ===")
rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=4,
                           random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test).clip(min=0)
results.append(metrics("Random Forest", y_test, pred_rf))
print(f"  RMSE: {rmse(y_test, pred_rf):,.2f}")

# Feature importance — RF
fi_rf = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(20)
plt.figure(figsize=(9,6))
fi_rf.sort_values().plot(kind="barh", color="#1f77b4")
plt.title("Random Forest — Top 20 Feature Importances")
plt.xlabel("Importance")
save_plot("model_rf_feature_importance.png")

# ─────────────────────────────────────────────
# 10. XGBOOST
# ─────────────────────────────────────────────
print("\n=== 9. XGBoost ===")
xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=RANDOM_STATE, n_jobs=-1,
                               eval_metric="rmse", verbosity=0)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
pred_xgb = xgb_model.predict(X_test).clip(min=0)
results.append(metrics("XGBoost", y_test, pred_xgb))
print(f"  RMSE: {rmse(y_test, pred_xgb):,.2f}")

# Feature importance — XGB
fi_xgb = pd.Series(xgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(20)
plt.figure(figsize=(9,6))
fi_xgb.sort_values().plot(kind="barh", color="#FF7043")
plt.title("XGBoost — Top 20 Feature Importances")
plt.xlabel("Importance")
save_plot("model_xgb_feature_importance.png")

# ─────────────────────────────────────────────
# 11. PICK BEST STORE-DEPT FOR TIME SERIES MODELS
# ─────────────────────────────────────────────
# Need a series with enough history (>= 104 weeks) and no gaps
series_len = df.groupby(["Store","Dept"])["Date"].count()
eligible   = series_len[series_len >= 80]

if FORECAST_STORE and FORECAST_DEPT:
    s_id, d_id = FORECAST_STORE, FORECAST_DEPT
else:
    s_id, d_id = int(eligible.index[0][0]), int(eligible.index[0][1])

print(f"\n  Time series models will use Store={s_id}, Dept={d_id}")

series_full = (df[(df["Store"]==s_id) & (df["Dept"]==d_id)]
               .sort_values("Date").set_index("Date"))
y_full = series_full["Weekly_Sales"].astype(float)

# External regressors for this series
exog_cols_available = [c for c in ["Temperature","Fuel_Price","CPI","Unemployment","IsHoliday","MarkDown_Total"]
                       if c in series_full.columns]
exog_full = series_full[exog_cols_available].astype(float)

y_tr_ts   = y_full[y_full.index <= split_date]
y_te_ts   = y_full[y_full.index >  split_date]
exog_tr   = exog_full[exog_full.index <= split_date]
exog_te   = exog_full[exog_full.index >  split_date]

n_test_ts = len(y_te_ts)

# ─────────────────────────────────────────────
# 12. ARIMA
# ─────────────────────────────────────────────
print("\n=== 10. ARIMA ===")
arima_pred = np.array([])
try:
    arima_model = ARIMA(y_tr_ts, order=(2,1,2))
    arima_fit   = arima_model.fit()
    arima_pred  = arima_fit.forecast(steps=n_test_ts).clip(min=0).values
    results.append(metrics("ARIMA(2,1,2)", y_te_ts.values, arima_pred))
    print(f"  RMSE: {rmse(y_te_ts.values, arima_pred):,.2f}")
except Exception as e:
    print(f"  ARIMA failed: {e}")

# ─────────────────────────────────────────────
# 13. SARIMAX (ARX — with exogenous features)
# ─────────────────────────────────────────────
print("\n=== 11. SARIMAX (ARX) ===")
sarimax_pred = np.array([])
try:
    sarimax_model = SARIMAX(y_tr_ts,
                             exog=exog_tr,
                             order=(1,1,1),
                             seasonal_order=(1,0,1,52),
                             enforce_stationarity=False,
                             enforce_invertibility=False)
    sarimax_fit  = sarimax_model.fit(disp=False, maxiter=100)
    sarimax_pred = sarimax_fit.forecast(steps=n_test_ts, exog=exog_te).clip(min=0).values
    results.append(metrics("SARIMAX (ARX)", y_te_ts.values, sarimax_pred))
    print(f"  RMSE: {rmse(y_te_ts.values, sarimax_pred):,.2f}")
except Exception as e:
    print(f"  SARIMAX failed: {e}")

# ─────────────────────────────────────────────
# 14. PROPHET
# ─────────────────────────────────────────────
print("\n=== 12. Prophet ===")
prophet_pred = np.array([])
try:
    p_train = y_tr_ts.reset_index().rename(columns={"Date":"ds","Weekly_Sales":"y"})
    p_model = Prophet(weekly_seasonality=True,
                      yearly_seasonality=True,
                      daily_seasonality=False,
                      seasonality_mode="multiplicative",
                      changepoint_prior_scale=0.05)

    # Add regressors
    for col in exog_cols_available:
        p_model.add_regressor(col)
        p_train[col] = exog_tr[col].values

    p_model.fit(p_train)

    future = pd.DataFrame({"ds": y_te_ts.index})
    for col in exog_cols_available:
        future[col] = exog_te[col].values

    forecast    = p_model.predict(future)
    prophet_pred = forecast["yhat"].clip(lower=0).values
    results.append(metrics("Prophet", y_te_ts.values, prophet_pred))
    print(f"  RMSE: {rmse(y_te_ts.values, prophet_pred):,.2f}")
except Exception as e:
    print(f"  Prophet failed: {e}")

# ─────────────────────────────────────────────
# 15. LSTM
# ─────────────────────────────────────────────
print("\n=== 13. LSTM ===")
lstm_pred_inv = np.array([])
try:
    mm = MinMaxScaler(feature_range=(0,1))
    y_scaled = mm.fit_transform(y_full.values.reshape(-1,1)).flatten()

    win = LSTM_WINDOW
    X_seq, y_seq, dates_seq = [], [], []
    for i in range(win, len(y_scaled)):
        X_seq.append(y_scaled[i-win:i])
        y_seq.append(y_scaled[i])
        dates_seq.append(y_full.index[i])

    X_seq      = np.array(X_seq).reshape(-1, win, 1)
    y_seq      = np.array(y_seq)
    dates_seq  = pd.DatetimeIndex(dates_seq)

    tr_mask = dates_seq <= split_date
    te_mask = ~tr_mask

    X_tr_seq = X_seq[tr_mask]; y_tr_seq = y_seq[tr_mask]
    X_te_seq = X_seq[te_mask]; y_te_seq = y_seq[te_mask]
    dates_te_seq = dates_seq[te_mask]

    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(win,1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    lstm_model.fit(X_tr_seq, y_tr_seq,
                   epochs=LSTM_EPOCHS, batch_size=16,
                   validation_split=0.1, callbacks=[es], verbose=0)

    lstm_pred_sc  = lstm_model.predict(X_te_seq, verbose=0).flatten()
    lstm_pred_inv = mm.inverse_transform(lstm_pred_sc.reshape(-1,1)).flatten().clip(min=0)
    lstm_actual   = mm.inverse_transform(y_te_seq.reshape(-1,1)).flatten()

    results.append(metrics("LSTM", lstm_actual, lstm_pred_inv))
    print(f"  RMSE: {rmse(lstm_actual, lstm_pred_inv):,.2f}")
except Exception as e:
    print(f"  LSTM failed: {e}")

# ─────────────────────────────────────────────
# 16. MODEL COMPARISON
# ─────────────────────────────────────────────
print("\n=== 14. Model Comparison ===")

comp = (pd.DataFrame(results)
          .sort_values("RMSE")
          .reset_index(drop=True))
comp.index += 1
comp.to_csv("model_comparison.csv", index=False)
print("\n" + comp.to_string(index=False))

# Bar charts for RMSE, MAE, R2
for metric_col, color, title in [
    ("RMSE", "#E53935", "RMSE — Lower is Better"),
    ("MAE",  "#FB8C00", "MAE  — Lower is Better"),
    ("R2",   "#43A047", "R²   — Higher is Better"),
]:
    plt.figure(figsize=(12,5))
    vals  = comp[metric_col].values
    names = comp["Model"].values
    bars  = plt.barh(names, vals, color=color, edgecolor="white")
    plt.title(f"Model Comparison: {title}")
    plt.xlabel(metric_col)
    plt.gca().invert_yaxis()
    for b, v in zip(bars, vals):
        plt.text(v + max(vals)*0.005, b.get_y()+b.get_height()/2,
                 f"{v:.4f}" if metric_col=="R2" else f"{v:,.2f}",
                 va="center", fontsize=9)
    save_plot(f"comparison_{metric_col.lower()}_all_models.png")

# ─────────────────────────────────────────────
# 17. ACTUAL vs PREDICTED PLOTS
# ─────────────────────────────────────────────
print("\n=== 15. Actual vs Predicted Plots ===")

# For ML models: pick top store-dept in test set
pair = (test_df.groupby(["Store","Dept"])["Weekly_Sales"]
               .count().sort_values(ascending=False).index[0])
ms, md = int(pair[0]), int(pair[1])
sample = (test_df[(test_df["Store"]==ms) & (test_df["Dept"]==md)]
          .sort_values("Date").copy())
Xs = pd.get_dummies(sample[feat_cols], drop_first=True).reindex(columns=X_train.columns, fill_value=0)
Xs = Xs.replace([np.inf,-np.inf], np.nan).fillna(0)

ml_preds = {
    "Linear Regression": lr.predict(scaler.transform(Xs)).clip(min=0),
    "Random Forest":     rf.predict(Xs).clip(min=0),
    "XGBoost":           xgb_model.predict(Xs).clip(min=0),
}

colors_ml = {"Linear Regression":"#9C27B0","Random Forest":"#1565C0","XGBoost":"#E65100"}

for model_name, preds in ml_preds.items():
    plt.figure(figsize=(12,4))
    plt.plot(sample["Date"], sample["Weekly_Sales"], label="Actual", color="black", linewidth=1.5)
    plt.plot(sample["Date"], preds, label=model_name, color=colors_ml[model_name],
             linewidth=1.5, linestyle="--")
    plt.title(f"Actual vs {model_name} (Store {ms}, Dept {md})")
    plt.xlabel("Date"); plt.ylabel("Weekly Sales ($)")
    plt.legend(); plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gcf().autofmt_xdate()
    safe = model_name.lower().replace(" ","_")
    save_plot(f"pred_{safe}_store{ms}_dept{md}.png")

# Time series models on their series
ts_preds = {}
if arima_pred.size > 0:
    ts_preds["ARIMA"] = (y_te_ts.index, y_te_ts.values, arima_pred, "#00838F")
if sarimax_pred.size > 0:
    ts_preds["SARIMAX (ARX)"] = (y_te_ts.index, y_te_ts.values, sarimax_pred, "#2E7D32")
if prophet_pred.size > 0:
    ts_preds["Prophet"] = (y_te_ts.index, y_te_ts.values, prophet_pred, "#AD1457")
if lstm_pred_inv.size > 0:
    ts_preds["LSTM"] = (dates_te_seq, lstm_actual, lstm_pred_inv, "#E65100")

for model_name, (dates, actual, pred, col) in ts_preds.items():
    plt.figure(figsize=(12,4))
    plt.plot(y_tr_ts.index[-40:], y_tr_ts.values[-40:], label="Train (last 40w)", color="#90A4AE", linewidth=1)
    plt.plot(dates, actual, label="Actual", color="black", linewidth=1.5)
    plt.plot(dates, pred, label=model_name, color=col, linewidth=1.5, linestyle="--")
    plt.axvline(x=split_date, color="red", linestyle=":", alpha=0.6, label="Split")
    plt.title(f"Actual vs {model_name} (Store {s_id}, Dept {d_id})")
    plt.xlabel("Date"); plt.ylabel("Weekly Sales ($)")
    plt.legend(); plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gcf().autofmt_xdate()
    safe = model_name.lower().replace(" ","_").replace("(","").replace(")","")
    save_plot(f"pred_{safe}_store{s_id}_dept{d_id}.png")

# All time series models on one chart
if ts_preds:
    plt.figure(figsize=(13,5))
    plt.plot(y_tr_ts.index[-40:], y_tr_ts.values[-40:], label="Train", color="#B0BEC5", linewidth=1)
    first = True
    for model_name, (dates, actual, pred, col) in ts_preds.items():
        if first:
            plt.plot(dates, actual, label="Actual", color="black", linewidth=2)
            first = False
        plt.plot(dates, pred, label=model_name, color=col, linewidth=1.5, linestyle="--")
    plt.axvline(x=split_date, color="red", linestyle=":", alpha=0.6, label="Split")
    plt.title(f"All Time Series Models — Store {s_id}, Dept {d_id}")
    plt.xlabel("Date"); plt.ylabel("Weekly Sales ($)")
    plt.legend(fontsize=8); plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gcf().autofmt_xdate()
    save_plot(f"pred_all_ts_models_store{s_id}_dept{d_id}.png")

# ─────────────────────────────────────────────
# 18. RESIDUALS — best ML model
# ─────────────────────────────────────────────
print("\n=== 16. Residual Analysis ===")
best_ml_name = comp[comp["Model"].isin(["Random Forest","XGBoost","Linear Regression"])].iloc[0]["Model"]
best_ml_pred = {"Random Forest": pred_rf, "XGBoost": pred_xgb, "Linear Regression": pred_lr}[best_ml_name]
residuals = y_test - best_ml_pred

plt.figure(figsize=(12,4))
plt.plot(residuals, color="#7B1FA2", alpha=0.7, linewidth=0.8)
plt.axhline(0, color="red", linewidth=1, linestyle="--")
plt.title(f"Residuals Over Time — {best_ml_name}")
plt.xlabel("Test Sample Index"); plt.ylabel("Residual ($)")
save_plot("residuals_over_time.png")

plt.figure(figsize=(7,4))
plt.hist(residuals, bins=60, color="#7B1FA2", edgecolor="white", linewidth=0.3)
plt.axvline(0, color="red", linewidth=1.5, linestyle="--")
plt.title(f"Residual Distribution — {best_ml_name}")
plt.xlabel("Residual ($)"); plt.ylabel("Count")
save_plot("residuals_distribution.png")

# ─────────────────────────────────────────────
# 19. SAVE PREDICTIONS CSV
# ─────────────────────────────────────────────
print("\n=== 17. Saving Predictions ===")
pred_out = test_df[["Store","Dept","Date","Weekly_Sales"]].copy()
pred_out["Pred_LR"]  = np.round(pred_lr, 2)
pred_out["Pred_RF"]  = np.round(pred_rf, 2)
pred_out["Pred_XGB"] = np.round(pred_xgb, 2)
pred_out.to_csv("predictions.csv", index=False)
print("  saved: predictions.csv")

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  PIPELINE COMPLETE")
print("="*55)
print("\nOutput files:")
print("  retail_master_merged.csv")
print("  model_comparison.csv")
print("  predictions.csv")
print(f"  plots/ ({len(os.listdir(PLOTS_DIR))} files)")
print("\nBest model by RMSE:", comp.iloc[0]["Model"])
print(f"  RMSE : {comp.iloc[0]['RMSE']:,.2f}")
print(f"  MAE  : {comp.iloc[0]['MAE']:,.2f}")
print(f"  R²   : {comp.iloc[0]['R2']:.4f}")
