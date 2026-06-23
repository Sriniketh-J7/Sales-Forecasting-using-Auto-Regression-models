# Walmart Store Sales Forecasting

End-to-end retail sales forecasting pipeline using **9 models** across AR, ML, and Deep Learning categories.

---

## Dataset

Download from Kaggle: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data

| File | Rename to |
|---|---|
| `train.csv` | `sales_data.csv` |
| `features.csv` | `features_data.csv` |
| `stores.csv` | `stores_data.csv` |

Place all 3 files in the same folder as `main.py`.

---

## Project Structure

```
walmart_forecast/
├── main.py                      # Full pipeline
├── app.py                       # Streamlit dashboard
├── requirements.txt
├── README.md
├── sales_data.csv               # (you add this)
├── features_data.csv            # (you add this)
├── stores_data.csv              # (you add this)
├── retail_master_merged.csv     # generated
├── model_comparison.csv         # generated
├── predictions.csv              # generated
└── plots/                       # generated (~22 plots)
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run Pipeline

```bash
python main.py
```

This runs the full pipeline end-to-end and generates all outputs.

---

## Run Dashboard

```bash
streamlit run app.py
```

Opens a local browser dashboard at `http://localhost:8501`

---

## Deploy on Streamlit Cloud (share live link)

1. Push your project folder to a **GitHub repo** (include `requirements.txt`, `main.py`, `app.py`, and all generated CSVs + plots)
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click Deploy — you get a public shareable link

---

## Models

| Model | Type | Notes |
|---|---|---|
| Baseline — Lag 1 Week | Baseline | Last week's sales as prediction |
| Baseline — Rolling Mean 4W | Baseline | 4-week average |
| Linear Regression | ML | All engineered features |
| Random Forest | ML | 200 trees, depth 20 |
| XGBoost | ML | 500 trees, LR 0.05 |
| ARIMA(2,1,2) | AR | Single series |
| SARIMAX (ARX) | AR + Exogenous | Seasonal + external features |
| Prophet | Time Series | Meta's trend/seasonality model |
| LSTM | Deep Learning | 2-layer LSTM, window=12 weeks |

---

## Output Files

| File | Description |
|---|---|
| `retail_master_merged.csv` | Cleaned & merged dataset |
| `model_comparison.csv` | RMSE, MAE, R² for all 9 models |
| `predictions.csv` | Test set predictions (LR, RF, XGB) |
| `plots/` | ~22 EDA + model + forecast plots |

---

## Features Engineered

- Date features: Year, Month, Quarter, WeekOfYear, IsMonthEnd, IsQuarterEnd
- Holiday flags: IsHoliday, Holiday_PrevWeek, Holiday_NextWeek
- Lag features: 1, 2, 4, 8, 13, 26, 52 weeks
- Rolling stats: Mean and Std for 4, 8, 13, 26 week windows
- Exponential weighted mean: span 4 and 13 weeks
- External deltas: Week-over-week change in Temperature, Fuel Price, CPI, Unemployment
- Markdown: Total markdown, Any markdown flag
- Store features: Size, Sales per sq ft, Store type dummies
- Hierarchical: Store total sales, Dept share of store
- Seasonal baseline: Dept average sales by week of year
