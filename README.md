# Walmart Store Sales Forecasting Using Auto-Regressive Models

An end-to-end retail sales forecasting project that predicts Walmart weekly sales using classical statistical models, machine learning algorithms, time series models and deep learning techniques. The project compares the performance of 9 forecasting approaches and provides an interactive Streamlit dashboard for visualization and analysis.

## 🌐 Live Demo
Explore the deployed Streamlit dashboard:

🔗 **Live Application:** https://sales-forecasting-ar.streamlit.app/

## Project Overview

Accurate demand forecasting is critical in retail operations. Forecasting future sales helps businesses optimize inventory management, workforce planning, supply chain operations, and promotional strategies.

This project builds a complete forecasting pipeline using Walmart's historical sales data and evaluates multiple forecasting approaches ranging from simple baselines to advanced deep learning models.

## 🎯 Objectives
- Forecast weekly sales at Store-Department level.
- Compare traditional Auto-Regressive models with Machine Learning and Deep Learning approaches.
- Engineer time-series and business-specific features to improve predictive performance.
- Visualize forecasts, trends, seasonality, and model performance through an interactive dashboard.

---

## 📊 Dataset

Source from Kaggle: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data

The dataset contains:
- Historical weekly sales
- Store information
- Economic indicators
- Holiday information
- Promotional markdown data

## Files Required
| Original File | Rename To |
|---|---|
| `train.csv` | `sales_data.csv` |
| `features.csv` | `features_data.csv` |
| `stores.csv` | `stores_data.csv` |

Place all 3 files in the same folder as `main.py`.

---

## Project Structure

```
walmart_forecast/
|
├── main.py                      # Full pipeline
├── app.py                       # Streamlit dashboard
├── requirements.txt
├── README.md
|
├── sales_data.csv               # (you add this)
├── features_data.csv            # (you add this)
├── stores_data.csv              # (you add this)
|
├── retail_master_merged.csv     # generated
├── model_comparison.csv         # generated
├── predictions.csv              # generated
|
└── plots/                       # generated (~22 plots)
```

---

## 🛠️Technology Stack

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Time Series Forecasting

* ARIMA
* SARIMAX
* Prophet

### Machine Learning

* Scikit-Learn
* XGBoost

### Deep Learning

* TensorFlow
* Keras (LSTM)

### Dashboard

* Streamlit

---

## 🚀 Setup

```bash
pip install -r requirements.txt
```

## ▶️ Run Forecasting Pipeline

```bash
python main.py
```

This runs the full pipeline end-to-end and generates all outputs.

## 📊 Launch Dashboard

```bash
streamlit run app.py
```
The dashboard provides:

* Dataset Overview
* Forecast Visualizations
* Model Comparison Charts
* Performance Metrics
* Downloadable Reports

Opens a local browser dashboard at `http://localhost:8501`

---

## Models(Model outputs and comparission files created)

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

## 📈 Evaluation Metrics

All models are evaluated using:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score

Performance comparison results are automatically exported for analysis.

---

## 📁 Generated Output Files

| File | Description |
|---|---|
| `retail_master_merged.csv` | Cleaned & merged dataset |
| `model_comparison.csv` | RMSE, MAE, R² for all 9 models |
| `predictions.csv` | Test set predictions (LR, RF, XGB) |
| `plots/` | ~22 EDA + model + forecast plots |

---

## ⚙️ Features Engineered

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

---

## 🔄 Forecasting Pipeline

1. Data Loading and Validation
2. Data Cleaning and Preprocessing
3. Feature Engineering
4. Exploratory Data Analysis
5. Train-Test Split
6. Model Training
7. Forecast Generation
8. Model Evaluation
9. Visualization and Reporting

---

## 🔮 Future Enhancements

* Hyperparameter Optimization
* Hierarchical Forecasting
* Automated Model Selection
* Real-Time Forecast API
* Cloud Deployment
* Multi-Store Forecasting Dashboard
