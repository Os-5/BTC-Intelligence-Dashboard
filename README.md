# 🪙 Bitcoin Intelligence Dashboard: Forecasting & Market Regimes

An end-to-end quantitative analytics system that combines **time-series forecasting**, **unsupervised machine learning**, and **interactive visualization** to model Bitcoin price dynamics and identify market regimes (Bull, Bear, Neutral). The system integrates statistical modeling, clustering algorithms, and feature engineering into a unified Streamlit-based decision support dashboard.

---

# 🚀 Key Features

## 📈 Time-Series Forecasting Engine

* Implements **Box-Cox transformation** for variance stabilization of OHLCV price data
* Uses **SARIMAX (Seasonal ARIMA with Exogenous Variables)** for long-term forecasting
* Generates forward-looking price predictions with confidence-aware evaluation
* Evaluated using:

  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)
  * R² Score

---

## 🤖 Market Regime Detection (Unsupervised Learning)

* **K-Means Clustering**

  * Identifies structured market regimes (Bull / Bear / Neutral)
  * Groups similar volatility-price behavior patterns

* **DBSCAN Density Clustering**

  * Detects anomalous market periods (black swan events)
  * Labels noise points as outlier regime (-1 cluster)

* **PCA Dimensionality Reduction**

  * Projects high-dimensional financial features into 2D space
  * Enables visual separation of market states

* **Cluster Validation Metrics**

  * Silhouette Score
  * Davies-Bouldin Index
  * Calinski-Harabasz Score

---

## 🖥️ Interactive Streamlit Dashboard

* Modern dark-themed UI for financial analytics
* Upload custom Bitcoin OHLCV datasets
* Adjust forecasting and clustering parameters dynamically
* Real-time visualization of:

  * Price trends
  * Market regimes
  * Anomaly detection results
  * PCA cluster maps

---

# 📂 Project Structure

```bash id="btc_structure"
├── main.ipynb              # Data pipeline, forecasting & ML training
├── streamlit_app.py        # Interactive dashboard application
├── sarimax_model.pkl       # Trained time-series forecasting model
├── kmeans_model.pkl        # K-Means clustering model
├── dbscan_model.pkl        # DBSCAN anomaly detection model
└── scaler.pkl             # Feature scaling transformer
```

---

# 🧠 Feature Engineering Pipeline

The system constructs a multi-dimensional financial feature space including:

* 📉 Log Returns (price stability normalization)
* 📊 Volatility indicators (rolling standard deviation)
* 📈 Trend features (moving averages, momentum signals)
* 🔄 Time-based decomposition (seasonality patterns)

---

# 📊 Forecasting Methodology

The forecasting pipeline applies:

* Box-Cox transformation for variance stabilization
* SARIMAX modeling for temporal dependency learning
* Multi-step forecasting for future price trajectories
* Residual evaluation for predictive reliability

---

# 📌 Mathematical Framework

## 📉 Silhouette Score (Cluster Validation)

[
\text{Silhouette Score} = \frac{b - a}{\max(a, b)}
]

Where:

* (a) = intra-cluster distance
* (b) = nearest-cluster distance

Higher values indicate stronger and more separable market regimes.

---

# ⚙️ Installation & Setup

## Prerequisites

* Python 3.9+
* pip package manager

---

## 1️⃣ Clone Repository

```bash id="btc_clone"
git clone https://github.com/your-username/bitcoin-intelligence-dashboard.git
cd bitcoin-intelligence-dashboard
```

---

## 2️⃣ Install Dependencies

```bash id="btc_install"
pip install streamlit pandas numpy scipy scikit-learn statsmodels matplotlib seaborn
```

---

# ▶️ Running the Application

Ensure trained model files are in the root directory:

* `sarimax_model.pkl`
* `kmeans_model.pkl`
* `dbscan_model.pkl`

Start the dashboard:

```bash id="btc_run"
streamlit run streamlit_app.py
```

---

# 🖥️ Dashboard Capabilities

## 📂 Data Upload

* Load custom OHLCV Bitcoin datasets

## 📈 Forecasting Module

* View SARIMAX-based predictions
* Analyze trend extrapolations

## 🤖 Market Regime Module

* Visualize clustering results
* Detect anomalies and regime shifts

## 🧭 Interactive Controls

* Adjust clustering parameters (K, epsilon)
* Modify rolling window sizes
* Switch between time granularities

---

# 🧰 Technologies Used

## Data Science & ML

* Pandas
* NumPy
* Scikit-learn
* Statsmodels

## Time-Series Analysis

* SARIMAX
* Box-Cox transformation

## Unsupervised Learning

* K-Means
* DBSCAN
* PCA

## Visualization & App

* Matplotlib
* Seaborn
* Streamlit

---

# 📌 Key Highlights

* End-to-end Bitcoin forecasting and regime detection system
* Hybrid statistical + ML modeling approach
* Real-time interactive financial dashboard
* Anomaly detection for black swan market events
* Fully reproducible ML pipeline with serialized models

---

# 📜 License

This project is intended for educational and research purposes.
