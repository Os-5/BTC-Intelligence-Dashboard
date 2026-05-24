import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import pickle
import os
import tempfile
from scipy import stats
from itertools import product
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BTC Intelligence Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg:       #0d0f14;
    --surface:  #161a23;
    --border:   #252b38;
    --accent:   #f7931a;   /* BTC orange */
    --accent2:  #3b82f6;   /* cool blue  */
    --danger:   #ef4444;
    --success:  #22c55e;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --mono:     'Space Mono', monospace;
    --sans:     'DM Sans', sans-serif;
}

/* ── Global ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans);
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stButton button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: var(--mono) !important;
    font-weight: 700;
    border: none;
    border-radius: 4px;
    letter-spacing: 0.05em;
}
[data-testid="stSidebar"] .stButton button:hover {
    opacity: 0.85;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"]  { color: var(--muted) !important; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stMetricValue"]  { color: var(--text)  !important; font-family: var(--mono); font-size: 1.5rem !important; }
[data-testid="stMetricDelta"]  { font-family: var(--mono); font-size: 0.85rem !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: var(--mono) !important;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 6px; }

/* ── Info / warning / error boxes ── */
[data-testid="stAlert"] { border-radius: 6px; }

/* ── Inputs & sliders ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem;
}

/* ── Section headers ── */
.section-header {
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* ── Tag chips ── */
.chip {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 0.72rem;
    font-weight: 700;
    margin: 2px;
}
.chip-orange { background: rgba(247,147,26,0.15); color: var(--accent); border: 1px solid rgba(247,147,26,0.4); }
.chip-blue   { background: rgba(59,130,246,0.15);  color: var(--accent2); border: 1px solid rgba(59,130,246,0.4); }
.chip-green  { background: rgba(34,197,94,0.12);   color: var(--success);  border: 1px solid rgba(34,197,94,0.4); }
.chip-red    { background: rgba(239,68,68,0.12);   color: var(--danger);   border: 1px solid rgba(239,68,68,0.4); }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #161a23 0%, #0d1117 60%, #1a1200 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
}
.hero h1 { font-family: var(--mono); font-size: 1.8rem; color: var(--accent); margin: 0 0 0.3rem; }
.hero p  { color: var(--muted); font-size: 0.9rem; margin: 0; }

/* ── Plot backgrounds ── */
.element-container iframe, .stPlotlyChart { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT THEME
# ─────────────────────────────────────────────────────────────────────────────

PLOT_STYLE = {
    'figure.facecolor': '#161a23',
    'axes.facecolor':   '#0d0f14',
    'axes.edgecolor':   '#252b38',
    'axes.labelcolor':  '#94a3b8',
    'axes.grid':        True,
    'grid.color':       '#252b38',
    'grid.linewidth':   0.6,
    'xtick.color':      '#64748b',
    'ytick.color':      '#64748b',
    'text.color':       '#e2e8f0',
    'legend.facecolor': '#161a23',
    'legend.edgecolor': '#252b38',
    'font.family':      'monospace',
    'font.size':        10,
}
plt.rcParams.update(PLOT_STYLE)

PALETTE = {'bear': '#ef4444', 'neutral': '#3b82f6', 'bull': '#22c55e'}
CLUSTER_COLORS = {0: '#ef4444', 1: '#3b82f6', 2: '#22c55e', -1: '#64748b'}

# ─────────────────────────────────────────────────────────────────────────────
# BACKEND FUNCTIONS  (mirror the notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    if 'Timestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.drop(columns=['Timestamp'], inplace=True)
    df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume_(BTC)'}, inplace=True)
    df.sort_values('Date', inplace=True)
    df['Weighted_Price'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df.set_index('Date', inplace=True)
    return df


def resample_data(df):
    return (df.resample('D').mean(),
            df.resample('ME').mean(),
            df.resample('QE-DEC').mean(),
            df.resample('YE-DEC').mean())


def apply_boxcox(df_monthly):
    df = df_monthly.copy()
    df['Weighted_Price_box'], lmbda = stats.boxcox(df['Weighted_Price'].dropna())
    df['diff'] = df['Weighted_Price_box'].diff()
    return df, lmbda


def inverse_boxcox(y, lmbda):
    return np.exp(np.log(lmbda * y + 1) / lmbda)


def select_model(df_monthly):
    params = list(product(range(2), range(2), range(2), range(2)))
    best_aic, best_model = np.inf, None
    for p, q, P, Q in params:
        try:
            m = sm.tsa.statespace.SARIMAX(
                df_monthly['Weighted_Price_box'],
                order=(p,1,q), seasonal_order=(P,1,Q,12)
            ).fit(disp=False)
            if m.aic < best_aic:
                best_aic, best_model = m.aic, m
        except: continue
    return best_model


def build_ml_features(df_monthly):
    df = df_monthly.copy()
    df['Return']      = df['Weighted_Price'].pct_change()
    df['Volatility']  = df['Return'].rolling(3).std()
    df['Momentum']    = df['Weighted_Price'].diff()
    df['Log_Price']   = np.log1p(df['Weighted_Price'])
    df['Price_Range'] = df['Weighted_Price'].rolling(3).std()
    df.dropna(inplace=True)
    return df


FEATURE_COLS = ['Weighted_Price','Return','Volatility','Momentum','Log_Price','Price_Range']


def run_kmeans(X_scaled, df_ml, k=3):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_ml = df_ml.copy()
    df_ml['KMeans_Cluster'] = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, df_ml['KMeans_Cluster'])
    dbi = davies_bouldin_score(X_scaled, df_ml['KMeans_Cluster'])
    ch  = calinski_harabasz_score(X_scaled, df_ml['KMeans_Cluster'])
    return km, df_ml, sil, dbi, ch


def run_dbscan(X_scaled, df_ml, eps=0.7, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    df_ml = df_ml.copy()
    df_ml['DBSCAN_Cluster'] = db.fit_predict(X_scaled)
    n_clusters = len(set(df_ml['DBSCAN_Cluster'])) - (1 if -1 in df_ml['DBSCAN_Cluster'].values else 0)
    n_noise    = (df_ml['DBSCAN_Cluster'] == -1).sum()
    return db, df_ml, n_clusters, n_noise


# ─────────────────────────────────────────────────────────────────────────────
# CACHED PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def full_pipeline(file_path):
    df = load_and_prepare_data(file_path)
    df_daily, df_monthly, df_quarterly, df_yearly = resample_data(df)
    df_monthly, lmbda = apply_boxcox(df_monthly)
    return df_daily, df_monthly, df_quarterly, df_yearly, lmbda


@st.cache_resource(show_spinner=False)
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown('<p class="section-header">Data Source</p>', unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV", type="csv",
                                    help="Needs: Timestamp, open, high, low, close, volume")
        csv_path = st.text_input("…or enter file path",
                                 value=r"D:/Applied/archive/btcusd_1-min_data.csv")

        st.markdown('<p class="section-header">Model</p>', unsafe_allow_html=True)
        model_path = st.text_input("Model .pkl path", value="best_sarimax_model.pkl")
        train_new  = st.checkbox("Train new SARIMAX (slow)", value=False)

        st.markdown('<p class="section-header">Forecast</p>', unsafe_allow_html=True)
        months   = st.slider("Horizon (months)", 1, 36, 12)
        history  = st.slider("History shown (months)", 6, 120, 36)
        show_ci  = st.checkbox("Show 95% CI", value=True)

        st.markdown('<p class="section-header">KMeans</p>', unsafe_allow_html=True)
        k_clusters = st.slider("Number of clusters (K)", 2, 8, 3)

        st.markdown('<p class="section-header">DBSCAN</p>', unsafe_allow_html=True)
        eps_val     = st.slider("ε (epsilon)", 0.3, 2.0, 0.7, step=0.05)
        min_samples = st.slider("min_samples", 2, 15, 5)

        st.markdown("---")
        run = st.button("▶  Run Analysis", use_container_width=True)

    return uploaded, csv_path, model_path, train_new, months, history, show_ci, k_clusters, eps_val, min_samples, run


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fig_overview(df_daily, df_monthly, df_quarterly, df_yearly):
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    fig.suptitle('BTC Weighted Price — Multi-Frequency Overview', fontsize=13, fontweight='bold', color='#e2e8f0')
    pairs = [
        (axes[0,0], df_daily,     'Daily',     '#f7931a'),
        (axes[0,1], df_monthly,   'Monthly',   '#3b82f6'),
        (axes[1,0], df_quarterly, 'Quarterly', '#22c55e'),
        (axes[1,1], df_yearly,    'Yearly',    '#a855f7'),
    ]
    for ax, data, title, color in pairs:
        ax.plot(data['Weighted_Price'], color=color, linewidth=1.5)
        ax.set_title(title, fontweight='bold', color='#e2e8f0')
        ax.set_ylabel('USD', color='#64748b')
    plt.tight_layout()
    return fig


def fig_forecast(df_monthly, model, lmbda, months, history, show_ci):
    fcast_obj = model.get_forecast(steps=months)
    fp        = inverse_boxcox(fcast_obj.predicted_mean, lmbda)
    conf      = fcast_obj.conf_int()
    lower     = inverse_boxcox(conf.iloc[:, 0], lmbda)
    upper     = inverse_boxcox(conf.iloc[:, 1], lmbda)
    last      = df_monthly.index[-1]
    dates     = pd.date_range(last + pd.offsets.MonthBegin(1), periods=months, freq='ME')

    hist_idx    = df_monthly.index[-history:]
    hist_prices = df_monthly['Weighted_Price'].iloc[-history:]
    n_total     = len(df_monthly)
    fit_vals    = inverse_boxcox(model.predict(start=max(0, n_total - history), end=n_total - 1), lmbda)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hist_idx, hist_prices,          color='#94a3b8', linewidth=1.8, label='History')
    ax.plot(hist_idx[-len(fit_vals):], fit_vals, color='#f7931a', linewidth=1.2,
            linestyle='--', label='In-sample fit')
    ax.plot(dates, fp,                      color='#ef4444', linewidth=2.2, label='Forecast')
    if show_ci:
        ax.fill_between(dates, lower, upper, alpha=0.15, color='#ef4444', label='95% CI')
    ax.plot([hist_idx[-1], dates[0]], [hist_prices.iloc[-1], fp.iloc[0]],
            color='#ef4444', linewidth=1.2, linestyle=':')
    ax.set_ylabel('Weighted Price (USD)')
    ax.set_title('Bitcoin Monthly Price — SARIMAX Forecast', color='#e2e8f0', fontweight='bold')
    ax.legend(facecolor='#161a23', edgecolor='#252b38', labelcolor='#e2e8f0')
    fig.tight_layout()
    return fig, fp, lower, upper, dates


def fig_kmeans(df_ml, X_scaled, X_pca, pca, k):
    colors = df_ml['KMeans_Cluster'].map(CLUSTER_COLORS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'KMeans Clustering (K={k}) — Market Regimes', fontsize=12, fontweight='bold', color='#e2e8f0')

    axes[0].scatter(df_ml.index, df_ml['Weighted_Price'], c=colors, s=28, alpha=0.85, edgecolors='none')
    axes[0].set_title('Price Timeline', fontweight='bold', color='#e2e8f0')
    axes[0].set_ylabel('USD')
    patches = [mpatches.Patch(color=CLUSTER_COLORS[i], label=f'Cluster {i}') for i in range(k)]
    axes[0].legend(handles=patches, facecolor='#161a23', edgecolor='#252b38', labelcolor='#e2e8f0')

    for c in range(k):
        mask = df_ml['KMeans_Cluster'] == c
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {c}',
                        color=CLUSTER_COLORS[c], s=45, alpha=0.85, edgecolors='none')
    axes[1].set_title('PCA Projection', fontweight='bold', color='#e2e8f0')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    axes[1].legend(facecolor='#161a23', edgecolor='#252b38', labelcolor='#e2e8f0')

    plt.tight_layout()
    return fig


def fig_dbscan(df_ml, X_pca, pca, eps, min_s):
    unique = sorted(df_ml['DBSCAN_Cluster'].unique())
    cmap   = cm.get_cmap('tab10', len(unique))
    lc     = {lbl: ('#64748b' if lbl == -1 else cmap(i)) for i, lbl in enumerate(unique)}
    db_col = df_ml['DBSCAN_Cluster'].map(lc)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'DBSCAN (ε={eps}, min_samples={min_s})', fontsize=12, fontweight='bold', color='#e2e8f0')

    axes[0].scatter(df_ml.index, df_ml['Weighted_Price'], c=db_col, s=28, alpha=0.85, edgecolors='none')
    axes[0].set_title('Price Timeline', fontweight='bold', color='#e2e8f0')
    axes[0].set_ylabel('USD')
    patches = [mpatches.Patch(color=lc[l], label=('Noise ⚠' if l == -1 else f'Cluster {l}')) for l in unique]
    axes[0].legend(handles=patches, facecolor='#161a23', edgecolor='#252b38', labelcolor='#e2e8f0', fontsize=8)

    for lbl in unique:
        mask = df_ml['DBSCAN_Cluster'] == lbl
        name = 'Noise' if lbl == -1 else f'Cluster {lbl}'
        marker = 'x' if lbl == -1 else 'o'
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], label=name, color=lc[lbl],
                        s=45, alpha=0.85, edgecolors='none', marker=marker)
    axes[1].set_title('PCA Projection + Noise', fontweight='bold', color='#e2e8f0')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    axes[1].legend(facecolor='#161a23', edgecolor='#252b38', labelcolor='#e2e8f0', fontsize=8)

    plt.tight_layout()
    return fig


def fig_elbow(X_scaled):
    inertias, sils = [], []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X_scaled, lbl))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(K_range), inertias, 'o-', color='#f7931a', linewidth=2)
    axes[0].set_title('Elbow — Inertia vs K', fontweight='bold', color='#e2e8f0')
    axes[0].set_xlabel('K'); axes[0].set_ylabel('Inertia')

    axes[1].plot(list(K_range), sils, 's-', color='#3b82f6', linewidth=2)
    axes[1].set_title('Silhouette Score vs K', fontweight='bold', color='#e2e8f0')
    axes[1].set_xlabel('K'); axes[1].set_ylabel('Silhouette')

    plt.tight_layout()
    return fig


def fig_kdist(X_scaled):
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_dist = np.sort(distances[:, 4])[::-1]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(k_dist, color='#a855f7', linewidth=1.5)
    ax.axhline(y=0.7, color='#ef4444', linestyle='--', label='ε = 0.7 (default)')
    ax.set_title('k-Distance Graph — Choose ε at the Elbow', fontweight='bold', color='#e2e8f0')
    ax.set_xlabel('Points (sorted)'); ax.set_ylabel('5-NN Distance')
    ax.legend(facecolor='#161a23', edgecolor='#252b38', labelcolor='#e2e8f0')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Hero banner
    st.markdown("""
    <div class="hero">
        <h1>₿ BTC Intelligence Dashboard</h1>
        <p>Time Series Forecasting · KMeans Market Regimes · DBSCAN Anomaly Detection</p>
    </div>
    """, unsafe_allow_html=True)

    (uploaded, csv_path, model_path, train_new,
     months, history, show_ci,
     k_clusters, eps_val, min_samples, run) = render_sidebar()

    if not run:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div style="background:#161a23;border:1px solid #252b38;border-top:3px solid #f7931a;
                        border-radius:8px;padding:1.4rem;">
            <div style="font-family:monospace;font-size:0.7rem;color:#f7931a;letter-spacing:0.12em;
                        text-transform:uppercase;margin-bottom:0.8rem;">01 · Forecast</div>
            <p style="color:#94a3b8;font-size:0.88rem;line-height:1.6;margin:0;">
            SARIMAX model with Box-Cox transformation.<br>
            Grid-search over ARIMA orders to minimise AIC.
            </p></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div style="background:#161a23;border:1px solid #252b38;border-top:3px solid #3b82f6;
                        border-radius:8px;padding:1.4rem;">
            <div style="font-family:monospace;font-size:0.7rem;color:#3b82f6;letter-spacing:0.12em;
                        text-transform:uppercase;margin-bottom:0.8rem;">02 · KMeans</div>
            <p style="color:#94a3b8;font-size:0.88rem;line-height:1.6;margin:0;">
            Segment monthly data into Bear / Neutral / Bull regimes.<br>
            Elbow + Silhouette scoring to find optimal K.
            </p></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div style="background:#161a23;border:1px solid #252b38;border-top:3px solid #a855f7;
                        border-radius:8px;padding:1.4rem;">
            <div style="font-family:monospace;font-size:0.7rem;color:#a855f7;letter-spacing:0.12em;
                        text-transform:uppercase;margin-bottom:0.8rem;">03 · DBSCAN</div>
            <p style="color:#94a3b8;font-size:0.88rem;line-height:1.6;margin:0;">
            Density-based clustering — no K required.<br>
            Flags anomalous months as noise automatically.
            </p></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("👈 Configure options in the sidebar, then click **▶ Run Analysis**.")
        return

    # ── Resolve file path ─────────────────────────────────────────────────────
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        tmp.write(uploaded.read()); tmp.close()
        data_path = tmp.name
    else:
        candidates = [csv_path,
                      r'D:/Nlp/archive/btcusd_1-min_data.csv',
                      r'D:/Applied/archive/btcusd_1-min_data.csv']
        data_path = next((p for p in candidates if os.path.exists(p)), csv_path)

    if not os.path.exists(data_path):
        st.error(f"CSV not found: `{data_path}`"); return

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner('Loading & resampling data…'):
        try:
            df_daily, df_monthly, df_quarterly, df_yearly, lmbda = full_pipeline(data_path)
        except Exception as e:
            st.error(f"Data error: {e}"); return

    # ── Load or train model ───────────────────────────────────────────────────
    if train_new:
        with st.spinner('Training SARIMAX (this may take a few minutes)…'):
            model = select_model(df_monthly)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            st.success(f'Model trained & saved → `{model_path}`')
    else:
        if not os.path.exists(model_path):
            st.error(f"Model not found: `{model_path}` — enable **Train new SARIMAX** in sidebar."); return
        with st.spinner('Loading model…'):
            model = load_model(model_path)

    # ── Compute forecast metrics ──────────────────────────────────────────────
    in_sample  = inverse_boxcox(model.predict(start=0, end=len(df_monthly)-1), lmbda)
    actual     = df_monthly['Weighted_Price'].values[:len(in_sample)]
    mask       = ~np.isnan(actual) & ~np.isnan(in_sample)
    a, p_vals  = actual[mask], in_sample[mask]

    mae  = mean_absolute_error(a, p_vals)
    rmse = np.sqrt(mean_squared_error(a, p_vals))
    r2   = r2_score(a, p_vals)
    mape = np.mean(np.abs((a - p_vals) / a)) * 100

    latest_price  = df_monthly['Weighted_Price'].iloc[-1]
    fcast_obj     = model.get_forecast(steps=months)
    fp            = inverse_boxcox(fcast_obj.predicted_mean, lmbda)
    forecast_next = fp.iloc[0]
    pct_change    = (forecast_next - latest_price) / latest_price * 100

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Key Metrics</p>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Latest Price",       f"${latest_price:,.0f}")
    k2.metric("Next Month Fcst",    f"${forecast_next:,.0f}",  f"{pct_change:+.1f}%")
    k3.metric("Model AIC",          f"{model.aic:.1f}")
    k4.metric("MAE",                f"${mae:,.0f}")
    k5.metric("RMSE",               f"${rmse:,.0f}")
    k6.metric("R²",                 f"{r2:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈  Forecast",
        "📊  EDA Overview",
        "🟠  KMeans Regimes",
        "🟣  DBSCAN Anomalies",
    ])

    # ── TAB 1 · Forecast ──────────────────────────────────────────────────────
    with tab1:
        col_chart, col_info = st.columns([3, 1])

        with col_chart:
            st.markdown('<p class="section-header">Price Forecast</p>', unsafe_allow_html=True)
            f_fig, fp, lower, upper, dates = fig_forecast(
                df_monthly, model, lmbda, months, history, show_ci)
            st.pyplot(f_fig); plt.close(f_fig)

        with col_info:
            st.markdown('<p class="section-header">Model Info</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <span class="chip chip-orange">SARIMAX</span>
            <span class="chip chip-blue">order {model.model.order}</span>
            <span class="chip chip-green">seasonal {model.model.seasonal_order}</span>
            <br><br>
            <table style="width:100%;font-family:monospace;font-size:0.82rem;color:#94a3b8;border-collapse:collapse;">
            <tr><td style="padding:4px 0;color:#64748b">AIC</td><td style="text-align:right">{model.aic:.2f}</td></tr>
            <tr><td style="padding:4px 0;color:#64748b">BIC</td><td style="text-align:right">{model.bic:.2f}</td></tr>
            <tr><td style="padding:4px 0;color:#64748b">λ (Box-Cox)</td><td style="text-align:right">{lmbda:.5f}</td></tr>
            <tr><td style="padding:4px 0;color:#64748b">MAE</td><td style="text-align:right">${mae:,.0f}</td></tr>
            <tr><td style="padding:4px 0;color:#64748b">RMSE</td><td style="text-align:right">${rmse:,.0f}</td></tr>
            <tr><td style="padding:4px 0;color:#64748b">MAPE</td><td style="text-align:right">{mape:.2f}%</td></tr>
            <tr><td style="padding:4px 0;color:#64748b">R²</td><td style="text-align:right">{r2:.4f}</td></tr>
            </table>
            """, unsafe_allow_html=True)

        st.markdown('<p class="section-header">Forecast Table</p>', unsafe_allow_html=True)
        df_table = pd.DataFrame({'Date': dates.strftime('%Y-%m'), 'Forecast (USD)': fp.values.round(2)})
        if show_ci:
            df_table['Lower 95%'] = lower.values.round(2)
            df_table['Upper 95%'] = upper.values.round(2)
        st.dataframe(df_table, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download forecast CSV",
                           data=df_table.to_csv(index=False),
                           file_name="btc_forecast.csv", mime="text/csv")

    # ── TAB 2 · EDA ───────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<p class="section-header">Multi-Frequency Overview</p>', unsafe_allow_html=True)
        ov_fig = fig_overview(df_daily, df_monthly, df_quarterly, df_yearly)
        st.pyplot(ov_fig); plt.close(ov_fig)

        st.markdown('<p class="section-header">Return Distribution</p>', unsafe_allow_html=True)
        fig_ret, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].hist(df_monthly['Weighted_Price'].dropna(), bins=30,
                     color='#f7931a', edgecolor='none', alpha=0.85)
        axes[0].set_title('Monthly Price Distribution', fontweight='bold', color='#e2e8f0')
        axes[0].set_xlabel('Price (USD)')

        log_ret = np.log(df_daily['Weighted_Price'] / df_daily['Weighted_Price'].shift(1)).dropna()
        axes[1].hist(log_ret, bins=60, color='#3b82f6', edgecolor='none', alpha=0.85)
        axes[1].set_title('Daily Log Returns', fontweight='bold', color='#e2e8f0')
        axes[1].set_xlabel('Log Return')

        plt.tight_layout()
        st.pyplot(fig_ret); plt.close(fig_ret)

    # ── TAB 3 · KMeans ───────────────────────────────────────────────────────
    with tab3:
        st.markdown('<p class="section-header">Elbow & Silhouette — Choose K</p>', unsafe_allow_html=True)

        df_ml = build_ml_features(df_monthly)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_ml[FEATURE_COLS])
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        elbow_fig = fig_elbow(X_scaled)
        st.pyplot(elbow_fig); plt.close(elbow_fig)

        st.markdown(f'<p class="section-header">KMeans Results (K={k_clusters})</p>', unsafe_allow_html=True)
        km_model, df_ml, sil, dbi, ch = run_kmeans(X_scaled, df_ml, k=k_clusters)
        km_fig = fig_kmeans(df_ml, X_scaled, X_pca, pca, k_clusters)
        st.pyplot(km_fig); plt.close(km_fig)

        e1, e2, e3 = st.columns(3)
        e1.metric("Silhouette ↑",     f"{sil:.4f}")
        e2.metric("Davies–Bouldin ↓", f"{dbi:.4f}")
        e3.metric("Calinski–Harabasz ↑", f"{ch:.1f}")

        st.markdown('<p class="section-header">Cluster Profiles</p>', unsafe_allow_html=True)
        profile = df_ml.groupby('KMeans_Cluster')[FEATURE_COLS].mean().round(2)
        profile.index = [f'Cluster {i}' for i in profile.index]
        st.dataframe(profile, use_container_width=True)

    # ── TAB 4 · DBSCAN ────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<p class="section-header">k-Distance Graph — Tune ε</p>', unsafe_allow_html=True)
        kd_fig = fig_kdist(X_scaled)
        st.pyplot(kd_fig); plt.close(kd_fig)

        st.markdown(f'<p class="section-header">DBSCAN Results (ε={eps_val}, min_samples={min_samples})</p>',
                    unsafe_allow_html=True)
        db_model, df_ml_db, n_clust, n_noise = run_dbscan(X_scaled, df_ml, eps=eps_val, min_samples=min_samples)
        db_fig = fig_dbscan(df_ml_db, X_pca, pca, eps_val, min_samples)
        st.pyplot(db_fig); plt.close(db_fig)

        d1, d2, d3 = st.columns(3)
        d1.metric("Clusters Found",     n_clust)
        d2.metric("Noise Points",       n_noise)
        d3.metric("Noise %",            f"{n_noise/len(df_ml_db)*100:.1f}%")

        anomalies = df_ml_db[df_ml_db['DBSCAN_Cluster'] == -1][
            ['Weighted_Price','Return','Volatility','Momentum']].round(4)
        if len(anomalies):
            st.markdown('<p class="section-header">Anomalous Months</p>', unsafe_allow_html=True)
            st.dataframe(anomalies, use_container_width=True)

        # Side-by-side comparison
        st.markdown('<p class="section-header">KMeans vs DBSCAN — Timeline Comparison</p>',
                    unsafe_allow_html=True)
        fig_cmp, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        fig_cmp.suptitle('Market Regime Comparison', fontsize=12, fontweight='bold', color='#e2e8f0')

        col_km = df_ml['KMeans_Cluster'].map(CLUSTER_COLORS)
        axes[0].scatter(df_ml.index, df_ml['Weighted_Price'],
                        c=col_km, s=25, alpha=0.85, edgecolors='none')
        axes[0].set_title(f'KMeans (K={k_clusters})', fontweight='bold', color='#e2e8f0')
        axes[0].set_ylabel('USD')

        db_unique = sorted(df_ml_db['DBSCAN_Cluster'].unique())
        db_cmap   = cm.get_cmap('tab10', len(db_unique))
        db_lc     = {l: ('#64748b' if l == -1 else db_cmap(i)) for i, l in enumerate(db_unique)}
        db_colors = df_ml_db['DBSCAN_Cluster'].map(db_lc)
        axes[1].scatter(df_ml_db.index, df_ml_db['Weighted_Price'],
                        c=db_colors, s=25, alpha=0.85, edgecolors='none')
        axes[1].set_title(f'DBSCAN (ε={eps_val}, min_samples={min_samples})', fontweight='bold', color='#e2e8f0')
        axes[1].set_ylabel('USD')

        plt.tight_layout()
        st.pyplot(fig_cmp); plt.close(fig_cmp)


if __name__ == '__main__':
    main()
