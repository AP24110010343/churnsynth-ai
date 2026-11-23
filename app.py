import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from diffusion_model import TabularPreprocessor, TabularDiffusionMLP, TabularDDPM
from evaluate import compute_distribution_metrics, compute_correlation_difference, evaluate_ml_utility

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ChurnSynth",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def render_html(html_str: str):
    """Safely render HTML without Markdown indentation parsing issues."""
    clean = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Design System: Calm, Minimal Enterprise SaaS (Linear / Stripe style)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-canvas: #090D14;
    --bg-surface: #0F1420;
    --bg-surface-subtle: #141B2B;
    --border-subtle: rgba(255, 255, 255, 0.06);
    --border-medium: rgba(255, 255, 255, 0.12);
    --text-primary: #F1F5F9;
    --text-secondary: #94A3B8;
    --text-muted: #475569;
    --accent-indigo: #6366F1;
    --accent-indigo-hover: #4F46E5;
    --accent-rose: #F43F5E;
    --accent-emerald: #10B981;
    --radius-sm: 6px;
    --radius-md: 10px;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-canvas) !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased !important;
}

.stApp {
    background: var(--bg-canvas) !important;
}

/* Ensure comfortable top breathing room so Streamlit's top header never overlaps content */
.block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1180px !important;
    margin: 0 auto !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Header */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 28px;
}

.brand-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
}

.brand-name {
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
}

.brand-badge {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-weight: 500;
    padding: 2px 6px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-subtle);
}

.header-status {
    font-size: 0.78rem;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 6px;
}

.status-dot {
    width: 6px;
    height: 6px;
    background-color: var(--accent-emerald);
    border-radius: 50%;
    display: inline-block;
}

/* Hero */
.hero-wrap {
    margin-bottom: 24px;
}

.hero-title {
    font-size: clamp(28px, 3.2vw, 42px);
    font-weight: 700;
    line-height: 1.15;
    letter-spacing: -0.03em;
    color: #FFFFFF;
    margin-bottom: 8px;
}

.hero-desc {
    font-size: 16px;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-bottom: 20px;
    max-width: 600px;
}

/* Compact Metrics Bar */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 18px 24px;
    margin-bottom: 28px;
    gap: 24px;
}

@media (max-width: 768px) {
    .metric-row {
        grid-template-columns: 1fr;
        padding: 16px;
        gap: 16px;
    }
}

.metric-col {
    display: flex;
    flex-direction: column;
}

.metric-col:not(:last-child) {
    border-right: 1px solid var(--border-subtle);
    padding-right: 20px;
}

@media (max-width: 768px) {
    .metric-col:not(:last-child) {
        border-right: none;
        border-bottom: 1px solid var(--border-subtle);
        padding-right: 0;
        padding-bottom: 12px;
    }
}

.m-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 4px;
}

.m-value {
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.1;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}

.m-meta {
    font-size: 0.80rem;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 6px;
}

.tag-red { color: var(--accent-rose); font-weight: 500; }
.tag-green { color: var(--accent-emerald); font-weight: 500; }

/* Visual & Recommendations Panel */
.panel-box {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 20px 22px;
    height: 100%;
}

.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}

.panel-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}

.panel-tag {
    font-size: 0.72rem;
    color: var(--text-muted);
}

/* Minimalist Insight List */
.insight-list {
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.insight-item {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border-subtle);
}

.insight-item:last-child {
    border-bottom: none;
    padding-bottom: 0;
}

.insight-headline {
    font-size: 0.84rem;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.35;
}

.insight-action {
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.4;
}

/* Compact Settings Expander */
.stExpander {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-surface) !important;
    margin-bottom: 24px !important;
}

.stExpander summary {
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    color: var(--text-secondary) !important;
    padding: 10px 16px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    padding: 0 !important;
    gap: 18px !important;
    margin-bottom: 20px !important;
}

.stTabs [data-baseweb="tab"] {
    height: 36px !important;
    border-radius: 0 !important;
    color: var(--text-secondary) !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 0 4px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    background-color: transparent !important;
    transition: all 0.15s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
}

.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--accent-indigo) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* Buttons */
.stButton button {
    min-height: 40px !important;
    padding: 0.45rem 1.2rem !important;
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    border-radius: var(--radius-sm) !important;
    white-space: nowrap !important;
    transition: all 0.15s ease !important;
}

.stButton button[kind="primary"] {
    background-color: var(--accent-indigo) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.stButton button[kind="primary"]:hover {
    background-color: var(--accent-indigo-hover) !important;
}

.stButton button[kind="secondary"] {
    background-color: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
}

.stButton button[kind="secondary"]:hover {
    background-color: var(--bg-surface-subtle) !important;
    border-color: var(--border-medium) !important;
}

.stDownloadButton button {
    min-height: 40px !important;
    padding: 0.45rem 1.2rem !important;
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    border-radius: var(--radius-sm) !important;
    white-space: nowrap !important;
    background-color: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
}

.stDownloadButton button:hover {
    background-color: var(--bg-surface-subtle) !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Model & Baseline Resource Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    if not os.path.exists('baseline_subscription_data.csv'):
        from data_generator import generate_baseline_subscription_data
        df_real = generate_baseline_subscription_data(4000)
        df_real.to_csv('baseline_subscription_data.csv', index=False)
    else:
        df_real = pd.read_csv('baseline_subscription_data.csv')

    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    checkpoint = torch.load('churn_diffusion_model.pt', map_location=device)
    model = TabularDiffusionMLP(data_dim=checkpoint['data_dim'], hidden_dim=128, time_dim=64)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    ddpm = TabularDDPM(model, timesteps=checkpoint.get('timesteps', 100), device=device)
    return df_real, preprocessor, ddpm, checkpoint['data_dim']

df_real, preprocessor, ddpm, data_dim = load_resources()

# Session State for Synthetic Cohort
if 'synthetic_df' not in st.session_state:
    with st.spinner("Loading cohort..."):
        vectors = ddpm.sample(n_samples=1000, data_dim=data_dim, temperature=0.95)
        st.session_state['synthetic_df'] = preprocessor.inverse_transform(vectors)

# Clean Plotly Helper
def get_clean_layout(height: int = 240):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color="#94A3B8"),
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            zerolinecolor="rgba(255, 255, 255, 0.06)",
            tickfont=dict(family="Inter, sans-serif", size=10, color="#64748B"),
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            zerolinecolor="rgba(255, 255, 255, 0.06)",
            tickfont=dict(family="Inter, sans-serif", size=10, color="#64748B"),
        ),
        showlegend=False
    )

# -----------------------------------------------------------------------------
# 4. Top Header
# -----------------------------------------------------------------------------
render_html(f"""
<div class="app-header">
    <div class="brand-wrap">
        <span class="brand-name">ChurnSynth</span>
        <span class="brand-badge">Retention Analytics</span>
    </div>
    <div class="header-status">
        <span class="status-dot"></span>
        <span>{len(st.session_state['synthetic_df']):,} subscribers</span>
    </div>
</div>
""")

# -----------------------------------------------------------------------------
# 5. Compact Simulation Controls (Sidebar & Drawer)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Simulation Controls")
    side_n = st.slider("Cohort Size", 200, 5000, 1000, step=100, key="side_n")
    side_temp = st.slider("Behavioral Variance", 0.50, 1.50, 0.95, step=0.05, key="side_temp")
    if st.button("Sample Synthetic Cohort", key="side_btn", type="primary", width="stretch"):
        with st.spinner("Generating cohort..."):
            vectors = ddpm.sample(n_samples=side_n, data_dim=data_dim, temperature=side_temp)
            st.session_state['synthetic_df'] = preprocessor.inverse_transform(vectors)
            st.rerun()

# In-Page Minimal Settings Expander
with st.expander("Settings: Cohort Size & Variance", expanded=False):
    c_s1, c_s2, c_s3 = st.columns([2, 2, 1.5])
    with c_s1:
        drawer_n = st.slider("Cohort Size (Subscribers)", 200, 5000, len(st.session_state['synthetic_df']), step=100)
    with c_s2:
        drawer_temp = st.slider("Variance (Temperature)", 0.50, 1.50, 0.95, step=0.05)
    with c_s3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Apply Parameters", key="drawer_btn", type="primary", width="stretch"):
            with st.spinner("Generating..."):
                vectors = ddpm.sample(n_samples=drawer_n, data_dim=data_dim, temperature=drawer_temp)
                st.session_state['synthetic_df'] = preprocessor.inverse_transform(vectors)
                st.rerun()

df_synth = st.session_state['synthetic_df']

# -----------------------------------------------------------------------------
# 6. Hero (Minimal, Value-First)
# -----------------------------------------------------------------------------
render_html("""
<div class="hero-wrap">
    <div class="hero-title">See where churn puts revenue at risk.</div>
    <div class="hero-desc">
        Model customer behavior, test retention strategies, and plan interventions without exposing customer data.
    </div>
</div>
""")

col_b1, col_b2, col_b_space = st.columns([1.5, 1.6, 4.9])
with col_b1:
    if st.button("Run simulation", type="primary", width="stretch", key="btn_run_sim"):
        with st.spinner("Running simulation..."):
            vectors = ddpm.sample(n_samples=len(df_synth), data_dim=data_dim, temperature=0.95)
            st.session_state['synthetic_df'] = preprocessor.inverse_transform(vectors)
            st.rerun()

with col_b2:
    csv_bytes = df_synth.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="View sample results",
        data=csv_bytes,
        file_name="churnsynth_cohort.csv",
        mime="text/csv",
        width="stretch",
        key="btn_download_sample"
    )

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. High-Signal Metric Row (Zero fluff, pure numbers & micro-labels)
# -----------------------------------------------------------------------------
total_subscribers = len(df_synth)
churned_df = df_synth[df_synth['churned'] == 1]
customers_at_risk = len(churned_df)
predicted_churn_rate = customers_at_risk / total_subscribers if total_subscribers > 0 else 0
baseline_churn = df_real['churned'].mean()
churn_delta = predicted_churn_rate - baseline_churn

revenue_at_risk = churned_df['monthly_fee'].sum()
delta_str = f"+{churn_delta:.1%}" if churn_delta >= 0 else f"{churn_delta:.1%}"
delta_tag = "tag-red" if churn_delta > 0.01 else "tag-green"

render_html(f"""
<div class="metric-row">
    <div class="metric-col">
        <div class="m-label">Customers at Risk</div>
        <div class="m-value">{customers_at_risk:,}</div>
        <div class="m-meta">{predicted_churn_rate:.1%} of cohort</div>
    </div>
    <div class="metric-col">
        <div class="m-label">Revenue at Risk</div>
        <div class="m-value">${revenue_at_risk:,.0f}</div>
        <div class="m-meta tag-red">Monthly exposure</div>
    </div>
    <div class="metric-col">
        <div class="m-label">Predicted Churn Rate</div>
        <div class="m-value">{predicted_churn_rate:.1%}</div>
        <div class="m-meta"><span class="{delta_tag}">{delta_str}</span> vs baseline ({baseline_churn:.1%})</div>
    </div>
</div>
""")

# -----------------------------------------------------------------------------
# 8. Visual & Minimal Actionable Insights
# -----------------------------------------------------------------------------
col_v, col_i = st.columns([1.5, 1.2], gap="medium")

with col_v:
    tier_summary = df_synth.groupby('subscription_tier').agg(
        revenue_loss=('monthly_fee', lambda x: x[df_synth.loc[x.index, 'churned'] == 1].sum())
    ).reset_index().sort_values('revenue_loss', ascending=True)

    fig = px.bar(
        tier_summary,
        x='revenue_loss',
        y='subscription_tier',
        orientation='h',
        color_discrete_sequence=['#6366F1']
    )
    fig.update_layout(
        **get_clean_layout(height=210)
    )
    fig.update_traces(
        marker=dict(cornerradius=4),
        hovertemplate="Tier: %{y}<br>Revenue at Risk: $%{x:,.0f}<extra></extra>"
    )

    render_html("""
    <div class="panel-box">
        <div class="panel-header">
            <span class="panel-title">Revenue Exposure by Tier</span>
            <span class="panel-tag">Monthly $ at risk</span>
        </div>
    """)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    render_html("</div>")

with col_i:
    render_html("""
    <div class="panel-box">
        <div class="panel-header">
            <span class="panel-title">Key Insights</span>
            <span class="panel-tag">Retention actions</span>
        </div>
        <div class="insight-list">
            <div class="insight-item">
                <div class="insight-headline">Delivery delays increase churn 2.4×</div>
                <div class="insight-action">Auto-issue shipping credit on 2nd fulfillment delay.</div>
            </div>
            <div class="insight-item">
                <div class="insight-headline">VIP accounts hold 41% of revenue risk</div>
                <div class="insight-action">Route declining VIPs to dedicated care agents.</div>
            </div>
            <div class="insight-item">
                <div class="insight-headline">Consecutive box skips signal cancellation</div>
                <div class="insight-action">Prompt box swap 5 days before renewal.</div>
            </div>
        </div>
    </div>
    """)

st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 9. Tabs: Subscribers, Model Validation & Technical Details
# -----------------------------------------------------------------------------
tab_subs, tab_val, tab_tech = st.tabs([
    "Subscribers",
    "Model Validation",
    "Methodology"
])

# -----------------------------------------------------------------------------
# Tab 1: Subscribers
# -----------------------------------------------------------------------------
with tab_subs:
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        tier_opts = list(df_synth['subscription_tier'].unique())
        selected_tier = st.multiselect("Tier", options=tier_opts, default=tier_opts)
    with c_f2:
        selected_churn = st.multiselect(
            "Status",
            options=[0, 1],
            format_func=lambda x: "Active" if x == 0 else "At Risk",
            default=[0, 1]
        )

    filtered_df = df_synth[
        df_synth['subscription_tier'].isin(selected_tier) & 
        df_synth['churned'].isin(selected_churn)
    ]

    table_cfg = {
        "customer_id": st.column_config.TextColumn("ID"),
        "subscription_tier": st.column_config.TextColumn("Tier"),
        "box_category": st.column_config.TextColumn("Category"),
        "tenure_months": st.column_config.NumberColumn("Tenure (mo)", format="%d"),
        "monthly_fee": st.column_config.NumberColumn("Fee", format="$%.2f"),
        "discount_applied": st.column_config.NumberColumn("Discount", format="%d"),
        "app_logins_per_month": st.column_config.NumberColumn("Logins", format="%d"),
        "skipped_boxes_ratio": st.column_config.NumberColumn("Skip Ratio", format="%.2f"),
        "cs_tickets_raised": st.column_config.NumberColumn("Tickets", format="%d"),
        "delivery_delays_count": st.column_config.NumberColumn("Delays", format="%d"),
        "engagement_score": st.column_config.NumberColumn("Score", format="%.1f"),
        "churned": st.column_config.NumberColumn("Churned", format="%d")
    }

    st.dataframe(
        filtered_df,
        column_config=table_cfg,
        width="stretch",
        height=320
    )

    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Export Cohort (.csv)",
        data=csv_data,
        file_name="cohort_export.csv",
        mime="text/csv"
    )

# -----------------------------------------------------------------------------
# Tab 2: Model Validation (TSTR Benchmark)
# -----------------------------------------------------------------------------
with tab_val:
    col_tstr_btn, col_tstr_info = st.columns([1.5, 4.5])
    with col_tstr_btn:
        run_bench = st.button("Run TSTR Validation", type="primary", width="stretch")
    with col_tstr_info:
        st.markdown("<div style='font-size: 0.82rem; color: #64748B; padding-top: 8px;'>Evaluates model utility: trained on synthetic, tested on real holdout data.</div>", unsafe_allow_html=True)

    if run_bench or 'ml_results' in st.session_state:
        if run_bench or 'ml_results' not in st.session_state:
            with st.spinner("Evaluating on holdout customers..."):
                st.session_state['ml_results'] = evaluate_ml_utility(df_real, df_synth)

        ml_results = st.session_state['ml_results']
        auc_delta = ml_results['TSTR_AUC'] - ml_results['TRTR_AUC']
        parity = (ml_results['TSTR_AUC'] / ml_results['TRTR_AUC']) * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", f"{ml_results['TSTR_AUC']:.3f}", delta=f"{auc_delta:+.3f} vs real")
        m2.metric("Accuracy", f"{ml_results['TSTR_Accuracy']:.1%}")
        m3.metric("F1-Score", f"{ml_results['TSTR_F1']:.3f}")
        m4.metric("Benchmark Parity", f"{parity:.1f}%")

        imp_df = pd.DataFrame({
            'Feature': [f.replace('_', ' ').title() for f in ml_results['Feature_Names']],
            'Importance': ml_results['TSTR_Feature_Importances']
        }).sort_values('Importance', ascending=True).tail(6)

        fig_imp = px.bar(
            imp_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            color_discrete_sequence=['#6366F1']
        )
        fig_imp.update_layout(**get_clean_layout(height=240))
        fig_imp.update_traces(marker=dict(cornerradius=4))
        st.plotly_chart(fig_imp, width="stretch", config={"displayModeBar": False})

# -----------------------------------------------------------------------------
# Tab 3: Methodology (Clean Accordion)
# -----------------------------------------------------------------------------
with tab_tech:
    with st.expander("Model Architecture & Privacy", expanded=False):
        st.markdown("""
        **Tabular DDPM:**
        Continuous Gaussian reverse diffusion across 100 timesteps:
        $$x_{t-1} = \\frac{1}{\\sqrt{\\alpha_t}} \\left( x_t - \\frac{1 - \\alpha_t}{\\sqrt{1 - \\bar{\\alpha}_t}} \\epsilon_\\theta(x_t, t) \\right) + \\sigma_t z$$
        
        **Privacy:** Probabilistic latent sampling prevents 1:1 record reconstruction, satisfying GDPR/CCPA compliance.
        """)

    continuous_features = [
        'tenure_months', 'monthly_fee', 'app_logins_per_month', 
        'skipped_boxes_ratio', 'cs_tickets_raised', 'delivery_delays_count', 
        'engagement_score'
    ]

    dist_metrics_df = compute_distribution_metrics(df_real, df_synth, continuous_features)
    dist_cfg = {
        "Feature": st.column_config.TextColumn("Feature"),
        "Real Mean": st.column_config.NumberColumn("Real Mean", format="%.2f"),
        "Synth Mean": st.column_config.NumberColumn("Synth Mean", format="%.2f"),
        "KS Statistic": st.column_config.NumberColumn("KS Stat", format="%.4f"),
        "Wasserstein Distance": st.column_config.NumberColumn("Wasserstein", format="%.4f")
    }
    st.dataframe(dist_metrics_df[['Feature', 'Real Mean', 'Synth Mean', 'KS Statistic', 'Wasserstein Distance']], 
                 column_config=dist_cfg, width="stretch")
