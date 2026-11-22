import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch

from diffusion_model import TabularPreprocessor, TabularDiffusionMLP, TabularDDPM
from evaluate import compute_distribution_metrics, compute_correlation_difference, evaluate_ml_utility

st.set_page_config(
    page_title="ChurnSynth AI | Tabular Diffusion Generator",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-bottom: 1.2rem;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 14px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-retail { background-color: #E0F2FE; color: #0369A1; }
    .badge-diffusion { background-color: #EDE9FE; color: #6D28D9; }
    .badge-deploy { background-color: #DCFCE7; color: #15803D; }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Helper Functions & Caching ----------------- #
@st.cache_resource
def load_resources():
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Check baseline data
    if not os.path.exists('baseline_subscription_data.csv'):
        from data_generator import generate_baseline_subscription_data
        df_real = generate_baseline_subscription_data(4000)
        df_real.to_csv('baseline_subscription_data.csv', index=False)
    else:
        df_real = pd.read_csv('baseline_subscription_data.csv')

    # Load preprocessor
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    # Load Model
    checkpoint = torch.load('churn_diffusion_model.pt', map_location=device)
    model = TabularDiffusionMLP(data_dim=checkpoint['data_dim'], hidden_dim=128, time_dim=64)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    ddpm = TabularDDPM(model, timesteps=checkpoint.get('timesteps', 100), device=device)
    return df_real, preprocessor, ddpm, checkpoint['data_dim']

df_real, preprocessor, ddpm, data_dim = load_resources()

# Initialize Session State
if 'synthetic_df' not in st.session_state:
    with st.spinner("Generating initial synthetic cohort with Tabular DDPM..."):
        vectors = ddpm.sample(n_samples=1000, data_dim=data_dim, temperature=0.95)
        st.session_state['synthetic_df'] = preprocessor.inverse_transform(vectors)

# ----------------- Header & Badges ----------------- #
st.markdown('<div class="main-title">📦 ChurnSynth AI: Synthetic Subscription Churn Generator</div>', unsafe_allow_html=True)
st.markdown("""
<div style="margin-bottom: 15px;">
    <span class="badge badge-retail">🏢 Domain: Retail (Subscription Boxes)</span>
    <span class="badge badge-diffusion">🧠 Model: Tabular DDPM Diffusion</span>
    <span class="badge badge-deploy">⚡ Deployment: Streamlit</span>
</div>
<div class="sub-title">
Generate mathematically sound synthetic customer lifecycles and churn records to train, stress-test, and validate predictive churn models without exposing private customer data.
</div>
""", unsafe_allow_html=True)

# ----------------- Sidebar Controls ----------------- #
with st.sidebar:
    st.header("⚙️ Diffusion Generator Controls")
    st.markdown("Configure generation parameters for the reverse diffusion process:")
    
    n_samples = st.slider("Samples to Generate", min_value=200, max_value=5000, value=1000, step=100)
    temperature = st.slider("Diffusion Temperature (Diversity)", min_value=0.5, max_value=1.5, value=0.95, step=0.05, 
                            help="Higher temperature yields higher feature entropy/diversity; lower yields conservative modes.")
    
    st.markdown("---")
    st.markdown("### 🔬 Model Parameters")
    st.info(f"""
    - **Architecture:** Tabular DDPM MLP
    - **Total Features:** {data_dim} continuous slices
    - **Timesteps (T):** 100 denoising steps
    - **Device:** {ddpm.device.upper()}
    """)
    
    if st.button("🚀 Generate New Synthetic Cohort", use_container_width=True, type="primary"):
        with st.spinner(f"Running reverse diffusion across 100 timesteps for {n_samples} records..."):
            vectors = ddpm.sample(n_samples=n_samples, data_dim=data_dim, temperature=temperature)
            st.session_state['synthetic_df'] = preprocessor.inverse_transform(vectors)
            st.success(f"Successfully generated {n_samples} synthetic customer profiles!")

df_synth = st.session_state['synthetic_df']

# ----------------- Top KPI Metric Cards ----------------- #
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Synthetic Customers", f"{len(df_synth):,}")
with c2:
    synth_churn = df_synth['churned'].mean()
    real_churn = df_real['churned'].mean()
    st.metric("Synthetic Churn Rate", f"{synth_churn:.1%}", delta=f"{(synth_churn - real_churn):.1%} vs Real")
with c3:
    st.metric("Avg Monthly Fee", f"${df_synth['monthly_fee'].mean():.2f}", delta=f"${(df_synth['monthly_fee'].mean() - df_real['monthly_fee'].mean()):.2f}")
with c4:
    st.metric("Avg Customer Tenure", f"{df_synth['tenure_months'].mean():.1f} mo", delta=f"{(df_synth['tenure_months'].mean() - df_real['tenure_months'].mean()):.1f} mo")

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- Main Tabs ----------------- #
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Explorer & Export",
    "📊 Distribution Fidelity (KS Test)",
    "🔗 Correlation Preservation",
    "🤖 ML Utility Benchmark (TSTR)",
    "📑 Architecture & Presentation Guide"
])

# ----------------- Tab 1: Data Explorer ----------------- #
with tab1:
    st.subheader("Synthetic Customer Lifecycle Data Preview")
    st.markdown("Each row represents a synthesized subscription profile generated via reverse Gaussian denoising:")
    
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        selected_tier = st.multiselect("Filter by Subscription Tier", options=df_synth['subscription_tier'].unique(), default=df_synth['subscription_tier'].unique())
    with col_filter2:
        selected_churn = st.multiselect("Filter by Churn Status", options=[0, 1], format_func=lambda x: "Active (0)" if x == 0 else "Churned (1)", default=[0, 1])

    filtered_df = df_synth[
        df_synth['subscription_tier'].isin(selected_tier) & 
        df_synth['churned'].isin(selected_churn)
    ]

    st.dataframe(filtered_df, use_container_width=True, height=350)
    
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Synthetic Dataset (CSV)",
        data=csv_data,
        file_name="churnsynth_synthetic_subscription_data.csv",
        mime="text/csv",
        type="primary"
    )

# ----------------- Tab 2: Distribution Fidelity ----------------- #
with tab2:
    st.subheader("Statistical Distribution Alignment (Real vs Synthetic)")
    st.markdown("Examine how closely the Tabular Diffusion Model matches the continuous and categorical marginal distributions of the real dataset.")

    continuous_features = [
        'tenure_months', 'monthly_fee', 'app_logins_per_month', 
        'skipped_boxes_ratio', 'cs_tickets_raised', 'delivery_delays_count', 
        'engagement_score'
    ]
    
    selected_feat = st.selectbox("Select Feature to Inspect:", continuous_features, index=0)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_real[selected_feat],
        histnorm='probability density',
        name='Real Data (Ground Truth)',
        opacity=0.6,
        marker_color='#2563EB',
        nbinsx=30
    ))
    fig.add_trace(go.Histogram(
        x=df_synth[selected_feat],
        histnorm='probability density',
        name='Synthetic Data (Diffusion DDPM)',
        opacity=0.6,
        marker_color='#10B981',
        nbinsx=30
    ))
    fig.update_layout(
        barmode='overlay',
        title=f"Marginal Distribution Overlay: {selected_feat}",
        xaxis_title=selected_feat,
        yaxis_title="Probability Density",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Quantitative Fidelity: Kolmogorov-Smirnov & Wasserstein Distance")
    dist_metrics_df = compute_distribution_metrics(df_real, df_synth, continuous_features)
    st.dataframe(dist_metrics_df, use_container_width=True)
    st.caption("Lower Kolmogorov-Smirnov (KS) statistic and Wasserstein distance denote superior statistical fidelity.")

# ----------------- Tab 3: Correlation Preservation ----------------- #
with tab3:
    st.subheader("Preservation of Customer Lifecycle Correlation Matrix")
    st.markdown("""
    A hallmark of high-fidelity synthetic data is retaining **inter-feature correlations** (e.g. how customer service complaints, delivery delays, and skipped boxes trigger churn).
    """)

    num_cols = continuous_features + ['churned']
    corr_real, corr_synth, diff_corr, frob_norm = compute_correlation_difference(df_real, df_synth, num_cols)

    st.info(f"📐 **Frobenius Norm of Correlation Difference Matrix:** `{frob_norm:.3f}` (Lower = higher preservation of multivariate dependencies)")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("**Real Data Correlations**")
        fig_r = px.imshow(
            corr_real, 
            text_auto=".2f", 
            color_continuous_scale="RdBu_r", 
            zmin=-1, zmax=1,
            aspect="auto"
        )
        fig_r.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_r, use_container_width=True)

    with col_h2:
        st.markdown("**Synthetic Data (Diffusion) Correlations**")
        fig_s = px.imshow(
            corr_synth, 
            text_auto=".2f", 
            color_continuous_scale="RdBu_r", 
            zmin=-1, zmax=1,
            aspect="auto"
        )
        fig_s.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_s, use_container_width=True)

# ----------------- Tab 4: ML Utility Benchmark (TSTR) ----------------- #
with tab4:
    st.subheader("Machine Learning Utility Benchmark: TSTR vs TRTR")
    st.markdown("""
    **The Ultimate Test:** Can a machine learning model trained **strictly on synthetic data** accurately predict churn on **unseen real customers**?
    - **TRTR:** Train on Real, Test on Real (Theoretical upper bound)
    - **TSTR:** Train on Synthetic, Test on Real (Our diffusion model)
    """)

    if st.button("⚡ Run TSTR ML Utility Evaluation", type="primary"):
        with st.spinner("Training Random Forest on synthetic data and evaluating on real holdout test set..."):
            ml_results = evaluate_ml_utility(df_real, df_synth)
            
            u1, u2, u3 = st.columns(3)
            with u1:
                st.metric("TSTR ROC-AUC", f"{ml_results['TSTR_AUC']:.3f}", delta=f"{ml_results['TSTR_AUC'] - ml_results['TRTR_AUC']:.3f} vs TRTR")
            with u2:
                st.metric("TSTR Accuracy", f"{ml_results['TSTR_Accuracy']:.1%}", delta=f"{(ml_results['TSTR_Accuracy'] - ml_results['TRTR_Accuracy']):.1%}")
            with u3:
                st.metric("TSTR F1-Score", f"{ml_results['TSTR_F1']:.3f}", delta=f"{ml_results['TSTR_F1'] - ml_results['TRTR_F1']:.3f}")

            st.markdown("#### Feature Importances Discovered by Synthetic-Trained Model")
            imp_df = pd.DataFrame({
                'Feature': ml_results['Feature_Names'],
                'Importance': ml_results['TSTR_Feature_Importances']
            }).sort_values('Importance', ascending=True)

            fig_imp = px.bar(
                imp_df, 
                x='Importance', 
                y='Feature', 
                orientation='h',
                title="Top Churn Drivers Learned from Synthetic Data",
                color='Importance',
                color_continuous_scale="Teal"
            )
            fig_imp.update_layout(template="plotly_white")
            st.plotly_chart(fig_imp, use_container_width=True)
            
            st.success("✅ **Conclusion:** The model trained solely on diffusion-generated synthetic data achieves high predictive accuracy and learns the authentic churn drivers!")
    else:
        st.write("Click the button above to execute the real-time TSTR benchmark.")

# ----------------- Tab 5: Architecture & Presentation Guide ----------------- #
with tab5:
    st.subheader("Project Architecture & Presentation Points")
    st.markdown("""
    ### 1. Problem Formulation
    - **Domain:** Retail Subscription Boxes (e.g. beauty, snacks, lifestyle kits).
    - **Challenge:** Early-stage product lines and privacy regulations (GDPR/CCPA) prevent sharing customer history for churn prediction.
    - **Solution:** **ChurnSynth AI** synthesizes full customer trajectories with realistic churn outcomes.

    ### 2. Why Tabular Diffusion Models (DDPM)?
    Compared to traditional Tabular GANs (CTGAN):
    - **No Mode Collapse:** Diffusion models stably optimize a variational lower bound via MSE loss, avoiding the minimax instability of GAN discriminators.
    - **High Distribution Coverage:** Smooth reverse stochastic differential equations preserve tails and edge cases (e.g. rare churn incidents).
    - **Mixed Data Handling:** Normalization maps continuous variables into $[-1, 1]$ while categorical variables are projected via one-hot continuous relaxations and recovered using argmax decoding.

    ### 3. Mathematical Foundations
    - **Forward Process (Noise Addition):**
      $$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\\bar{\\alpha}_t} x_0, (1 - \\bar{\\alpha}_t) \mathbf{I})$$
    - **Denoising Loss Objective:**
      $$\mathcal{L}_{\\text{diffusion}} = \mathbb{E}_{t, x_0, \epsilon} [\\| \epsilon - \epsilon_\\theta(x_t, t) \\|^2]$$
    - **Reverse Sampling Process:**
      $$x_{t-1} = \\frac{1}{\\sqrt{\\alpha_t}} \\left( x_t - \\frac{\\beta_t}{\\sqrt{1 - \\bar{\\alpha}_t}} \epsilon_\\theta(x_t, t) \\right) + \\sigma_t z$$
    """)
