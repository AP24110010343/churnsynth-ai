# 📦 ChurnSynth AI: Synthetic Subscription Churn Generator

> **Domain:** Retail (Subscription Boxes)  
> **Model:** Tabular Denoising Diffusion Probabilistic Model (DDPM)  
> **Deployment:** Streamlit Interactive Dashboard  
> **Target Review:** Team 2 | 25th Nov (11:00 AM – 1:00 PM) | CV-block 911  

---

## 🎯 Problem Statement & Motivation
Subscription box businesses (such as gourmet snacks, beauty kits, apparel, or gaming merchandise) frequently launch new product lines or tiers. However:
1. **Cold-Start Problem:** New lines lack historical customer lifecycle & churn records.
2. **Privacy & Compliance:** Data protection regulations (GDPR / CCPA) prevent sharing real subscriber behavioral data with third-party model developers.
3. **Imbalanced Churn:** Real churn events are sparse, leading to undertrained predictive models.

**ChurnSynth AI** solves this by using **Tabular Diffusion Models (DDPM)** to generate rich, statistically faithful synthetic customer lifecycles and churn events without replicating real user PII.

---

## 🏗️ Architecture & Technical Highlights

```
Raw Subscription Schema (Mixed Tabular)
                  │
                  ▼
       [TabularPreprocessor]
 ┌────────────────────────────────────┐
 │ • Continuous: MinMax scaled [-1, 1]│
 │ • Categorical: One-Hot Simplex     │
 └────────────────────────────────────┘
                  │
                  ▼ (Forward Diffusion: Linear Beta Schedule T=100)
    Noisy Vector x_t ~ q(x_t | x_0)
                  │
                  ▼
    [TabularDiffusionMLP (Denoising Network)]
 ┌────────────────────────────────────┐
 │ • Sinusoidal Timestep Embeddings   │
 │ • Residual MLP with LayerNorm      │
 │ • SiLU Non-linear Activations      │
 └────────────────────────────────────┘
                  │
                  ▼ (Reverse Denoising Process)
       Denoised Vector x_0
                  │
                  ▼
       [Argmax & Range Inversion]
                  │
                  ▼
  High-Fidelity Synthetic Dataset (.csv)
```

### Why Diffusion Models instead of GANs (CTGAN)?
- **No Minimax Instability:** GANs suffer from mode collapse (generating only a subset of customer types) and hyperparameter sensitivity.
- **Superior Probability Density Coverage:** Diffusion models optimize an exact variational lower bound via MSE loss, reliably reproducing rare churn triggers.
- **Controlled Generation:** Support adjusting noise temperature during sampling to explore conservative vs exploratory customer behaviors.

---

## 🚀 Quick Start Guide

### 1. Launch the Application
Simply run the startup script:
```bash
cd /Users/vegirajusrujayvarma/.gemini/antigravity/scratch/churnsynth_ai
./run_app.sh
```
Or with python:
```bash
./venv/bin/streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 2. Retrain the Diffusion Model (Optional)
If you want to re-train the model weights from scratch:
```bash
./venv/bin/python train_offline.py
```

---

## 📊 Application Features

1. **📋 Data Explorer & Instant Export:**
   - Filter by subscription tier (`Basic`, `Standard`, `Premium`) and churn status.
   - 1-click download of generated CSV (`churnsynth_synthetic_subscription_data.csv`).
2. **📈 Distribution Alignment (Real vs Synthetic):**
   - Interactive Plotly histogram overlays.
   - Rigorous quantitative verification: Kolmogorov-Smirnov (KS) test statistics and Wasserstein distance.
3. **🔗 Correlation Matrix Preservation:**
   - Side-by-side heatmaps proving that key churn drivers (e.g. delivery delays, customer service complaints, skipped boxes) remain accurately correlated with churn.
4. **🤖 ML Utility Benchmark (TSTR vs TRTR):**
   - **TSTR (Train on Synthetic, Test on Real):** Trains a Random Forest churn model purely on synthetic data and tests it on unseen real customer data.
   - Measures ROC-AUC, Accuracy, F1-Score, and top feature importances.
5. **📑 Built-in Presentation & Defense Guide:**
   - Mathematical equations and talking points ready for your evaluation panel.

---

## 🎤 Key Presentation Talking Points (For Nov 25)

1. **Explain the Domain:** Retail subscription boxes live or die by customer retention. Testing churn algorithms requires data before launching campaigns.
2. **Explain the Model:** *"We implemented a Denoising Diffusion Probabilistic Model (DDPM) customized for tabular data. It continuously denoises random Gaussian noise through 100 timesteps conditioned on sinusoidal time embeddings."*
3. **Show the ML Utility (TSTR):** *"The gold standard of synthetic data is whether models trained on it generalize to the real world. Our TSTR benchmark shows that a churn classifier trained strictly on our synthetic data achieves competitive ROC-AUC against one trained on real data."*
