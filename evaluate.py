import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

def compute_distribution_metrics(real_df: pd.DataFrame, synth_df: pd.DataFrame, continuous_cols: list):
    """
    Computes statistical similarity between real and synthetic continuous features.
    """
    results = []
    for col in continuous_cols:
        real_vals = real_df[col].dropna().values
        synth_vals = synth_df[col].dropna().values
        
        # KS-test
        ks_stat, p_val = ks_2samp(real_vals, synth_vals)
        # Wasserstein Distance
        w_dist = wasserstein_distance(real_vals, synth_vals)
        
        results.append({
            'Feature': col,
            'Real Mean': np.mean(real_vals),
            'Synth Mean': np.mean(synth_vals),
            'Real Std': np.std(real_vals),
            'Synth Std': np.std(synth_vals),
            'KS Statistic': round(ks_stat, 4),
            'KS p-value': round(p_val, 4),
            'Wasserstein Distance': round(w_dist, 4)
        })
    return pd.DataFrame(results)

def compute_correlation_difference(real_df: pd.DataFrame, synth_df: pd.DataFrame, numeric_cols: list):
    """
    Measures how well the joint correlation structure is retained (Frobenius norm difference).
    """
    corr_real = real_df[numeric_cols].corr()
    corr_synth = synth_df[numeric_cols].corr()
    diff_matrix = corr_synth - corr_real
    frobenius_norm = np.linalg.norm(diff_matrix.values, 'fro')
    return corr_real, corr_synth, diff_matrix, frobenius_norm

def evaluate_ml_utility(real_df: pd.DataFrame, synth_df: pd.DataFrame):
    """
    TSTR (Train on Synthetic, Test on Real) benchmark vs
    TRTR (Train on Real, Test on Real) baseline.
    """
    feature_cols = [c for c in real_df.columns if c not in ['customer_id', 'churned']]
    
    # One-hot encode for ML model
    combined = pd.concat([real_df[feature_cols], synth_df[feature_cols]], ignore_index=True)
    combined_encoded = pd.get_dummies(combined, drop_first=True)
    
    n_real = len(real_df)
    X_real = combined_encoded.iloc[:n_real]
    y_real = real_df['churned'].values
    
    X_synth = combined_encoded.iloc[n_real:]
    y_synth = synth_df['churned'].values

    # Split Real into Train (80%) and Test (20%)
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
    )

    # 1. TRTR Benchmark (Train on Real, Test on Real)
    rf_real = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_real.fit(X_real_train, y_real_train)
    y_pred_trtr = rf_real.predict(X_real_test)
    y_proba_trtr = rf_real.predict_proba(X_real_test)[:, 1]

    # 2. TSTR Benchmark (Train on Synthetic, Test on Real)
    rf_synth = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_synth.fit(X_synth, y_synth)
    y_pred_tstr = rf_synth.predict(X_real_test)
    y_proba_tstr = rf_synth.predict_proba(X_real_test)[:, 1]

    metrics = {
        'TRTR_AUC': roc_auc_score(y_real_test, y_proba_trtr),
        'TRTR_Accuracy': accuracy_score(y_real_test, y_pred_trtr),
        'TRTR_F1': f1_score(y_real_test, y_pred_trtr, zero_division=0),
        'TSTR_AUC': roc_auc_score(y_real_test, y_proba_tstr),
        'TSTR_Accuracy': accuracy_score(y_real_test, y_pred_tstr),
        'TSTR_F1': f1_score(y_real_test, y_pred_tstr, zero_division=0),
        'TSTR_Precision': precision_score(y_real_test, y_pred_tstr, zero_division=0),
        'TSTR_Recall': recall_score(y_real_test, y_pred_tstr, zero_division=0),
        'Feature_Names': list(X_real.columns),
        'TSTR_Feature_Importances': rf_synth.feature_importances_,
        'Confusion_Matrix': confusion_matrix(y_real_test, y_pred_tstr)
    }

    return metrics
