import os
import sys
import unittest
import pickle
import numpy as np
import pandas as pd
import torch

# Add parent directory to sys.path
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from data_generator import generate_baseline_subscription_data
from diffusion_model import TabularPreprocessor, TabularDiffusionMLP, TabularDDPM
from evaluate import compute_distribution_metrics, compute_correlation_difference, evaluate_ml_utility


class TestDataGenerator(unittest.TestCase):
    """Exhaustive tests for baseline data generator and stored dataset."""

    def test_generate_baseline_subscription_data_schema_and_types(self):
        n_samples = 500
        df = generate_baseline_subscription_data(n_samples=n_samples, random_seed=42)
        
        self.assertEqual(len(df), n_samples)
        self.assertEqual(df.isna().sum().sum(), 0, "Generated data contains NaN values")

        expected_cols = [
            'customer_id', 'subscription_tier', 'box_category', 'tenure_months',
            'monthly_fee', 'discount_applied', 'app_logins_per_month',
            'skipped_boxes_ratio', 'cs_tickets_raised', 'delivery_delays_count',
            'engagement_score', 'churned'
        ]
        self.assertListEqual(list(df.columns), expected_cols)

    def test_generate_baseline_subscription_data_ranges(self):
        df = generate_baseline_subscription_data(n_samples=1000, random_seed=123)

        # Categorical sets
        valid_tiers = {'Basic', 'Standard', 'Premium'}
        self.assertTrue(set(df['subscription_tier'].unique()).issubset(valid_tiers))

        valid_categories = {'Beauty & Care', 'Gourmet Snacks', 'Eco Home', 'Fitness & Wellness', 'Book & Craft'}
        self.assertTrue(set(df['box_category'].unique()).issubset(valid_categories))

        # Numeric ranges
        self.assertTrue((df['tenure_months'] >= 1).all() and (df['tenure_months'] <= 48).all())
        self.assertTrue((df['monthly_fee'] >= 12.0).all() and (df['monthly_fee'] <= 65.0).all())
        self.assertTrue(set(df['discount_applied'].unique()).issubset({0, 1}))
        self.assertTrue((df['app_logins_per_month'] >= 0).all() and (df['app_logins_per_month'] <= 30).all())
        self.assertTrue((df['skipped_boxes_ratio'] >= 0.0).all() and (df['skipped_boxes_ratio'] <= 0.8).all())
        self.assertTrue((df['cs_tickets_raised'] >= 0).all() and (df['cs_tickets_raised'] <= 7).all())
        self.assertTrue((df['delivery_delays_count'] >= 0).all() and (df['delivery_delays_count'] <= 5).all())
        self.assertTrue((df['engagement_score'] >= 5.0).all() and (df['engagement_score'] <= 98.0).all())
        self.assertTrue(set(df['churned'].unique()).issubset({0, 1}))

    def test_baseline_subscription_csv_file_on_disk(self):
        csv_path = os.path.join(PROJECT_DIR, 'baseline_subscription_data.csv')
        self.assertTrue(os.path.exists(csv_path), "baseline_subscription_data.csv file is missing")

        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 4000)
        self.assertEqual(df.isna().sum().sum(), 0, "baseline_subscription_data.csv contains NaNs")
        self.assertEqual(df.shape[1], 12)
        
        churn_rate = df['churned'].mean()
        self.assertGreater(churn_rate, 0.05)
        self.assertLess(churn_rate, 0.35)


class TestPreprocessor(unittest.TestCase):
    """Exhaustive tests for TabularPreprocessor persistence, transformation, and inversion."""

    @classmethod
    def setUpClass(cls):
        cls.pkl_path = os.path.join(PROJECT_DIR, 'preprocessor.pkl')
        cls.csv_path = os.path.join(PROJECT_DIR, 'baseline_subscription_data.csv')
        cls.df_real = pd.read_csv(cls.csv_path)

    def test_preprocessor_pickle_loading(self):
        self.assertTrue(os.path.exists(self.pkl_path), "preprocessor.pkl is missing")
        with open(self.pkl_path, 'rb') as f:
            preprocessor = pickle.load(f)

        self.assertIsInstance(preprocessor, TabularPreprocessor)
        self.assertEqual(preprocessor.total_dim, 17)
        self.assertEqual(len(preprocessor.continuous_cols), 9)
        self.assertEqual(len(preprocessor.categorical_cols), 2)

    def test_preprocessor_transform_dimensions_and_nans(self):
        with open(self.pkl_path, 'rb') as f:
            preprocessor = pickle.load(f)

        X = preprocessor.transform(self.df_real)
        self.assertEqual(X.shape, (len(self.df_real), 17))
        self.assertFalse(np.isnan(X).any(), "Transformed array contains NaN values")
        self.assertFalse(np.isinf(X).any(), "Transformed array contains infinite values")
        self.assertEqual(X.dtype, np.float32)

    def test_preprocessor_inversion_fidelity(self):
        with open(self.pkl_path, 'rb') as f:
            preprocessor = pickle.load(f)

        X = preprocessor.transform(self.df_real)
        df_reconstructed = preprocessor.inverse_transform(X)

        self.assertEqual(len(df_reconstructed), len(self.df_real))
        self.assertEqual(df_reconstructed.isna().sum().sum(), 0)

        # Categorical columns exact match
        for cat_col in preprocessor.categorical_cols:
            self.assertTrue(
                (df_reconstructed[cat_col].values == self.df_real[cat_col].values).all(),
                f"Categorical column '{cat_col}' failed exact round-trip inversion"
            )

        # Continuous columns close numerical match
        np.testing.assert_allclose(
            df_reconstructed['monthly_fee'].values,
            self.df_real['monthly_fee'].values,
            atol=0.01,
            err_msg="monthly_fee inverted values deviated beyond 0.01 tolerance"
        )
        np.testing.assert_allclose(
            df_reconstructed['engagement_score'].values,
            self.df_real['engagement_score'].values,
            atol=0.1,
            err_msg="engagement_score inverted values deviated beyond 0.1 tolerance"
        )

        # Integer discrete features exact match
        for int_col in ['tenure_months', 'discount_applied', 'cs_tickets_raised', 'delivery_delays_count', 'churned']:
            self.assertTrue(
                (df_reconstructed[int_col].values == self.df_real[int_col].values).all(),
                f"Discrete column '{int_col}' failed round-trip inversion equality"
            )

    def test_preprocessor_synthetic_vector_inversion(self):
        with open(self.pkl_path, 'rb') as f:
            preprocessor = pickle.load(f)

        # Random synthetic vector batch
        np.random.seed(42)
        mock_synth_vectors = np.random.uniform(-1.5, 1.5, size=(100, 17)).astype(np.float32)
        df_synth = preprocessor.inverse_transform(mock_synth_vectors)

        self.assertEqual(len(df_synth), 100)
        self.assertEqual(df_synth.isna().sum().sum(), 0)
        self.assertTrue(set(df_synth['subscription_tier'].unique()).issubset({'Basic', 'Standard', 'Premium'}))
        self.assertTrue(set(df_synth['churned'].unique()).issubset({0, 1}))
        self.assertTrue(set(df_synth['discount_applied'].unique()).issubset({0, 1}))


class TestDiffusionModel(unittest.TestCase):
    """Exhaustive tests for TabularDiffusionMLP architecture, checkpoint, and DDPM sampling."""

    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = os.path.join(PROJECT_DIR, 'churn_diffusion_model.pt')
        cls.pkl_path = os.path.join(PROJECT_DIR, 'preprocessor.pkl')
        with open(cls.pkl_path, 'rb') as f:
            cls.preprocessor = pickle.load(f)

    def test_checkpoint_structure(self):
        self.assertTrue(os.path.exists(self.ckpt_path), "churn_diffusion_model.pt is missing")
        checkpoint = torch.load(self.ckpt_path, map_location='cpu')

        expected_keys = {'model_state_dict', 'data_dim', 'timesteps', 'continuous_cols', 'categorical_cols'}
        self.assertTrue(expected_keys.issubset(checkpoint.keys()))
        self.assertEqual(checkpoint['data_dim'], 17)
        self.assertEqual(checkpoint['timesteps'], 100)

    def test_tabular_diffusion_mlp_architecture_and_forward_pass(self):
        checkpoint = torch.load(self.ckpt_path, map_location='cpu')
        model = TabularDiffusionMLP(data_dim=checkpoint['data_dim'], hidden_dim=128, time_dim=64)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        batch_size = 16
        x = torch.randn(batch_size, checkpoint['data_dim'])
        t = torch.randint(0, checkpoint['timesteps'], (batch_size,)).long()

        with torch.no_grad():
            pred = model(x, t)

        self.assertEqual(pred.shape, (batch_size, checkpoint['data_dim']))
        self.assertFalse(torch.isnan(pred).any())
        self.assertFalse(torch.isinf(pred).any())

    def test_ddpm_sampling_execution(self):
        checkpoint = torch.load(self.ckpt_path, map_location='cpu')
        model = TabularDiffusionMLP(data_dim=checkpoint['data_dim'], hidden_dim=128, time_dim=64)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        ddpm = TabularDDPM(model, timesteps=checkpoint['timesteps'], device='cpu')

        n_samples = 50
        samples = ddpm.sample(n_samples=n_samples, data_dim=checkpoint['data_dim'], temperature=0.95)

        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(samples.shape, (n_samples, checkpoint['data_dim']))
        self.assertFalse(np.isnan(samples).any(), "Sampled vectors contain NaN")
        self.assertFalse(np.isinf(samples).any(), "Sampled vectors contain Inf")

        # Inversion of sampled vectors into DataFrame
        df_synth = self.preprocessor.inverse_transform(samples)
        self.assertEqual(len(df_synth), n_samples)
        self.assertEqual(df_synth.isna().sum().sum(), 0)
        self.assertEqual(df_synth.shape[1], 12)
        self.assertTrue(set(df_synth['churned'].unique()).issubset({0, 1}))


class TestEvaluationPipeline(unittest.TestCase):
    """Exhaustive tests for compute_distribution_metrics, compute_correlation_difference, and evaluate_ml_utility."""

    @classmethod
    def setUpClass(cls):
        cls.csv_path = os.path.join(PROJECT_DIR, 'baseline_subscription_data.csv')
        cls.df_real = pd.read_csv(cls.csv_path)

        # Generate a high quality synthetic cohort for evaluation tests
        ckpt_path = os.path.join(PROJECT_DIR, 'churn_diffusion_model.pt')
        pkl_path = os.path.join(PROJECT_DIR, 'preprocessor.pkl')
        with open(pkl_path, 'rb') as f:
            preprocessor = pickle.load(f)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model = TabularDiffusionMLP(data_dim=checkpoint['data_dim'], hidden_dim=128, time_dim=64)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        ddpm = TabularDDPM(model, timesteps=100, device='cpu')
        
        # Sample 400 synthetic records
        torch.manual_seed(42)
        np.random.seed(42)
        samples = ddpm.sample(n_samples=400, data_dim=checkpoint['data_dim'], temperature=0.95)
        cls.df_synth = preprocessor.inverse_transform(samples)
        cls.continuous_cols = checkpoint['continuous_cols']

    def test_compute_distribution_metrics(self):
        metrics_df = compute_distribution_metrics(self.df_real, self.df_synth, self.continuous_cols)

        self.assertIsInstance(metrics_df, pd.DataFrame)
        self.assertEqual(len(metrics_df), len(self.continuous_cols))
        
        expected_cols = [
            'Feature', 'Real Mean', 'Synth Mean', 'Real Std', 'Synth Std',
            'KS Statistic', 'KS p-value', 'Wasserstein Distance'
        ]
        self.assertListEqual(list(metrics_df.columns), expected_cols)
        self.assertEqual(metrics_df.isna().sum().sum(), 0)

        # Range checks
        self.assertTrue(((metrics_df['KS Statistic'] >= 0.0) & (metrics_df['KS Statistic'] <= 1.0)).all())
        self.assertTrue(((metrics_df['KS p-value'] >= 0.0) & (metrics_df['KS p-value'] <= 1.0)).all())
        self.assertTrue((metrics_df['Wasserstein Distance'] >= 0.0).all())

    def test_compute_correlation_difference(self):
        numeric_cols = [
            'tenure_months', 'monthly_fee', 'discount_applied', 
            'app_logins_per_month', 'skipped_boxes_ratio', 
            'cs_tickets_raised', 'delivery_delays_count', 
            'engagement_score', 'churned'
        ]
        corr_real, corr_synth, diff_matrix, frobenius_norm = compute_correlation_difference(
            self.df_real, self.df_synth, numeric_cols
        )

        self.assertIsInstance(corr_real, pd.DataFrame)
        self.assertIsInstance(corr_synth, pd.DataFrame)
        self.assertIsInstance(diff_matrix, pd.DataFrame)
        self.assertIsInstance(float(frobenius_norm), float)

        self.assertEqual(corr_real.shape, (len(numeric_cols), len(numeric_cols)))
        self.assertEqual(corr_synth.shape, (len(numeric_cols), len(numeric_cols)))
        self.assertEqual(diff_matrix.shape, (len(numeric_cols), len(numeric_cols)))
        self.assertGreaterEqual(frobenius_norm, 0.0)
        
        # Diagonals of correlation matrix must be 1.0, difference diagonal must be 0.0
        np.testing.assert_allclose(np.diag(corr_real.values), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.diag(corr_synth.values), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.diag(diff_matrix.values), 0.0, atol=1e-5)

    def test_evaluate_ml_utility(self):
        metrics = evaluate_ml_utility(self.df_real, self.df_synth)

        self.assertIsInstance(metrics, dict)
        required_keys = [
            'TRTR_AUC', 'TRTR_Accuracy', 'TRTR_F1',
            'TSTR_AUC', 'TSTR_Accuracy', 'TSTR_F1',
            'TSTR_Precision', 'TSTR_Recall',
            'Feature_Names', 'TSTR_Feature_Importances',
            'Confusion_Matrix'
        ]
        for key in required_keys:
            self.assertIn(key, metrics)

        # Performance scores within [0, 1]
        for metric_name in ['TRTR_AUC', 'TRTR_Accuracy', 'TRTR_F1', 'TSTR_AUC', 'TSTR_Accuracy', 'TSTR_F1', 'TSTR_Precision', 'TSTR_Recall']:
            val = metrics[metric_name]
            self.assertIsInstance(val, (float, np.floating))
            self.assertTrue(0.0 <= val <= 1.0, f"{metric_name} = {val} outside [0, 1]")

        # Confusion Matrix check
        cm = metrics['Confusion_Matrix']
        self.assertIsInstance(cm, np.ndarray)
        self.assertEqual(cm.shape, (2, 2))
        self.assertGreater(cm.sum(), 0)

        # Feature importances check
        feature_names = metrics['Feature_Names']
        importances = metrics['TSTR_Feature_Importances']
        self.assertEqual(len(feature_names), len(importances))
        self.assertAlmostEqual(float(importances.sum()), 1.0, places=3)


if __name__ == '__main__':
    unittest.main()
