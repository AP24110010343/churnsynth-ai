import os
import sys
import unittest

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from streamlit.testing.v1 import AppTest

class TestInteractiveApp(unittest.TestCase):
    """Test full interactive flows of the Streamlit application."""

    def test_app_initial_render(self):
        at = AppTest.from_file(os.path.join(PROJECT_DIR, 'app.py'), default_timeout=60)
        at.run()
        self.assertEqual(len(at.exception), 0, f"Exceptions on initial render: {at.exception}")

        # Check session state initialized
        self.assertIn('synthetic_df', at.session_state)
        df_synth = at.session_state['synthetic_df']
        self.assertEqual(len(df_synth), 1000)
        self.assertEqual(df_synth.isna().sum().sum(), 0)

    def test_app_sample_button_interaction(self):
        at = AppTest.from_file(os.path.join(PROJECT_DIR, 'app.py'), default_timeout=60)
        at.run()
        self.assertEqual(len(at.exception), 0)

        # Set sample count to 200 for fast test execution
        if len(at.slider) > 0:
            # Look for sample size slider
            for s in at.slider:
                if "Sample Size" in s.label or "Cohort Size" in s.label or "Samples" in s.label:
                    s.set_value(200)
                    break

        # Click the "Sample Synthetic Cohort (Reverse SDE)" button
        sample_btn = None
        for b in at.button:
            if "Sample Synthetic Cohort" in b.label:
                sample_btn = b
                break
        
        self.assertIsNotNone(sample_btn, "Sample button not found")
        sample_btn.click().run()
        self.assertEqual(len(at.exception), 0, f"Exceptions during cohort resample: {at.exception}")
        self.assertIn('synthetic_df', at.session_state)

    def test_app_run_tstr_benchmark_interaction(self):
        at = AppTest.from_file(os.path.join(PROJECT_DIR, 'app.py'), default_timeout=60)
        at.run()
        self.assertEqual(len(at.exception), 0)

        # Click the "Run TSTR Validation" button
        tstr_btn = None
        for b in at.button:
            if "Run TSTR Validation" in b.label:
                tstr_btn = b
                break

        self.assertIsNotNone(tstr_btn, "TSTR button not found")
        tstr_btn.click().run()
        self.assertEqual(len(at.exception), 0, f"Exceptions during TSTR validation run: {at.exception}")
        self.assertIn('ml_results', at.session_state)
        ml_results = at.session_state['ml_results']
        self.assertIn('TSTR_AUC', ml_results)
        self.assertGreater(ml_results['TSTR_AUC'], 0.5)

if __name__ == '__main__':
    unittest.main()
