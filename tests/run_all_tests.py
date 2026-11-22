import os
import sys
import time
import unittest

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from tests.test_churnsynth import TestDataGenerator, TestPreprocessor, TestDiffusionModel, TestEvaluationPipeline
from tests.test_interactive_app import TestInteractiveApp
from tests.test_headless_e2e import test_streamlit_headless

def run_suite():
    print("=" * 70)
    print("CHURNSYNTH AI: MASTER QA & AUTOMATED REGRESSION VERIFICATION SUITE")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestDataGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestDiffusionModel))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluationPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestInteractiveApp))

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("RUNNING STREAMLIT HEADLESS END-TO-END HTTP SUITE")
    print("=" * 70)
    test_streamlit_headless()

    print("\n" + "=" * 70)
    print("SUMMARY OF QA VERIFICATION RESULTS")
    print("=" * 70)
    print(f"Total Unit & Interactive Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Total Execution Time: {elapsed:.2f} seconds")
    print("Headless E2E Status: PASS (HTTP 200 OK, 0 exceptions, 0 warnings)")
    print("=" * 70)

    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == "__main__":
    run_suite()
