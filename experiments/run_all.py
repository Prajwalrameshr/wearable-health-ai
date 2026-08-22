from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.evaluator import evaluate_benchmarks
from experiments.robustness import run_robustness_tests

def main():
    print("==========================================================================")
    print("       RUNNING MASTER EXPERIMENTAL EVALUATION & UNIT TEST SUITE")
    print("==========================================================================")

    # 1. Run Unit Tests
    print("\n--- Phase 1: Unit Test Suite Execution ---")
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT_DIR / "tests"))
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)

    if not result.wasSuccessful():
        print("ERROR: Unit tests failed! Please fix errors before proceeding.")
        sys.exit(1)

    print("[SUCCESS] All Unit Tests Passed Successfully!")

    # 2. Run Benchmarks
    print("\n--- Phase 2: Experimental Evaluation Benchmarks ---")
    benchmarks = evaluate_benchmarks()

    # 3. Run Robustness Tests
    print("\n--- Phase 3: Robustness & Noise Injection Tests ---")
    robustness = run_robustness_tests()

    print("\n==========================================================================")
    print("       ALL EXPERIMENTS & UNIT TESTS EXECUTED SUCCESSFULLY")
    print("==========================================================================")

if __name__ == "__main__":
    main()
