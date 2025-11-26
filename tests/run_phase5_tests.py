"""
Phase 5 Test Runner: Comprehensive testing for label leakage fix.

This script runs all Phase 5 tests to validate the causal implementation
eliminates label leakage and ensures the model learns genuine reasoning.

Usage:
    python tests/run_phase5_tests.py [--verbose] [--skip-integration]

Test categories:
1. Unit tests (test_causal_packing.py) - Fast, can run anytime
2. Integration tests (test_copy_detection.py) - Requires implementation
3. Comparison tests (test_side_by_side_comparison.py) - Generates reports
4. Validation tests (test_heldout_validation.py) - Evaluates on held-out set

Exit codes:
    0 = All tests passed
    1 = Some tests failed
    2 = Tests skipped (implementation not ready)
"""

import sys
import argparse
from pathlib import Path
import subprocess
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_test_file(test_file: Path, verbose: bool = False) -> tuple[int, int, int]:
    """
    Run a single test file and return results.

    Args:
        test_file: Path to test file
        verbose: Whether to show verbose output

    Returns:
        Tuple of (passed, failed, skipped) counts
    """
    print(f"\n{'='*70}")
    print(f"Running: {test_file.name}")
    print(f"{'='*70}\n")

    # Run the test file
    cmd = [sys.executable, str(test_file)]

    if verbose:
        result = subprocess.run(cmd, cwd=project_root)
    else:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

    # Parse output to get counts (simplified - actual implementation would parse)
    # For now, just return based on exit code
    if result.returncode == 0:
        return (1, 0, 0)  # Assume passed if exit code 0
    else:
        return (0, 1, 0)  # Assume failed if exit code != 0


def main():
    parser = argparse.ArgumentParser(description="Run Phase 5 label leakage tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-integration", action="store_true", help="Skip integration tests")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    args = parser.parse_args()

    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("="*70)
    print("PHASE 5: COMPREHENSIVE TESTING FOR LABEL LEAKAGE FIX")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {project_root}")
    print()

    # Define test files
    unit_tests = [
        project_root / "tests" / "unit" / "test_causal_packing.py",
    ]

    integration_tests = [
        project_root / "tests" / "integration" / "test_copy_detection.py",
        project_root / "tests" / "integration" / "test_side_by_side_comparison.py",
        project_root / "tests" / "integration" / "test_heldout_validation.py",
    ]

    total_passed = 0
    total_failed = 0
    total_skipped = 0

    # Run unit tests
    print("\n" + "="*70)
    print("UNIT TESTS (T5.1)")
    print("="*70)

    for test_file in unit_tests:
        if not test_file.exists():
            print(f"⊘ SKIPPED: {test_file.name} (file not found)")
            total_skipped += 1
            continue

        passed, failed, skipped = run_test_file(test_file, args.verbose)
        total_passed += passed
        total_failed += failed
        total_skipped += skipped

    # Run integration tests (unless skipped)
    if not args.skip_integration and not args.unit_only:
        print("\n" + "="*70)
        print("INTEGRATION TESTS (T5.2, T5.3, T5.4)")
        print("="*70)

        for test_file in integration_tests:
            if not test_file.exists():
                print(f"⊘ SKIPPED: {test_file.name} (file not found)")
                total_skipped += 1
                continue

            passed, failed, skipped = run_test_file(test_file, args.verbose)
            total_passed += passed
            total_failed += failed
            total_skipped += skipped

    # Print summary
    print("\n" + "="*70)
    print("PHASE 5 TEST SUMMARY")
    print("="*70)
    print(f"Total passed:  {total_passed}")
    print(f"Total failed:  {total_failed}")
    print(f"Total skipped: {total_skipped}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Determine exit code
    if total_failed > 0:
        print("\n✗ FAILED: Some tests failed")
        print("\nNext steps:")
        print("1. Review failed tests above")
        print("2. Fix issues in implementation")
        print("3. Re-run tests")
        return 1
    elif total_passed == 0 and total_skipped > 0:
        print("\n⊘ SKIPPED: All tests skipped (implementation not ready)")
        print("\nNext steps:")
        print("1. Complete Phase 2 (Causal Packing Implementation)")
        print("2. Re-run these tests")
        return 2
    else:
        print("\n✓ SUCCESS: All available tests passed!")
        if total_skipped > 0:
            print(f"\n({total_skipped} test(s) skipped pending full implementation)")
        print("\nNext steps:")
        print("1. Complete any skipped implementations")
        print("2. Run full integration tests")
        print("3. Proceed to Phase 6 (Documentation)")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
