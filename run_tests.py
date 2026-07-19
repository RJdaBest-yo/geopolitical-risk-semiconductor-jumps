"""
Side Study test runner
"""
import sys, os, time, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Main project on path (for src.jump_diffusion_engine, etc.)
_main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, _main_dir)

from modules.forecasting import _run_self_tests as test_forecast
from modules.behavioral_adaptation import _run_self_tests as test_adaptation


def main():
    print("=" * 60)
    print("  Side Study - Test Suite")
    print("=" * 60)

    total_start = time.time()

    tests = [
        ("Module 1: Forecasting & Lead Time", test_forecast),
        ("Module 2: Behavioral Adaptation", test_adaptation),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n{'-' * 60}")
        print(f">> Running: {name}")
        print(f"{'-' * 60}")
        try:
            t0 = time.time()
            test_fn()
            elapsed = time.time() - t0
            print(f"\n[PASS] {name} ({elapsed:.1f}s)")
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, total {total_elapsed:.1f}s")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll side study modules passed!")


if __name__ == "__main__":
    main()
