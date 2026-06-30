#!/usr/bin/env python3
"""
Simple test runner for BoostARoota.
Tries pytest first, falls back to running test functions directly.
"""
import sys
import traceback

def run_with_pytest():
    try:
        import pytest
        print("Running full test suite with pytest...")
        return pytest.main(["-q", "tests/test_boostaroota.py"])
    except ImportError:
        return None

def run_manually():
    print("pytest not found, running tests manually...\n")
    # Import test functions
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_boostaroota", "tests/test_boostaroota.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["test_boostaroota"] = mod
    spec.loader.exec_module(mod)

    test_fns = [getattr(mod, name) for name in dir(mod) if name.startswith('test_') and callable(getattr(mod, name))]
    failed = 0
    passed = 0
    for fn in test_fns:
        try:
            fn()
            print(f"{fn.__name__}: PASS")
            passed += 1
        except Exception as e:
            failed += 1
            print(f"{fn.__name__}: FAIL - {e}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed+failed} tests")
    return 1 if failed else 0

if __name__ == "__main__":
    result = run_with_pytest()
    if result is None:
        result = run_manually()
    sys.exit(result)

