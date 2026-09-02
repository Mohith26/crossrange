"""Test runner. No framework, so it runs anywhere Python does."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = []
FAIL = []


def check(label, condition):
    if condition:
        PASS.append(label)
    else:
        FAIL.append(label)
        print("  FAIL  " + label)


def close(label, got, want, tol):
    ok = abs(got - want) <= tol
    if ok:
        PASS.append(label)
    else:
        FAIL.append(label)
        print("  FAIL  %s: got %r want %r (tol %r)" % (label, got, want, tol))


def main():
    import test_frames
    import test_validation
    import test_sitl

    groups = [
        ("frames and atmosphere", test_frames.register(check, close)),
        ("physics validation", test_validation.register(check, close)),
        ("software in the loop", test_sitl.register(check, close)),
    ]

    for name, tests in groups:
        print("\n" + name)
        for fn in tests:
            try:
                fn()
            except Exception as exc:
                FAIL.append(fn.__name__)
                print("  ERROR %s: %r" % (fn.__name__, exc))

    print("\n%d checks passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        print("failures:")
        for f in FAIL:
            print("  - " + f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

