"""Pre-flight compatibility check for the EEGConformer training package.

Run this BEFORE acquire/preprocess/train. It fails fast with an actionable
message if the installed dependency set will break the pipeline (the most
common failure mode is a Braindecode ↔ MOABB version skew that surfaces as
`ImportError: cannot import name 'BNCI2014001' from 'moabb.datasets'`).
"""
from __future__ import annotations

import importlib
import sys
from typing import Iterable

# (module, expected_version, hard) — hard=True means mismatch aborts.
EXPECTED: list[tuple[str, str, bool]] = [
    ("torch",         "2.4.1",   True),
    ("braindecode",   "1.1.1",   True),
    ("moabb",         "1.1.1",   True),
    ("mne",           "1.7.1",   True),
    ("numpy",         "1.26.4",  True),
    ("scipy",         "1.13.1",  False),
    ("sklearn",       "1.5.2",   False),  # package name: scikit-learn
    ("onnx",          "1.16.2",  False),
    ("onnxruntime",   "1.19.2",  False),
]


def _ver(mod: str) -> str:
    m = importlib.import_module(mod)
    return getattr(m, "__version__", "?")


def _check_versions() -> list[str]:
    errors: list[str] = []
    print(f"[compat] python={sys.version.split()[0]}")
    for name, expected, hard in EXPECTED:
        try:
            got = _ver(name)
        except Exception as e:
            msg = f"  ✗ {name}: not importable ({e})"
            print(msg); errors.append(msg); continue
        ok = got == expected
        flag = "✓" if ok else ("✗" if hard else "!")
        print(f"  {flag} {name}: got={got}  expected={expected}")
        if not ok and hard:
            errors.append(f"{name} version mismatch: {got} != {expected}")
    return errors


def _check_moabb_alias() -> list[str]:
    """The single most common breakage: BNCI2014_001 must be importable."""
    errors: list[str] = []
    try:
        from moabb.datasets import BNCI2014_001  # noqa: F401
        print("  ✓ moabb.datasets.BNCI2014_001 resolves")
    except ImportError as e:
        msg = (
            f"  ✗ moabb.datasets.BNCI2014_001 not importable: {e}\n"
            "      → Your MOABB is too old. Reinstall: pip install -r requirements.txt"
        )
        print(msg); errors.append(msg)
    # Some Braindecode versions still reach for the OLD alias internally.
    # Touch the import path the training scripts actually use.
    try:
        from braindecode.models import EEGConformer  # noqa: F401
        print("  ✓ braindecode.models.EEGConformer resolves")
    except ImportError as e:
        errors.append(f"braindecode import failed: {e}")
    return errors


def _check_python() -> list[str]:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10) or (major, minor) > (3, 12):
        return [f"Python {major}.{minor} is unsupported (need 3.10–3.12)."]
    return []


def main(argv: Iterable[str] | None = None) -> int:
    print("=" * 60)
    print("EEGConformer training package — compatibility check")
    print("=" * 60)
    errors: list[str] = []
    errors += _check_python()
    errors += _check_versions()
    errors += _check_moabb_alias()
    print("=" * 60)
    if errors:
        print("[compat] FAILED:")
        for e in errors:
            print(f"  - {e}")
        print("\nFix: pip install --force-reinstall -r requirements.txt")
        return 1
    print("[compat] OK — environment matches the pinned contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())