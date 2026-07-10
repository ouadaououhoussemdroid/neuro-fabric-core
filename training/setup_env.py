#!/usr/bin/env python3
"""
Validate Python environment setup for Neuro-Fabric training pipeline.
Run after: pip install -r training/requirements.txt
"""

import sys
import importlib
from pathlib import Path

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def check_python_version():
    """Verify Python 3.11+"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 11:
        print(f"{GREEN}✅{RESET} Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"{RED}❌{RESET} Python 3.11+ required (you have {version.major}.{version.minor})")
        return False

def check_module(name: str, import_name: str = None) -> tuple[bool, str]:
    """Check if module is installed and get version."""
    import_name = import_name or name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, None

def main():
    print(f"{YELLOW}Verifying Neuro-Fabric Python Environment{RESET}")
    print("=" * 50)
    print()

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    print()

    # Required packages
    required = [
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("braindecode", "braindecode"),
        ("moabb", "moabb"),
        ("mne", "mne"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("pandas", "pandas"),
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
        ("pyyaml", "yaml"),
    ]

    print(f"{YELLOW}Checking required packages:{RESET}")
    all_ok = True

    for display_name, import_name in required:
        found, version = check_module(import_name)
        if found:
            print(f"  {GREEN}✅{RESET} {display_name:20s} {version}")
        else:
            print(f"  {RED}❌{RESET} {display_name:20s} NOT INSTALLED")
            all_ok = False

    print()

    if all_ok:
        print(f"{GREEN}═════════════════════════════════════════{RESET}")
        print(f"{GREEN}✅ All dependencies verified!{RESET}")
        print(f"{GREEN}═════════════════════════════════════════{RESET}")
        print()
        print("Ready to start training. Next steps:")
        print(f"  1. {YELLOW}cd training{RESET}")
        print(f"  2. {YELLOW}bash scripts/run_all.sh{RESET} (or run individual scripts)")
        print()
        return 0
    else:
        print(f"{RED}═════════════════════════════════════════{RESET}")
        print(f"{RED}❌ Some dependencies are missing!{RESET}")
        print(f"{RED}═════════════════════════════════════════{RESET}")
        print()
        print("Fix with:")
        print(f"  {YELLOW}pip install --upgrade pip{RESET}")
        print(f"  {YELLOW}pip install -r training/requirements.txt --no-cache-dir{RESET}")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
