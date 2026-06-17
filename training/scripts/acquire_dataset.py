"""Download BCI Competition IV 2a via MOABB.

Caches under training/cache/moabb. Safe to re-run (MOABB no-ops if files
are present).
"""
from __future__ import annotations

import argparse
import os

from _common import Paths, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)

    os.environ.setdefault("MNE_DATA", str(paths.cache))
    os.environ.setdefault("MOABB_DATA", str(paths.cache))

    import moabb
    from moabb.datasets import BNCI2014_001

    moabb.set_log_level("WARNING")
    ds = BNCI2014_001()
    subjects = cfg["dataset"]["subjects"]
    print(f"[acquire] downloading BCI-IV-2a subjects={subjects} → {paths.cache}")
    ds.get_data(subjects=subjects)
    print("[acquire] done.")


if __name__ == "__main__":
    main()