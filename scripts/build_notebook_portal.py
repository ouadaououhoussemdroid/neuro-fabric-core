"""T-026 — Notebook portal build step.

Converts Jupyter notebooks in training/notebooks/ to standalone HTML files
and writes them to public/research/notebooks/ so they can be served
statically under /research/notebooks/:id.

Usage:
    python scripts/build_notebook_portal.py

Requirements:
    pip install nbconvert jupyterlab
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "training" / "notebooks"
OUTPUT_DIR = REPO_ROOT / "public" / "research" / "notebooks"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Try nbconvert; fall back to a simple manifest if it's not installed.
    try:
        from nbconvert import HTMLExporter
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError:
        print("[portal] nbconvert not installed — generating manifest only")
        _generate_manifest()
        return

    exporter = HTMLExporter(template_name="classic")
    # Execute notebooks before conversion so outputs are present.
    exporter.register_preprocessor(ExecutePreprocessor(timeout=600), enabled=True)

    converted = []
    for nb_path in NOTEBOOKS_DIR.glob("*.ipynb"):
        try:
            html, _ = exporter.from_filename(str(nb_path))
            out_path = OUTPUT_DIR / f"{nb_path.stem}.html"
            out_path.write_text(html, encoding="utf-8")
            converted.append(nb_path.stem)
            print(f"[portal] converted {nb_path.name} → {out_path}")
        except Exception as e:
            print(f"[portal] FAILED {nb_path.name}: {e}")

    _generate_manifest(converted)
    print(f"[portal] {len(converted)} notebooks → {OUTPUT_DIR}")


def _generate_manifest(notebook_ids: list[str] | None = None) -> None:
    """Write a JSON manifest listing available notebooks."""
    if notebook_ids is None:
        notebook_ids = [p.stem for p in NOTEBOOKS_DIR.glob("*.ipynb")]
    manifest = {
        "notebooks": [
            {"id": nid, "title": nid.replace("_", " ").title(), "url": f"/research/notebooks/{nid}.html"}
            for nid in notebook_ids
        ]
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[portal] manifest → {manifest_path}")


if __name__ == "__main__":
    main()
