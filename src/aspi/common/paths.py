"""Repo-root-anchored path helpers.

Finds the repo root by walking up from this file until `pyproject.toml`
is found. All data/results default paths are anchored here so that the
same command produces the same output regardless of which directory it
was run from.
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root directory.

    Walks upward from this file until a directory containing
    ``pyproject.toml`` is found. Raises RuntimeError if none is found.
    """
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    raise RuntimeError(
        "Could not find repo root (no pyproject.toml found by walking up "
        f"from {Path(__file__).resolve()})"
    )


# Default locations for runtime artifacts. Every CLI accepts overrides,
# but these are the baked-in defaults and are repo-root-anchored.
_ROOT = repo_root()

DATA_DIR = _ROOT / "data"
RESULTS_DIR = _ROOT / "results"
EVAL_RESULTS_DIR = RESULTS_DIR / "eval"
JUDGED_RESULTS_DIR = RESULTS_DIR / "judged"
PLOTS_DIR = RESULTS_DIR / "plots"
ERRORS_DIR = RESULTS_DIR / "errors"


def ensure_dirs(*dirs: Path) -> None:
    """Make sure the given directories exist (no-op if already present)."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
