"""Helpers for reading C4_user variants from per-group result dicts.

Was previously duplicated as ``_get_c4_user_variants`` in ``aspi_metrics.py``
and ``_get_c4v`` in ``analyze_results.py``. Single source of truth now.
"""

from __future__ import annotations


def get_c4_user_variants(result: dict) -> list[dict]:
    """Return the C4_user variant list for a group result dict.

    Handles both the current schema (``c4_user_variants``) and the
    legacy pre-C2_user schema (``c4_variants``). Returns an empty list
    if neither key is present.
    """
    if "c4_user_variants" in result:
        return result["c4_user_variants"]
    if "c4_variants" in result:
        return result["c4_variants"]
    return []
