"""Shared design-score computation for analog-design tasks.

``compute_basic_score`` implements the 4 binary format flags used across all
des_* scorers (exact_k, all_smiles_valid, no_duplicates, none_equal_lead).

It now also returns ratio-based subscores for finer-grained evaluation.
"""

from __future__ import annotations


def compute_basic_score(
    n_submitted: int,
    n_valid: int,
    analog_details: list[dict],
    k: int,
) -> tuple[float, dict]:
    """Return ``(basic_score, design_subscores)`` from analog metadata.

    Parameters
    ----------
    n_submitted:
        Number of analog SMILES strings the agent produced.
    n_valid:
        Number of those that parsed as valid RDKit molecules.
    analog_details:
        Per-analog detail dicts, each containing ``is_duplicate`` and
        ``equals_lead`` boolean fields.
    k:
        Expected number of analogs.

    Returns
    -------
    basic_score : float
        Mean of the four binary subscores, rounded to 4 decimal places.
    design_subscores : dict
        Contains both binary (0/1) and ratio (0.0-1.0) subscores.
    """
    n_duplicates = sum(1 for d in analog_details if d.get("is_duplicate"))
    n_equals_lead = sum(1 for d in analog_details if d.get("equals_lead"))
    n_unique = k - n_duplicates
    n_not_lead = k - n_equals_lead

    # Binary subscores (original Allen logic — all-or-nothing)
    exact_k = n_submitted == k
    all_smiles_valid = n_valid == k
    no_duplicates = n_unique == k
    none_equal_lead = n_not_lead == k

    binary_subscores = {
        "exact_k": 1.0 if exact_k else 0.0,
        "all_smiles_valid": 1.0 if all_smiles_valid else 0.0,
        "no_duplicates": 1.0 if no_duplicates else 0.0,
        "none_equal_lead": 1.0 if none_equal_lead else 0.0,
    }
    binary_basic_score = round(sum(binary_subscores.values()) / 4, 4)

    # Ratio subscores (proportional — more granular)
    denom = max(k, 1)
    ratio_subscores = {
        "submit_rate": round(min(n_submitted, k) / denom, 4),
        "valid_rate": round(n_valid / denom, 4),
        "unique_rate": round(n_unique / denom, 4),
        "not_lead_rate": round(n_not_lead / denom, 4),
    }
    ratio_basic_score = round(sum(ratio_subscores.values()) / 4, 4)

    design_subscores = {
        # Binary (backward compatible)
        "exact_k": binary_subscores["exact_k"],
        "all_smiles_valid": binary_subscores["all_smiles_valid"],
        "no_duplicates": binary_subscores["no_duplicates"],
        "none_equal_lead": binary_subscores["none_equal_lead"],
        # Ratio (new)
        "submit_rate": ratio_subscores["submit_rate"],
        "valid_rate": ratio_subscores["valid_rate"],
        "unique_rate": ratio_subscores["unique_rate"],
        "not_lead_rate": ratio_subscores["not_lead_rate"],
        # Aggregates
        "binary_basic_score": binary_basic_score,
        "ratio_basic_score": ratio_basic_score,
    }

    # basic_score uses ratio version for better granularity
    return ratio_basic_score, design_subscores
