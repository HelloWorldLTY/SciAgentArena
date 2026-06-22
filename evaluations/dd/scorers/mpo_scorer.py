"""Scorer for des_06: Osimertinib MPO (Multi-Property Objective) optimization.

Evaluation protocol:
  - Input : Osimertinib (reference molecule)
  - Output: k analog molecules Y
  - Optimization goal    : Osimertinib_MPO(Y) >= target_mpo_min (default 0.5)
  - Oracle               : TDC Oracle("Osimertinib_MPO")

  The Osimertinib MPO oracle (from GuacaMol) computes the geometric mean of
  four desirability functions:
    1. Structural Similarity 1 — FCFP4 Tanimoto; ClippedScoreModifier(upper_x=0.8)
       (score = clip(sim / 0.8, 0, 1); saturates at sim >= 0.8)
    2. Structural Similarity 2 — ECFP6 Tanimoto; MinGaussian(mu=0.85, sigma=0.1)
       (score = 1.0 for sim <= 0.85, Gaussian decay above 0.85 — penalizes
       molecules too similar to Osimertinib; favors near-but-not-identical analogs)
    3. TPSA — MaxGaussian(mu=100, sigma=10)
       (score = 1.0 for TPSA >= 100, Gaussian decay below 100)
    4. logP — MinGaussian(mu=1, sigma=1)
       (score = 1.0 for logP <= 1, Gaussian decay above 1 — strongly penalizes
       logP > 2; the dominant bottleneck for Osimertinib-like molecules)

Metrics returned
----------------
  basic_score  : mean of 4 binary format flags
                  (exact_k, all_smiles_valid, no_duplicates, none_equal_reference)
  design_result : dict with constraint subscores
    opt_success               : 1.0 if at least one analog passes, else 0.0
    opt_success_rate          : fraction of k analogs with MPO >= target.
                                Range [0.0, 1.0].
    best_oracle_value         : highest MPO score of a passing analog, or None.
    best_oracle_delta         : best_oracle_value minus reference MPO, or None.
    mpo_success_rate          : fraction with MPO >= target_mpo_min.
"""

from __future__ import annotations

import json

from scorers.design_scores import compute_basic_score
from scorers.penalized_logp_scorer import (
    _extract_smiles_list,
    _parse_json,
    _safe_float,
    canonicalize_smiles,
)

# TDC Oracle cache
_oracle_cache: dict[str, object] = {}


def _get_tdc_oracle(name: str):
    if name not in _oracle_cache:
        from tdc import Oracle

        _oracle_cache[name] = Oracle(name=name)
    return _oracle_cache[name]


# ---------------------------------------------------------------------------
# Per-candidate helper for multi-round evaluation (des_07)
# ---------------------------------------------------------------------------


def evaluate_mpo_candidate(
    smiles: str,
    scoring_conf: dict,
    seen_canonical: set | None = None,
) -> dict:
    """Per-candidate evaluation for the iterative des_07 Osimertinib MPO loop."""

    raw = str(smiles).strip() if smiles is not None else ""
    result: dict = {
        "submitted_smiles": raw or None,
        "canonical_smiles": None,
        "valid": False,
        "duplicate": False,
        "oracle_score": None,
        "final_score": None,
        "improved_ok": False,
        "success": False,
        "sanitize_error": None,
    }

    canonical, err = canonicalize_smiles(raw)
    if canonical is None:
        result["sanitize_error"] = err
        return result

    try:
        from rdkit import Chem
    except Exception as exc:
        raise RuntimeError(f"rdkit not available: {exc}") from exc

    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        result["sanitize_error"] = "rdkit could not parse canonical smiles"
        return result

    is_duplicate = seen_canonical is not None and canonical in seen_canonical

    target_mpo_min = _safe_float(scoring_conf.get("target_mpo_min", 0.5), 0.5)
    mpo_oracle = _get_tdc_oracle("Osimertinib_MPO")
    mpo_val = float(mpo_oracle(canonical))
    improved_ok = mpo_val >= target_mpo_min
    success = improved_ok

    result.update(
        {
            "canonical_smiles": canonical,
            "valid": True,
            "duplicate": is_duplicate,
            "oracle_score": mpo_val,
            "final_score": mpo_val,
            "improved_ok": improved_ok,
            "success": success,
        }
    )
    return result


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def score_mpo_optimization(agent_stdout: str, task: dict) -> dict:
    """Score the Osimertinib MPO optimization task.

    Expected ``task["scoring_logic"]`` fields:
      k                : int   — expected number of analogs (default 100)
      reference_smiles : str   — Osimertinib SMILES for none_equal_reference check
      target_mpo_min   : float — minimum MPO score for success (default 0.5)
    """
    payload, parse_err = _parse_json(agent_stdout)
    _empty_result = {
        "opt_success": 0.0,
        "opt_success_rate": 0.0,
        "best_oracle_value": None,
        "best_oracle_delta": None,
        "mpo_success_rate": 0.0,
    }
    if parse_err:
        return {
            "basic_score": 0.0,
            "design_result": _empty_result,
            "details": {"error": f"JSON parse error: {parse_err}"},
        }

    scoring_conf = task.get("scoring_logic", {})

    try:
        from rdkit import Chem
    except Exception as exc:
        return {
            "basic_score": 0.0,
            "design_result": _empty_result,
            "details": {"error": f"rdkit not available: {exc}"},
        }

    k = int(scoring_conf.get("k", 100))
    reference_smiles_raw = str(scoring_conf.get("reference_smiles", "")).strip()
    target_mpo_min = _safe_float(scoring_conf.get("target_mpo_min", 0.5), 0.5)

    # Canonicalise reference for equals-check
    reference_canonical: str | None = None
    if reference_smiles_raw:
        ref_mol = Chem.MolFromSmiles(reference_smiles_raw)
        if ref_mol is not None:
            reference_canonical = Chem.MolToSmiles(ref_mol)

    mpo_oracle = _get_tdc_oracle("Osimertinib_MPO")
    reference_mpo = (
        float(mpo_oracle(reference_canonical)) if reference_canonical else 0.0
    )

    analog_smiles: list[str] = []
    if isinstance(payload, dict):
        analog_smiles = _extract_smiles_list(
            payload.get("analogs", payload.get("molecules", []))
        )
    elif isinstance(payload, list):
        analog_smiles = _extract_smiles_list(payload)

    n_valid = 0
    n_mpo_pass = 0
    n_success = 0

    best_mpo: float | None = None
    best_delta: float | None = None
    best_smiles: str | None = None

    seen_canonical: set[str] = set()
    analog_details: list[dict] = []

    for i, raw_smi in enumerate(analog_smiles[:k]):
        mol = Chem.MolFromSmiles(raw_smi)
        is_valid = mol is not None
        sanitize_err = None
        if is_valid:
            try:
                Chem.SanitizeMol(mol)
            except Exception as exc:
                is_valid = False
                sanitize_err = str(exc)

        if not is_valid:
            analog_details.append(
                {
                    "rank": i + 1,
                    "smiles_raw": raw_smi,
                    "is_valid": False,
                    "is_duplicate": False,
                    "equals_lead": False,
                    "format_ok": False,
                    "mpo_score": None,
                    "mpo_delta": None,
                    "improved_ok": False,
                    "success": False,
                    "sanitize_error": sanitize_err,
                }
            )
            continue

        n_valid += 1
        canonical = Chem.MolToSmiles(mol)

        is_duplicate = canonical in seen_canonical
        seen_canonical.add(canonical)
        equals_lead = (
            reference_canonical is not None and canonical == reference_canonical
        )

        format_ok = not is_duplicate and not equals_lead

        mpo_val = float(mpo_oracle(canonical))
        improved_ok = mpo_val >= target_mpo_min
        if improved_ok:
            n_mpo_pass += 1

        success = improved_ok
        if success:
            n_success += 1
            delta = mpo_val - reference_mpo
            if best_mpo is None or mpo_val > best_mpo:
                best_mpo = mpo_val
                best_delta = delta
                best_smiles = canonical

        analog_details.append(
            {
                "rank": i + 1,
                "smiles_raw": raw_smi,
                "smiles_canonical": canonical,
                "is_valid": True,
                "is_duplicate": is_duplicate,
                "equals_lead": equals_lead,
                "format_ok": format_ok,
                "mpo_score": round(mpo_val, 4),
                "mpo_delta": round(mpo_val - reference_mpo, 4),
                "improved_ok": improved_ok,
                "success": success,
            }
        )

    for i in range(len(analog_smiles[:k]), k):
        analog_details.append(
            {
                "rank": i + 1,
                "smiles_raw": None,
                "is_valid": False,
                "is_duplicate": False,
                "equals_lead": False,
                "format_ok": False,
                "mpo_score": None,
                "mpo_delta": None,
                "improved_ok": False,
                "success": False,
            }
        )

    n_submitted = len(analog_smiles)

    basic_score, design_subscores = compute_basic_score(
        n_submitted, n_valid, analog_details, k
    )
    design_result = {
        "opt_success": 1.0 if n_success > 0 else 0.0,
        "opt_success_rate": round(n_success / k, 6),
        "best_oracle_value": round(best_mpo, 4) if best_mpo is not None else None,
        "best_oracle_delta": round(best_delta, 4) if best_delta is not None else None,
        "mpo_success_rate": round(n_mpo_pass / k, 6),
    }

    return {
        "basic_score": basic_score,
        "design_result": design_result,
        "details": {
            "reference_smiles": reference_smiles_raw,
            "reference_canonical": reference_canonical,
            "reference_mpo": round(reference_mpo, 4),
            "target_mpo_min": target_mpo_min,
            "k": k,
            "n_submitted": n_submitted,
            "n_valid": n_valid,
            "n_mpo_pass": n_mpo_pass,
            "n_success": n_success,
            "design_subscores": design_subscores,
            "oracle_success": n_success > 0,
            "best_smiles": best_smiles,
            "analog_details": analog_details,
        },
    }
