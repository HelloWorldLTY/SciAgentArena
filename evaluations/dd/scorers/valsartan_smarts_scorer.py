"""Scorer for des_04: Valsartan SMARTS goal-directed generation.

Evaluation protocol:
  - Input : no lead molecule — pure goal-directed generation.
  - Output: k analog molecules Y from the oracle stream.
  - Structural goal : contain the Valsartan amide-biphenyl SMARTS motif.
  - Property goal   : match Sitagliptin's logP / TPSA / Bertz profile.
  - Oracle          : TDC Oracle("Valsartan_SMARTS")
                      geometric mean of 4 components; returns 0.0 exactly
                      whenever the SMARTS motif is absent.

The design mirrors ``mpo_scorer.score_mpo_optimization``:
  - ``score_valsartan_smarts_optimization`` parses a synthesized
    ``{"analogs": [...]}`` JSON string supplied by ``batch_runner``,
    scores each candidate with the TDC oracle, and aggregates.
  - ``evaluate_valsartan_candidate`` is retained for any iterative
    single-candidate callers (kept API-compatible with the archived
    legacy scorer).
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

# Benchmark SMARTS — Valsartan amide-biphenyl motif used by GuacaMol.
VALSARTAN_SMARTS = "CN(C=O)Cc1ccc(c2ccccc2)cc1"


# TDC Oracle cache (lazy to avoid importing TDC at module load).
_oracle_cache: dict[str, object] = {}


def _get_tdc_oracle(name: str):
    if name not in _oracle_cache:
        from tdc import Oracle

        _oracle_cache[name] = Oracle(name=name)
    return _oracle_cache[name]


# ---------------------------------------------------------------------------
# Per-candidate helper (iterative callers)
# ---------------------------------------------------------------------------


def evaluate_valsartan_candidate(
    smiles: str,
    scoring_conf: dict,
    seen_canonical: set | None = None,
) -> dict:
    """Per-candidate evaluation for the Valsartan SMARTS task."""
    raw = str(smiles).strip() if smiles is not None else ""
    result: dict = {
        "submitted_smiles": raw or None,
        "canonical_smiles": None,
        "valid": False,
        "duplicate": False,
        "oracle_score": None,
        "smarts_ok": False,
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

    smarts_pattern = str(scoring_conf.get("smarts_pattern", VALSARTAN_SMARTS))
    smarts_query = Chem.MolFromSmarts(smarts_pattern)
    smarts_ok = smarts_query is not None and mol.HasSubstructMatch(smarts_query)

    target_score_min = _safe_float(scoring_conf.get("target_score_min", 0.5), 0.5)
    oracle = _get_tdc_oracle("Valsartan_SMARTS")
    oracle_score = float(oracle(canonical))
    success = oracle_score >= target_score_min

    result.update(
        {
            "canonical_smiles": canonical,
            "valid": True,
            "duplicate": is_duplicate,
            "oracle_score": oracle_score,
            "smarts_ok": smarts_ok,
            "success": success,
        }
    )
    return result


# ---------------------------------------------------------------------------
# Scorer entry point used by the code-optimizer harness
# ---------------------------------------------------------------------------


def score_valsartan_smarts_optimization(agent_stdout: str, task: dict) -> dict:
    """Score the Valsartan_SMARTS code-optimizer task.

    Expected ``task["scoring_logic"]`` fields:
      k                : int   — expected number of analogs (default 10)
      reference_smiles : str   — Valsartan SMILES for the equals-reference check
      smarts_pattern   : str   — required SMARTS motif (default VALSARTAN_SMARTS)
      target_score_min : float — minimum oracle score for success (default 0.5)
    """
    payload, parse_err = _parse_json(agent_stdout)
    _empty_result = {
        "opt_success": 0.0,
        "opt_success_rate": 0.0,
        "best_oracle_value": None,
        "best_oracle_delta": None,
        "smarts_success_rate": 0.0,
        "score_success_rate": 0.0,
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

    k = int(scoring_conf.get("k", 10))
    reference_smiles_raw = str(scoring_conf.get("reference_smiles", "")).strip()
    smarts_pattern = str(scoring_conf.get("smarts_pattern", VALSARTAN_SMARTS))
    target_score_min = _safe_float(scoring_conf.get("target_score_min", 0.5), 0.5)

    reference_canonical: str | None = None
    if reference_smiles_raw:
        ref_mol = Chem.MolFromSmiles(reference_smiles_raw)
        if ref_mol is not None:
            reference_canonical = Chem.MolToSmiles(ref_mol)

    smarts_query = Chem.MolFromSmarts(smarts_pattern)
    if smarts_query is None:
        return {
            "basic_score": 0.0,
            "design_result": _empty_result,
            "details": {"error": f"invalid smarts_pattern: {smarts_pattern}"},
        }

    valsartan_oracle = _get_tdc_oracle("Valsartan_SMARTS")
    reference_score = (
        float(valsartan_oracle(reference_canonical)) if reference_canonical else 0.0
    )

    analog_smiles: list[str] = []
    if isinstance(payload, dict):
        analog_smiles = _extract_smiles_list(
            payload.get("analogs", payload.get("molecules", []))
        )
    elif isinstance(payload, list):
        analog_smiles = _extract_smiles_list(payload)

    n_valid = 0
    n_smarts_match = 0
    n_score_pass = 0
    n_success = 0

    best_score: float | None = None
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
                    "equals_reference": False,
                    "format_ok": False,
                    "smarts_match": False,
                    "oracle_score": None,
                    "score_above_threshold": False,
                    "success": False,
                    "sanitize_error": sanitize_err,
                }
            )
            continue

        n_valid += 1
        canonical = Chem.MolToSmiles(mol)

        is_duplicate = canonical in seen_canonical
        seen_canonical.add(canonical)
        equals_reference = (
            reference_canonical is not None and canonical == reference_canonical
        )

        format_ok = not is_duplicate and not equals_reference

        smarts_match = len(mol.GetSubstructMatches(smarts_query)) > 0
        if smarts_match:
            n_smarts_match += 1

        oracle_score = float(valsartan_oracle(canonical))
        above_threshold = oracle_score >= target_score_min
        if above_threshold:
            n_score_pass += 1

        success = above_threshold and smarts_match and not equals_reference
        if success:
            n_success += 1
            if best_score is None or oracle_score > best_score:
                best_score = oracle_score
                best_smiles = canonical

        analog_details.append(
            {
                "rank": i + 1,
                "smiles_raw": raw_smi,
                "smiles_canonical": canonical,
                "is_valid": True,
                "is_duplicate": is_duplicate,
                "equals_reference": equals_reference,
                "format_ok": format_ok,
                "smarts_match": smarts_match,
                "oracle_score": round(oracle_score, 6),
                "score_above_threshold": above_threshold,
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
                "equals_reference": False,
                "format_ok": False,
                "smarts_match": False,
                "oracle_score": None,
                "score_above_threshold": False,
                "success": False,
            }
        )

    n_submitted = len(analog_smiles)
    basic_score, design_subscores = compute_basic_score(
        n_submitted, n_valid, analog_details, k
    )

    best_delta = (
        round(best_score - reference_score, 6) if best_score is not None else None
    )

    design_result = {
        "opt_success": 1.0 if n_success > 0 else 0.0,
        "opt_success_rate": round(n_success / k, 6),
        "best_oracle_value": round(best_score, 6) if best_score is not None else None,
        "best_oracle_delta": best_delta,
        "smarts_success_rate": round(n_smarts_match / k, 6),
        "score_success_rate": round(n_score_pass / k, 6),
    }

    return {
        "basic_score": basic_score,
        "design_result": design_result,
        "details": {
            "reference_smiles": reference_smiles_raw,
            "reference_canonical": reference_canonical,
            "reference_oracle_score": round(reference_score, 6),
            "smarts_pattern": smarts_pattern,
            "target_score_min": target_score_min,
            "k": k,
            "n_submitted": n_submitted,
            "n_valid": n_valid,
            "n_smarts_match": n_smarts_match,
            "n_score_pass": n_score_pass,
            "n_success": n_success,
            "design_subscores": design_subscores,
            "oracle_success": n_success > 0,
            "best_smiles": best_smiles,
            "analog_details": analog_details,
        },
    }
