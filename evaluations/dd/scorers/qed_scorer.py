"""Scorer for des_02: constrained QED optimization.

Evaluation protocol:
  - Input : lead molecule X (QED in [0.7, 0.8] from ZINC)
  - Output: top-k analog molecules Y recovered from the oracle stream
  - Similarity constraint : Tanimoto(X, Y) >= 0.4  (Morgan fp, radius=2, 2048 bits)
  - Optimization goal    : QED(Y) in [0.9, 1.0]
  - Oracle               : TDC Oracle("QED")

Metrics returned
----------------
  basic_score  : mean of 4 binary format flags
                  (exact_k, all_smiles_valid, no_duplicates, none_equal_lead)
  design_result : dict with constraint subscores
    opt_success       : 1.0 if at least one analog passed both constraints, else 0.0
    opt_success_rate  : fraction of k analogs satisfying BOTH similarity AND QED-range
                        constraints. Range [0.0, 1.0].
    best_oracle_value : QED of the best passing analog, or None if no analog passed.
    best_oracle_delta : best_oracle_value minus lead's QED, or None if no analog passed.
    sim_success_rate  : fraction with Tanimoto >= similarity_threshold.
    qed_success_rate  : fraction with QED(Y) in [target_qed_min, target_qed_max].
"""

from __future__ import annotations

import json

from scorers.design_scores import compute_basic_score
from scorers.drd2_scorer import (
    _extract_smiles_list,
    _get_tdc_oracle,
    _parse_json,
    _safe_float,
)

# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def score_qed_optimization(agent_stdout: str, task: dict) -> dict:
    """Score the constrained QED optimization task.

    Expected ``task["scoring_logic"]`` fields:
      lead_smiles          : str   — updated per lead by adapt_task
      k                    : int   — expected number of analogs (default 10)
      similarity_threshold : float — minimum Tanimoto to lead (default 0.4)
      target_qed_min       : float — lower bound of QED target window (default 0.9)
      target_qed_max       : float — upper bound of QED target window (default 1.0)
      fingerprint_radius   : int   (default 2)
      fingerprint_nbits    : int   (default 2048)
    """

    payload, parse_err = _parse_json(agent_stdout)
    if parse_err:
        return {
            "basic_score": 0.0,
            "design_result": {
                "opt_success": 0.0,
                "opt_success_rate": 0.0,
                "best_oracle_value": None,
                "best_oracle_delta": None,
                "sim_success_rate": 0.0,
                "qed_success_rate": 0.0,
            },
            "details": {"error": f"JSON parse error: {parse_err}"},
        }

    scoring_conf = task.get("scoring_logic", {})

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdMolDescriptors
    except Exception as exc:  # noqa: BLE001
        return {
            "basic_score": 0.0,
            "design_result": {
                "opt_success": 0.0,
                "opt_success_rate": 0.0,
                "best_oracle_value": None,
                "best_oracle_delta": None,
                "sim_success_rate": 0.0,
                "qed_success_rate": 0.0,
            },
            "details": {"error": f"rdkit not available: {exc}"},
        }

    k = int(scoring_conf.get("k", 10))
    lead_smiles_raw = str(scoring_conf.get("lead_smiles", "")).strip()
    sim_threshold = _safe_float(scoring_conf.get("similarity_threshold", 0.4), 0.4)
    target_qed_min = _safe_float(scoring_conf.get("target_qed_min", 0.9), 0.9)
    target_qed_max = _safe_float(scoring_conf.get("target_qed_max", 1.0), 1.0)
    fp_radius = int(scoring_conf.get("fingerprint_radius", 2))
    fp_nbits = int(scoring_conf.get("fingerprint_nbits", 2048))

    lead_mol = Chem.MolFromSmiles(lead_smiles_raw)
    if lead_mol is None:
        return {
            "basic_score": 0.0,
            "design_result": {
                "opt_success": 0.0,
                "opt_success_rate": 0.0,
                "best_oracle_value": None,
                "best_oracle_delta": None,
                "sim_success_rate": 0.0,
                "qed_success_rate": 0.0,
            },
            "details": {"error": f"lead_smiles is invalid: {lead_smiles_raw}"},
        }
    try:
        Chem.SanitizeMol(lead_mol)
    except Exception as exc:  # noqa: BLE001
        return {
            "basic_score": 0.0,
            "design_result": {
                "opt_success": 0.0,
                "opt_success_rate": 0.0,
                "best_oracle_value": None,
                "best_oracle_delta": None,
                "sim_success_rate": 0.0,
                "qed_success_rate": 0.0,
            },
            "details": {"error": f"lead_smiles failed sanitization: {exc}"},
        }
    lead_canonical = Chem.MolToSmiles(lead_mol)
    lead_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        lead_mol, fp_radius, nBits=fp_nbits
    )

    qed_oracle = _get_tdc_oracle("QED")
    lead_qed = float(qed_oracle(lead_canonical))

    analog_smiles: list[str] = []
    if isinstance(payload, dict):
        analog_smiles = _extract_smiles_list(payload.get("analogs", []))
    elif isinstance(payload, list):
        analog_smiles = _extract_smiles_list(payload)

    n_valid = 0
    n_similar = 0  # Tanimoto >= sim_threshold
    n_in_qed = 0  # QED in [target_qed_min, target_qed_max]
    n_success = 0  # both similar AND in qed window

    best_qed: float | None = None
    best_qed_delta: float | None = None
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
            except Exception as exc:  # noqa: BLE001
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
                    "tanimoto": None,
                    "similar_ok": False,
                    "qed": None,
                    "qed_delta": None,
                    "qed_in_window": False,
                    "success": False,
                    "sanitize_error": sanitize_err,
                }
            )
            continue

        n_valid += 1
        canonical = Chem.MolToSmiles(mol)

        is_duplicate = canonical in seen_canonical
        seen_canonical.add(canonical)
        equals_lead = canonical == lead_canonical

        format_ok = not is_duplicate and not equals_lead

        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, fp_radius, nBits=fp_nbits
        )
        tanimoto = float(DataStructs.TanimotoSimilarity(lead_fp, fp))
        similar_ok = tanimoto >= sim_threshold
        if similar_ok:
            n_similar += 1

        qed_val = float(qed_oracle(canonical))
        qed_in_window = target_qed_min <= qed_val <= target_qed_max
        if qed_in_window:
            n_in_qed += 1

        success = similar_ok and qed_in_window and not equals_lead
        if success:
            n_success += 1
            if best_qed is None or qed_val > best_qed:
                best_qed = qed_val
                best_qed_delta = qed_val - lead_qed
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
                "tanimoto": round(tanimoto, 4),
                "similar_ok": similar_ok,
                "qed": round(qed_val, 4),
                "qed_delta": round(qed_val - lead_qed, 4),
                "qed_in_window": qed_in_window,
                "success": success,
            }
        )

    # Pad details for missing analogs (fewer than k submitted)
    for i in range(len(analog_smiles[:k]), k):
        analog_details.append(
            {
                "rank": i + 1,
                "smiles_raw": None,
                "is_valid": False,
                "is_duplicate": False,
                "equals_lead": False,
                "format_ok": False,
                "tanimoto": None,
                "similar_ok": False,
                "qed": None,
                "qed_delta": None,
                "qed_in_window": False,
                "success": False,
            }
        )

    n_submitted = len(analog_smiles)
    basic_score, design_subscores = compute_basic_score(
        n_submitted,
        n_valid,
        analog_details,
        k,
    )
    design_result = {
        "opt_success": 1.0 if n_success > 0 else 0.0,
        "opt_success_rate": round(n_success / k, 6),
        "best_oracle_value": round(best_qed, 4) if best_qed is not None else None,
        "best_oracle_delta": (
            round(best_qed_delta, 4) if best_qed_delta is not None else None
        ),
        "sim_success_rate": round(n_similar / k, 6),
        "qed_success_rate": round(n_in_qed / k, 6),
    }

    return {
        "basic_score": basic_score,
        "design_result": design_result,
        "details": {
            "lead_smiles": lead_smiles_raw,
            "lead_canonical": lead_canonical,
            "lead_qed": round(lead_qed, 4),
            "similarity_threshold": sim_threshold,
            "target_qed_min": target_qed_min,
            "target_qed_max": target_qed_max,
            "k": k,
            "n_submitted": n_submitted,
            "n_valid": n_valid,
            "n_similar": n_similar,
            "n_in_qed": n_in_qed,
            "n_success": n_success,
            "design_subscores": design_subscores,
            "oracle_success": n_success > 0,
            "best_qed": round(best_qed, 4) if best_qed is not None else None,
            "best_smiles": best_smiles,
            "analog_details": analog_details,
        },
    }
