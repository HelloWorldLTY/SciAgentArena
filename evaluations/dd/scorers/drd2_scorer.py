"""Scorer for des_03: constrained DRD2 activity optimization.

Evaluation protocol:
  - Input : lead molecule X (DRD2 < 0.05 — inactive, from ZINC)
  - Output: 50 analog molecules Y
  - Similarity constraint : Tanimoto(X, Y) >= 0.4  (Morgan fp, radius=2, 2048 bits)
  - Optimization goal    : DRD2(Y) >= 0.5  (active)
  - Oracle               : TDC Oracle("DRD2")

Metrics returned
----------------
  basic_score  : mean of 4 binary format flags
                  (exact_k, all_smiles_valid, no_duplicates, none_equal_lead)
  design_result : dict with constraint subscores
    opt_success_rate : fraction of k analogs satisfying BOTH similarity AND DRD2
                       activity constraints. Range [0.0, 1.0].
    best_oracle_value: DRD2 of the best passing analog, or None if no analog passed.
    best_oracle_delta: best_oracle_value minus lead's DRD2, or None if no analog passed.
    sim_success_rate : fraction with Tanimoto >= similarity_threshold.
    drd2_success_rate: fraction with DRD2(Y) >= target_drd2_min.

Details also include:
  oracle_success : bool — True if at least one analog passed both constraints
  lead_drd2      : DRD2 score of the lead molecule (recomputed from SMILES)
  best_smiles    : SMILES of the best passing analog
  n_success      : number of analogs satisfying both constraints
  n_active       : number of analogs with DRD2 >= target_drd2_min
"""

from __future__ import annotations

import json

from scorers.design_scores import compute_basic_score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json(agent_stdout: str) -> tuple[dict | list | None, str | None]:
    try:
        payload = json.loads(agent_stdout.strip())
        return payload, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float(default)


def _extract_smiles_list(items) -> list[str]:
    out: list[str] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, str):
            smi = item.strip()
            if smi:
                out.append(smi)
            continue
        if not isinstance(item, dict):
            continue
        for key in ("smiles", "canonical_smiles", "mol_smiles"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                out.append(value.strip())
                break
    return out


# TDC oracle cache — lazy initialisation to avoid loading models at import
_oracle_cache: dict[str, object] = {}


def _get_tdc_oracle(name: str):
    if name not in _oracle_cache:
        from tdc import Oracle

        _oracle_cache[name] = Oracle(name=name)
    return _oracle_cache[name]


# ---------------------------------------------------------------------------
# Per-candidate helpers for multi-round evaluation (des_03)
# ---------------------------------------------------------------------------


def build_drd2_context(
    lead_smiles: str,
    similarity_threshold: float = 0.4,
    fingerprint_radius: int = 2,
    fingerprint_nbits: int = 2048,
    target_drd2_min: float = 0.5,
) -> dict:
    """Build reusable lead context for the iterative des_03 DRD2 loop."""
    from scorers.penalized_logp_scorer import canonicalize_smiles

    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"rdkit not available: {exc}") from exc

    lead_canonical, err = canonicalize_smiles(lead_smiles)
    if lead_canonical is None:
        raise ValueError(f"lead_smiles is invalid: {err}")
    lead_mol = Chem.MolFromSmiles(lead_canonical)
    if lead_mol is None:
        raise ValueError(f"lead_smiles is invalid: {lead_smiles}")
    lead_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        lead_mol,
        fingerprint_radius,
        nBits=fingerprint_nbits,
    )
    drd2_oracle = _get_tdc_oracle("DRD2")
    lead_drd2 = float(drd2_oracle(lead_canonical))
    return {
        "lead_smiles": lead_smiles,
        "lead_canonical": lead_canonical,
        "lead_drd2": lead_drd2,
        "lead_oracle_value": lead_drd2,
        "oracle_name": "DRD2",
        "target_min": float(target_drd2_min),
        "similarity_threshold": float(similarity_threshold),
        "fingerprint_radius": int(fingerprint_radius),
        "fingerprint_nbits": int(fingerprint_nbits),
        "lead_fp": lead_fp,
    }


def evaluate_drd2_candidate(
    smiles: str,
    context: dict,
    *,
    is_duplicate: bool = False,
) -> dict:
    """Return exact per-candidate feedback for the iterative des_03 DRD2 loop."""
    from scorers.penalized_logp_scorer import canonicalize_smiles

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdMolDescriptors
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"rdkit not available: {exc}") from exc

    raw = str(smiles).strip() if smiles is not None else ""
    result: dict = {
        "submitted_smiles": raw or None,
        "canonical_smiles": None,
        "valid": False,
        "duplicate": bool(is_duplicate),
        "equals_lead": False,
        "format_ok": False,
        "tanimoto": None,
        "similar_ok": False,
        "oracle_score": None,
        "oracle_delta": None,
        "improved_ok": False,
        "success": False,
        "sanitize_error": None,
    }

    canonical, err = canonicalize_smiles(raw)
    if canonical is None:
        result["sanitize_error"] = err
        return result
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        result["sanitize_error"] = "rdkit could not parse canonical smiles"
        return result

    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol,
        int(context["fingerprint_radius"]),
        nBits=int(context["fingerprint_nbits"]),
    )
    tanimoto = float(DataStructs.TanimotoSimilarity(context["lead_fp"], fp))
    drd2_oracle = _get_tdc_oracle("DRD2")
    drd2_val = float(drd2_oracle(canonical))
    delta = drd2_val - float(context["lead_drd2"])
    equals_lead = canonical == context["lead_canonical"]
    similar_ok = tanimoto >= float(context["similarity_threshold"])
    target_min = float(context.get("target_min", 0.5))
    improved_ok = drd2_val >= target_min
    success = similar_ok and improved_ok

    result.update(
        {
            "canonical_smiles": canonical,
            "valid": True,
            "equals_lead": equals_lead,
            "format_ok": (not bool(is_duplicate)) and (not equals_lead),
            "tanimoto": tanimoto,
            "similar_ok": similar_ok,
            "oracle_score": drd2_val,
            "oracle_delta": delta,
            "improved_ok": improved_ok,
            "success": success,
        }
    )
    return result


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def score_drd2_optimization(agent_stdout: str, task: dict) -> dict:
    """Score the constrained DRD2 activity optimization task.

    Expected ``task["scoring_logic"]`` fields:
      lead_smiles          : str   — updated per lead by adapt_task
      k                    : int   — expected number of analogs (default 50)
      similarity_threshold : float — minimum Tanimoto to lead (default 0.4)
      target_drd2_min      : float — minimum DRD2 for success (default 0.5)
      fingerprint_radius   : int   (default 2)
      fingerprint_nbits    : int   (default 2048)
    """

    # --- Parse JSON output --------------------------------------------------
    payload, parse_err = _parse_json(agent_stdout)
    if parse_err:
        return {
            "basic_score": 0.0,
            "design_result": {
                "opt_success_rate": 0.0,
                "best_oracle_value": None,
                "best_oracle_delta": None,
                "sim_success_rate": 0.0,
                "drd2_success_rate": 0.0,
            },
            "details": {"error": f"JSON parse error: {parse_err}"},
        }

    scoring_conf = task.get("scoring_logic", {})

    # --- RDKit imports -------------------------------------------------------
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdMolDescriptors
    except Exception as exc:  # noqa: BLE001
        return {
            "basic_score": 0.0,
            "design_result": {
                "opt_success_rate": 0.0,
                "best_oracle_value": None,
                "best_oracle_delta": None,
                "sim_success_rate": 0.0,
                "drd2_success_rate": 0.0,
            },
            "details": {"error": f"rdkit not available: {exc}"},
        }

    # --- Configuration -------------------------------------------------------
    k = int(scoring_conf.get("k", 50))
    lead_smiles_raw = str(scoring_conf.get("lead_smiles", "")).strip()
    sim_threshold = _safe_float(scoring_conf.get("similarity_threshold", 0.4), 0.4)
    target_drd2_min = _safe_float(scoring_conf.get("target_drd2_min", 0.5), 0.5)
    fp_radius = int(scoring_conf.get("fingerprint_radius", 2))
    fp_nbits = int(scoring_conf.get("fingerprint_nbits", 2048))

    # --- Validate lead -------------------------------------------------------
    lead_mol = Chem.MolFromSmiles(lead_smiles_raw)
    if lead_mol is None:
        return {
            "basic_score": 0.0,
            "design_result": {
                "opt_success_rate": 0.0,
                "best_oracle_value": None,
                "best_oracle_delta": None,
                "sim_success_rate": 0.0,
                "drd2_success_rate": 0.0,
            },
            "details": {"error": f"lead_smiles is invalid: {lead_smiles_raw}"},
        }
    try:
        Chem.SanitizeMol(lead_mol)
    except Exception as exc:  # noqa: BLE001
        return {
            "basic_score": 0.0,
            "design_result": {
                "opt_success_rate": 0.0,
                "best_oracle_value": None,
                "best_oracle_delta": None,
                "sim_success_rate": 0.0,
                "drd2_success_rate": 0.0,
            },
            "details": {"error": f"lead_smiles failed sanitization: {exc}"},
        }
    lead_canonical = Chem.MolToSmiles(lead_mol)
    lead_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        lead_mol, fp_radius, nBits=fp_nbits
    )

    # Compute lead's DRD2 via TDC (recomputed from SMILES — robust)
    drd2_oracle = _get_tdc_oracle("DRD2")
    lead_drd2 = float(drd2_oracle(lead_canonical))

    # --- Extract analogs from payload ----------------------------------------
    analog_smiles: list[str] = []
    if isinstance(payload, dict):
        analog_smiles = _extract_smiles_list(payload.get("analogs", []))
    elif isinstance(payload, list):
        analog_smiles = _extract_smiles_list(payload)

    # --- Score each analog ---------------------------------------------------
    n_valid = 0
    n_similar = 0  # Tanimoto >= sim_threshold
    n_active = 0  # DRD2 >= target_drd2_min
    n_success = 0  # both similar AND active

    best_drd2: float | None = None
    best_drd2_delta: float | None = None
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
                    "drd2": None,
                    "drd2_delta": None,
                    "drd2_active": False,
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

        # Format check: valid + not-duplicate + not-equal-to-lead
        format_ok = not is_duplicate and not equals_lead

        # Tanimoto similarity
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, fp_radius, nBits=fp_nbits
        )
        tanimoto = float(DataStructs.TanimotoSimilarity(lead_fp, fp))
        similar_ok = tanimoto >= sim_threshold
        if similar_ok:
            n_similar += 1

        # DRD2 activity check
        drd2_val = float(drd2_oracle(canonical))
        drd2_active = drd2_val >= target_drd2_min
        if drd2_active:
            n_active += 1

        # Optimization success: both constraints satisfied
        success = similar_ok and drd2_active
        if success:
            n_success += 1
            if best_drd2 is None or drd2_val > best_drd2:
                best_drd2 = drd2_val
                best_drd2_delta = drd2_val - lead_drd2
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
                "drd2": round(drd2_val, 4),
                "drd2_delta": round(drd2_val - lead_drd2, 4),
                "drd2_active": drd2_active,
                "success": success,
            }
        )

    # Pad details for any missing analogs (fewer than k submitted)
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
                "drd2": None,
                "drd2_delta": None,
                "drd2_active": False,
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
        "best_oracle_value": round(best_drd2, 4) if best_drd2 is not None else None,
        "best_oracle_delta": (
            round(best_drd2_delta, 4) if best_drd2_delta is not None else None
        ),
        "sim_success_rate": round(n_similar / k, 6),
        "drd2_success_rate": round(n_active / k, 6),
    }

    return {
        "basic_score": basic_score,
        "design_result": design_result,
        "details": {
            "lead_smiles": lead_smiles_raw,
            "lead_canonical": lead_canonical,
            "lead_drd2": round(lead_drd2, 4),
            "similarity_threshold": sim_threshold,
            "target_drd2_min": target_drd2_min,
            "k": k,
            "n_submitted": n_submitted,
            "n_valid": n_valid,
            "n_similar": n_similar,
            "n_active": n_active,
            "n_success": n_success,
            "design_subscores": design_subscores,
            "oracle_success": n_success > 0,
            "best_drd2": round(best_drd2, 4) if best_drd2 is not None else None,
            "best_smiles": best_smiles,
            "analog_details": analog_details,
        },
    }
