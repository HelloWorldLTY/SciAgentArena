"""Scorer for des_01: constrained penalized logP optimization.

Evaluation protocol:
  - Input : lead molecule X (low penalized logP from ZINC)
  - Output: 50 analog molecules Y
  - Similarity constraint : Tanimoto(X, Y) >= 0.4  (Morgan fp, radius=2, 2048 bits)
  - Optimization goal    : penalized_logP(Y) > penalized_logP(X)
  - Oracle               : TDC Oracle("LogP")  — already the penalized logP
                           (logP − SA − ring_penalty, computed via RDKit)

Metrics returned
----------------
  basic_score  : mean of 4 binary format flags
                  (exact_k, all_smiles_valid, no_duplicates, none_equal_lead)
  design_result : dict with constraint subscores
    opt_success_rate          : fraction of k analogs satisfying BOTH similarity AND
                                improvement constraints. Range [0.0, 1.0].
    best_oracle_value         : absolute penalized_logP of the best passing analog,
                                or None if no analog passed.
    best_oracle_delta         : best_oracle_value minus lead's penalized_logP,
                                or None if no analog passed.
    sim_success_rate          : fraction with Tanimoto >= similarity_threshold.
    penalized_logp_success_rate: fraction with penalized_logP(Y) > penalized_logP(X).

Details also include:
  oracle_success : bool — True if at least one analog passed both constraints
  best_smiles    : canonical SMILES of the best passing analog
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

_PENALIZED_LOGP_STATS = {
    "logP_mean": 2.4570953396190123,
    "logP_std": 1.434324401111988,
    "SA_mean": -3.0525811293166134,
    "SA_std": 0.8335207024513095,
    "cycle_mean": -0.0485696876403053,
    "cycle_std": 0.2860212110245455,
}


def _get_tdc_oracle(name: str):
    if name not in _oracle_cache:
        from tdc import Oracle

        _oracle_cache[name] = Oracle(name=name)
    return _oracle_cache[name]


def canonicalize_smiles(smiles: str) -> tuple[str | None, str | None]:
    """Return ``(canonical_smiles, error)`` for *smiles*."""
    try:
        from rdkit import Chem
    except Exception as exc:  # noqa: BLE001
        return None, f"rdkit not available: {exc}"

    raw = str(smiles).strip()
    if not raw:
        return None, "empty smiles"

    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return None, "rdkit could not parse smiles"
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    return Chem.MolToSmiles(mol), None


def compute_penalized_logp_breakdown(smiles: str) -> dict | None:
    """Return the exact TDC penalized-logP component breakdown for *smiles*."""
    canonical, _ = canonicalize_smiles(smiles)
    if canonical is None:
        return None

    try:
        import networkx as nx
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from tdc.chem_utils.oracle.oracle import calculateScore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Missing penalized logP dependencies: {exc}") from exc

    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return None

    log_p = float(Descriptors.MolLogP(mol))
    sa_score = float(calculateScore(mol))
    sa_component = -sa_score

    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        largest_cycle = 0
    else:
        largest_cycle = max(len(cycle) for cycle in cycle_list)

    cycle_overflow = max(largest_cycle - 6, 0)
    cycle_penalty = -float(cycle_overflow)

    normalized_log_p = (
        log_p - _PENALIZED_LOGP_STATS["logP_mean"]
    ) / _PENALIZED_LOGP_STATS["logP_std"]
    normalized_sa = (
        sa_component - _PENALIZED_LOGP_STATS["SA_mean"]
    ) / _PENALIZED_LOGP_STATS["SA_std"]
    normalized_cycle = (
        cycle_penalty - _PENALIZED_LOGP_STATS["cycle_mean"]
    ) / _PENALIZED_LOGP_STATS["cycle_std"]
    penalized_logp = normalized_log_p + normalized_sa + normalized_cycle

    return {
        "canonical_smiles": canonical,
        "mol_logp": log_p,
        "sa_score": sa_score,
        "sa_component": sa_component,
        "largest_cycle_size": largest_cycle,
        "cycle_overflow": cycle_overflow,
        "cycle_penalty": cycle_penalty,
        "normalized_logp": normalized_log_p,
        "normalized_sa": normalized_sa,
        "normalized_cycle": normalized_cycle,
        "penalized_logp": penalized_logp,
    }


def build_penalized_logp_context(
    lead_smiles: str,
    similarity_threshold: float = 0.4,
    fingerprint_radius: int = 2,
    fingerprint_nbits: int = 2048,
    min_improvement_delta: float = 0.0,
) -> dict:
    """Build reusable lead context for iterative des_01 scoring."""
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

    lead_breakdown = compute_penalized_logp_breakdown(lead_canonical)
    if lead_breakdown is None:
        raise ValueError(f"Could not compute penalized logP for lead: {lead_smiles}")

    lead_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        lead_mol,
        fingerprint_radius,
        nBits=fingerprint_nbits,
    )
    return {
        "lead_smiles": lead_smiles,
        "lead_canonical": lead_canonical,
        "lead_penalized_logp": float(lead_breakdown["penalized_logp"]),
        "lead_oracle_value": float(lead_breakdown["penalized_logp"]),
        "oracle_name": "penalized_logP",
        "lead_breakdown": lead_breakdown,
        "similarity_threshold": float(similarity_threshold),
        "fingerprint_radius": int(fingerprint_radius),
        "fingerprint_nbits": int(fingerprint_nbits),
        "min_improvement_delta": float(min_improvement_delta),
        "lead_fp": lead_fp,
    }


def evaluate_penalized_logp_candidate(
    smiles: str,
    context: dict,
    *,
    is_duplicate: bool = False,
) -> dict:
    """Return exact per-candidate feedback for the iterative des_01 loop."""
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdMolDescriptors
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"rdkit not available: {exc}") from exc

    raw = str(smiles).strip() if smiles is not None else ""
    result = {
        "submitted_smiles": raw or None,
        "canonical_smiles": None,
        "valid": False,
        "duplicate": bool(is_duplicate),
        "equals_lead": False,
        "format_ok": False,
        "tanimoto": None,
        "similar_ok": False,
        "oracle_score": None,
        "final_score": None,
        "penalized_logp_delta": None,
        "improved_ok": False,
        "success": False,
        "components": None,
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

    breakdown = compute_penalized_logp_breakdown(canonical)
    if breakdown is None:
        result["sanitize_error"] = "failed to compute penalized logP"
        return result

    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol,
        int(context["fingerprint_radius"]),
        nBits=int(context["fingerprint_nbits"]),
    )
    tanimoto = float(DataStructs.TanimotoSimilarity(context["lead_fp"], fp))
    penlogp_val = float(breakdown["penalized_logp"])
    delta = penlogp_val - float(context["lead_penalized_logp"])
    equals_lead = canonical == context["lead_canonical"]
    similar_ok = tanimoto >= float(context["similarity_threshold"])
    min_delta = float(context.get("min_improvement_delta", 0.0))
    improved_ok = penlogp_val > float(context["lead_penalized_logp"]) + min_delta
    success = similar_ok and improved_ok

    result.update(
        {
            "canonical_smiles": canonical,
            "valid": True,
            "equals_lead": equals_lead,
            "format_ok": (not bool(is_duplicate)) and (not equals_lead),
            "tanimoto": tanimoto,
            "similar_ok": similar_ok,
            "oracle_score": penlogp_val,
            "final_score": penlogp_val,
            "penalized_logp_delta": delta,
            "improved_ok": improved_ok,
            "success": success,
            "components": breakdown,
        }
    )
    return result


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def score_penalized_logp_optimization(agent_stdout: str, task: dict) -> dict:
    """Score the JT-VAE constrained penalized logP optimization task.

    Expected ``task["scoring_logic"]`` fields:
      lead_smiles          : str   — updated per lead by adapt_task
      k                    : int   — expected number of analogs (default 50)
      similarity_threshold : float — minimum Tanimoto to lead (default 0.4)
      fingerprint_radius   : int   (default 2)
      fingerprint_nbits    : int   (default 2048)
    """

    # --- Parse JSON output --------------------------------------------------
    payload, parse_err = _parse_json(agent_stdout)
    _empty_result = {
        "opt_success_rate": 0.0,
        "best_oracle_value": None,
        "best_oracle_delta": None,
        "sim_success_rate": 0.0,
        "penalized_logp_success_rate": 0.0,
    }
    if parse_err:
        return {
            "basic_score": 0.0,
            "design_result": _empty_result,
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
            "design_result": _empty_result,
            "details": {"error": f"rdkit not available: {exc}"},
        }

    # --- Configuration -------------------------------------------------------
    k = int(scoring_conf.get("k", 50))
    lead_smiles_raw = str(scoring_conf.get("lead_smiles", "")).strip()
    sim_threshold = _safe_float(scoring_conf.get("similarity_threshold", 0.4), 0.4)
    fp_radius = int(scoring_conf.get("fingerprint_radius", 2))
    fp_nbits = int(scoring_conf.get("fingerprint_nbits", 2048))
    min_improvement_delta = _safe_float(
        scoring_conf.get("min_improvement_delta", 0.0), 0.0
    )

    # --- Validate lead -------------------------------------------------------
    lead_mol = Chem.MolFromSmiles(lead_smiles_raw)
    if lead_mol is None:
        return {
            "basic_score": 0.0,
            "design_result": _empty_result,
            "details": {"error": f"lead_smiles is invalid: {lead_smiles_raw}"},
        }
    try:
        Chem.SanitizeMol(lead_mol)
    except Exception as exc:  # noqa: BLE001
        return {
            "basic_score": 0.0,
            "design_result": _empty_result,
            "details": {"error": f"lead_smiles failed sanitization: {exc}"},
        }
    lead_canonical = Chem.MolToSmiles(lead_mol)
    lead_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        lead_mol, fp_radius, nBits=fp_nbits
    )

    # Compute lead's penalized logP via TDC (recomputed from SMILES — robust)
    logp_oracle = _get_tdc_oracle("LogP")
    lead_penalized_logp = float(logp_oracle(lead_canonical))

    # --- Extract analogs from payload ----------------------------------------
    analog_smiles: list[str] = []
    if isinstance(payload, dict):
        analog_smiles = _extract_smiles_list(payload.get("analogs", []))
    elif isinstance(payload, list):
        analog_smiles = _extract_smiles_list(payload)

    # --- Score each analog ---------------------------------------------------
    n_valid = 0
    n_format_ok = 0  # valid + unique + not-lead  (used for correctness)
    n_similar = 0  # Tanimoto >= sim_threshold
    n_improved = 0  # penalized_logP(Y) > penalized_logP(X)
    n_success = 0  # both similar AND improved  (used for optimization_success)

    best_improvement: float | None = None
    best_penalized_logp: float | None = None
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
                    "penalized_logp": None,
                    "penalized_logp_delta": None,
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
        equals_lead = canonical == lead_canonical

        # Format check: valid + not-duplicate + not-equal-to-lead
        format_ok = not is_duplicate and not equals_lead
        if format_ok:
            n_format_ok += 1

        # Tanimoto similarity
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, fp_radius, nBits=fp_nbits
        )
        tanimoto = float(DataStructs.TanimotoSimilarity(lead_fp, fp))
        similar_ok = tanimoto >= sim_threshold
        if similar_ok:
            n_similar += 1

        # Penalized logP improvement (must exceed lead by at least min_improvement_delta)
        penlogp_val = float(logp_oracle(canonical))
        improved_ok = penlogp_val > lead_penalized_logp + min_improvement_delta
        if improved_ok:
            n_improved += 1

        # Optimization success: both constraints satisfied
        success = similar_ok and improved_ok
        if success:
            n_success += 1
            improvement = penlogp_val - lead_penalized_logp
            if best_improvement is None or improvement > best_improvement:
                best_improvement = improvement
                best_penalized_logp = penlogp_val
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
                "penalized_logp": round(penlogp_val, 4),
                "penalized_logp_delta": round(penlogp_val - lead_penalized_logp, 4),
                "improved_ok": improved_ok,
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
                "penalized_logp": None,
                "penalized_logp_delta": None,
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
        "best_oracle_value": (
            round(best_penalized_logp, 4) if best_penalized_logp is not None else None
        ),
        "best_oracle_delta": (
            round(best_improvement, 4) if best_improvement is not None else None
        ),
        "sim_success_rate": round(n_similar / k, 6),
        "penalized_logp_success_rate": round(n_improved / k, 6),
    }

    return {
        "basic_score": basic_score,
        "design_result": design_result,
        "details": {
            "lead_smiles": lead_smiles_raw,
            "lead_canonical": lead_canonical,
            "lead_penalized_logp": round(lead_penalized_logp, 4),
            "similarity_threshold": sim_threshold,
            "min_improvement_delta": min_improvement_delta,
            "k": k,
            "n_submitted": n_submitted,
            "n_valid": n_valid,
            "n_similar": n_similar,
            "n_improved": n_improved,
            "n_success": n_success,
            "design_subscores": design_subscores,
            "oracle_success": n_success > 0,
            "best_smiles": best_smiles,
            "analog_details": analog_details,
        },
    }
