"""Scorer for des_05: Erlotinib decoration hopping.

Task intent (GuacaMol/TDC Deco Hop):
  - Keep quinazoline scaffold
  - Remove two Erlotinib decorations
  - Keep pharmacophore similarity high

The TDC Oracle("Deco Hop") returns a continuous score:
  oracle = mean([
      ClippedScoreModifier(upper_x=0.85)(PHCO_tanimoto),
      deco1_absent,       # 1.0 if CS([#6])(=O)=O absent, else 0.0
      deco2_absent,       # 1.0 if [#7]-c1ccc2ncsc2c1 absent, else 0.0
      scaffold_present,   # 1.0 if quinazoline present, else 0.0
  ])

Success = oracle_score >= target_oracle_min (default 0.8).
"""

from __future__ import annotations

import json

from scorers.design_scores import compute_basic_score

ERLOTINIB_SMILES = "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C"
DECO1_SMARTS = "CS([#6])(=O)=O"
DECO2_SMARTS = "[#7]-c1ccc2ncsc2c1"
SCAFFOLD_SMARTS = "[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12"

# TDC Oracle cache
_oracle_cache: dict[str, object] = {}


def _get_tdc_oracle(name: str):
    if name not in _oracle_cache:
        from tdc import Oracle

        _oracle_cache[name] = Oracle(name=name)
    return _oracle_cache[name]


def _parse_json(agent_stdout: str) -> tuple[dict | list | None, str | None]:
    try:
        payload = json.loads(agent_stdout.strip())
        return payload, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _safe_float(value, default: float) -> float:
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


def _get_phco_fingerprint(mol):
    from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

    return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)


def evaluate_deco_hop_candidate(
    smiles: str,
    scoring_conf: dict,
    seen_canonical: set | None = None,
) -> dict:
    """Per-candidate evaluation for the iterative des_05 Deco Hop loop."""
    from scorers.penalized_logp_scorer import canonicalize_smiles

    raw = str(smiles).strip() if smiles is not None else ""
    result: dict = {
        "submitted_smiles": raw or None,
        "canonical_smiles": None,
        "valid": False,
        "duplicate": False,
        "oracle_score": None,
        "scaffold_ok": False,
        "deco1_absent": False,
        "deco2_absent": False,
        "phco_sim": None,
        "success": False,
        "sanitize_error": None,
    }

    canonical, err = canonicalize_smiles(raw)
    if canonical is None:
        result["sanitize_error"] = err
        return result

    try:
        from rdkit import Chem, DataStructs
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"rdkit not available: {exc}") from exc

    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        result["sanitize_error"] = "rdkit could not parse canonical smiles"
        return result

    is_duplicate = seen_canonical is not None and canonical in seen_canonical
    target_oracle_min = _safe_float(scoring_conf.get("target_oracle_min", 0.8), 0.8)

    # Call TDC Oracle for the composite score
    oracle = _get_tdc_oracle("Deco Hop")
    oracle_score = float(oracle(canonical))

    # Compute SMARTS flags as informational feedback
    reference_smiles_raw = str(
        scoring_conf.get("reference_smiles", ERLOTINIB_SMILES)
    ).strip()
    ref_mol = Chem.MolFromSmiles(reference_smiles_raw)
    reference_phco_fp = _get_phco_fingerprint(ref_mol) if ref_mol is not None else None

    scaffold_smarts = str(scoring_conf.get("scaffold_smarts", SCAFFOLD_SMARTS)).strip()
    forbidden_deco1_smarts = str(
        scoring_conf.get("forbidden_deco1_smarts", DECO1_SMARTS)
    ).strip()
    forbidden_deco2_smarts = str(
        scoring_conf.get("forbidden_deco2_smarts", DECO2_SMARTS)
    ).strip()

    deco1_query = Chem.MolFromSmarts(forbidden_deco1_smarts)
    deco2_query = Chem.MolFromSmarts(forbidden_deco2_smarts)
    scaffold_query = Chem.MolFromSmarts(scaffold_smarts)

    deco1_absent = deco1_query is not None and not mol.HasSubstructMatch(deco1_query)
    deco2_absent = deco2_query is not None and not mol.HasSubstructMatch(deco2_query)
    scaffold_ok = scaffold_query is not None and mol.HasSubstructMatch(scaffold_query)

    phco_sim = None
    if reference_phco_fp is not None:
        phco_fp = _get_phco_fingerprint(mol)
        phco_sim = float(DataStructs.TanimotoSimilarity(phco_fp, reference_phco_fp))

    success = oracle_score >= target_oracle_min

    result.update(
        {
            "canonical_smiles": canonical,
            "valid": True,
            "duplicate": is_duplicate,
            "oracle_score": round(oracle_score, 4),
            "scaffold_ok": scaffold_ok,
            "deco1_absent": deco1_absent,
            "deco2_absent": deco2_absent,
            "phco_sim": round(phco_sim, 4) if phco_sim is not None else None,
            "success": success,
        }
    )
    return result


def score_deco_hop(agent_stdout: str, task: dict) -> dict:
    payload, parse_err = _parse_json(agent_stdout)
    _empty_result = {
        "opt_success": 0.0,
        "opt_success_rate": 0.0,
        "best_oracle_value": None,
        "best_oracle_delta": None,
        "scaffold_present_rate": 0.0,
        "deco1_absent_rate": 0.0,
        "deco2_absent_rate": 0.0,
    }
    if parse_err:
        return {
            "basic_score": 0.0,
            "design_result": _empty_result,
            "details": {"error": f"JSON parse error: {parse_err}"},
        }

    scoring_conf = task.get("scoring_logic", {})

    try:
        from rdkit import Chem, DataStructs
    except Exception as exc:  # noqa: BLE001
        return {
            "basic_score": 0.0,
            "design_result": _empty_result,
            "details": {"error": f"rdkit not available: {exc}"},
        }

    k = int(scoring_conf.get("k", 50))
    reference_smiles_raw = str(
        scoring_conf.get("reference_smiles", ERLOTINIB_SMILES)
    ).strip()
    target_oracle_min = _safe_float(scoring_conf.get("target_oracle_min", 0.8), 0.8)
    scaffold_smarts = str(scoring_conf.get("scaffold_smarts", SCAFFOLD_SMARTS)).strip()
    forbidden_deco1_smarts = str(
        scoring_conf.get("forbidden_deco1_smarts", DECO1_SMARTS)
    ).strip()
    forbidden_deco2_smarts = str(
        scoring_conf.get("forbidden_deco2_smarts", DECO2_SMARTS)
    ).strip()

    ref_mol = Chem.MolFromSmiles(reference_smiles_raw)
    if ref_mol is None:
        return {
            "basic_score": 0.0,
            "design_result": _empty_result,
            "details": {
                "error": f"reference_smiles is invalid: {reference_smiles_raw}"
            },
        }
    try:
        Chem.SanitizeMol(ref_mol)
    except Exception as exc:  # noqa: BLE001
        return {
            "basic_score": 0.0,
            "design_result": _empty_result,
            "details": {"error": f"reference_smiles failed sanitization: {exc}"},
        }

    reference_canonical = Chem.MolToSmiles(ref_mol)
    reference_phco_fp = _get_phco_fingerprint(ref_mol)

    deco1_query = Chem.MolFromSmarts(forbidden_deco1_smarts)
    deco2_query = Chem.MolFromSmarts(forbidden_deco2_smarts)
    scaffold_query = Chem.MolFromSmarts(scaffold_smarts)
    if deco1_query is None or deco2_query is None or scaffold_query is None:
        return {
            "basic_score": 0.0,
            "design_result": _empty_result,
            "details": {
                "error": "SMARTS parse error for Deco Hop patterns",
                "scaffold_smarts": scaffold_smarts,
                "forbidden_deco1_smarts": forbidden_deco1_smarts,
                "forbidden_deco2_smarts": forbidden_deco2_smarts,
            },
        }

    # TDC Oracle
    oracle = _get_tdc_oracle("Deco Hop")
    lead_oracle_score = float(oracle(reference_canonical))

    analog_smiles: list[str] = []
    if isinstance(payload, dict):
        analog_smiles = _extract_smiles_list(payload.get("analogs", []))
    elif isinstance(payload, list):
        analog_smiles = _extract_smiles_list(payload)

    n_valid = 0
    n_success = 0
    n_deco1_absent = 0
    n_deco2_absent = 0
    n_scaffold_present = 0

    best_oracle: float | None = None
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
                    "oracle_score": None,
                    "oracle_delta": None,
                    "phco_similarity": None,
                    "deco1_absent": False,
                    "deco2_absent": False,
                    "scaffold_present": False,
                    "success": False,
                    "sanitize_error": sanitize_err,
                }
            )
            continue

        n_valid += 1
        canonical = Chem.MolToSmiles(mol)
        is_duplicate = canonical in seen_canonical
        seen_canonical.add(canonical)
        equals_lead = canonical == reference_canonical
        format_ok = not is_duplicate and not equals_lead

        # TDC Oracle score
        oracle_score = float(oracle(canonical))
        oracle_delta = oracle_score - lead_oracle_score

        # Informational SMARTS flags
        phco_fp = _get_phco_fingerprint(mol)
        phco_similarity = float(
            DataStructs.TanimotoSimilarity(phco_fp, reference_phco_fp)
        )

        deco1_absent = not mol.HasSubstructMatch(deco1_query)
        deco2_absent = not mol.HasSubstructMatch(deco2_query)
        scaffold_present = mol.HasSubstructMatch(scaffold_query)

        if deco1_absent:
            n_deco1_absent += 1
        if deco2_absent:
            n_deco2_absent += 1
        if scaffold_present:
            n_scaffold_present += 1

        success = oracle_score >= target_oracle_min
        if success:
            n_success += 1
            if best_oracle is None or oracle_score > best_oracle:
                best_oracle = oracle_score
                best_delta = oracle_delta
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
                "oracle_score": round(oracle_score, 4),
                "oracle_delta": round(oracle_delta, 4),
                "phco_similarity": round(phco_similarity, 4),
                "deco1_absent": deco1_absent,
                "deco2_absent": deco2_absent,
                "scaffold_present": scaffold_present,
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
                "oracle_score": None,
                "oracle_delta": None,
                "phco_similarity": None,
                "deco1_absent": False,
                "deco2_absent": False,
                "scaffold_present": False,
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
        "best_oracle_value": round(best_oracle, 4) if best_oracle is not None else None,
        "best_oracle_delta": round(best_delta, 4) if best_delta is not None else None,
        "scaffold_present_rate": round(n_scaffold_present / k, 6),
        "deco1_absent_rate": round(n_deco1_absent / k, 6),
        "deco2_absent_rate": round(n_deco2_absent / k, 6),
    }

    return {
        "basic_score": basic_score,
        "design_result": design_result,
        "details": {
            "reference_smiles": reference_smiles_raw,
            "reference_canonical": reference_canonical,
            "lead_oracle_score": round(lead_oracle_score, 4),
            "target_oracle_min": target_oracle_min,
            "k": k,
            "n_submitted": n_submitted,
            "n_valid": n_valid,
            "n_success": n_success,
            "design_subscores": design_subscores,
            "oracle_success": n_success > 0,
            "best_smiles": best_smiles,
            "analog_details": analog_details,
        },
    }
