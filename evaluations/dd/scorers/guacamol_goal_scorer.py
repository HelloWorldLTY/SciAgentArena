"""Generic GuacaMol goal-directed scorer for code-optimizer tasks.

This scorer covers the four GuacaMol benchmarks added in this branch:

- ``celecoxib_rediscovery``
- ``aripiprazole_similarity``
- ``isomers_c11h24``
- ``median1``

It is configured entirely from the task JSON's ``scoring_logic`` block,
so a new GuacaMol benchmark only needs a new task JSON pointing at this
function (and the existing :class:`BudgetedGuacamolOracle`).

Expected ``task["scoring_logic"]`` fields
-----------------------------------------
tdc_oracle_name : str
    Name passed to ``tdc.Oracle(name=...)``. Required.
target_score_min : float
    Threshold the candidate's oracle score must clear for ``success``.
target_smiles_a : str | None
    Optional primary target (rediscovery / similarity / median's first ref).
target_smiles_b : str | None
    Optional secondary target (median's second ref).
exclude_exact_target : bool, default False
    If True, candidates equal to ``target_smiles_a`` are not counted as
    success.  If False (standard GuacaMol), exact target IS a legitimate
    success.
basic_score_mode : str, default "strict"
    "strict" applies all four format checks (exact_k, all_smiles_valid,
    no_duplicates, none_equal_lead).  "allow_target" forces
    ``equals_lead=False`` for every analog so exact target does not get
    penalised by the format ratio (rediscovery / standard similarity).
k : int, default 10
    Expected number of analogs.

Output schema (matches mpo_scorer / qed_scorer)
-----------------------------------------------
``{basic_score, design_result, details}`` where
``design_result.best_oracle_delta = best_oracle_value - target_score_min``,
i.e. positive numbers mean "exceeded the success threshold by this much".
"""

from __future__ import annotations

from typing import Any

from scorers.design_scores import compute_basic_score
from scorers.penalized_logp_scorer import (
    _extract_smiles_list,
    _parse_json,
    _safe_float,
)

# Lazy-loaded TDC oracle cache shared with the oracle wrapper.
_oracle_cache: dict[str, object] = {}


def _get_tdc_oracle(name: str):
    if name not in _oracle_cache:
        from tdc import Oracle

        _oracle_cache[name] = Oracle(name=name)
    return _oracle_cache[name]


def score_guacamol_goal(agent_stdout: str, task: dict) -> dict:
    """Score a GuacaMol-style goal-directed task from the agent stream.

    The harness in ``runners/batch_runner.py`` reconstructs a
    ``{"analogs": [...]}`` JSON string from the optimizer stream and
    passes it as ``agent_stdout``.  This scorer re-runs the TDC oracle
    on every candidate so that the final score is independent of any
    cached values written into the stream.
    """
    payload, parse_err = _parse_json(agent_stdout)
    empty_design = {
        "opt_success": 0.0,
        "opt_success_rate": 0.0,
        "best_oracle_value": None,
        "best_oracle_delta": None,
        "oracle_pass_rate": 0.0,
    }
    if parse_err:
        return {
            "basic_score": 0.0,
            "design_result": empty_design,
            "details": {"error": f"JSON parse error: {parse_err}"},
        }

    scoring_conf = task.get("scoring_logic", {})

    try:
        from rdkit import Chem
    except Exception as exc:
        return {
            "basic_score": 0.0,
            "design_result": empty_design,
            "details": {"error": f"rdkit not available: {exc}"},
        }

    tdc_oracle_name = str(scoring_conf.get("tdc_oracle_name", "")).strip()
    if not tdc_oracle_name:
        return {
            "basic_score": 0.0,
            "design_result": empty_design,
            "details": {"error": "scoring_logic.tdc_oracle_name is required"},
        }

    k = int(scoring_conf.get("k", 10))
    target_score_min = _safe_float(scoring_conf.get("target_score_min", 0.5), 0.5)
    exclude_exact_target = bool(scoring_conf.get("exclude_exact_target", False))
    basic_score_mode = str(scoring_conf.get("basic_score_mode", "strict")).strip()
    if basic_score_mode not in ("strict", "allow_target"):
        basic_score_mode = "strict"

    target_smiles_a_raw = str(scoring_conf.get("target_smiles_a", "")).strip()
    target_smiles_b_raw = str(scoring_conf.get("target_smiles_b", "")).strip()

    # Canonicalise references for equals-checks (free, no oracle calls).
    # If a target SMILES is non-empty but unparseable, that's a config error —
    # silently treating it as "no target" would break equals_target_a /
    # exclude_exact_target / strict basic-score logic downstream.
    target_canonical_a: str | None = None
    if target_smiles_a_raw:
        ref_mol = Chem.MolFromSmiles(target_smiles_a_raw)
        if ref_mol is None:
            return {
                "basic_score": 0.0,
                "design_result": empty_design,
                "details": {
                    "error": (
                        "scoring_logic.target_smiles_a is non-empty but RDKit "
                        f"cannot parse it: {target_smiles_a_raw!r}"
                    )
                },
            }
        target_canonical_a = Chem.MolToSmiles(ref_mol)

    target_canonical_b: str | None = None
    if target_smiles_b_raw:
        ref_mol = Chem.MolFromSmiles(target_smiles_b_raw)
        if ref_mol is None:
            return {
                "basic_score": 0.0,
                "design_result": empty_design,
                "details": {
                    "error": (
                        "scoring_logic.target_smiles_b is non-empty but RDKit "
                        f"cannot parse it: {target_smiles_b_raw!r}"
                    )
                },
            }
        target_canonical_b = Chem.MolToSmiles(ref_mol)

    oracle = _get_tdc_oracle(tdc_oracle_name)

    analog_smiles: list[str] = []
    if isinstance(payload, dict):
        analog_smiles = _extract_smiles_list(
            payload.get("analogs", payload.get("molecules", []))
        )
    elif isinstance(payload, list):
        analog_smiles = _extract_smiles_list(payload)

    n_valid = 0
    n_oracle_pass = 0
    n_success = 0

    best_oracle_value: float | None = None
    best_oracle_delta: float | None = None
    best_smiles: str | None = None

    seen_canonical: set[str] = set()
    analog_details: list[dict[str, Any]] = []

    for i, raw_smi in enumerate(analog_smiles[:k]):
        mol = Chem.MolFromSmiles(raw_smi) if raw_smi else None
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
                    "oracle_score": None,
                    "oracle_delta": None,
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

        # Real equals-target checks (independent of basic_score_mode).
        equals_target_a = (
            target_canonical_a is not None and canonical == target_canonical_a
        )
        equals_target_b = (
            target_canonical_b is not None and canonical == target_canonical_b
        )
        equals_any_target = equals_target_a or equals_target_b
        # `equals_lead` is what compute_basic_score reads. In allow_target
        # mode we override this to False so the format ratio is not
        # penalised when the candidate IS a target. In strict mode either
        # of the configured targets counts as "equals_lead" — covers
        # multi-reference tasks like median (Camphor + Menthol).
        if basic_score_mode == "allow_target":
            equals_lead_for_basic = False
        else:
            equals_lead_for_basic = equals_any_target

        format_ok = (not is_duplicate) and (not equals_lead_for_basic)

        oracle_score = float(oracle(canonical))
        improved_ok = oracle_score >= target_score_min
        if improved_ok:
            n_oracle_pass += 1

        # exclude_exact_target excludes EITHER target (A or B) — important
        # for multi-reference tasks where copying any reference is trivial.
        if exclude_exact_target:
            success = improved_ok and not equals_any_target
        else:
            success = improved_ok

        if success:
            n_success += 1
            delta = oracle_score - target_score_min
            if best_oracle_value is None or oracle_score > best_oracle_value:
                best_oracle_value = oracle_score
                best_oracle_delta = delta
                best_smiles = canonical

        analog_details.append(
            {
                "rank": i + 1,
                "smiles_raw": raw_smi,
                "smiles_canonical": canonical,
                "is_valid": True,
                "is_duplicate": is_duplicate,
                "equals_lead": equals_lead_for_basic,
                "equals_target_a": equals_target_a,
                "equals_target_b": equals_target_b,
                "format_ok": format_ok,
                "oracle_score": round(oracle_score, 4),
                "oracle_delta": round(oracle_score - target_score_min, 4),
                "improved_ok": improved_ok,
                "success": success,
            }
        )

    # Pad analog_details up to k so compute_basic_score's ratio is correct.
    for i in range(len(analog_smiles[:k]), k):
        analog_details.append(
            {
                "rank": i + 1,
                "smiles_raw": None,
                "is_valid": False,
                "is_duplicate": False,
                "equals_lead": False,
                "equals_target_a": False,
                "equals_target_b": False,
                "format_ok": False,
                "oracle_score": None,
                "oracle_delta": None,
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
            round(best_oracle_value, 4) if best_oracle_value is not None else None
        ),
        "best_oracle_delta": (
            round(best_oracle_delta, 4) if best_oracle_delta is not None else None
        ),
        "oracle_pass_rate": round(n_oracle_pass / k, 6),
    }

    return {
        "basic_score": basic_score,
        "design_result": design_result,
        "details": {
            "tdc_oracle_name": tdc_oracle_name,
            "target_score_min": target_score_min,
            "exclude_exact_target": exclude_exact_target,
            "basic_score_mode": basic_score_mode,
            "target_smiles_a": target_smiles_a_raw or None,
            "target_canonical_a": target_canonical_a,
            "target_smiles_b": target_smiles_b_raw or None,
            "target_canonical_b": target_canonical_b,
            "k": k,
            "n_submitted": n_submitted,
            "n_valid": n_valid,
            "n_oracle_pass": n_oracle_pass,
            "n_success": n_success,
            "design_subscores": design_subscores,
            "oracle_success": n_success > 0,
            "best_smiles": best_smiles,
            "analog_details": analog_details,
        },
    }
