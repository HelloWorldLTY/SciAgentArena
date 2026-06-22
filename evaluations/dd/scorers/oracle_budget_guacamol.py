"""Budget-aware oracle wrapper for GuacaMol code-optimizer tasks.

A single generic oracle class (:class:`BudgetedGuacamolOracle`) covers
the four GuacaMol-style benchmarks added in this branch:

- ``celecoxib_rediscovery``  — ECFP4 Tanimoto vs. Celecoxib
- ``aripiprazole_similarity`` — FCFP4 Tanimoto vs. Aripiprazole, clipped
- ``isomers_c11h24``          — geometric-mean Gaussian over (nC, nH, nAtoms)
- ``median1``                 — sqrt(Tanimoto(camphor) * Tanimoto(menthol))

The TDC oracle name is configurable so that adding more GuacaMol tasks
(Troglitazone rediscovery, Albuterol similarity, Median 2, …) only
requires a new task JSON.
"""

from __future__ import annotations

from typing import Any

from scorers.oracle_budget import BudgetedOracleBase, OracleBudgetExceededError

# Internal placeholder used as base-class anchor when the task has no real
# reference SMILES (e.g. isomers_c11h24).  Methane is the smallest valid
# molecule and is *not* a legal answer for any of our tasks (e.g. it is not
# C11H24, not similar to Celecoxib, etc.), so it cannot leak into the prompt
# as a trivial cheat.
_DEFAULT_PLACEHOLDER_SMILES = "C"


# Lazy-loaded TDC oracle cache shared across instances.
_oracle_cache: dict[str, object] = {}


def _get_tdc_oracle(name: str):
    if name not in _oracle_cache:
        from tdc import Oracle

        _oracle_cache[name] = Oracle(name=name)
    return _oracle_cache[name]


class BudgetedGuacamolOracle(BudgetedOracleBase):
    """Generic budget-counting wrapper for GuacaMol scalar oracles.

    Parameters
    ----------
    tdc_oracle_name :
        Name passed to ``tdc.Oracle(name=...)``.  Lower-case names from
        ``tdc/metadata.py::guacamol_oracle`` (e.g. ``"celecoxib_rediscovery"``,
        ``"aripiprazole_similarity"``, ``"isomers_c11h24"``, ``"median1"``).
    budget :
        Hard cap on :meth:`score` calls.  Every call (even invalid /
        duplicate) consumes one unit.
    target_score_min :
        Threshold the oracle score must clear for ``improved_ok``/``success``.
    target_smiles_a :
        Optional primary target SMILES.  Used by rediscovery / similarity
        (it is the molecule to match) and by median tasks (Camphor in
        ``median1``).  When provided it serves as the base class anchor
        and an ECFP4 fingerprint is precomputed for free Tanimoto checks.
    target_smiles_b :
        Optional secondary target SMILES.  Used only by median tasks.
        ECFP4 fingerprint is precomputed for free Tanimoto checks.
    exclude_exact_target :
        If True, candidates equal to ``target_smiles_a`` are NOT counted
        as success (analog-style tasks).  If False, exact target is
        a legitimate success (standard rediscovery / similarity).
    stream_path :
        Optional override of the JSONL stream path; otherwise read from
        the ``AGENT4S_OPTIMIZER_STREAM`` environment variable.
    """

    STREAM_EXTRA_FIELDS = (
        "oracle_score",
        "improved_ok",
        "equals_target_a",
        "equals_target_b",
        "debug_tanimoto_primary",
        "debug_tanimoto_secondary",
    )

    # Allowed fingerprint types for the free debug Tanimoto helper.
    # Each maps to the actual RDKit primitive used at construction.
    _SUPPORTED_FP_TYPES: tuple[str, ...] = ("ecfp", "fcfp", "ap")

    def __init__(
        self,
        tdc_oracle_name: str,
        budget: int,
        *,
        target_score_min: float = 0.5,
        target_smiles_a: str | None = None,
        target_smiles_b: str | None = None,
        exclude_exact_target: bool = False,
        fingerprint_type: str = "ecfp",
        fingerprint_radius: int = 2,
        stream_path: str | None = None,
    ) -> None:
        if not tdc_oracle_name:
            raise ValueError("tdc_oracle_name is required")
        self.tdc_oracle_name = str(tdc_oracle_name)
        self.target_score_min = float(target_score_min)
        self.exclude_exact_target = bool(exclude_exact_target)

        fp_type = str(fingerprint_type).strip().lower()
        if fp_type not in self._SUPPORTED_FP_TYPES:
            raise ValueError(
                f"fingerprint_type must be one of {self._SUPPORTED_FP_TYPES}, "
                f"got {fingerprint_type!r}"
            )
        self.fingerprint_type = fp_type
        self.fingerprint_radius = int(fingerprint_radius)

        # Validate target SMILES early — if a target was provided but cannot be
        # parsed, it's a config error (would silently break equals_target /
        # exclude_exact_target / strict-mode basic-score checks downstream).
        try:
            from rdkit import Chem
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"rdkit not available: {exc}") from exc

        if target_smiles_a is not None and str(target_smiles_a).strip():
            if Chem.MolFromSmiles(str(target_smiles_a).strip()) is None:
                raise ValueError(
                    f"target_smiles_a is non-empty but RDKit cannot parse it: "
                    f"{target_smiles_a!r}"
                )
        if target_smiles_b is not None and str(target_smiles_b).strip():
            if Chem.MolFromSmiles(str(target_smiles_b).strip()) is None:
                raise ValueError(
                    f"target_smiles_b is non-empty but RDKit cannot parse it: "
                    f"{target_smiles_b!r}"
                )

        self._target_a_input = target_smiles_a
        self._target_b_input = target_smiles_b

        # Resolve the SMILES handed to BudgetedOracleBase.  When the task has
        # no real anchor (isomers), use an internal placeholder so callers
        # never have to leak a legal answer through the task input.
        anchor = (target_smiles_a or "").strip()
        if not anchor:
            anchor = _DEFAULT_PLACEHOLDER_SMILES
            self._has_real_anchor = False
        else:
            self._has_real_anchor = True

        super().__init__(
            reference_smiles=anchor,
            budget=budget,
            stream_path=stream_path,
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _compute_fp(self, mol: Any):
        """Compute the configured debug fingerprint for a molecule.

        - ``ecfp``: sparse-count Morgan (radius=fingerprint_radius), matching
          the ECFP fingerprint family TDC GuacaMol rediscovery / median /
          isomers oracles use internally (helper: smiles_2_fingerprint_ECFP4).
        - ``fcfp``: feature-based sparse-count Morgan, matching the FCFP
          fingerprint family TDC GuacaMol similarity oracles use internally
          (helper: smiles_2_fingerprint_FCFP4).
        - ``ap``: sparse atom-pair fingerprint with maxLength=10, matching
          TDC's smiles_2_fingerprint_AP helper used by mestranol_similarity,
          fexofenadine_mpo, and ranolazine_mpo. Uses
          ``AllChem.GetAtomPairFingerprint(mol, maxLength=10)``, NOT the
          hashed variant — Tanimoto values differ between the two.
        """
        if self.fingerprint_type == "ecfp":
            return self._AllChem.GetMorganFingerprint(
                mol,
                self.fingerprint_radius,
            )
        if self.fingerprint_type == "fcfp":
            return self._AllChem.GetMorganFingerprint(
                mol,
                self.fingerprint_radius,
                useFeatures=True,
            )
        # "ap" — match TDC's smiles_2_fingerprint_AP exactly.
        return self._AllChem.GetAtomPairFingerprint(mol, maxLength=10)

    def _init_reference(self, canonical: str, mol: Any) -> None:
        from rdkit import DataStructs
        from rdkit.Chem import AllChem, rdMolDescriptors

        self._DataStructs = DataStructs
        self._AllChem = AllChem
        self._rdMolDescriptors = rdMolDescriptors

        # Free debug fingerprint type / radius is configurable so we can match
        # whichever fingerprint the underlying TDC oracle actually uses
        # (ECFP4 for rediscovery / median1 / isomers, FCFP4 for most
        # similarity tasks, ECFP6 for median2, AP for mestranol_similarity).
        if self._has_real_anchor:
            self.target_canonical_a = canonical
            self.target_fp_a = self._compute_fp(mol)
        else:
            self.target_canonical_a = None
            self.target_fp_a = None

        # Optional secondary target fingerprint (median tasks).
        self.target_canonical_b = None
        self.target_fp_b = None
        if self._target_b_input:
            mol_b = self._Chem.MolFromSmiles(self._target_b_input)
            if mol_b is not None:
                self.target_canonical_b = self._Chem.MolToSmiles(mol_b)
                self.target_fp_b = self._compute_fp(mol_b)

    def _session_start_extras(self) -> dict[str, Any]:
        extras: dict[str, Any] = {
            "tdc_oracle_name": self.tdc_oracle_name,
            "target_score_min": self.target_score_min,
            "exclude_exact_target": self.exclude_exact_target,
            "has_real_anchor": self._has_real_anchor,
            "fingerprint_type": self.fingerprint_type,
            "fingerprint_radius": self.fingerprint_radius,
        }
        if self._has_real_anchor:
            extras["target_canonical_a"] = self.target_canonical_a
        if self.target_canonical_b is not None:
            extras["target_canonical_b"] = self.target_canonical_b
        return extras

    def _empty_result(self) -> dict[str, Any]:
        return {
            "valid": False,
            "canonical_smiles": None,
            "duplicate": False,
            "equals_target_a": False,
            "equals_target_b": False,
            "oracle_score": None,
            "improved_ok": False,
            "debug_tanimoto_primary": None,
            "debug_tanimoto_secondary": None,
            "success": False,
            "error": None,
        }

    def _evaluate_valid(self, canonical: str, mol: Any) -> dict[str, Any]:
        oracle_score = float(_get_tdc_oracle(self.tdc_oracle_name)(canonical))
        improved_ok = oracle_score >= self.target_score_min

        equals_target_a = self._has_real_anchor and canonical == self.target_canonical_a
        equals_target_b = (
            self.target_canonical_b is not None and canonical == self.target_canonical_b
        )

        # Free Tanimoto helpers using the configured fingerprint type / radius
        # (matches the fingerprint TDC's underlying oracle uses for this task).
        # The optimizer can use these to pre-rank candidates without spending
        # oracle budget.
        cand_fp = None
        debug_tanimoto_primary: float | None = None
        if self.target_fp_a is not None:
            cand_fp = self._compute_fp(mol)
            debug_tanimoto_primary = float(
                self._DataStructs.TanimotoSimilarity(self.target_fp_a, cand_fp)
            )

        debug_tanimoto_secondary: float | None = None
        if self.target_fp_b is not None:
            if cand_fp is None:
                cand_fp = self._compute_fp(mol)
            debug_tanimoto_secondary = float(
                self._DataStructs.TanimotoSimilarity(self.target_fp_b, cand_fp)
            )

        # exclude_exact_target removes BOTH targets when set (median uses
        # target A and B; rediscovery/similarity only uses A so equals_target_b
        # is False there).
        if self.exclude_exact_target:
            success = improved_ok and not (equals_target_a or equals_target_b)
        else:
            success = improved_ok

        return {
            "valid": True,
            "canonical_smiles": canonical,
            "duplicate": False,
            "equals_target_a": equals_target_a,
            "equals_target_b": equals_target_b,
            "oracle_score": oracle_score,
            "improved_ok": improved_ok,
            "debug_tanimoto_primary": debug_tanimoto_primary,
            "debug_tanimoto_secondary": debug_tanimoto_secondary,
            "success": success,
            "error": None,
        }
