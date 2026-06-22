"""Budget-aware oracle wrapper for Scaffold Hop code-optimizer tasks."""

from __future__ import annotations

from typing import Any

from scorers.oracle_budget import BudgetedOracleBase, OracleBudgetExceededError
from scorers.scaffold_hop_scorer import (
    DECORATION_SMARTS,
    SCAFFOLD_SMARTS,
    _get_phco_fingerprint,
    _get_tdc_oracle,
)


class BudgetedScaffoldHopOracle(BudgetedOracleBase):
    """Budget-counting Scaffold Hop oracle bound to a reference molecule."""

    STREAM_EXTRA_FIELDS = (
        "oracle_score",
        "oracle_delta",
        "decoration_ok",
        "scaffold_absent",
        "phco_sim",
        "improved_ok",
    )

    def __init__(
        self,
        reference_smiles: str,
        budget: int,
        decoration_smarts: str = DECORATION_SMARTS,
        forbidden_scaffold_smarts: str = SCAFFOLD_SMARTS,
        target_oracle_min: float = 0.7,
        stream_path: str | None = None,
    ) -> None:
        self.decoration_smarts = str(decoration_smarts).strip()
        self.forbidden_scaffold_smarts = str(forbidden_scaffold_smarts).strip()
        self.target_oracle_min = float(target_oracle_min)
        super().__init__(
            reference_smiles=reference_smiles,
            budget=budget,
            stream_path=stream_path,
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _init_reference(self, canonical: str, mol: Any) -> None:
        self._decoration_query = self._Chem.MolFromSmarts(self.decoration_smarts)
        self._scaffold_query = self._Chem.MolFromSmarts(self.forbidden_scaffold_smarts)
        if self._decoration_query is None or self._scaffold_query is None:
            raise ValueError("invalid SMARTS provided to BudgetedScaffoldHopOracle")

        self.reference_oracle_score = float(_get_tdc_oracle("Scaffold Hop")(canonical))
        self.reference_phco_fp = _get_phco_fingerprint(mol)

    def _session_start_extras(self) -> dict[str, Any]:
        return {
            "reference_oracle_score": self.reference_oracle_score,
            "target_oracle_min": self.target_oracle_min,
        }

    def _empty_result(self) -> dict[str, Any]:
        return {
            "valid": False,
            "canonical_smiles": None,
            "duplicate": False,
            "equals_reference": False,
            "oracle_score": None,
            "oracle_delta": None,
            "decoration_ok": False,
            "scaffold_absent": False,
            "phco_sim": None,
            "improved_ok": False,
            "success": False,
            "error": None,
        }

    def _evaluate_valid(self, canonical: str, mol: Any) -> dict[str, Any]:
        from rdkit import DataStructs

        oracle_score = float(_get_tdc_oracle("Scaffold Hop")(canonical))

        phco_sim: float | None
        try:
            phco_fp = _get_phco_fingerprint(mol)
            phco_sim = float(
                DataStructs.TanimotoSimilarity(phco_fp, self.reference_phco_fp)
            )
        except Exception:
            phco_sim = None

        equals_reference = canonical == self.reference_canonical
        decoration_ok = mol.HasSubstructMatch(self._decoration_query)
        scaffold_absent = not mol.HasSubstructMatch(self._scaffold_query)
        improved_ok = oracle_score >= self.target_oracle_min
        success = improved_ok and not equals_reference

        return {
            "valid": True,
            "canonical_smiles": canonical,
            "duplicate": False,
            "equals_reference": equals_reference,
            "oracle_score": oracle_score,
            "oracle_delta": oracle_score - self.reference_oracle_score,
            "decoration_ok": decoration_ok,
            "scaffold_absent": scaffold_absent,
            "phco_sim": phco_sim,
            "improved_ok": improved_ok,
            "success": success,
            "error": None,
        }
