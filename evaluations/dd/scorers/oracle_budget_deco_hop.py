"""Budget-aware oracle wrapper for Deco Hop code-optimizer tasks."""

from __future__ import annotations

from typing import Any

from scorers.deco_hop_scorer import (
    DECO1_SMARTS,
    DECO2_SMARTS,
    SCAFFOLD_SMARTS,
    _get_phco_fingerprint,
    _get_tdc_oracle,
)
from scorers.oracle_budget import BudgetedOracleBase, OracleBudgetExceededError


class BudgetedDecoHopOracle(BudgetedOracleBase):
    """Budget-counting Deco Hop oracle bound to a single reference molecule."""

    STREAM_EXTRA_FIELDS = (
        "oracle_score",
        "oracle_delta",
        "scaffold_ok",
        "deco1_absent",
        "deco2_absent",
        "phco_sim",
        "improved_ok",
    )

    def __init__(
        self,
        reference_smiles: str,
        budget: int,
        scaffold_smarts: str = SCAFFOLD_SMARTS,
        forbidden_deco1_smarts: str = DECO1_SMARTS,
        forbidden_deco2_smarts: str = DECO2_SMARTS,
        target_oracle_min: float = 0.8,
        stream_path: str | None = None,
    ) -> None:
        self.scaffold_smarts = str(scaffold_smarts).strip()
        self.forbidden_deco1_smarts = str(forbidden_deco1_smarts).strip()
        self.forbidden_deco2_smarts = str(forbidden_deco2_smarts).strip()
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
        self._scaffold_query = self._Chem.MolFromSmarts(self.scaffold_smarts)
        self._deco1_query = self._Chem.MolFromSmarts(self.forbidden_deco1_smarts)
        self._deco2_query = self._Chem.MolFromSmarts(self.forbidden_deco2_smarts)
        if (
            self._scaffold_query is None
            or self._deco1_query is None
            or self._deco2_query is None
        ):
            raise ValueError("invalid SMARTS provided to BudgetedDecoHopOracle")

        self.reference_oracle_score = float(_get_tdc_oracle("Deco Hop")(canonical))
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
            "scaffold_ok": False,
            "deco1_absent": False,
            "deco2_absent": False,
            "phco_sim": None,
            "improved_ok": False,
            "success": False,
            "error": None,
        }

    def _evaluate_valid(self, canonical: str, mol: Any) -> dict[str, Any]:
        from rdkit import DataStructs

        oracle_score = float(_get_tdc_oracle("Deco Hop")(canonical))

        phco_sim: float | None
        try:
            phco_fp = _get_phco_fingerprint(mol)
            phco_sim = float(
                DataStructs.TanimotoSimilarity(phco_fp, self.reference_phco_fp)
            )
        except Exception:
            phco_sim = None

        equals_reference = canonical == self.reference_canonical
        scaffold_ok = mol.HasSubstructMatch(self._scaffold_query)
        deco1_absent = not mol.HasSubstructMatch(self._deco1_query)
        deco2_absent = not mol.HasSubstructMatch(self._deco2_query)
        improved_ok = oracle_score >= self.target_oracle_min
        success = improved_ok and not equals_reference

        return {
            "valid": True,
            "canonical_smiles": canonical,
            "duplicate": False,
            "equals_reference": equals_reference,
            "oracle_score": oracle_score,
            "oracle_delta": oracle_score - self.reference_oracle_score,
            "scaffold_ok": scaffold_ok,
            "deco1_absent": deco1_absent,
            "deco2_absent": deco2_absent,
            "phco_sim": phco_sim,
            "improved_ok": improved_ok,
            "success": success,
            "error": None,
        }
