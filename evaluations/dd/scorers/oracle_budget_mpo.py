"""Budget-aware oracle wrapper for Osimertinib MPO code-optimizer tasks."""

from __future__ import annotations

from typing import Any

from scorers.mpo_scorer import _get_tdc_oracle
from scorers.oracle_budget import BudgetedOracleBase, OracleBudgetExceededError


class BudgetedMPOOracle(BudgetedOracleBase):
    """Budget-counting Osimertinib MPO oracle bound to a reference molecule."""

    STREAM_EXTRA_FIELDS = (
        "mpo",
        "oracle_score",
        "mpo_delta",
        "improved_ok",
    )

    def __init__(
        self,
        reference_smiles: str,
        budget: int,
        target_mpo_min: float = 0.5,
        stream_path: str | None = None,
    ) -> None:
        self.target_mpo_min = float(target_mpo_min)
        super().__init__(
            reference_smiles=reference_smiles,
            budget=budget,
            stream_path=stream_path,
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _init_reference(self, canonical: str, mol: Any) -> None:
        self.reference_mpo = float(_get_tdc_oracle("Osimertinib_MPO")(canonical))

    def _session_start_extras(self) -> dict[str, Any]:
        return {
            "reference_mpo": self.reference_mpo,
            "target_mpo_min": self.target_mpo_min,
        }

    def _empty_result(self) -> dict[str, Any]:
        return {
            "valid": False,
            "canonical_smiles": None,
            "duplicate": False,
            "equals_reference": False,
            "mpo": None,
            "oracle_score": None,
            "mpo_delta": None,
            "improved_ok": False,
            "success": False,
            "error": None,
        }

    def _evaluate_valid(self, canonical: str, mol: Any) -> dict[str, Any]:
        mpo = float(_get_tdc_oracle("Osimertinib_MPO")(canonical))
        equals_reference = canonical == self.reference_canonical
        improved_ok = mpo >= self.target_mpo_min
        success = improved_ok and not equals_reference

        return {
            "valid": True,
            "canonical_smiles": canonical,
            "duplicate": False,
            "equals_reference": equals_reference,
            "mpo": mpo,
            "oracle_score": mpo,
            "mpo_delta": mpo - self.reference_mpo,
            "improved_ok": improved_ok,
            "success": success,
            "error": None,
        }
