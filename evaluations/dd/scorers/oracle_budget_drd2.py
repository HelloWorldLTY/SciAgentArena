"""Budget-aware oracle wrapper for DRD2 code-optimizer tasks."""

from __future__ import annotations

from typing import Any

from scorers.drd2_scorer import _get_tdc_oracle
from scorers.oracle_budget import BudgetedOracleBase, OracleBudgetExceededError


class BudgetedDRD2Oracle(BudgetedOracleBase):
    """Budget-counting DRD2 oracle bound to a single lead molecule.

    Parameters
    ----------
    lead_smiles :
        SMILES of the lead molecule.
    budget :
        Hard cap on :meth:`score` calls (every call counts, including
        invalid / duplicate).
    similarity_threshold :
        Tanimoto cutoff for the ``similar_ok`` flag. Defaults to 0.4.
    fingerprint_radius, fingerprint_nbits :
        Morgan fingerprint parameters.
    target_drd2_min :
        Minimum DRD2 score that counts as ``improved_ok``. Defaults to
        0.5 to match the scorer in ``scoring_logic.target_drd2_min``.
    """

    STREAM_EXTRA_FIELDS = (
        "tanimoto",
        "similar_ok",
        "drd2",
        "oracle_score",
        "drd2_delta",
        "improved_ok",
    )

    def __init__(
        self,
        lead_smiles: str,
        budget: int,
        similarity_threshold: float = 0.4,
        fingerprint_radius: int = 2,
        fingerprint_nbits: int = 2048,
        target_drd2_min: float = 0.5,
        stream_path: str | None = None,
    ) -> None:
        self.similarity_threshold = float(similarity_threshold)
        self.fingerprint_radius = int(fingerprint_radius)
        self.fingerprint_nbits = int(fingerprint_nbits)
        self.target_drd2_min = float(target_drd2_min)
        super().__init__(
            reference_smiles=lead_smiles,
            budget=budget,
            stream_path=stream_path,
        )
        self.lead_smiles = self.reference_smiles
        self.lead_canonical = self.reference_canonical

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _init_reference(self, canonical: str, mol: Any) -> None:
        from rdkit import DataStructs
        from rdkit.Chem import rdMolDescriptors

        self._DataStructs = DataStructs
        self._rdMolDescriptors = rdMolDescriptors

        self.lead_drd2 = float(_get_tdc_oracle("DRD2")(canonical))
        self.lead_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            self.fingerprint_radius,
            nBits=self.fingerprint_nbits,
        )

    def _session_start_extras(self) -> dict[str, Any]:
        return {
            "lead_canonical": self.reference_canonical,
            "lead_drd2": self.lead_drd2,
            "similarity_threshold": self.similarity_threshold,
            "target_drd2_min": self.target_drd2_min,
        }

    def _empty_result(self) -> dict[str, Any]:
        return {
            "valid": False,
            "canonical_smiles": None,
            "duplicate": False,
            "equals_lead": False,
            "tanimoto": None,
            "similar_ok": False,
            "drd2": None,
            "oracle_score": None,
            "drd2_delta": None,
            "improved_ok": False,
            "success": False,
            "error": None,
        }

    def _evaluate_valid(self, canonical: str, mol: Any) -> dict[str, Any]:
        fp = self._rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            self.fingerprint_radius,
            nBits=self.fingerprint_nbits,
        )
        tanimoto = float(self._DataStructs.TanimotoSimilarity(self.lead_fp, fp))
        drd2 = float(_get_tdc_oracle("DRD2")(canonical))
        equals_lead = canonical == self.reference_canonical
        similar_ok = tanimoto >= self.similarity_threshold
        improved_ok = drd2 >= self.target_drd2_min
        success = similar_ok and improved_ok and not equals_lead

        return {
            "valid": True,
            "canonical_smiles": canonical,
            "duplicate": False,
            "equals_lead": equals_lead,
            "tanimoto": tanimoto,
            "similar_ok": similar_ok,
            "drd2": drd2,
            "oracle_score": drd2,
            "drd2_delta": drd2 - self.lead_drd2,
            "improved_ok": improved_ok,
            "success": success,
            "error": None,
        }

    # ------------------------------------------------------------------
    # Free helpers (no budget cost)
    # ------------------------------------------------------------------
    def fingerprint(self, smiles: str):
        mol = self._Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self._rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            self.fingerprint_radius,
            nBits=self.fingerprint_nbits,
        )

    def tanimoto_to_lead(self, smiles: str) -> float | None:
        fp = self.fingerprint(smiles)
        if fp is None:
            return None
        return float(self._DataStructs.TanimotoSimilarity(self.lead_fp, fp))
