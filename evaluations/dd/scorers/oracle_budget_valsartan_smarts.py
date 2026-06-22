"""Budget-aware oracle wrapper for the Valsartan_SMARTS code-optimizer task.

This task is pure goal-directed generation: no lead molecule, no similarity
constraint. The oracle is the TDC Oracle("Valsartan_SMARTS"), a [0, 1]
geometric mean of:

  1. SMARTS match (binary, required): motif
     ``CN(C=O)Cc1ccc(c2ccccc2)cc1``
  2. logP Gaussian centered at 2.02, σ 0.20
  3. TPSA Gaussian centered at 77.04, σ 5.0
  4. Bertz Gaussian centered at 896.38, σ 30.0

``reference_smiles`` is Valsartan itself. It is used only to populate the
``equals_reference`` flag and ``reference_oracle_score`` (Valsartan scores
≈ 0 because its logP / TPSA / Bertz are far from the Sitagliptin targets).
"""

from __future__ import annotations

from typing import Any

from scorers.oracle_budget import BudgetedOracleBase, OracleBudgetExceededError
from scorers.valsartan_smarts_scorer import VALSARTAN_SMARTS, _get_tdc_oracle


class BudgetedValsartanSMARTSOracle(BudgetedOracleBase):
    """Budget-counting Valsartan SMARTS oracle.

    Parameters
    ----------
    reference_smiles :
        Valsartan SMILES — used for the ``equals_reference`` check only.
    budget :
        Hard cap on :meth:`score` calls.
    smarts_pattern :
        Required substructure SMARTS. Defaults to the Valsartan
        amide-biphenyl motif (GuacaMol Valsartan_SMARTS benchmark).
    target_score_min :
        Minimum Valsartan_SMARTS oracle score for ``improved_ok`` /
        ``success``. Defaults to 0.5.
    """

    STREAM_EXTRA_FIELDS = (
        "oracle_score",
        "smarts_ok",
        "logp",
        "tpsa",
        "bertz",
        "improved_ok",
    )

    def __init__(
        self,
        reference_smiles: str,
        budget: int,
        smarts_pattern: str = VALSARTAN_SMARTS,
        target_score_min: float = 0.5,
        stream_path: str | None = None,
    ) -> None:
        self.smarts_pattern = str(smarts_pattern)
        self.target_score_min = float(target_score_min)
        super().__init__(
            reference_smiles=reference_smiles,
            budget=budget,
            stream_path=stream_path,
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _init_reference(self, canonical: str, mol: Any) -> None:
        from rdkit.Chem import AllChem  # noqa: F401  (ensures rdkit is loaded)

        smarts_mol = self._Chem.MolFromSmarts(self.smarts_pattern)
        if smarts_mol is None:
            raise ValueError(f"Invalid SMARTS pattern: {self.smarts_pattern}")
        self._smarts_mol = smarts_mol

        self.reference_oracle_score = float(
            _get_tdc_oracle("Valsartan_SMARTS")(canonical)
        )

    def _session_start_extras(self) -> dict[str, Any]:
        return {
            "reference_oracle_score": self.reference_oracle_score,
            "smarts_pattern": self.smarts_pattern,
            "target_score_min": self.target_score_min,
        }

    def _empty_result(self) -> dict[str, Any]:
        return {
            "valid": False,
            "canonical_smiles": None,
            "duplicate": False,
            "equals_reference": False,
            "oracle_score": None,
            "smarts_ok": False,
            "logp": None,
            "tpsa": None,
            "bertz": None,
            "improved_ok": False,
            "success": False,
            "error": None,
        }

    def _evaluate_valid(self, canonical: str, mol: Any) -> dict[str, Any]:
        from rdkit.Chem import Crippen, GraphDescriptors
        from rdkit.Chem.rdMolDescriptors import CalcTPSA

        smarts_ok = mol.HasSubstructMatch(self._smarts_mol)
        logp = float(Crippen.MolLogP(mol))
        tpsa = float(CalcTPSA(mol))
        bertz = float(GraphDescriptors.BertzCT(mol))

        oracle_score = float(_get_tdc_oracle("Valsartan_SMARTS")(canonical))

        equals_reference = canonical == self.reference_canonical
        improved_ok = oracle_score >= self.target_score_min
        success = improved_ok and smarts_ok and not equals_reference

        return {
            "valid": True,
            "canonical_smiles": canonical,
            "duplicate": False,
            "equals_reference": equals_reference,
            "oracle_score": oracle_score,
            "smarts_ok": smarts_ok,
            "logp": logp,
            "tpsa": tpsa,
            "bertz": bertz,
            "improved_ok": improved_ok,
            "success": success,
            "error": None,
        }
