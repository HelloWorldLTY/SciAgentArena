"""Budget-aware oracle wrappers for code-optimizer tasks.

This module provides

- :class:`OracleBudgetExceededError`, :class:`BudgetedOracleBase`,
  and the streaming-sink env var :data:`STREAM_ENV_VAR`, shared by every
  budgeted oracle in this package;
- :class:`BudgetedPenalizedLogPOracle`, the concrete oracle used by the
  ``des_01_code_optimizer_penalized_logp`` task.

Design contract — every budgeted oracle is bound to a single reference
molecule (the "lead" in similarity-constrained tasks) and counts every
call to :meth:`score` against a hard budget, regardless of whether the
candidate was valid, invalid, or a duplicate.  Each call is appended as
one JSON line to an optional streaming sink so that ``batch_runner`` can
recover partial results when the subprocess is killed by the wall-clock
timeout.
"""

from __future__ import annotations

import json
import os
from typing import Any, TextIO

from scorers.penalized_logp_scorer import (
    canonicalize_smiles,
    compute_penalized_logp_breakdown,
)

# Env var pointing to a JSONL file that the oracle appends to line-buffered
# after every score() call.  ``batch_runner`` uses this to synthesize the
# final submission from the optimizer's actual evaluations.
STREAM_ENV_VAR = "AGENT4S_OPTIMIZER_STREAM"


class OracleBudgetExceededError(RuntimeError):
    """Raised when the optimizer attempts to query the oracle past its budget."""


class BudgetedOracleBase:
    """Shared machinery for all budgeted oracles.

    Subclass contract
    -----------------
    1. Override :meth:`_init_reference` to precompute reference-molecule
       data (free, no budget cost) and set any subclass-specific public
       attributes (e.g. ``lead_fp``, ``lead_drd2``).
    2. Override :meth:`_session_start_extras` to return a dict of extra
       fields for the streaming JSONL ``session_start`` header.
    3. Override :meth:`_empty_result` to return the default dict for
       invalid / empty SMILES — all task-specific fields set to the
       appropriate null (``None`` / ``False``).
    4. Override :meth:`_evaluate_valid` to compute the result dict for a
       successfully parsed candidate.
    5. Declare :attr:`STREAM_EXTRA_FIELDS` — tuple of result-dict keys to
       emit on every ``score`` call, in addition to the common fields.
    """

    STREAM_EXTRA_FIELDS: tuple[str, ...] = ()

    _COMMON_STREAM_FIELDS: tuple[str, ...] = (
        "canonical_smiles",
        "valid",
        "duplicate",
        "success",
        "calls_used",
        "budget",
        "error",
    )

    def __init__(
        self,
        reference_smiles: str,
        budget: int,
        *,
        stream_path: str | None = None,
    ) -> None:
        # Set before any raise so __del__ on partially-constructed objects
        # doesn't blow up with AttributeError.
        self._stream_fp: TextIO | None = None

        try:
            from rdkit import Chem
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"rdkit not available: {exc}") from exc
        self._Chem = Chem

        raw = "" if reference_smiles is None else str(reference_smiles).strip()
        if not raw:
            raise ValueError("reference_smiles must be a non-empty SMILES string")

        canonical, err = canonicalize_smiles(raw)
        if canonical is None:
            raise ValueError(f"reference_smiles is invalid: {err}")
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            raise ValueError(f"reference_smiles is invalid: {reference_smiles}")

        self.reference_smiles = reference_smiles
        self.reference_canonical = canonical

        self._init_reference(canonical, mol)

        self.budget = int(budget)
        self.calls_used = 0
        self._cache: dict[str, dict[str, Any]] = {}

        resolved = stream_path or os.environ.get(STREAM_ENV_VAR)
        self.stream_path = resolved
        if resolved:
            try:
                self._stream_fp = open(resolved, "a", buffering=1)
                header = {
                    "event": "session_start",
                    "reference_canonical": self.reference_canonical,
                    "budget": self.budget,
                    **self._session_start_extras(),
                }
                self._stream_fp.write(json.dumps(header) + "\n")
                self._stream_fp.flush()
            except Exception:
                self._stream_fp = None

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def _init_reference(self, canonical: str, mol: Any) -> None:
        raise NotImplementedError

    def _session_start_extras(self) -> dict[str, Any]:
        return {}

    def _empty_result(self) -> dict[str, Any]:
        raise NotImplementedError

    def _evaluate_valid(self, canonical: str, mol: Any) -> dict[str, Any]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def calls_remaining(self) -> int:
        return max(0, self.budget - self.calls_used)

    def score(self, smiles: str) -> dict[str, Any]:
        if self.calls_used >= self.budget:
            raise OracleBudgetExceededError(
                f"Oracle budget exhausted ({self.budget} calls used)."
            )
        self.calls_used += 1

        raw = "" if smiles is None else str(smiles).strip()

        def _finalize(result: dict[str, Any]) -> dict[str, Any]:
            result["calls_used"] = self.calls_used
            result["budget"] = self.budget
            self._emit(result, raw)
            return result

        if not raw:
            result = self._empty_result()
            result["error"] = "empty smiles"
            return _finalize(result)

        canonical, err = canonicalize_smiles(raw)
        if canonical is None:
            result = self._empty_result()
            result["error"] = err or "sanitize_failed"
            return _finalize(result)

        if canonical in self._cache:
            cached = dict(self._cache[canonical])
            cached["duplicate"] = True
            return _finalize(cached)

        mol = self._Chem.MolFromSmiles(canonical)
        if mol is None:
            result = self._empty_result()
            result["canonical_smiles"] = canonical
            result["error"] = "rdkit could not parse canonical smiles"
            return _finalize(result)

        try:
            result = self._evaluate_valid(canonical, mol)
        except Exception as exc:
            result = self._empty_result()
            result["canonical_smiles"] = canonical
            result["error"] = f"evaluate_valid failed: {exc}"
            return _finalize(result)

        self._cache[canonical] = dict(result)
        return _finalize(result)

    # ------------------------------------------------------------------
    # Streaming sink
    # ------------------------------------------------------------------
    def _emit(self, result: dict[str, Any], raw_smiles: str) -> None:
        if self._stream_fp is None:
            return
        record: dict[str, Any] = {"event": "score", "smiles_raw": raw_smiles}
        for field in self._COMMON_STREAM_FIELDS:
            record[field] = result.get(field)
        for field in self.STREAM_EXTRA_FIELDS:
            record[field] = result.get(field)
        try:
            self._stream_fp.write(json.dumps(record) + "\n")
            self._stream_fp.flush()
        except Exception:
            pass

    def close(self) -> None:
        fp = getattr(self, "_stream_fp", None)
        if fp is not None:
            try:
                fp.close()
            finally:
                self._stream_fp = None

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


class BudgetedPenalizedLogPOracle(BudgetedOracleBase):
    """Budget-counting penalized-logP oracle bound to a single lead molecule.

    Parameters
    ----------
    lead_smiles :
        SMILES of the lead molecule. Must be RDKit-parsable.
    budget :
        Maximum number of :meth:`score` calls allowed. Each call — even
        for invalid or duplicate SMILES — consumes one unit of budget.
    similarity_threshold :
        Tanimoto cutoff used to populate the ``similar_ok`` flag on each
        result. Defaults to 0.4.
    fingerprint_radius, fingerprint_nbits :
        Morgan fingerprint parameters used for similarity and exposed so
        the optimizer can reproduce the scorer's similarity metric exactly.
    min_improvement_delta :
        Minimum penalized_logP improvement (candidate minus lead) required
        to flag ``improved_ok``. Defaults to 0.0 (any strict improvement);
        task JSONs set this to 1.0 so that sub-noise gains do not count as
        success.
    """

    STREAM_EXTRA_FIELDS = (
        "tanimoto",
        "similar_ok",
        "penalized_logp",
        "penalized_logp_delta",
        "improved_ok",
    )

    def __init__(
        self,
        lead_smiles: str,
        budget: int,
        similarity_threshold: float = 0.4,
        fingerprint_radius: int = 2,
        fingerprint_nbits: int = 2048,
        min_improvement_delta: float = 0.0,
        stream_path: str | None = None,
    ) -> None:
        self.similarity_threshold = float(similarity_threshold)
        self.fingerprint_radius = int(fingerprint_radius)
        self.fingerprint_nbits = int(fingerprint_nbits)
        self.min_improvement_delta = float(min_improvement_delta)
        super().__init__(
            reference_smiles=lead_smiles,
            budget=budget,
            stream_path=stream_path,
        )
        # Historic aliases used by task prompts and agent stubs.
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

        breakdown = compute_penalized_logp_breakdown(canonical)
        if breakdown is None:
            raise ValueError(f"Could not compute penalized logP for lead: {canonical}")
        self.lead_penalized_logp = float(breakdown["penalized_logp"])
        self.lead_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            self.fingerprint_radius,
            nBits=self.fingerprint_nbits,
        )

    def _session_start_extras(self) -> dict[str, Any]:
        return {
            "lead_canonical": self.reference_canonical,
            "lead_penalized_logp": self.lead_penalized_logp,
            "similarity_threshold": self.similarity_threshold,
            "min_improvement_delta": self.min_improvement_delta,
        }

    def _empty_result(self) -> dict[str, Any]:
        return {
            "valid": False,
            "canonical_smiles": None,
            "duplicate": False,
            "equals_lead": False,
            "tanimoto": None,
            "similar_ok": False,
            "penalized_logp": None,
            "penalized_logp_delta": None,
            "improved_ok": False,
            "success": False,
            "components": None,
            "error": None,
        }

    def _evaluate_valid(self, canonical: str, mol: Any) -> dict[str, Any]:
        breakdown = compute_penalized_logp_breakdown(canonical)
        if breakdown is None:
            result = self._empty_result()
            result["canonical_smiles"] = canonical
            result["error"] = "failed to compute penalized logP"
            return result

        fp = self._rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            self.fingerprint_radius,
            nBits=self.fingerprint_nbits,
        )
        tanimoto = float(self._DataStructs.TanimotoSimilarity(self.lead_fp, fp))
        penlogp = float(breakdown["penalized_logp"])
        equals_lead = canonical == self.reference_canonical
        similar_ok = tanimoto >= self.similarity_threshold
        improved_ok = penlogp > self.lead_penalized_logp + self.min_improvement_delta
        success = similar_ok and improved_ok and not equals_lead

        return {
            "valid": True,
            "canonical_smiles": canonical,
            "duplicate": False,
            "equals_lead": equals_lead,
            "tanimoto": tanimoto,
            "similar_ok": similar_ok,
            "penalized_logp": penlogp,
            "penalized_logp_delta": penlogp - self.lead_penalized_logp,
            "improved_ok": improved_ok,
            "success": success,
            "components": breakdown,
            "error": None,
        }

    # ------------------------------------------------------------------
    # Free helpers (no budget cost)
    # ------------------------------------------------------------------
    def fingerprint(self, smiles: str):
        """Morgan fingerprint of *smiles*, or None if invalid. Free."""
        mol = self._Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self._rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            self.fingerprint_radius,
            nBits=self.fingerprint_nbits,
        )

    def tanimoto_to_lead(self, smiles: str) -> float | None:
        """Tanimoto(lead, smiles), or None if invalid. Free."""
        fp = self.fingerprint(smiles)
        if fp is None:
            return None
        return float(self._DataStructs.TanimotoSimilarity(self.lead_fp, fp))
