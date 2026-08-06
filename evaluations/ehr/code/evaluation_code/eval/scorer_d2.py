"""D2 Cohort Construction scorer.

Evaluates inclusion/exclusion criteria, index event, cohort size, and attrition.
Uses regex matching for criteria and optional golden cohort ID comparison.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from eval.normalizers import normalize_text
from eval.schemas import CohortAnswer, CohortGolden, DimensionResult, FieldScore


def _criteria_match(agent_criteria: list[str], golden_patterns: list[str]) -> tuple[int, int, list[str]]:
    """Check how many golden criteria patterns are covered by agent criteria.

    Returns (matched, total, missing_patterns).
    """
    agent_text = " ".join(agent_criteria).lower()
    agent_text_norm = normalize_text(" ".join(agent_criteria))
    matched = 0
    missing = []

    for pattern in golden_patterns:
        alternatives = [alt.strip() for alt in pattern.split("|")]
        found = False
        for alt in alternatives:
            alt_lower = alt.lower()
            alt_escaped = re.escape(alt_lower)
            if re.search(alt_escaped, agent_text) or re.search(alt_escaped, agent_text_norm):
                found = True
                break
            # Also check expanded form
            alt_norm = normalize_text(alt)
            if re.search(re.escape(alt_norm), agent_text_norm):
                found = True
                break
        if found:
            matched += 1
        else:
            missing.append(pattern)

    return matched, len(golden_patterns), missing


def _score_index_event(agent_value: str, canonical: str, aliases: list[str]) -> FieldScore:
    """Score the index event against canonical + aliases."""
    if not agent_value.strip():
        return FieldScore(score=0.0, reason="index_event is empty")

    agent_norm = normalize_text(agent_value)
    canonical_norm = normalize_text(canonical)

    if canonical_norm in agent_norm or agent_norm in canonical_norm:
        return FieldScore(score=1.0, reason=f"matches canonical: {canonical}")

    for alias in aliases:
        alias_norm = normalize_text(alias)
        if alias_norm in agent_norm or agent_norm in alias_norm:
            return FieldScore(score=1.0, reason=f"matches alias: {alias}")

    return FieldScore(score=0.0, reason=f"no match for: {agent_value}")


def _load_cohort_ids(path: Path, id_col: str = "hadm_id") -> set[int] | None:
    """Load cohort IDs from a CSV or Parquet file. Returns None if not loadable."""
    if not path.exists():
        return None
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        if id_col not in df.columns:
            # Try common alternatives
            for alt in ("hadm_id", "HADM_ID", "subject_id", "SUBJECT_ID"):
                if alt in df.columns:
                    id_col = alt
                    break
            else:
                return None
        return set(df[id_col].dropna().astype(int))
    except Exception:
        return None


def _compute_f1(agent_ids: set[int], golden_ids: set[int]) -> tuple[float, float, float]:
    """Compute precision, recall, F1 between agent and golden cohort IDs."""
    if not agent_ids or not golden_ids:
        return 0.0, 0.0, 0.0
    tp = len(agent_ids & golden_ids)
    precision = tp / len(agent_ids) if agent_ids else 0.0
    recall = tp / len(golden_ids) if golden_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _score_cohort_size(
    agent_n_final: int,
    golden_csv: str,
    tolerance: float,
    project_root: str | Path | None = None,
    agent_cohort_file: str = "",
    run_dir: str | Path | None = None,
) -> FieldScore:
    """Score cohort against golden using patient-level F1 when possible, count otherwise."""
    # Resolve golden CSV
    golden_path = Path(golden_csv) if golden_csv else None
    if golden_path and project_root:
        golden_path = Path(project_root) / golden_csv

    if not golden_path or not golden_path.exists():
        if agent_n_final > 0:
            return FieldScore(score=0.5, reason=f"n_final={agent_n_final}, no golden to compare")
        return FieldScore(score=0.0, reason="n_final=0 and no golden cohort")

    golden_ids = _load_cohort_ids(golden_path)
    golden_n = len(golden_ids) if golden_ids else len(pd.read_csv(golden_path))

    if agent_n_final == 0:
        return FieldScore(score=0.0, reason=f"n_final=0, golden has {golden_n}")

    # Try patient-level F1 scoring
    agent_ids = None
    if agent_cohort_file and run_dir:
        agent_path = Path(run_dir) / Path(agent_cohort_file).name
        agent_ids = _load_cohort_ids(agent_path)

    if agent_ids is not None and golden_ids is not None:
        precision, recall, f1 = _compute_f1(agent_ids, golden_ids)
        detail = f"F1={f1:.3f} (P={precision:.3f}, R={recall:.3f}, agent={len(agent_ids)}, golden={len(golden_ids)})"
        if f1 >= 0.7:
            return FieldScore(score=1.0, reason=detail)
        elif f1 >= 0.4:
            return FieldScore(score=0.5, reason=detail)
        else:
            return FieldScore(score=0.0, reason=detail)

    # Fallback: count-based comparison
    ratio = abs(agent_n_final - golden_n) / golden_n
    if ratio <= tolerance:
        return FieldScore(
            score=1.0,
            reason=f"n_final={agent_n_final} within {tolerance:.0%} of golden {golden_n} (diff={ratio:.1%}) [count-based, no cohort file]",
        )
    elif ratio <= tolerance * 2:
        return FieldScore(
            score=0.5,
            reason=f"n_final={agent_n_final} partially matches golden {golden_n} (diff={ratio:.1%}) [count-based]",
        )
    else:
        return FieldScore(
            score=0.0,
            reason=f"n_final={agent_n_final} too far from golden {golden_n} (diff={ratio:.1%}) [count-based]",
        )


D2_WEIGHT = 15


def score_d2(
    agent_cohort: CohortAnswer,
    golden_cohort: CohortGolden,
    project_root: str | Path | None = None,
    agent_cohort_file: str = "",
    run_dir: str | Path | None = None,
) -> DimensionResult:
    """Score D2 Cohort Construction dimension.

    Args:
        agent_cohort_file: Filename of agent's cohort output (e.g. "cohort.parquet").
        run_dir: Directory where agent run artifacts are stored.
    """
    field_scores: dict[str, FieldScore] = {}

    # Inclusion criteria
    if agent_cohort.inclusion_criteria:
        matched, total, missing = _criteria_match(
            agent_cohort.inclusion_criteria, golden_cohort.inclusion_criteria_required
        )
        ratio = matched / total if total > 0 else 0.0
        if ratio >= 0.85:
            score = 1.0
        elif ratio >= 0.5:
            score = 0.5
        else:
            score = 0.0
        field_scores["inclusion_criteria"] = FieldScore(
            score=score,
            reason=f"{matched}/{total} required criteria matched" + (f", missing: {missing}" if missing else ""),
        )
    else:
        field_scores["inclusion_criteria"] = FieldScore(score=0.0, reason="no inclusion criteria provided")

    # Exclusion criteria
    if agent_cohort.exclusion_criteria and golden_cohort.exclusion_criteria_expected:
        matched, total, missing = _criteria_match(
            agent_cohort.exclusion_criteria, golden_cohort.exclusion_criteria_expected
        )
        ratio = matched / total if total > 0 else 0.0
        if ratio >= 0.85:
            score = 1.0
        elif ratio >= 0.5:
            score = 0.5
        else:
            score = 0.0
        field_scores["exclusion_criteria"] = FieldScore(
            score=score,
            reason=f"{matched}/{total} expected exclusion criteria matched" + (f", missing: {missing}" if missing else ""),
        )
    elif not golden_cohort.exclusion_criteria_expected:
        field_scores["exclusion_criteria"] = FieldScore(score=1.0, reason="no exclusion criteria required")
    else:
        field_scores["exclusion_criteria"] = FieldScore(score=0.0, reason="no exclusion criteria provided")

    # Index event
    field_scores["index_event"] = _score_index_event(
        agent_cohort.index_event,
        golden_cohort.index_event_canonical,
        golden_cohort.index_event_aliases,
    )

    # Cohort size / F1
    field_scores["n_final"] = _score_cohort_size(
        agent_cohort.n_final,
        golden_cohort.golden_cohort_csv,
        golden_cohort.n_final_tolerance,
        project_root,
        agent_cohort_file=agent_cohort_file,
        run_dir=run_dir,
    )

    # Aggregate
    scores = [fs.score for fs in field_scores.values()]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    if mean_score >= 0.85:
        grade = 2
    elif mean_score >= 0.5:
        grade = 1
    else:
        grade = 0

    return DimensionResult(
        dimension="D2",
        grade=grade,
        mean_field_score=round(mean_score, 4),
        weighted_score=round(grade * D2_WEIGHT, 1),
        field_scores=field_scores,
    )
