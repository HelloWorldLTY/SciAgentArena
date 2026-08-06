"""D5 Causal Method Selection and Execution scorer.

Evaluates whether the agent chose a defensible estimator and specified it
consistently. Checks estimator choice against task-allowed set, and verifies
that an analysis script artifact is referenced.
"""

from __future__ import annotations

from pathlib import Path

from eval.schemas import (
    DimensionResult,
    FieldScore,
    MethodAnswer,
    MethodGolden,
    ArtifactsAnswer,
)


def _score_estimator_choice(
    primary_estimator: str,
    golden: MethodGolden,
) -> FieldScore:
    """Score the primary estimator choice against allowed/partial/zero lists."""
    est = primary_estimator.strip().lower()

    if not est or est in ("n/a", "na", "none", ""):
        return FieldScore(score=0.0, reason="no estimator declared")

    allowed = [e.lower() for e in golden.allowed_estimators]
    partial = [e.lower() for e in golden.partial_credit_estimators]
    zero = [e.lower() for e in golden.zero_credit_estimators]

    if est in allowed:
        return FieldScore(score=1.0, reason=f"estimator '{primary_estimator}' is in allowed set")
    if est in partial:
        return FieldScore(score=0.5, reason=f"estimator '{primary_estimator}' gets partial credit")
    if est in zero:
        return FieldScore(score=0.0, reason=f"estimator '{primary_estimator}' is zero-credit for this task")

    # Unknown estimator — partial credit if it looks plausible
    return FieldScore(score=0.5, reason=f"estimator '{primary_estimator}' not in predefined sets")


def _score_estimation_family(
    family: str | None,
    golden: MethodGolden,
) -> FieldScore:
    """Score the estimation family."""
    if family is None or not str(family).strip():
        return FieldScore(score=0.0, reason="no estimation family declared")

    fam = str(family).strip().lower()
    allowed = [f.lower() for f in golden.allowed_families]

    if fam in allowed:
        return FieldScore(score=1.0, reason=f"family '{family}' is valid")

    return FieldScore(score=0.0, reason=f"family '{family}' not in allowed set")


def _score_adjustment_set(adjustment_set: list[str]) -> FieldScore:
    """Score whether an adjustment set is declared."""
    if not adjustment_set:
        return FieldScore(score=0.0, reason="no adjustment set declared")
    if len(adjustment_set) < 3:
        return FieldScore(score=0.5, reason=f"only {len(adjustment_set)} variables in adjustment set")
    return FieldScore(score=1.0, reason=f"{len(adjustment_set)} variables in adjustment set")


def _score_models_specified(method: MethodAnswer) -> FieldScore:
    """Score whether propensity and outcome models are specified."""
    has_propensity = bool(method.propensity_model.strip()) and method.propensity_model.strip().lower() not in ("n/a", "na", "none")
    has_outcome = bool(method.outcome_model.strip()) and method.outcome_model.strip().lower() not in ("n/a", "na", "none")

    if has_propensity and has_outcome:
        return FieldScore(score=1.0, reason="both propensity and outcome models specified")
    if has_propensity or has_outcome:
        return FieldScore(score=0.5, reason="only one model specified")
    return FieldScore(score=0.0, reason="no models specified")


def _score_analysis_script(
    artifacts: ArtifactsAnswer,
    run_dir: str | Path | None = None,
) -> FieldScore:
    """Score whether analysis script exists."""
    script = artifacts.analysis_script.strip()
    if not script:
        return FieldScore(score=0.0, reason="no analysis script referenced")

    if run_dir:
        script_path = Path(run_dir) / script
        if script_path.exists():
            return FieldScore(score=1.0, reason=f"analysis script exists: {script}")
        return FieldScore(score=0.5, reason=f"analysis script referenced but not found: {script}")

    return FieldScore(score=0.5, reason=f"analysis script referenced: {script} (existence not checked)")


D5_WEIGHT = 15


def score_d5(
    agent_method: MethodAnswer | None,
    agent_artifacts: ArtifactsAnswer,
    golden_method: MethodGolden,
    run_dir: str | Path | None = None,
) -> DimensionResult:
    """Score D5 Causal Method Selection dimension."""
    field_scores: dict[str, FieldScore] = {}

    if agent_method is None:
        return DimensionResult(
            dimension="D5",
            grade=0,
            mean_field_score=0.0,
            weighted_score=0.0,
            field_scores={"method": FieldScore(score=0.0, reason="no method block")},
        )

    field_scores["primary_estimator"] = _score_estimator_choice(
        agent_method.primary_estimator, golden_method
    )
    field_scores["estimation_family"] = _score_estimation_family(
        agent_method.estimation_family, golden_method
    )
    field_scores["adjustment_set"] = _score_adjustment_set(agent_method.adjustment_set)
    field_scores["models_specified"] = _score_models_specified(agent_method)
    field_scores["analysis_script"] = _score_analysis_script(agent_artifacts, run_dir)

    scores = [fs.score for fs in field_scores.values()]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    if mean_score >= 0.85:
        grade = 2
    elif mean_score >= 0.5:
        grade = 1
    else:
        grade = 0

    return DimensionResult(
        dimension="D5",
        grade=grade,
        mean_field_score=round(mean_score, 4),
        weighted_score=round(grade * D5_WEIGHT, 1),
        field_scores=field_scores,
    )
