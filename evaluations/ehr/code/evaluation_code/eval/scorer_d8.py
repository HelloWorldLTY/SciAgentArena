"""D8 Robustness, Failure Handling, and Abstention scorer.

Evaluates whether the agent provides robustness checks, handles failures
properly, and abstains correctly when appropriate.
"""

from __future__ import annotations

from eval.schemas import (
    DimensionResult,
    FinalDecisionAnswer,
    FieldScore,
    ResultAnswer,
    RobustnessAnswer,
    RobustnessGolden,
)


def _score_alternative_estimators(
    alternatives: list,
    min_required: int,
) -> FieldScore:
    """Score alternative estimator reporting."""
    n = len(alternatives)
    if n >= min_required:
        return FieldScore(score=1.0, reason=f"{n} alternative estimators reported (>= {min_required})")
    if n > 0:
        return FieldScore(score=0.5, reason=f"{n} alternative estimators (< {min_required} expected)")
    return FieldScore(score=0.0, reason="no alternative estimators reported")


def _score_sensitivity_analyses(
    analyses: list,
    min_required: int,
) -> FieldScore:
    """Score sensitivity analysis reporting."""
    n = len(analyses)
    if min_required == 0:
        if n > 0:
            return FieldScore(score=1.0, reason=f"{n} sensitivity analyses (bonus)")
        return FieldScore(score=1.0, reason="sensitivity analyses not required")
    if n >= min_required:
        return FieldScore(score=1.0, reason=f"{n} sensitivity analyses (>= {min_required})")
    if n > 0:
        return FieldScore(score=0.5, reason=f"{n} sensitivity analyses (< {min_required})")
    return FieldScore(score=0.0, reason="no sensitivity analyses reported")


def _score_final_decision(
    decision: FinalDecisionAnswer,
    result: ResultAnswer,
    golden: RobustnessGolden,
) -> FieldScore:
    """Score the final decision for internal consistency."""
    status = decision.status.strip().lower()

    if status == "success":
        # Check that result actually has values
        effect = result.primary_effect
        has_estimate = effect.estimate != 0.0 or effect.ci_95 != [0.0, 0.0]
        if has_estimate:
            return FieldScore(score=1.0, reason="status=success with actual results")
        return FieldScore(
            score=0.0,
            reason="status=success but no actual estimate — should have abstained",
        )

    if status == "abstain":
        if golden.abstain_requires_reason and not decision.abstain_reason.strip():
            return FieldScore(score=0.5, reason="abstained but no reason given")
        if decision.abstain_reason.strip():
            return FieldScore(score=1.0, reason=f"justified abstention: {decision.abstain_reason[:80]}")
        return FieldScore(score=0.5, reason="abstained without explanation")

    if status == "error":
        return FieldScore(score=0.5, reason="status=error — agent reported an error condition")

    return FieldScore(score=0.0, reason=f"unknown status: {status}")


def _score_confidence(confidence: str) -> FieldScore:
    """Score whether confidence is declared."""
    conf = confidence.strip().lower()
    if conf in ("high", "medium", "low"):
        return FieldScore(score=1.0, reason=f"confidence declared: {confidence}")
    return FieldScore(score=0.0, reason=f"invalid confidence: {confidence}")


D8_WEIGHT = 10


def score_d8(
    agent_robustness: RobustnessAnswer,
    agent_decision: FinalDecisionAnswer,
    agent_result: ResultAnswer,
    golden_robustness: RobustnessGolden,
) -> DimensionResult:
    """Score D8 Robustness and Abstention dimension."""
    field_scores: dict[str, FieldScore] = {}

    status = agent_decision.status.strip().lower()

    field_scores["final_decision"] = _score_final_decision(
        agent_decision, agent_result, golden_robustness
    )
    field_scores["confidence"] = _score_confidence(agent_decision.confidence)

    if status == "abstain":
        # For abstention, don't require robustness checks — score is based
        # on whether abstention is justified
        scores = [fs.score for fs in field_scores.values()]
        mean_score = sum(scores) / len(scores) if scores else 0.0
    else:
        # For success, also check robustness
        field_scores["alternative_estimators"] = _score_alternative_estimators(
            agent_robustness.alternative_estimators,
            golden_robustness.min_alternative_estimators,
        )
        field_scores["sensitivity_analyses"] = _score_sensitivity_analyses(
            agent_robustness.sensitivity_analyses,
            golden_robustness.min_sensitivity_analyses,
        )

        scores = [fs.score for fs in field_scores.values()]
        mean_score = sum(scores) / len(scores) if scores else 0.0

    if mean_score >= 0.85:
        grade = 2
    elif mean_score >= 0.5:
        grade = 1
    else:
        grade = 0

    return DimensionResult(
        dimension="D8",
        grade=grade,
        mean_field_score=round(mean_score, 4),
        weighted_score=round(grade * D8_WEIGHT, 1),
        field_scores=field_scores,
    )
