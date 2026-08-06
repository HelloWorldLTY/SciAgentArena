"""D7 Result Reporting and Uncertainty scorer.

Evaluates whether the agent produces a valid causal estimate with
uncertainty and clear interpretation, AND whether the reported estimate
is internally consistent with the agent's own cohort data.
"""

from __future__ import annotations

from eval.estimate_verifier import EstimateReport
from eval.schemas import (
    DimensionResult,
    FinalDecisionAnswer,
    FieldScore,
    ResultAnswer,
    ResultGolden,
)


def _score_measure(measure: str, valid_measures: list[str]) -> FieldScore:
    """Score the effect measure."""
    if not measure.strip() or measure.strip().lower() in ("n/a", "na", "none"):
        return FieldScore(score=0.0, reason="no effect measure declared")

    measure_lower = measure.strip().lower()
    valid_lower = [m.lower() for m in valid_measures]

    if measure_lower in valid_lower:
        return FieldScore(score=1.0, reason=f"measure '{measure}' is valid")

    return FieldScore(score=0.5, reason=f"measure '{measure}' not in expected set but may be valid")


def _score_estimate(estimate: float, ci_95: list[float]) -> FieldScore:
    """Score the numeric estimate."""
    if estimate == 0.0 and ci_95 == [0.0, 0.0]:
        return FieldScore(score=0.0, reason="estimate and CI are both zero (likely not computed)")

    if estimate != 0.0:
        return FieldScore(score=1.0, reason=f"estimate={estimate:.4f}")

    return FieldScore(score=0.5, reason="estimate is zero but CI may indicate a result")


def _score_ci(ci_95: list[float], requires_ci: bool) -> FieldScore:
    """Score the confidence interval."""
    if ci_95 == [0.0, 0.0]:
        if requires_ci:
            return FieldScore(score=0.0, reason="CI not reported (required)")
        return FieldScore(score=0.5, reason="CI not reported (optional)")

    if len(ci_95) != 2:
        return FieldScore(score=0.0, reason=f"CI has {len(ci_95)} elements, expected 2")

    if ci_95[0] > ci_95[1]:
        return FieldScore(score=0.0, reason=f"CI lower > upper: {ci_95}")

    return FieldScore(score=1.0, reason=f"CI=[{ci_95[0]:.4f}, {ci_95[1]:.4f}]")


def _score_direction(direction: str, valid_directions: list[str]) -> FieldScore:
    """Score the direction of effect."""
    if not direction.strip() or direction.strip().lower() in ("n/a", "na", "none"):
        return FieldScore(score=0.0, reason="no direction declared")

    dir_lower = direction.strip().lower()
    valid_lower = [d.lower() for d in valid_directions]

    if dir_lower in valid_lower:
        return FieldScore(score=1.0, reason=f"direction '{direction}' is valid")

    return FieldScore(score=0.0, reason=f"direction '{direction}' not in valid set")


def _score_interpretation(interpretation: str) -> FieldScore:
    """Score the interpretation text."""
    if not interpretation.strip() or interpretation.strip().lower() in ("n/a", "na", "none"):
        return FieldScore(score=0.0, reason="no interpretation provided")

    words = interpretation.split()
    if len(words) < 5:
        return FieldScore(score=0.5, reason="interpretation is very brief")

    return FieldScore(score=1.0, reason=f"interpretation provided ({len(words)} words)")


def _score_estimate_verification(
    agent_estimate: float,
    agent_direction: str,
    estimate_report: EstimateReport | None,
    agent_method: str,
) -> FieldScore:
    """Score internal consistency between the agent's reported estimate and
    an independent recomputation from the agent's own cohort file.

    Three checks, in order of importance:
    1. Sign consistency — does the recomputed ATE agree on direction?
    2. Magnitude plausibility — is the agent's estimate in the same ballpark?
    3. Exact match (only for IPW-family methods) — is it within 10% tolerance?

    If no cohort file exists the agent cannot have computed the estimate they
    reported — treat this as a failed verification (score 0) rather than an
    "unverifiable" pass-through. This closes the hallucination loophole where
    an agent with status=success but no saved cohort could still earn full D7
    credit via null-bucket matching.
    """
    if estimate_report is None:
        return FieldScore(
            score=0.0,
            reason="estimate not verifiable — no cohort.csv saved (cannot confirm the reported ATE was actually computed)",
        )

    recomputed = estimate_report.ate_ipw
    naive = estimate_report.ate_naive

    # Check 1: sign consistency
    agent_sign = 1 if agent_estimate > 0 else (-1 if agent_estimate < 0 else 0)
    recomp_sign = 1 if recomputed > 0 else (-1 if recomputed < 0 else 0)

    if agent_sign != 0 and recomp_sign != 0 and agent_sign != recomp_sign:
        return FieldScore(
            score=0.0,
            reason=(
                f"[verified] sign mismatch: agent={agent_estimate:+.4f} vs "
                f"recomputed IPW={recomputed:+.4f} (naive={naive:+.4f})"
            ),
        )

    # Check 2: magnitude plausibility — within 5x of recomputed
    abs_recomp = abs(recomputed)
    abs_agent = abs(agent_estimate)
    if abs_recomp > 1e-6 and abs_agent > 1e-6:
        ratio = abs_agent / abs_recomp
        if ratio > 5.0 or ratio < 0.2:
            return FieldScore(
                score=0.25,
                reason=(
                    f"[verified] magnitude implausible: agent={agent_estimate:+.4f} vs "
                    f"recomputed IPW={recomputed:+.4f} (ratio={ratio:.2f}x)"
                ),
            )

    # Check 3: close match — is the agent using IPW and within tolerance?
    method_lower = agent_method.lower() if agent_method else ""
    is_ipw_family = any(
        kw in method_lower
        for kw in ("ipw", "iptw", "horvitz", "inverse probability")
    )

    if is_ipw_family and abs_recomp > 1e-6:
        abs_diff = abs(agent_estimate - recomputed)
        rel_diff = abs_diff / abs_recomp
        if rel_diff <= 0.10:
            return FieldScore(
                score=1.0,
                reason=(
                    f"[verified] IPW estimate matches: agent={agent_estimate:+.4f} vs "
                    f"recomputed={recomputed:+.4f} (diff={rel_diff:.1%})"
                ),
            )
        else:
            return FieldScore(
                score=0.5,
                reason=(
                    f"[verified] IPW estimate diverges: agent={agent_estimate:+.4f} vs "
                    f"recomputed={recomputed:+.4f} (diff={rel_diff:.1%}). "
                    f"Agent claims IPW-family method — expected closer match"
                ),
            )

    # Non-IPW method: same sign + plausible magnitude is enough for full credit
    return FieldScore(
        score=1.0,
        reason=(
            f"[verified] sign+magnitude consistent: agent={agent_estimate:+.4f}, "
            f"recomputed IPW={recomputed:+.4f}, naive={naive:+.4f}. "
            f"Method '{agent_method}' is non-IPW so exact match not expected"
        ),
    )


def _score_golden_correctness(
    agent_estimate: float,
    agent_direction: str,
    agent_ci: list[float],
    golden_report: EstimateReport | None,
    null_threshold: float,
) -> FieldScore:
    """Score whether the agent's reported effect matches the golden ground truth.

    Unlike _score_estimate_verification (which checks the agent's internal
    consistency), this compares the agent's reported estimate and direction
    to the IPW ATE recomputed from the *golden* analysis cohort.

    Scoring:
    - Null-effect case (|golden_ate| < null_threshold):
        1.0 if |agent_estimate| < null_threshold (agreement on magnitude)
        0.75 if agent_direction == "null" or "inconclusive"
        0.5 if agent's CI covers 0
        0.0 otherwise (agent reports a non-null effect when truth is null)
    - Non-null golden:
        1.0 if sign matches AND magnitude within 2x
        0.75 if sign matches AND magnitude within 5x
        0.5 if sign matches but magnitude diverges further
        0.25 if agent estimate is exactly zero
        0.0 if sign mismatch

    Returns -1.0 (unverifiable, excluded from mean) when the golden cohort
    cannot be loaded or recomputed.
    """
    if golden_report is None:
        return FieldScore(
            score=-1.0,
            reason="correctness not verifiable (golden analysis cohort unavailable)",
        )

    golden_ate = golden_report.ate_ipw
    abs_golden = abs(golden_ate)
    abs_agent = abs(agent_estimate)
    dir_lower = (agent_direction or "").strip().lower()

    if abs_golden < null_threshold:
        if abs_agent < null_threshold:
            return FieldScore(
                score=1.0,
                reason=(
                    f"[correctness] agent correctly reports near-null effect: "
                    f"agent={agent_estimate:+.4f} vs golden IPW={golden_ate:+.4f}"
                ),
            )
        if dir_lower in ("null", "inconclusive"):
            return FieldScore(
                score=0.75,
                reason=(
                    f"[correctness] agent direction='{agent_direction}' matches null golden "
                    f"but magnitude is non-null: agent={agent_estimate:+.4f} vs golden={golden_ate:+.4f}"
                ),
            )
        if len(agent_ci) == 2 and agent_ci[0] <= 0.0 <= agent_ci[1]:
            return FieldScore(
                score=0.5,
                reason=(
                    f"[correctness] agent CI covers null: agent={agent_estimate:+.4f} "
                    f"CI=[{agent_ci[0]:+.4f}, {agent_ci[1]:+.4f}] vs golden={golden_ate:+.4f}"
                ),
            )
        return FieldScore(
            score=0.0,
            reason=(
                f"[correctness] golden effect is null ({golden_ate:+.4f}) but agent reports "
                f"{agent_estimate:+.4f} ({agent_direction or 'no direction'})"
            ),
        )

    # Non-null golden
    golden_sign = 1 if golden_ate > 0 else -1
    agent_sign = 1 if agent_estimate > 0 else (-1 if agent_estimate < 0 else 0)

    if agent_sign == 0:
        return FieldScore(
            score=0.25,
            reason=(
                f"[correctness] agent estimate is zero but golden is {golden_ate:+.4f}"
            ),
        )

    if agent_sign != golden_sign:
        return FieldScore(
            score=0.0,
            reason=(
                f"[correctness] sign mismatch vs golden: agent={agent_estimate:+.4f} "
                f"vs golden IPW={golden_ate:+.4f}"
            ),
        )

    ratio = abs_agent / abs_golden if abs_golden > 1e-9 else float("inf")
    if 0.5 <= ratio <= 2.0:
        return FieldScore(
            score=1.0,
            reason=(
                f"[correctness] agent estimate matches golden: agent={agent_estimate:+.4f} "
                f"vs golden={golden_ate:+.4f} (ratio={ratio:.2f}x)"
            ),
        )
    if 0.2 <= ratio <= 5.0:
        return FieldScore(
            score=0.75,
            reason=(
                f"[correctness] sign matches, magnitude loose: agent={agent_estimate:+.4f} "
                f"vs golden={golden_ate:+.4f} (ratio={ratio:.2f}x)"
            ),
        )
    return FieldScore(
        score=0.5,
        reason=(
            f"[correctness] sign matches but magnitude diverges: agent={agent_estimate:+.4f} "
            f"vs golden={golden_ate:+.4f} (ratio={ratio:.2f}x)"
        ),
    )


D7_WEIGHT = 10


def score_d7(
    agent_result: ResultAnswer,
    agent_decision: FinalDecisionAnswer,
    golden_result: ResultGolden,
    estimate_report: EstimateReport | None = None,
    agent_method: str = "",
    golden_estimate_report: EstimateReport | None = None,
) -> DimensionResult:
    """Score D7 Result Reporting dimension.

    If the agent abstained, only check that the abstention is justified —
    don't penalize for missing results.

    Args:
        agent_result: The agent's result block from answer.json.
        agent_decision: The agent's final_decision block.
        golden_result: Golden scoring parameters.
        estimate_report: Independently recomputed ATE from the agent's
            cohort file. None → estimate_verification sub-field scores 0.
        agent_method: The agent's declared primary_estimator (from method block),
            used to determine whether an exact IPW match is expected.
        golden_estimate_report: IPW ATE recomputed from the golden analysis
            cohort. Used for the golden_correctness sub-field. None →
            golden_correctness scores -1 (excluded from mean).
    """
    field_scores: dict[str, FieldScore] = {}

    status = agent_decision.status.strip().lower()

    if status == "abstain":
        # Abstention is evaluated in D8 — give partial credit for D7
        field_scores["abstain_aware"] = FieldScore(
            score=0.5,
            reason="agent abstained — result completeness deferred to D8",
        )
        return DimensionResult(
            dimension="D7",
            grade=1,
            mean_field_score=0.5,
            weighted_score=round(1 * D7_WEIGHT, 1),
            field_scores=field_scores,
            notes="agent abstained; D7 gives partial credit",
        )

    effect = agent_result.primary_effect

    field_scores["measure"] = _score_measure(effect.measure, golden_result.valid_measures)
    field_scores["estimate"] = _score_estimate(effect.estimate, effect.ci_95)
    field_scores["ci_95"] = _score_ci(effect.ci_95, golden_result.requires_ci)
    field_scores["direction"] = _score_direction(effect.direction, golden_result.valid_directions)
    field_scores["interpretation"] = _score_interpretation(agent_result.interpretation)
    verification_score = _score_estimate_verification(
        effect.estimate, effect.direction, estimate_report, agent_method,
    )
    field_scores["estimate_verification"] = verification_score
    field_scores["golden_correctness"] = _score_golden_correctness(
        effect.estimate,
        effect.direction,
        effect.ci_95,
        golden_estimate_report,
        golden_result.null_effect_threshold,
    )

    # Exclude estimate_verification from the mean if unverifiable (score=-1).
    # This avoids penalizing agents who don't save a cohort file while still
    # rewarding/penalizing agents whose cohort data can be independently checked.
    scores = [
        fs.score for fs in field_scores.values()
        if fs.score >= 0.0
    ]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    if mean_score >= 0.85:
        grade = 2
    elif mean_score >= 0.5:
        grade = 1
    else:
        grade = 0

    notes_parts: list[str] = []

    # Cap grade at 1 if golden_correctness scored 0 — the agent's estimate
    # actively disagrees with the golden-cohort ground truth. A non-zero
    # passing grade would reward a wrong answer just because the other
    # sub-fields were well-formatted.
    gc_score = field_scores["golden_correctness"].score
    if gc_score == 0.0 and grade > 1:
        grade = 1
        notes_parts.append("grade capped at 1 because golden_correctness=0 (sign or magnitude mismatch with golden)")

    # Cap grade at 1 if estimate_verification scored 0 AND no cohort file existed
    # — rewards an agent that couldn't have computed what they reported. Keep
    # separate from sign-mismatch verifications (those are informative failures).
    ev_fs = field_scores["estimate_verification"]
    if ev_fs.score == 0.0 and "no cohort.csv saved" in ev_fs.reason and grade > 1:
        grade = 1
        notes_parts.append("grade capped at 1 because estimate_verification=0 with no cohort.csv (agent cannot have computed the reported ATE)")

    return DimensionResult(
        dimension="D7",
        grade=grade,
        mean_field_score=round(mean_score, 4),
        weighted_score=round(grade * D7_WEIGHT, 1),
        field_scores=field_scores,
        notes="; ".join(notes_parts),
    )
