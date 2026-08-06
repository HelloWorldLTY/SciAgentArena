"""LLM-as-judge scorer for D1 estimand fields.

Uses a structured rubric prompt to score each semantic field. Deterministic
scorers are kept for trivially checkable fields (effect_type, time_horizon).
Results are cached by content hash to ensure reproducibility and save cost.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from openai import OpenAI


CACHE_DIR = Path(__file__).resolve().parent.parent / ".llm_cache" / "judge"


# ---------------------------------------------------------------------------
# Per-field rubrics (derived from benchmark-eval-spec.md)
# ---------------------------------------------------------------------------

FIELD_RUBRICS: dict[str, str] = {
    "population": """\
Score the agent's `estimand.population` against the golden answer.

This field should describe WHO the causal effect is about. It should cover \
the task-critical population-defining conditions, which may include:
- source population or care setting (inpatient, ICU, ED, outpatient)
- unit of analysis (hospitalized adults, admissions, ICU stays)
- clinical eligibility (the disease or condition defining the population)
- demographic constraints (adults, pediatric, age threshold)
- required data availability that defines the population (e.g. linked imaging)
- temporal eligibility relative to time zero

It should NOT be penalized for omitting:
- detailed exclusion lists, attrition steps, patient counts
- missing-data handling rules
- SQL-level or implementation-level cohort logic

Scoring:
- 1.0: captures all task-critical population conditions with no material contradiction
- 0.5: captures the main target population but misses one secondary condition, \
is too vague on one qualifier, or adds a mild extra restriction
- 0.0: misses or contradicts one or more core population-defining conditions, \
or defines a materially different target population

Core errors that should score 0.0:
- wrong care setting
- wrong disease or clinical condition
- wrong age group
- wrong time anchoring
- wrong modality requirement when essential to the task""",

    "treatment": """\
Score the agent's `estimand.treatment` against the golden answer.

This field should precisely define the intervention being studied, including:
- the specific drug or drug class
- the route of administration
- the dose criteria (if relevant)
- the timing window relative to time zero

Scoring:
- 1.0: captures the correct drug/class, route, and timing window
- 0.5: correct drug/class but vague on route or timing, or slightly overspecified
- 0.0: wrong drug, wrong route, wrong timing, or materially different intervention""",

    "comparator": """\
Score the agent's `estimand.comparator` against the golden answer.

This field should precisely define the control condition, i.e. what "no treatment" \
means operationally. It should specify:
- absence of the treatment
- the same route qualification as the treatment
- the same timing window as the treatment

Scoring:
- 1.0: correctly specifies absence of the treatment with matching route and timing
- 0.5: mostly correct but vague on one dimension (route, timing)
- 0.0: missing, uses an active comparator when the task asks for no-treatment, \
or materially different control condition""",

    "outcome": """\
Score the agent's `estimand.outcome` against the golden answer.

This field should define the measured outcome, including:
- what is being measured (the clinical endpoint)
- the direction of change that constitutes the outcome
- the assessment window

Scoring:
- 1.0: correct endpoint, correct direction, correct assessment window
- 0.5: correct endpoint but vague on assessment window or direction
- 0.0: wrong endpoint, wrong assessment window (e.g. 7 days instead of 24 hours), \
or contradicts the task question""",

    "time_zero": """\
Score the agent's `estimand.time_zero` against the golden answer.

This field should identify the clinical event that marks the start of follow-up.

Scoring:
- 1.0: semantically equivalent to the golden answer (allowing paraphrases like \
"hospital admission" = "admittime" = "time of admission")
- 0.5: close but imprecise (e.g. "when the patient arrives" when the task means \
hospital admission specifically)
- 0.0: wrong anchor event (e.g. discharge, ICU admission, first lab, when the \
task means hospital admission)""",
}


# ---------------------------------------------------------------------------
# Judge prompt template
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are an evaluator for a causal inference benchmark. Score the agent's answer \
for the field `estimand.{field_name}` by comparing it to the golden answer.

## Golden answer
{golden_value}

## Agent answer
{agent_value}

## Rubric
{rubric}

## Instructions
1. Compare the agent's answer to the golden answer using the rubric above.
2. Output ONLY a JSON object with exactly two keys:
   - "score": a float, one of 1.0, 0.5, or 0.0
   - "reason": a brief explanation (1-2 sentences) of why you assigned this score

Do not include any other text."""


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_key(field_name: str, golden_value: str, agent_value: str) -> str:
    payload = json.dumps({
        "field": field_name,
        "golden": golden_value,
        "agent": agent_value,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_cache(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_cache(key: str, result: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"{key}.json", "w") as f:
        json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Core judge function
# ---------------------------------------------------------------------------

def judge_field(
    field_name: str,
    golden_value: str,
    agent_value: str,
    model: str = "gpt-4o",
    use_cache: bool = True,
) -> dict:
    """Score a single estimand field using an LLM judge.

    Args:
        field_name: The field being scored (e.g. "population").
        golden_value: The canonical golden answer for this field.
        agent_value: The agent's answer for this field.
        model: LLM model to use for judging.
        use_cache: Whether to cache results.

    Returns:
        Dict with "score" (float) and "reason" (str).
    """
    if not agent_value or not agent_value.strip():
        return {"score": 0.0, "reason": "field is empty or missing"}

    rubric = FIELD_RUBRICS.get(field_name)
    if rubric is None:
        raise ValueError(f"No rubric defined for field: {field_name}")

    key = _cache_key(field_name, golden_value, agent_value)
    if use_cache:
        cached = _load_cache(key)
        if cached is not None:
            return cached

    prompt = JUDGE_PROMPT.format(
        field_name=field_name,
        golden_value=golden_value,
        agent_value=agent_value,
        rubric=rubric,
    )

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON, handling markdown code blocks
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
        # Validate structure
        result = {
            "score": float(result.get("score", 0.0)),
            "reason": str(result.get("reason", "")),
        }
        # Clamp score to valid values
        valid_scores = {0.0, 0.5, 1.0}
        if result["score"] not in valid_scores:
            # Round to nearest valid score
            result["score"] = min(valid_scores, key=lambda x: abs(x - result["score"]))
    except (json.JSONDecodeError, TypeError, ValueError):
        result = {"score": 0.0, "reason": f"Failed to parse judge output: {raw[:200]}"}

    if use_cache:
        _save_cache(key, result)

    return result
