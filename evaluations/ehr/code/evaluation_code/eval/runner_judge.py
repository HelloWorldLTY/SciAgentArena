"""CLI entry point for D1 evaluation using LLM-judge scorer.

Usage:
    python -m eval.runner_judge --answer runs/task_001/answer.json \
                                --golden tasks/task_001_diuretic_fluid/golden.json
"""

from __future__ import annotations

import argparse
import json
import sys

from eval.aggregation import aggregate_d1
from eval.loader import ValidationGateError, load_answer, load_golden
from eval.scorer_d1_judge import score_d1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate D1 Estimand (LLM-judge)")
    parser.add_argument("--answer", required=True, help="Path to agent's answer.json")
    parser.add_argument("--golden", required=True, help="Path to task's golden.json")
    parser.add_argument("--output", default=None, help="Path to write scores JSON (default: stdout)")
    parser.add_argument("--model", default="gpt-4o", help="LLM model for judge (default: gpt-4o)")
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM response caching")
    args = parser.parse_args(argv)

    use_cache = not args.no_cache

    # Load and validate
    try:
        answer = load_answer(args.answer)
    except ValidationGateError as e:
        print(f"VALIDATION FAILED: {e}", file=sys.stderr)
        result = {
            "dimension": "D1",
            "scorer": "llm_judge",
            "grade": 0,
            "mean_field_score": 0.0,
            "weighted_score": 0.0,
            "error": str(e),
        }
        _output(result, args.output)
        return 1

    try:
        golden = load_golden(args.golden)
    except ValidationGateError as e:
        print(f"GOLDEN FILE ERROR: {e}", file=sys.stderr)
        return 2

    # Score D1 with LLM judge
    print(f"Scoring with LLM judge (model={args.model})...", file=sys.stderr)
    field_scores = score_d1(
        answer.estimand, golden.estimand_golden,
        model=args.model, use_cache=use_cache,
    )

    # Aggregate
    d1_result = aggregate_d1(field_scores)

    # Build output with per-field details
    result_dict = json.loads(d1_result.model_dump_json())
    result_dict["scorer"] = "llm_judge"
    result_dict["model"] = args.model

    # Add readable field breakdown
    print("", file=sys.stderr)
    print("Field scores:", file=sys.stderr)
    for name, fs in field_scores.items():
        print(f"  {name}: {fs.score} — {fs.reason}", file=sys.stderr)
    print(f"\nGrade: {d1_result.grade} (mean={d1_result.mean_field_score:.2f})", file=sys.stderr)

    _output(result_dict, args.output)
    return 0


def _output(data: dict, path: str | None) -> None:
    output_str = json.dumps(data, indent=2)
    if path:
        with open(path, "w") as f:
            f.write(output_str + "\n")
        print(f"Scores written to {path}", file=sys.stderr)
    else:
        print(output_str)


if __name__ == "__main__":
    sys.exit(main())
