"""CLI entry point for D1 evaluation."""

from __future__ import annotations

import argparse
import json
import sys

from eval.aggregation import aggregate_d1
from eval.loader import ValidationGateError, load_answer, load_golden
from eval.scorer_d1 import score_d1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate D1 Estimand Specification")
    parser.add_argument("--answer", required=True, help="Path to agent's answer.json")
    parser.add_argument("--golden", required=True, help="Path to task's golden.json")
    parser.add_argument("--output", default=None, help="Path to write scores JSON (default: stdout)")
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

    # Score D1
    field_scores = score_d1(answer.estimand, golden.estimand_golden, use_cache=use_cache)

    # Aggregate
    d1_result = aggregate_d1(field_scores)

    # Output
    result_dict = json.loads(d1_result.model_dump_json())
    _output(result_dict, args.output)

    return 0


def _output(data: dict, path: str | None) -> None:
    """Write result to file or stdout."""
    output_str = json.dumps(data, indent=2)
    if path:
        with open(path, "w") as f:
            f.write(output_str)
            f.write("\n")
        print(f"Scores written to {path}", file=sys.stderr)
    else:
        print(output_str)


if __name__ == "__main__":
    sys.exit(main())
