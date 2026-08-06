"""CLI entry point for full D1-D8 evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from eval.aggregation import aggregate_all
from eval.loader import ValidationGateError, load_answer, load_golden
from eval.scorer_d1 import score_d1
from eval.scorer_d2 import score_d2
from eval.scorer_d3 import score_d3
from eval.scorer_d4 import score_d4
from eval.scorer_d5 import score_d5
from eval.scorer_d6 import score_d6
from eval.scorer_d7 import score_d7
from eval.scorer_d8 import score_d8
from eval.aggregation import aggregate_dimension
from eval.schemas import DimensionResult


def evaluate(
    answer_path: str,
    golden_path: str,
    project_root: str | None = None,
    run_dir: str | None = None,
    data_root: str | None = None,
    use_cache: bool = True,
    dimensions: list[str] | None = None,
) -> dict:
    """Run full D1-D8 evaluation and return results.

    Args:
        answer_path: Path to agent's answer.json.
        golden_path: Path to task's golden.json.
        project_root: Project root for resolving golden cohort CSV.
        run_dir: Run directory for checking analysis script existence.
        data_root: MIMIC data root for post-hoc temporal verification.
            If None, defaults to project_root.
        use_cache: Whether to cache LLM judge results.
        dimensions: Optional list of dimensions to score (e.g. ["D1", "D2"]).
            If None, scores all available.

    Returns:
        Dict with overall_score, per-dimension results, and errors.
    """
    # Load and validate
    answer = load_answer(answer_path)
    golden = load_golden(golden_path)

    all_dims = dimensions or ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
    results: list[DimensionResult] = []
    cohort_path_used: Path | None = None  # set by D6, reused by D7

    # D1: Estimand Specification
    if "D1" in all_dims:
        field_scores = score_d1(answer.estimand, golden.estimand_golden, use_cache=use_cache)
        results.append(aggregate_dimension("D1", field_scores))

    # D2: Cohort Construction
    if "D2" in all_dims:
        agent_cohort_file = ""
        if answer.artifacts:
            agent_cohort_file = answer.artifacts.cohort_file or ""
        results.append(score_d2(
            answer.cohort, golden.cohort_golden, project_root,
            agent_cohort_file=agent_cohort_file, run_dir=run_dir,
        ))

    # D3: Temporal Validity
    if "D3" in all_dims:
        # Run post-hoc temporal verification if data root is available
        verification_report = None
        if data_root and run_dir:
            try:
                from eval.temporal_verifier import (
                    TemporalVerifier, TASK_SOURCES, load_treatment_times,
                )
                sources = TASK_SOURCES.get(answer.task_id, [])
                if sources:
                    verifier = TemporalVerifier(data_root)
                    # Load agent cohort IDs
                    cohort_hadm_ids = set()
                    agent_cohort_df = None
                    if answer.artifacts and answer.artifacts.cohort_file:
                        cohort_path = Path(run_dir) / Path(answer.artifacts.cohort_file).name
                        if cohort_path.exists():
                            if cohort_path.suffix == ".parquet":
                                agent_cohort_df = pd.read_parquet(cohort_path)
                            else:
                                agent_cohort_df = pd.read_csv(cohort_path)
                            if "hadm_id" in agent_cohort_df.columns:
                                cohort_hadm_ids = set(agent_cohort_df["hadm_id"].astype(int))
                    if cohort_hadm_ids:
                        # Load per-patient treatment times for temporal boundary
                        treatment_times = load_treatment_times(data_root, cohort_hadm_ids)
                        verification_report = verifier.verify_all(
                            sources, cohort_hadm_ids, treatment_times, agent_cohort_df
                        )
            except Exception as e:
                print(f"Temporal verification skipped: {e}", file=sys.stderr)

        results.append(score_d3(
            answer.temporal, answer.variables, golden.temporal_golden,
            estimand_time_zero=answer.estimand.time_zero,
            verification_report=verification_report,
        ))

    # D4: Variable Construction
    if "D4" in all_dims:
        results.append(score_d4(answer.variables, golden.variables_golden))

    # D5: Causal Method
    if "D5" in all_dims:
        results.append(score_d5(answer.method, answer.artifacts, golden.method_golden, run_dir))

    # D6: Diagnostics
    if "D6" in all_dims:
        # Programmatic verification: auto-detect a cohort file and recompute SMDs
        # and propensity overlap from the actual data. If no verifiable cohort
        # file is found, balance and overlap subscores are zeroed — we do NOT
        # trust the agent's self-attested diagnostics.
        balance_report = None
        overlap_range = None
        if run_dir:
            from eval.balance_verifier import load_and_verify

            candidates: list[str] = []
            if answer.artifacts and answer.artifacts.cohort_file:
                candidates.append(Path(answer.artifacts.cohort_file).name)
            # Fall back to common filenames (first hit wins)
            for name in (
                "cohort.parquet", "cohort.csv",
                "cohort_final.parquet", "cohort_final.csv",
                "cohort_analysis.parquet", "cohort_analysis.csv",
                "analytic_cohort.parquet", "analytic_cohort.csv",
                "cohort_intermediate.csv",
            ):
                if name not in candidates:
                    candidates.append(name)

            for name in candidates:
                p = Path(run_dir) / name
                if not p.exists():
                    continue
                try:
                    balance_report = load_and_verify(p)
                except Exception as e:
                    print(f"load_and_verify({p.name}) failed: {e}", file=sys.stderr)
                    continue
                if balance_report is None:
                    continue
                cohort_path_used = p
                # Also extract propensity-score range for independent overlap check
                try:
                    import pandas as pd
                    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                    if "propensity_score" in df.columns:
                        ps = df["propensity_score"].dropna()
                        if len(ps):
                            overlap_range = (float(ps.min()), float(ps.max()))
                except Exception as e:
                    print(f"overlap recompute failed: {e}", file=sys.stderr)
                break

            if cohort_path_used is not None:
                print(
                    f"D6 verified against {cohort_path_used.name} "
                    f"(declared={answer.artifacts.cohort_file if answer.artifacts else None})",
                    file=sys.stderr,
                )
            else:
                print(
                    f"D6: no verifiable cohort file found in {run_dir} — balance/overlap zeroed",
                    file=sys.stderr,
                )

        results.append(score_d6(
            answer.diagnostics, golden.diagnostics_golden, balance_report, overlap_range,
        ))

    # D7: Result Reporting + Estimate Verification
    if "D7" in all_dims:
        # Recompute ATE from the agent's cohort file for internal consistency
        estimate_report = None
        if run_dir:
            from eval.estimate_verifier import load_and_recompute

            # Reuse the same cohort file that D6 found, or search again
            cohort_for_d7 = cohort_path_used
            if cohort_for_d7 is None:
                candidates_d7: list[str] = []
                if answer.artifacts and answer.artifacts.cohort_file:
                    candidates_d7.append(Path(answer.artifacts.cohort_file).name)
                for name in (
                    "cohort.parquet", "cohort.csv",
                    "cohort_final.parquet", "cohort_final.csv",
                    "cohort_analysis.parquet", "cohort_analysis.csv",
                    "analytic_cohort.parquet", "analytic_cohort.csv",
                    "cohort_intermediate.csv",
                ):
                    if name not in candidates_d7:
                        candidates_d7.append(name)
                for name in candidates_d7:
                    p = Path(run_dir) / name
                    if p.exists():
                        cohort_for_d7 = p
                        break

            if cohort_for_d7 is not None:
                try:
                    estimate_report = load_and_recompute(cohort_for_d7)
                    if estimate_report is not None:
                        print(
                            f"D7 estimate recomputed from {Path(cohort_for_d7).name}: "
                            f"IPW ATE={estimate_report.ate_ipw:+.4f}, "
                            f"naive={estimate_report.ate_naive:+.4f} "
                            f"(n_t={estimate_report.n_treated}, n_c={estimate_report.n_control})",
                            file=sys.stderr,
                        )
                except Exception as e:
                    print(f"D7 estimate recomputation failed: {e}", file=sys.stderr)

        agent_method = ""
        if answer.method:
            agent_method = answer.method.primary_estimator or ""

        golden_estimate_report = None
        golden_cohort_csv = golden.result_golden.golden_analysis_cohort_csv
        if golden_cohort_csv:
            from eval.estimate_verifier import load_and_recompute

            golden_cohort_path = Path(golden_cohort_csv)
            if not golden_cohort_path.is_absolute() and project_root:
                golden_cohort_path = Path(project_root) / golden_cohort_csv
            if golden_cohort_path.exists():
                try:
                    golden_estimate_report = load_and_recompute(golden_cohort_path)
                    if golden_estimate_report is not None:
                        print(
                            f"D7 golden ATE from {golden_cohort_path.name}: "
                            f"IPW={golden_estimate_report.ate_ipw:+.4f}, "
                            f"naive={golden_estimate_report.ate_naive:+.4f} "
                            f"(n_t={golden_estimate_report.n_treated}, "
                            f"n_c={golden_estimate_report.n_control})",
                            file=sys.stderr,
                        )
                except Exception as e:
                    print(f"D7 golden recomputation failed: {e}", file=sys.stderr)
            else:
                print(
                    f"D7 golden cohort not found at {golden_cohort_path}",
                    file=sys.stderr,
                )

        results.append(score_d7(
            answer.result, answer.final_decision, golden.result_golden,
            estimate_report=estimate_report, agent_method=agent_method,
            golden_estimate_report=golden_estimate_report,
        ))

    # D8: Robustness
    if "D8" in all_dims:
        results.append(score_d8(
            answer.robustness, answer.final_decision, answer.result, golden.robustness_golden
        ))

    # Aggregate
    overall = aggregate_all(results)

    # Add per-dimension field details
    detail = {}
    for dr in results:
        detail[dr.dimension] = {
            "grade": dr.grade,
            "mean_field_score": dr.mean_field_score,
            "weighted_score": dr.weighted_score,
            "hard_gate_failed": dr.hard_gate_failed,
            "notes": dr.notes,
            "field_scores": {
                k: {"score": v.score, "reason": v.reason, "disqualified": v.disqualified}
                for k, v in dr.field_scores.items()
            },
        }

    return {
        "task_id": answer.task_id,
        "run_id": answer.run_id or "",
        "overall_score": overall["overall_score"],
        "max_score": overall["max_score"],
        "percentage": overall["percentage"],
        "dimensions": detail,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full D1-D8 Evaluation")
    parser.add_argument("--answer", required=True, help="Path to agent's answer.json")
    parser.add_argument("--golden", required=True, help="Path to task's golden.json")
    parser.add_argument("--project-root", default=None, help="Project root for golden cohort CSV")
    parser.add_argument("--run-dir", default=None, help="Run dir for checking analysis script")
    parser.add_argument("--data-root", default=None, help="MIMIC data root for temporal verification")
    parser.add_argument("--output", default=None, help="Path to write scores JSON")
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM response caching")
    parser.add_argument("--dimensions", nargs="*", help="Specific dimensions to score (e.g. D1 D2)")
    args = parser.parse_args(argv)

    use_cache = not args.no_cache

    # Auto-detect project root from golden path if not given
    project_root = args.project_root
    if project_root is None:
        golden_path = Path(args.golden).resolve()
        # golden.json is at tasks/task_xxx/golden.json -> project root is 2 levels up
        candidate = golden_path.parent.parent.parent
        if (candidate / "docs").exists():
            project_root = str(candidate)

    # Auto-detect run dir from answer path
    run_dir = args.run_dir
    if run_dir is None:
        run_dir = str(Path(args.answer).resolve().parent)

    # Auto-detect data root
    data_root = args.data_root or project_root

    try:
        result = evaluate(
            answer_path=args.answer,
            golden_path=args.golden,
            project_root=project_root,
            run_dir=run_dir,
            data_root=data_root,
            use_cache=use_cache,
            dimensions=args.dimensions,
        )
    except ValidationGateError as e:
        print(f"VALIDATION FAILED: {e}", file=sys.stderr)
        result = {
            "task_id": "unknown",
            "run_id": "",
            "overall_score": 0.0,
            "max_score": 200,
            "percentage": 0.0,
            "error": str(e),
            "dimensions": {},
        }
        _output(result, args.output)
        return 1

    _output(result, args.output)
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
