#!/usr/bin/env python3
"""
Evaluate performance for each task and the whole pipeline.

Usage:
    # Summarize all jobs under var/jobs/
    python evaluate.py

    # Summarize a specific job
    python evaluate.py --job <job-id>

    # Summarize only completed jobs for a specific benchmark
    python evaluate.py --benchmark single-cell

    # Output as JSON
    python evaluate.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


JOBS_DIR = Path(__file__).resolve().parent / "var" / "jobs"

# Canonical task names per benchmark, used to group score_summary keys
SINGLE_CELL_TASKS = {
    "task1":  "Read & Harmonize Metadata",
    "task2":  "QC Metrics",
    "task3":  "Cell & Gene Filtering",
    "task4":  "Doublet Detection",
    "task5":  "Normalization",
    "task6":  "HVG + PCA",
    "task7":  "Neighbors + UMAP",
    "task8":  "Batch Effect Correction",
    "task9":  "Clustering Evaluation",
    "task10": "Differential Expression",
    "task11": "Trajectory Preservation",
}

SPATIAL_TASKS = {
    "task1":  "Read & Spatial Metadata",
    "task2":  "QC Metrics",
    "task3":  "Filtering",
    "task4":  "Normalization",
    "task5":  "HVG + PCA",
    "task6":  "Neighbors + UMAP",
    "task7":  "Spatial Graph",
    "task8":  "Batch Effect Correction",
    "task9":  "Clustering + Spatial Autocorrelation",
    "task10": "Spatially Variable Genes",
    "task11": "Neighborhood Enrichment",
}

PERTURBATION_PREDICTION_TASKS = {
    "task1": "Load & Validate Perturbation Data",
    "task2": "Prediction Format Validation",
    "task3": "Perturbation Prediction Quality",
}


def load_result(job_dir: Path) -> Optional[Dict[str, Any]]:
    result_path = job_dir / "result.json"
    if not result_path.exists():
        return None
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_job(job_dir: Path) -> Optional[Dict[str, Any]]:
    job_path = job_dir / "job.json"
    if not job_path.exists():
        return None
    with open(job_path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_scores_by_task(score_summary: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Group flat score_summary keys into per-task dicts.

    e.g. {"task9_ARI": 0.35, "task9_AMI": 0.67, "task1": 1.0}
    -> {"task1": {"score": 1.0}, "task9": {"ARI": 0.35, "AMI": 0.67}}
    """
    grouped: Dict[str, Dict[str, float]] = {}
    for key, value in score_summary.items():
        # Keys like "task8_total", "task9_ARI", "task10_jaccard"
        parts = key.split("_", 1)
        task_id = parts[0]
        metric = parts[1] if len(parts) > 1 else "score"
        grouped.setdefault(task_id, {})[metric] = value
    return grouped


def task_aggregate(metrics: Dict[str, float]) -> float:
    """Average all sub-metrics for a task into one number."""
    vals = [v for v in metrics.values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def print_job_report(job_id: str, result: Dict[str, Any], job_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Print a per-task and overall report for one job. Returns structured summary."""
    status = result.get("status", "unknown")
    dataset = result.get("dataset", "unknown")
    benchmark = result.get("benchmark", "single-cell")
    overall = result.get("overall_average", 0.0)
    score_summary = result.get("score_summary", {})

    if benchmark == "spatial":
        task_names = SPATIAL_TASKS
    elif benchmark == "perturbation-prediction":
        task_names = PERTURBATION_PREDICTION_TASKS
    else:
        task_names = SINGLE_CELL_TASKS
    grouped = group_scores_by_task(score_summary)

    mode = job_meta.get("mode", "?") if job_meta else "?"
    created = job_meta.get("createdAt", "") if job_meta else ""

    print(f"\n{'='*72}")
    print(f"  Job:       {job_id}")
    print(f"  Status:    {status}")
    print(f"  Benchmark: {benchmark}")
    print(f"  Dataset:   {dataset}")
    print(f"  Mode:      {mode}")
    if created:
        print(f"  Created:   {created}")
    print(f"{'='*72}")

    if status == "failed":
        msg = result.get("message", "")
        print(f"  FAILED: {msg}")
        print(f"{'='*72}")
        return {
            "job_id": job_id, "status": status, "benchmark": benchmark,
            "dataset": dataset, "overall_average": 0.0, "task_scores": {},
        }

    task_scores: Dict[str, float] = {}

    print(f"\n  {'Task':<8} {'Name':<38} {'Agg Score':>10}")
    print(f"  {'-'*8} {'-'*38} {'-'*10}")

    for task_id in sorted(task_names.keys(), key=lambda t: int(t.replace("task", ""))):
        name = task_names.get(task_id, task_id)
        metrics = grouped.get(task_id, {})
        agg = task_aggregate(metrics)
        task_scores[task_id] = agg
        print(f"  {task_id:<8} {name:<38} {agg:>10.4f}")

        # Show sub-metrics if more than one
        if len(metrics) > 1:
            for mname, mval in sorted(metrics.items()):
                print(f"  {'':8}   {mname:<36} {mval:>10.4f}")

    print(f"\n  {'Overall Average':<48} {overall:>10.4f}")
    print(f"{'='*72}")

    return {
        "job_id": job_id,
        "status": status,
        "benchmark": benchmark,
        "dataset": dataset,
        "mode": mode,
        "overall_average": overall,
        "task_scores": task_scores,
        "score_summary": score_summary,
    }


def print_pipeline_summary(summaries: List[Dict[str, Any]]) -> None:
    """Print an aggregate summary across all evaluated jobs."""
    completed = [s for s in summaries if s["status"] == "completed"]
    if not completed:
        print("\nNo completed jobs to summarize.")
        return

    print(f"\n{'#'*72}")
    print(f"  PIPELINE SUMMARY  ({len(completed)} completed / {len(summaries)} total jobs)")
    print(f"{'#'*72}")

    # Aggregate per-task scores across jobs
    all_tasks: Dict[str, List[float]] = {}
    for s in completed:
        for tid, score in s.get("task_scores", {}).items():
            all_tasks.setdefault(tid, []).append(score)

    overall_scores = [s["overall_average"] for s in completed]

    print(f"\n  {'Task':<8} {'Mean':>10} {'Min':>10} {'Max':>10} {'N':>5}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*5}")

    for tid in sorted(all_tasks.keys(), key=lambda t: int(t.replace("task", ""))):
        vals = all_tasks[tid]
        print(f"  {tid:<8} {sum(vals)/len(vals):>10.4f} {min(vals):>10.4f} {max(vals):>10.4f} {len(vals):>5}")

    mean_overall = sum(overall_scores) / len(overall_scores)
    print(f"\n  {'Overall':48} {mean_overall:>10.4f}")
    print(f"{'#'*72}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate per-task and pipeline performance from job results.")
    parser.add_argument("--job", type=str, default=None, help="Evaluate a single job by ID.")
    parser.add_argument("--benchmark", type=str, default=None,
                        choices=["single-cell", "spatial", "perturbation-prediction"],
                        help="Filter to a specific benchmark type.")
    parser.add_argument("--jobs-dir", type=str, default=str(JOBS_DIR), help="Path to the jobs directory.")
    parser.add_argument("--json", action="store_true", help="Output results as JSON instead of human-readable tables.")
    args = parser.parse_args()

    jobs_dir = Path(args.jobs_dir)
    if not jobs_dir.is_dir():
        print(f"Jobs directory not found: {jobs_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect job directories
    if args.job:
        job_dirs = [jobs_dir / args.job]
        if not job_dirs[0].is_dir():
            print(f"Job not found: {args.job}", file=sys.stderr)
            sys.exit(1)
    else:
        job_dirs = sorted([d for d in jobs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    summaries: List[Dict[str, Any]] = []

    for job_dir in job_dirs:
        job_id = job_dir.name
        job_meta = load_job(job_dir)
        result = load_result(job_dir)

        if result is None:
            # No result.json — check job.json for failed status
            if job_meta and job_meta.get("status") == "failed":
                result = {
                    "status": "failed",
                    "message": job_meta.get("error", "No result produced"),
                    "score_summary": {},
                    "overall_average": 0.0,
                }
            else:
                if not args.json:
                    print(f"\nSkipping {job_id}: no result.json found")
                continue

        benchmark = result.get("benchmark") or (job_meta or {}).get("datasetId", "")
        if args.benchmark and benchmark != args.benchmark:
            continue

        if not args.json:
            summary = print_job_report(job_id, result, job_meta)
        else:
            # Build summary silently for JSON output
            score_summary = result.get("score_summary", {})
            grouped = group_scores_by_task(score_summary)
            task_scores = {tid: task_aggregate(m) for tid, m in grouped.items()}
            summary = {
                "job_id": job_id,
                "status": result.get("status", "unknown"),
                "benchmark": benchmark,
                "dataset": result.get("dataset", "unknown"),
                "mode": (job_meta or {}).get("mode", "?"),
                "overall_average": result.get("overall_average", 0.0),
                "task_scores": task_scores,
                "score_summary": score_summary,
            }
        summaries.append(summary)

    if args.json:
        pipeline = {}
        completed = [s for s in summaries if s["status"] == "completed"]
        if completed:
            all_tasks: Dict[str, List[float]] = {}
            for s in completed:
                for tid, score in s.get("task_scores", {}).items():
                    all_tasks.setdefault(tid, []).append(score)
            pipeline = {
                "n_completed": len(completed),
                "n_total": len(summaries),
                "mean_overall": sum(s["overall_average"] for s in completed) / len(completed),
                "per_task_mean": {tid: sum(v)/len(v) for tid, v in sorted(all_tasks.items())},
            }
        output = {"jobs": summaries, "pipeline_summary": pipeline}
        print(json.dumps(output, indent=2))
    else:
        if len(summaries) > 1:
            print_pipeline_summary(summaries)


if __name__ == "__main__":
    main()
