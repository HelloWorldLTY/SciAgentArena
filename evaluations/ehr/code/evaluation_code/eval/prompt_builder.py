"""Build agent prompts from task.json + dataset_manifest.json + answer schema."""

from __future__ import annotations

import json
from pathlib import Path

from eval.schemas import TaskSpec

# The full answer.json schema from the benchmark spec, used as-is in the prompt.
ANSWER_SCHEMA = {
    "task_id": "string",
    "run_id": "string",
    "estimand": {
        "population": "string",
        "treatment": "string",
        "comparator": "string",
        "outcome": "string",
        "time_zero": "ISO 8601 timestamp or description",
        "time_horizon": "ISO 8601 duration (e.g. P1D)",
        "effect_type": "one of: ATE, ATT, CATE, risk_difference, risk_ratio, odds_ratio, hazard_ratio",
    },
    "cohort": {
        "inclusion_criteria": ["string"],
        "exclusion_criteria": ["string"],
        "index_event": "string",
        "n_initial": 0,
        "n_final": 0,
        "attrition": [{"step": "string", "n_remaining": 0}],
    },
    "temporal": {
        "time_zero": "ISO 8601 timestamp",
        "allowed_data_end": "ISO 8601 timestamp",
        "post_t0_features_used": [],
    },
    "variables": {
        "treatment": {"name": "string", "definition": "string"},
        "outcome": {"name": "string", "definition": "string"},
        "confounders": [{"name": "string", "source": "string", "time_window": "pre_t0"}],
        "effect_modifiers": [{"name": "string", "source": "string", "time_window": "pre_t0"}],
        "imaging_features_used": ["string"],
    },
    "method": {
        "primary_estimator": "string",
        "estimation_family": "one of: matching, ipw, aipw, tmle, g_formula, dml, iv",
        "propensity_model": "string",
        "outcome_model": "string",
        "adjustment_set": ["string"],
    },
    "diagnostics": {
        "balance": {"max_smd_before": 0.0, "max_smd_after": 0.0},
        "overlap": {"min_propensity": 0.0, "max_propensity": 1.0},
        "missingness": {"rows_dropped": 0, "imputation_used": False},
        "positivity_violations": [],
        "model_failures": [],
    },
    "result": {
        "primary_effect": {
            "measure": "risk_difference",
            "estimate": 0.0,
            "ci_95": [0.0, 0.0],
            "p_value": 0.0,
            "direction": "one of: beneficial, harmful, null, mixed, inconclusive",
        },
        "heterogeneity": [],
        "interpretation": "string",
    },
    "robustness": {
        "alternative_estimators": [],
        "sensitivity_analyses": [],
    },
    "artifacts": {
        "analysis_script": "analysis.py",
        "cohort_file": "cohort.parquet",
        "log_file": "run.log",
    },
    "final_decision": {
        "status": "one of: success, abstain, error",
        "confidence": "one of: high, medium, low",
        "abstain_reason": "",
    },
}


def load_task(task_dir: str | Path) -> TaskSpec:
    """Load a task.json from a task directory."""
    task_dir = Path(task_dir)
    task_file = task_dir / "task.json"
    with open(task_file) as f:
        return TaskSpec.model_validate(json.load(f))


def _load_manifest(workspace_root: str | Path, manifest_rel_path: str) -> dict:
    """Load the dataset manifest."""
    manifest_path = Path(workspace_root) / manifest_rel_path
    with open(manifest_path) as f:
        return json.load(f)


def _format_data_summary(manifest: dict) -> str:
    """Extract a concise data summary from the dataset manifest."""
    lines = []

    for ds_key, ds in manifest.get("datasets", {}).items():
        lines.append(f"### {ds.get('dataset_name', ds_key)}")
        lines.append(f"- Version: {ds.get('source_version', 'unknown')}")
        lines.append(f"- Modality: {ds.get('modality', 'unknown')}")
        lines.append(f"- Base path: `{ds.get('base_path', 'N/A')}`")

        # List task-relevant tables
        tables = ds.get("task_relevant_tables", {})
        if tables:
            lines.append("- Tables:")
            for tname, tinfo in tables.items():
                desc = tinfo.get("description", "")
                path = tinfo.get("path", "")
                header = tinfo.get("header", [])
                lines.append(f"  - **{tname}**: {desc}")
                lines.append(f"    - Path: `{path}`")
                if header:
                    lines.append(f"    - Columns: {', '.join(header)}")

        # List metadata tables (for imaging)
        meta_tables = ds.get("metadata_tables", {})
        if meta_tables:
            lines.append("- Metadata tables:")
            for tname, tinfo in meta_tables.items():
                desc = tinfo.get("description", "")
                path = tinfo.get("path", "")
                lines.append(f"  - **{tname}**: {desc}")
                lines.append(f"    - Path: `{path}`")

        # JPG package metadata
        jpg = ds.get("jpg_package", {})
        jpg_meta = jpg.get("metadata_tables", {})
        if jpg_meta:
            lines.append("- Imaging label tables:")
            for tname, tinfo in jpg_meta.items():
                desc = tinfo.get("description", "")
                path = tinfo.get("path", "")
                header = tinfo.get("header", [])
                lines.append(f"  - **{tname}**: {desc}")
                lines.append(f"    - Path: `{path}`")
                if header:
                    lines.append(f"    - Columns: {', '.join(header)}")

        lines.append("")

    # Cross-dataset linkage
    linkage = manifest.get("cross_dataset_linkage", {})
    if linkage:
        lines.append("### Cross-Dataset Linkage")
        lines.append(f"- Primary shared key: `{linkage.get('primary_shared_identifier', '')}`")
        strategy = linkage.get("recommended_cross_modal_link_strategy", [])
        for s in strategy:
            lines.append(f"  - {s}")
        lines.append("")

    # Known limitations
    limitations = manifest.get("known_local_limitations", [])
    if limitations:
        lines.append("### Known Limitations")
        for lim in limitations:
            lines.append(f"- {lim}")
        lines.append("")

    return "\n".join(lines)


def _format_hints(task: TaskSpec) -> str:
    """Format optional hints as bullet points."""
    hints = task.hints
    lines = []
    if hints.treatment_route:
        lines.append(f"- Treatment route: {hints.treatment_route}")
    if hints.imaging_modality:
        lines.append(f"- Imaging modality: {hints.imaging_modality}")
    if hints.outcome_window:
        lines.append(f"- Outcome window: {hints.outcome_window}")
    for k, v in hints.custom.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) if lines else "None provided."


def _format_constraints(task: TaskSpec) -> str:
    """Format constraints as bullet points."""
    c = task.constraints
    lines = [
        f"- Allowed effect types: {', '.join(c.allowed_effect_types)}",
        f"- Allowed estimation families: {', '.join(c.allowed_estimation_families)}",
        f"- Timestamps: {c.time_format}",
        f"- Durations: {c.duration_format}",
    ]
    return "\n".join(lines)


def build_prompt(
    task: TaskSpec,
    workspace_root: str | Path = "/home/developer/mimic",
) -> str:
    """Build the full agent prompt from a task spec.

    Args:
        task: The parsed TaskSpec.
        workspace_root: Root of the workspace (for resolving manifest path).

    Returns:
        The complete prompt string to send to the agent.
    """
    workspace_root = Path(workspace_root)
    manifest = _load_manifest(workspace_root, task.data_manifest_path)
    data_summary = _format_data_summary(manifest)
    hints_text = _format_hints(task)
    constraints_text = _format_constraints(task)
    schema_json = json.dumps(ANSWER_SCHEMA, indent=2)

    prompt = f"""\
# Causal Inference Benchmark Task

## Task ID
`{task.task_id}`

## Clinical Question
{task.question}

## Your Objective
You are a causal inference agent. Your job is to:
1. Translate the clinical question above into a precise causal estimand.
2. Construct the appropriate study cohort from the available data.
3. Select and execute a defensible causal estimation strategy.
4. Report your results with uncertainty estimates and diagnostics.
5. Abstain if the causal question is not identifiable from the data.

## Required Output
You must produce a file called `answer.json` as your primary scored artifact. \
This file must conform exactly to the schema below. All scored fields must appear \
in this single machine-readable file.

You may also produce:
- `analysis.py` — your executable analysis script
- `summary.md` — an optional human-readable explanation

The evaluator scores `answer.json`, not your chat transcript or summary.

## answer.json Schema
```json
{schema_json}
```

## Field-Level Rules
- All timestamps must be ISO 8601.
- All durations must use ISO 8601 duration format (e.g. `P1D`, `PT24H`).
- Every declared variable must have a source or provenance string.
- `post_t0_features_used` must be an explicit list. Empty list means none were used.
- `adjustment_set` must list actual variable names used by the model.
- `estimate`, `ci_95`, and `p_value` must be numeric, not strings.
- `interpretation` must be a short declarative sentence.
- If you abstain, the result block may be present but must not claim a definitive effect.
- If `status` is `"success"`, then `result.primary_effect` must be fully populated.
- If `status` is `"abstain"`, then `abstain_reason` must be non-empty.

## Hints
{hints_text}

## Constraints
{constraints_text}

## Available Data
{data_summary}\
## Important Reminders
- Use only pre-treatment (pre-time-zero) data for covariates. Any post-treatment \
variable use will invalidate your analysis.
- Use both EHR and imaging data where appropriate to the task.
- If diagnostics indicate that causal assumptions are violated, you should abstain \
or downgrade your confidence rather than report a misleading result.
- Your `answer.json` must be valid JSON with all required fields populated.
"""
    return prompt


def build_prompt_from_dir(
    task_dir: str | Path,
    workspace_root: str | Path = "/home/developer/mimic",
) -> str:
    """Convenience: load task.json from a directory and build the prompt."""
    task = load_task(task_dir)
    return build_prompt(task, workspace_root)
