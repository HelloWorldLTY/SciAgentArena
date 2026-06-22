"""registry.py — unified, category-aware task index.

Discovers every task JSON across the three task roots, normalizes its
metadata, and exposes one flat registry. Tasks fall into five categories:

    C1  Chemical Data Preprocessing
    C2  Chemical Data Analysis
    C3  Molecule Optimization
    C4  Chemical Safety Assessment
    C5  Chemical Claim Validation

Two task-declared fields drive all routing, so the dispatcher in
:mod:`evaluate` needs no per-category special cases:

    constraints.output_format  ->  which runner executes the task
    scoring_logic.scorer       ->  which scorer grades it (``module.function``)
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Where task definitions live. C1/C3/C4 (+ C2 vision) batch tasks and C2
# interactive tasks share the master skeleton; C5 stays self-contained.
TASK_GLOBS = [
    "tasks_batch/*/*.json",
    "tasks_interactive/*/*.json",
]

# Authoritative mapping from the task's ``type`` field to a category.
CATEGORY_BY_TYPE = {
    "chemical_data_preprocessing": ("C1", "Chemical Data Preprocessing"),
    "chemical_data_analysis": ("C2", "Chemical Data Analysis"),
    "molecule_optimization": ("C3", "Molecule Optimization"),
    "chemical_safety_assessment": ("C4", "Chemical Safety Assessment"),
    "chemical_claim_validation": ("C5", "Chemical Claim Validation"),
}

# Fallback when ``type`` is missing/unknown: infer the type from the task-id
# prefix (prefixes are unchanged from the original suite).
CATEGORY_BY_PREFIX = {
    "tech_": "chemical_data_preprocessing",
    "ana_": "chemical_data_analysis",
    "des_": "molecule_optimization",
    "reg_": "chemical_safety_assessment",
    "val_": "chemical_claim_validation",
}

# output_format values that must run inside a Jupyter kernel (figure or
# named-variable introspection). Everything else runs as a stdout/stream batch.
NOTEBOOK_FORMATS = {"notebook_variables", "matplotlib_figure"}

# Scorer modules whose oracles depend on PyTDC (GuacaMol / DRD2 / penalized
# logP, etc.). These C3 design tasks need `uv pip install pytdc` before they
# can be executed end-to-end.
TDC_SCORER_MODULES = {
    "deco_hop_scorer",
    "drd2_scorer",
    "qed_scorer",
    "penalized_logp_scorer",
    "mpo_scorer",
    "scaffold_hop_scorer",
    "valsartan_smarts_scorer",
    "guacamol_goal_scorer",
}


def _runner_for(output_format: str) -> str:
    if output_format in NOTEBOOK_FORMATS:
        return "notebook_runner"
    if output_format == "oracle_stream":
        return "design_runner"
    return "batch_runner"


# runner module -> coarse execution mode label
_MODE_BY_RUNNER = {
    "notebook_runner": "notebook",
    "design_runner": "design",
    "batch_runner": "batch",
}


def _category(task: dict, task_id: str) -> tuple[str, str]:
    ty = task.get("type", "")
    if ty in CATEGORY_BY_TYPE:
        return CATEGORY_BY_TYPE[ty]
    for prefix, mapped_type in CATEGORY_BY_PREFIX.items():
        if task_id.startswith(prefix):
            return CATEGORY_BY_TYPE[mapped_type]
    return ("C?", "Unknown")


def build_registry() -> list[dict]:
    """Scan all task roots and return a list of normalized registry entries."""
    entries: list[dict] = []
    for pattern in TASK_GLOBS:
        for path in sorted(glob.glob(str(ROOT / pattern))):
            rel = os.path.relpath(path, ROOT).replace("\\", "/")
            if "/data/" in rel:  # skip non-task JSON living under a data/ dir
                continue
            try:
                task = json.load(open(path, encoding="utf-8"))
            except Exception as exc:  # malformed task JSON — surface, don't crash
                entries.append(
                    {
                        "task_id": Path(path).stem,
                        "task_path": rel,
                        "error": f"parse_error: {exc}",
                    }
                )
                continue

            task_id = task.get("id", Path(path).stem)
            output_format = task.get("constraints", {}).get(
                "output_format", "json_stdout"
            )
            scorer = task.get("scoring_logic", {}).get("scorer", "")
            scorer_module = scorer.split(".")[0] if scorer else ""
            category, label = _category(task, task_id)
            runner = _runner_for(output_format)

            entries.append(
                {
                    "task_id": task_id,
                    "category": category,
                    "category_label": label,
                    "type": task.get("type", ""),
                    "mode": _MODE_BY_RUNNER.get(runner, "batch"),
                    "runner": runner,
                    "output_format": output_format,
                    "scorer": scorer,
                    "scorer_module": scorer_module,
                    "difficulty": task.get("metadata", {}).get("difficulty", ""),
                    "needs_tdc": output_format == "oracle_stream"
                    or scorer_module in TDC_SCORER_MODULES,
                    "task_path": rel,
                }
            )
    return entries


def registry_by_id() -> dict[str, dict]:
    """Return the registry keyed by task_id (parse-error entries excluded)."""
    return {e["task_id"]: e for e in build_registry() if "error" not in e}


if __name__ == "__main__":
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser(description="Dump the unified task registry.")
    ap.add_argument("--json", action="store_true", help="emit full registry as JSON")
    args = ap.parse_args()

    reg = build_registry()
    if args.json:
        print(json.dumps(reg, indent=2))
    else:
        by_cat = Counter(e.get("category", "C?") for e in reg)
        by_mode = Counter(e.get("mode", "?") for e in reg)
        for cat in sorted(by_cat):
            print(f"{cat}: {by_cat[cat]} tasks")
        print(f"modes: {dict(by_mode)}")
        print(f"total: {len(reg)} tasks")
