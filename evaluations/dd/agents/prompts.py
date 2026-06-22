"""Build the (system, user) prompt pair an LLM agent receives for a task.

The system instruction adapts to the task's execution mode (its
``constraints.output_format``) so the generated script matches the contract the
matching runner expects — stdout JSON for batch, a ``fig`` variable for figure
tasks, named variables for variable tasks, the budgeted-oracle stream for
design tasks.
"""

from __future__ import annotations

import json

_BASE = (
    "You are an expert autonomous scientist solving a cheminformatics task. "
    "Write a single, complete, self-contained, executable Python script.\n\n"
    "STRICT CONSTRAINTS:\n"
    "1. OFFLINE: do not import requests, urllib, http, socket, openai, "
    "anthropic, or google, and do not access the network.\n"
    "2. LIBRARIES: rdkit, pandas, numpy, scipy, scikit-learn, networkx, "
    "matplotlib and the Python standard library are available.\n"
    "3. ROBUSTNESS: handle invalid SMILES / missing data gracefully (try/except).\n"
    "4. DATA: process the real inputs provided; never fabricate dummy data.\n"
)

_BY_FORMAT = {
    "json_stdout": (
        "OUTPUT (batch mode): read the task input from the AGENT4S_INPUT_JSON "
        "environment variable (a JSON object). Print ONLY the final result to "
        "stdout as valid JSON. Do not print debug text to stdout (use stderr).\n"
    ),
    "matplotlib_figure": (
        "OUTPUT (notebook figure mode): your code runs inside a notebook cell. "
        "The task input is preloaded as the variable `_TASK_INPUT` (a dict) and "
        "matplotlib.pyplot is available as plt. Do NOT read env vars or argv. "
        "Build the requested plot and leave the final matplotlib Figure in a "
        "variable named `fig`.\n"
    ),
    "notebook_variables": (
        "OUTPUT (notebook variables mode): your code runs inside a notebook "
        "cell. The task input is preloaded as the variable `_TASK_INPUT` (a "
        "dict). Do NOT read env vars or argv, and do not print the answer. "
        "Assign each result the task asks for to a top-level variable using "
        "exactly the name requested.\n"
    ),
    "oracle_stream": (
        "OUTPUT (design mode): read the task input from AGENT4S_INPUT_JSON. "
        "Follow the task's oracle instructions exactly — import the budgeted "
        "oracle from the scorers.oracle_budget_* module named in the task and "
        "call oracle.score(smiles) up to the oracle budget. Do not import any "
        "other module under scorers/. stdout is ignored.\n"
    ),
}


def build_system_instruction(task: dict) -> str:
    output_format = task.get("constraints", {}).get("output_format", "json_stdout")
    return _BASE + "\n" + _BY_FORMAT.get(output_format, _BY_FORMAT["json_stdout"])


def build_user_prompt(task: dict) -> str:
    output_format = task.get("constraints", {}).get("output_format", "json_stdout")
    inp = task.get("input", {}) or {}
    parts = [task.get("prompt", "").strip()]

    if output_format in ("json_stdout", "oracle_stream"):
        parts.append(
            "\nTask input (delivered via AGENT4S_INPUT_JSON):\n"
            + json.dumps(inp, indent=2)
        )
    else:  # notebook modes: input is injected as _TASK_INPUT
        keys = list(inp.keys()) if isinstance(inp, dict) else []
        parts.append(f"\nThe variable `_TASK_INPUT` is a dict with keys: {keys}")

    forbidden = task.get("constraints", {}).get("forbidden_imports", [])
    if forbidden:
        parts.append("\nForbidden imports: " + ", ".join(forbidden))
    return "\n".join(parts)
