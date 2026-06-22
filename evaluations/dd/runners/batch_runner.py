"""batch_runner — Execute an agent .py script in a subprocess and score its output.

Usage::

    python -m runners.batch_runner <task.json> <agent.py>

The runner:
1. Reads the task JSON to get input data, constraints, and scoring config.
2. Passes task input to the agent via the AGENT4S_INPUT_JSON env var.
3. Runs the agent script in a subprocess with a timeout.
4. Captures stdout (expected to be JSON) and scores it via the scorer
   referenced in the task's ``scoring_logic.scorer`` field.
5. Prints a final score dict to stdout.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dynamic scorer loading
# ---------------------------------------------------------------------------


def load_scorer(dotted_name: str):
    """Import ``scorers.<module>.<function>`` from a dotted string.

    Example: ``"oracle_chem.molecular_weight_score"``
    → imports ``scorers.oracle_chem`` and returns the ``molecular_weight_score``
    callable.
    """
    parts = dotted_name.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"scorer must be '<module>.<function>', got '{dotted_name}'")
    module_name, func_name = parts
    mod = importlib.import_module(f"scorers.{module_name}")
    return getattr(mod, func_name)


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------


def run_agent(
    agent_path: str,
    input_json: str,
    timeout: int,
    forbidden_imports: list[str] | None = None,
) -> dict:
    """Run *agent_path* in a child process and return an execution record.

    Returns a dict with keys: ``stdout``, ``stderr``, ``returncode``,
    ``timed_out``, ``wall_time``.
    """
    env = os.environ.copy()
    env["AGENT4S_INPUT_JSON"] = input_json

    # Block networking via env hint (informational — not enforced at OS level)
    env["AGENT4S_NO_NETWORK"] = "1"

    def _sanitize_agent_source(path: str) -> str:
        try:
            text = Path(path).read_text()
        except Exception:
            return ""
        stripped = text.lstrip()
        if not stripped.startswith("```"):
            return ""
        # Prefer ```python fenced blocks if present, otherwise strip outer fences.
        if "```python" in stripped:
            start_idx = stripped.find("```python") + len("```python")
            end_idx = stripped.find("```", start_idx)
            if end_idx != -1:
                return stripped[start_idx:end_idx].strip()
        # Fallback: remove leading/trailing fences.
        first = stripped.find("```")
        last = stripped.rfind("```")
        if last > first:
            return stripped[first + 3 : last].strip()
        return ""

    start = time.monotonic()
    try:
        sanitized = _sanitize_agent_source(agent_path)
        exec_path = agent_path
        tmp_file = None
        if sanitized:
            tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
            tmp_file.write(sanitized)
            tmp_file.flush()
            exec_path = tmp_file.name
        runner_code = (
            "try:\n"
            "    import sitecustomize  # noqa: F401\n"
            "except Exception:\n"
            "    pass\n"
            "import builtins\n"
            "import io\n"
            "import os\n"
            "from pathlib import Path\n"
            "_agent4s_original_open = builtins.open\n"
            "def _agent4s_open(file, *args, **kwargs):\n"
            "    if str(file) == 'AGENT4S_INPUT_JSON':\n"
            "        return io.StringIO(os.environ.get('AGENT4S_INPUT_JSON', '{}'))\n"
            "    return _agent4s_original_open(file, *args, **kwargs)\n"
            "builtins.open = _agent4s_open\n"
            "_agent4s_original_path_open = Path.open\n"
            "def _agent4s_path_open(self, *args, **kwargs):\n"
            "    if str(self) == 'AGENT4S_INPUT_JSON':\n"
            "        return io.StringIO(os.environ.get('AGENT4S_INPUT_JSON', '{}'))\n"
            "    return _agent4s_original_path_open(self, *args, **kwargs)\n"
            "Path.open = _agent4s_path_open\n"
            "import runpy\n"
            f"runpy.run_path({exec_path!r}, run_name='__main__')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        wall = time.monotonic() - start
        if tmp_file is not None:
            try:
                os.unlink(tmp_file.name)
            except Exception:
                pass
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "timed_out": False,
            "wall_time": round(wall, 3),
        }
    except subprocess.TimeoutExpired as exc:
        wall = time.monotonic() - start
        return {
            "stdout": (exc.stdout or b"").decode(errors="replace"),
            "stderr": (exc.stderr or b"").decode(errors="replace"),
            "returncode": -1,
            "timed_out": True,
            "wall_time": round(wall, 3),
        }


# ---------------------------------------------------------------------------
# Forbidden-import check (lightweight, regex-free)
# ---------------------------------------------------------------------------


def check_forbidden_imports(agent_path: str, forbidden: list[str]) -> list[str]:
    """Return list of forbidden modules that appear as imports in *agent_path*."""
    violations: list[str] = []
    try:
        source = Path(agent_path).read_text()
    except OSError:
        return violations
    for mod in forbidden:
        # Check for "import <mod>" or "from <mod>"
        if f"import {mod}" in source or f"from {mod}" in source:
            violations.append(mod)
    return violations


# ---------------------------------------------------------------------------
# Waterfall scoring
# ---------------------------------------------------------------------------


def score(task: dict, exec_record: dict) -> dict:
    """Apply waterfall scoring: executability → validity → correctness.

    Returns the final score dict.
    """
    result = {
        "task_id": task["id"],
        "executability": 0.0,
        "validity": 0.0,
        "correctness": 0.0,
        "strategic_success": 0.0,
        "details": {},
        "wall_time_seconds": exec_record["wall_time"],
    }

    # --- Gate 1: Executability ---
    if exec_record["timed_out"]:
        result["details"]["error"] = "Agent timed out"
        return result
    if exec_record["returncode"] != 0:
        result["details"][
            "error"
        ] = f"Agent exited with code {exec_record['returncode']}"
        result["details"]["stderr"] = exec_record["stderr"][:2000]
        return result

    result["executability"] = 1.0

    # --- Gate 2+3: Validity & Correctness (delegated to scorer) ---
    scorer_name = task.get("scoring_logic", {}).get("scorer")
    if not scorer_name:
        result["details"]["error"] = "No scorer defined in task JSON"
        return result

    try:
        scorer_fn = load_scorer(scorer_name)
    except Exception as exc:
        result["details"]["error"] = f"Failed to load scorer: {exc}"
        return result

    try:
        scorer_result = scorer_fn(exec_record["stdout"], task)
    except Exception as exc:
        result["details"]["error"] = f"Scorer raised an exception: {exc}"
        return result

    result["validity"] = scorer_result.get("validity", 0.0)
    result["correctness"] = scorer_result.get("correctness", 0.0)
    result["strategic_success"] = scorer_result.get("strategic_success", 0.0)
    result["details"] = scorer_result.get("details", {})

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run an agent script against a batch task and score it.",
    )
    parser.add_argument("task_json", help="Path to the task JSON file")
    parser.add_argument("agent_py", help="Path to the agent Python script")
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print the score JSON"
    )
    args = parser.parse_args(argv)

    # Load task
    with open(args.task_json) as f:
        task = json.load(f)

    # Check forbidden imports before running
    forbidden = task.get("constraints", {}).get("forbidden_imports", [])
    violations = check_forbidden_imports(args.agent_py, forbidden)
    if violations:
        result = {
            "task_id": task["id"],
            "executability": 0.0,
            "validity": 0.0,
            "correctness": 0.0,
            "strategic_success": 0.0,
            "details": {"error": f"Forbidden imports detected: {violations}"},
            "wall_time_seconds": 0.0,
        }
        print(json.dumps(result, indent=2 if args.pretty else None))
        return

    # Forward the full task input payload to the agent. Ground truth never
    # lives in task["input"] (it is under scoring_logic / ground_truth_file),
    # so forwarding the whole payload is safe and lets tasks whose input is
    # not a plain file_path (e.g. vision image paths, inline molecule lists)
    # reach the agent intact.
    raw_input = task.get("input", {})
    input_json = json.dumps(raw_input if isinstance(raw_input, dict) else {})
    timeout = task.get("constraints", {}).get("timeout_seconds", 30)

    # Execute
    exec_record = run_agent(args.agent_py, input_json, timeout)

    # Score
    result = score(task, exec_record)
    print(json.dumps(result, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
