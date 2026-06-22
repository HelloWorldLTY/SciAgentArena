"""design_runner — Execute a design agent and score its oracle-call stream.

Usage::

    python -m runners.design_runner <task.json> <agent.py>

Runs the agent in a subprocess with the full ``task["input"]`` payload
forwarded via the AGENT4S_INPUT_JSON env var (lead SMILES, oracle budget,
etc.). The agent's budgeted ``oracle.score(...)`` calls are recorded to a
stream; the scorer named in ``scoring_logic.scorer`` reads that stream and
returns the final score dict, printed to stdout.
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
    """Import ``scorers.<module>.<function>`` from a dotted string."""
    parts = dotted_name.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"scorer must be '<module>.<function>', got '{dotted_name}'")
    module_name, func_name = parts
    mod = importlib.import_module(f"scorers.{module_name}")
    return getattr(mod, func_name)


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------


def _sanitize_agent_source(path: str) -> str:
    """If the file starts with a markdown fence, extract the Python block."""
    try:
        text = Path(path).read_text()
    except Exception:
        return ""
    stripped = text.lstrip()
    if not stripped.startswith("```"):
        return ""
    if "```python" in stripped:
        start_idx = stripped.find("```python") + len("```python")
        end_idx = stripped.find("```", start_idx)
        if end_idx != -1:
            return stripped[start_idx:end_idx].strip()
    first = stripped.find("```")
    last = stripped.rfind("```")
    if last > first:
        return stripped[first + 3 : last].strip()
    return ""


def run_agent(
    agent_path: str,
    input_json: str,
    timeout: int,
    project_root: str | None = None,
    stream_path: str | None = None,
) -> dict:
    """Run *agent_path* in a child process and return an execution record.

    Returns a dict with keys: ``stdout``, ``stderr``, ``returncode``,
    ``timed_out``, ``wall_time``, ``stream_path``.
    """
    env = os.environ.copy()
    env["AGENT4S_INPUT_JSON"] = input_json
    env["AGENT4S_NO_NETWORK"] = "1"
    if stream_path:
        env["AGENT4S_OPTIMIZER_STREAM"] = stream_path
        # Truncate any pre-existing file from a previous run so reads are clean.
        try:
            open(stream_path, "w").close()
        except OSError:
            pass

    # Make `import scorers.*` and `import runners.*` work from the agent,
    # even if it lives outside the repo root.
    root = project_root or str(Path(__file__).resolve().parent.parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root + (os.pathsep + existing if existing else "")

    start = time.monotonic()
    tmp_file = None
    try:
        sanitized = _sanitize_agent_source(agent_path)
        exec_path = agent_path
        if sanitized:
            tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
            tmp_file.write(sanitized)
            tmp_file.flush()
            exec_path = tmp_file.name

        # Inline runner so that ``open("AGENT4S_INPUT_JSON")`` behaves as
        # a convenience handle to the env payload.
        runner_code = (
            "try:\n"
            "    import sitecustomize  # noqa: F401\n"
            "except Exception:\n"
            "    pass\n"
            "import builtins\n"
            "import io\n"
            "import os\n"
            "from pathlib import Path\n"
            "_orig_open = builtins.open\n"
            "def _open(file, *args, **kwargs):\n"
            "    if str(file) == 'AGENT4S_INPUT_JSON':\n"
            "        return io.StringIO(os.environ.get('AGENT4S_INPUT_JSON', '{}'))\n"
            "    return _orig_open(file, *args, **kwargs)\n"
            "builtins.open = _open\n"
            "_orig_path_open = Path.open\n"
            "def _path_open(self, *args, **kwargs):\n"
            "    if str(self) == 'AGENT4S_INPUT_JSON':\n"
            "        return io.StringIO(os.environ.get('AGENT4S_INPUT_JSON', '{}'))\n"
            "    return _orig_path_open(self, *args, **kwargs)\n"
            "Path.open = _path_open\n"
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
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "timed_out": False,
            "wall_time": round(wall, 3),
            "stream_path": stream_path,
        }
    except subprocess.TimeoutExpired as exc:
        wall = time.monotonic() - start
        return {
            "stdout": (
                (exc.stdout or b"").decode(errors="replace")
                if isinstance(exc.stdout, (bytes, bytearray))
                else (exc.stdout or "")
            ),
            "stderr": (
                (exc.stderr or b"").decode(errors="replace")
                if isinstance(exc.stderr, (bytes, bytearray))
                else (exc.stderr or "")
            ),
            "returncode": -1,
            "timed_out": True,
            "wall_time": round(wall, 3),
            "stream_path": stream_path,
        }
    finally:
        if tmp_file is not None:
            try:
                os.unlink(tmp_file.name)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Forbidden-import check
# ---------------------------------------------------------------------------


def check_forbidden_imports(agent_path: str, forbidden: list[str]) -> list[str]:
    """Return list of forbidden modules that appear as imports in *agent_path*."""
    violations: list[str] = []
    try:
        source = Path(agent_path).read_text()
    except OSError:
        return violations
    for mod in forbidden:
        if f"import {mod}" in source or f"from {mod}" in source:
            violations.append(mod)
    return violations


# ---------------------------------------------------------------------------
# Safe input payload
# ---------------------------------------------------------------------------


def build_safe_input(task: dict) -> dict:
    """Forward every JSON-serializable field from ``task['input']``.

    Prevents leaking scorer-only metadata (ground-truth files, etc.) while
    still giving optimizer tasks the fields they need (lead SMILES,
    similarity threshold, oracle budget, etc.).
    """
    raw = task.get("input", {})
    if not isinstance(raw, dict):
        return {}
    safe: dict = {}
    for key, value in raw.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        safe[key] = value
    return safe


# ---------------------------------------------------------------------------
# Stream-based recovery (for optimizer tasks)
# ---------------------------------------------------------------------------


def _read_stream_records(stream_path: str | None) -> list[dict]:
    if not stream_path:
        return []
    try:
        text = Path(stream_path).read_text()
    except OSError:
        return []
    records: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("event") == "score":
            records.append(obj)
    return records


def _oracle_value_from_record(rec: dict) -> float | None:
    for key in ("oracle_score", "penalized_logp", "drd2", "mpo"):
        val = rec.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None


def _compute_auc_top_k(
    records: list[dict],
    budget: int,
    k: int,
) -> tuple[float, int]:
    """PMO-style AUC top-k over the oracle-call trajectory.

    At each budget step n, take the mean of the current top-k oracle values
    observed so far (over valid records only; invalid calls advance the step
    counter but do not update the running top-k). Plateau-fill the remaining
    budget steps with the final top-k mean so early-quitting optimizers do not
    get a free pass. Return (auc, n_calls_recorded).
    """
    if budget <= 0 or k <= 0 or not records:
        return 0.0, len(records)

    capped = records[:budget]
    top_k: list[float] = []  # kept sorted descending
    step_means: list[float] = []
    for rec in capped:
        v = _oracle_value_from_record(rec) if rec.get("valid") else None
        if v is not None:
            if len(top_k) < k:
                top_k.append(v)
                top_k.sort(reverse=True)
            elif v > top_k[-1]:
                top_k[-1] = v
                top_k.sort(reverse=True)
        step_means.append(sum(top_k) / len(top_k) if top_k else 0.0)

    last = step_means[-1]
    while len(step_means) < budget:
        step_means.append(last)

    return sum(step_means) / budget, len(capped)


def _stream_recovered_stdout(records: list[dict], k: int) -> str | None:
    """Synthesize a ``{"analogs": [...]}`` JSON string from streamed scores.

    Selects the top-k molecules the agent passed to ``oracle.score()`` ranked
    by oracle value, prioritizing successes when ties exist. Dedups by
    canonical SMILES. Always fills up to k slots when at least k unique valid
    candidates were scored — this gives the scorer the optimizer's actual
    "top-k exploration" view, not just the bare list of clear successes.
    Returns ``None`` only when no parsable, valid SMILES were ever scored.

    Note on metric semantics with this fill behavior:
      * ``opt_success_rate`` (= n_success / k) is unchanged; padding does
        not introduce new successes.
      * ``submit_rate`` / ``valid_rate`` now reflect "did the agent make
        ≥k successful oracle.score() calls with parseable SMILES?", which
        is the right exploration-breadth proxy for code-gen tasks where
        the harness — not the agent — curates the final list.
    """
    if not records:
        return None
    seen: set[str] = set()
    # Tuples are (sort_key=(success_flag, oracle_value), canonical_smiles).
    # Sorting descending puts successes first, then non-success valids,
    # each ranked by oracle value within their bucket.
    pool: list[tuple[tuple[int, float], str]] = []
    for rec in records:
        canonical = rec.get("canonical_smiles")
        if not canonical or canonical in seen:
            continue
        if not rec.get("valid"):
            continue
        oracle_value = rec.get("oracle_score")
        if oracle_value is None:
            oracle_value = rec.get("penalized_logp")
        if oracle_value is None:
            oracle_value = rec.get("drd2")
        if oracle_value is None:
            oracle_value = rec.get("mpo")
        if oracle_value is None:
            continue
        seen.add(canonical)
        success_flag = 1 if rec.get("success") else 0
        pool.append(((success_flag, float(oracle_value)), canonical))
    if not pool:
        return None
    pool.sort(reverse=True, key=lambda t: t[0])
    chosen = [smi for _, smi in pool[:k]]
    return json.dumps({"analogs": chosen})


# ---------------------------------------------------------------------------
# Waterfall scoring
# ---------------------------------------------------------------------------


def score(task: dict, exec_record: dict) -> dict:
    """Apply waterfall scoring: executability → validity → success_rate.

    Candidate selection is always driven by the oracle stream: the top-k
    successful SMILES (by penalized_logp) from ``score()`` calls the agent
    made are fed to the scorer. The agent's stdout is ignored. Whether the
    subprocess exited cleanly only affects ``executability``.
    """
    result = {
        "task_id": task["id"],
        "executability": 0.0,
        "validity": 0.0,
        "success_rate": 0.0,
        "strategic_success": 0.0,
        "details": {},
        "wall_time_seconds": exec_record["wall_time"],
        "auc_top_k": 0.0,
        "n_oracle_calls": 0,
        "invalid_smiles_rate": 0.0,
        "n_invalid_smiles": 0,
    }

    stream_records = _read_stream_records(exec_record.get("stream_path"))
    result["details"]["n_stream_records"] = len(stream_records)

    budget = int(task.get("input", {}).get("oracle_budget", 0))
    k_for_auc = int(task.get("scoring_logic", {}).get("k", 10))
    auc_val, n_calls = _compute_auc_top_k(stream_records, budget, k_for_auc)
    result["auc_top_k"] = round(auc_val, 6)
    result["n_oracle_calls"] = n_calls
    n_invalid = sum(1 for r in stream_records if not r.get("valid"))
    result["invalid_smiles_rate"] = (
        round(n_invalid / n_calls, 6) if n_calls > 0 else 0.0
    )
    result["n_invalid_smiles"] = n_invalid

    if exec_record["timed_out"]:
        result["details"]["error"] = "Agent timed out"
        result["details"]["stderr"] = exec_record.get("stderr", "")[:2000]
    elif exec_record["returncode"] != 0:
        result["details"][
            "error"
        ] = f"Agent exited with code {exec_record['returncode']}"
        result["details"]["stderr"] = exec_record["stderr"][:2000]
    else:
        result["executability"] = 1.0

    k = int(task.get("scoring_logic", {}).get("k", 10))
    synth = _stream_recovered_stdout(stream_records, k)
    if synth is None:
        result["details"]["error"] = (
            result["details"].get("error")
            or "Oracle stream empty: agent made no score() calls"
        )
        return result
    stdout_for_scorer = synth
    result["details"]["n_submitted_analogs"] = len(json.loads(synth)["analogs"])

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
        scorer_result = scorer_fn(stdout_for_scorer, task)
    except Exception as exc:
        result["details"]["error"] = f"Scorer raised an exception: {exc}"
        return result

    # Scorers on this branch return {basic_score, design_result, details}.
    # Map those to the waterfall schema without losing information.
    if "basic_score" in scorer_result or "design_result" in scorer_result:
        design = scorer_result.get("design_result", {}) or {}
        result["validity"] = float(scorer_result.get("basic_score", 0.0))
        result["success_rate"] = float(design.get("opt_success_rate", 0.0))
        result["strategic_success"] = float(design.get("opt_success", 0.0))
        result["details"] = {
            "design_result": design,
            **(scorer_result.get("details", {}) or {}),
        }
    else:
        # Batch scorers return ``correctness``; map it onto our
        # ``success_rate`` waterfall field.
        result["validity"] = float(scorer_result.get("validity", 0.0))
        result["success_rate"] = float(scorer_result.get("correctness", 0.0))
        result["strategic_success"] = float(scorer_result.get("strategic_success", 0.0))
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
    parser.add_argument(
        "--stream-path",
        default=None,
        help=(
            "Path for the oracle's JSONL stream. If omitted, a temp file is "
            "used and cleaned up automatically."
        ),
    )
    args = parser.parse_args(argv)

    with open(args.task_json) as f:
        task = json.load(f)

    forbidden = task.get("constraints", {}).get("forbidden_imports", [])
    violations = check_forbidden_imports(args.agent_py, forbidden)
    if violations:
        result = {
            "task_id": task["id"],
            "executability": 0.0,
            "validity": 0.0,
            "success_rate": 0.0,
            "strategic_success": 0.0,
            "details": {"error": f"Forbidden imports detected: {violations}"},
            "wall_time_seconds": 0.0,
        }
        print(json.dumps(result, indent=2 if args.pretty else None))
        return

    safe_input = build_safe_input(task)
    input_json = json.dumps(safe_input)
    timeout = task.get("constraints", {}).get("timeout_seconds", 30)

    stream_path = args.stream_path
    tmp_stream = None
    if stream_path is None:
        tmp_stream = tempfile.NamedTemporaryFile(
            mode="w", suffix=".stream.jsonl", delete=False
        )
        tmp_stream.close()
        stream_path = tmp_stream.name

    try:
        exec_record = run_agent(
            args.agent_py,
            input_json,
            timeout,
            stream_path=stream_path,
        )
        result = score(task, exec_record)
        result["stderr_tail"] = exec_record.get("stderr", "")[-2000:]
        result["stream_path"] = stream_path
        print(json.dumps(result, indent=2 if args.pretty else None))
    finally:
        if tmp_stream is not None:
            try:
                os.unlink(tmp_stream.name)
            except OSError:
                pass


if __name__ == "__main__":
    main()
