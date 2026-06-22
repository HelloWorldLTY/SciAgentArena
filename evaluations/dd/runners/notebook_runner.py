"""notebook_runner — Execute agent code in a Jupyter kernel and score output.

Usage::

    python -m runners.notebook_runner <task.json> <agent.py|agent.ipynb>

Supports two output modes (determined by ``task.constraints.output_format``):

- **matplotlib_figure** (default): Agent produces a matplotlib figure.
  Inspector cell extracts axis labels, title, element counts, colorbar.
- **notebook_variables**: Agent defines named Python variables.
  Inspector cell extracts variable values/shapes as tagged JSON.

The runner:
1. Reads the task JSON and the agent source (.py or .ipynb).
2. Builds an in-memory notebook with two cells:
   - **Cell 1 (Agent):** Preamble (inject input data, set Agg backend) + agent code.
   - **Cell 2 (Inspector):** Hidden cell that introspects figures or variables.
3. Executes the notebook via ``nbclient.NotebookClient``.
4. Parses the inspector JSON from cell output.
5. Scores via the scorer referenced in the task JSON.
6. Prints a final score dict to stdout.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

try:
    import nbformat
    from nbclient import NotebookClient

    HAS_NB = True
except ImportError:
    HAS_NB = False

# Sentinel tags for inspector output
_INSPECTOR_START = "__INSPECTOR_JSON_START__"
_INSPECTOR_END = "__INSPECTOR_JSON_END__"
_VARIABLE_START = "__VARIABLE_JSON_START__"
_VARIABLE_END = "__VARIABLE_JSON_END__"


# ---------------------------------------------------------------------------
# Dynamic scorer loading (shared pattern with batch_runner)
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
# Notebook construction
# ---------------------------------------------------------------------------


def _build_preamble(task: dict) -> str:
    """Return Python source that injects task input and configures matplotlib."""
    input_json = json.dumps(task.get("input", {}))
    return (
        "import matplotlib as _mpl\n"
        "_mpl.use('Agg')  # non-interactive backend\n"
        "import matplotlib.pyplot as plt\n"
        "import json as _json\n"
        f"_TASK_INPUT = _json.loads({input_json!r})\n"
    )


def _build_inspector_cell() -> str:
    """Return Python source for the figure inspector cell.

    The inspector introspects the current matplotlib figure and emits
    a tagged JSON block that the runner parses from cell output.
    """
    return f"""\
import json as _ins_json
import matplotlib.pyplot as _ins_plt

# Grab the figure — prefer a variable named `fig`, fall back to gcf()
try:
    _ins_fig = fig
except NameError:
    _ins_fig = _ins_plt.gcf()

_ins_ax = _ins_fig.axes[0] if _ins_fig.axes else None

_ins_result = {{}}
if _ins_ax is not None:
    _ins_result["x_label"] = _ins_ax.get_xlabel()
    _ins_result["y_label"] = _ins_ax.get_ylabel()
    _ins_result["title"] = _ins_ax.get_title()
    _ins_result["n_collections"] = len(_ins_ax.collections)
    _ins_result["n_lines"] = len(_ins_ax.lines)
    _ins_result["n_patches"] = len(_ins_ax.patches)

    # Count scatter points across all PathCollections
    _ins_n_pts = 0
    for _c in _ins_ax.collections:
        try:
            _ins_n_pts += len(_c.get_offsets())
        except Exception:
            pass
    _ins_result["n_scatter_points"] = _ins_n_pts

    # Check for colorbar — look for extra axes with a mappable
    _ins_result["has_colorbar"] = len(_ins_fig.axes) > 1

print("{_INSPECTOR_START}")
print(_ins_json.dumps(_ins_result))
print("{_INSPECTOR_END}")
"""


def _build_variable_inspector_cell(
    variable_names: list[str], check_plot: bool = False
) -> str:
    """Return Python source for the variable inspector cell.

    For each variable name, checks if it exists in the namespace and
    records its value/shape. Emits tagged JSON that the runner parses.
    """
    var_names_json = json.dumps(variable_names)

    code = f"""\
import json as _vi_json
import sys as _vi_sys

_vi_result = {{}}
_vi_var_names = {var_names_json}

for _vi_name in _vi_var_names:
    try:
        _vi_val = eval(_vi_name)
    except NameError:
        _vi_result[_vi_name] = {{"exists": False}}
        continue

    _vi_entry = {{"exists": True}}

    # numpy array
    try:
        import numpy as _vi_np
        if isinstance(_vi_val, _vi_np.ndarray):
            _vi_entry["shape"] = list(_vi_val.shape)
            if _vi_val.size <= 10000:
                _vi_entry["value"] = _vi_val.tolist()
            else:
                _vi_entry["value"] = "array_too_large"
            _vi_result[_vi_name] = _vi_entry
            continue
    except ImportError:
        pass

    # pandas DataFrame or Series
    try:
        import pandas as _vi_pd
        if isinstance(_vi_val, _vi_pd.DataFrame):
            _vi_entry["shape"] = list(_vi_val.shape)
            _vi_entry["value"] = "dataframe"
            _vi_result[_vi_name] = _vi_entry
            continue
        if isinstance(_vi_val, _vi_pd.Series):
            _vi_entry["shape"] = [len(_vi_val)]
            _vi_entry["value"] = "series"
            _vi_result[_vi_name] = _vi_entry
            continue
    except ImportError:
        pass

    # list or tuple
    if isinstance(_vi_val, (list, tuple)):
        _vi_entry["shape"] = [len(_vi_val)]
        if len(_vi_val) <= 10000:
            _vi_entry["value"] = list(_vi_val)
        else:
            _vi_entry["value"] = "list_too_large"
        _vi_result[_vi_name] = _vi_entry
        continue

    # scalar (int, float, str, bool)
    if isinstance(_vi_val, (int, float, str, bool)):
        _vi_entry["value"] = _vi_val
        _vi_result[_vi_name] = _vi_entry
        continue

    # fallback: try to convert
    try:
        _vi_entry["value"] = str(_vi_val)
    except Exception:
        _vi_entry["value"] = "unconvertible"
    _vi_result[_vi_name] = _vi_entry
"""

    if check_plot:
        code += """
# Check if a matplotlib plot exists with content
try:
    import matplotlib.pyplot as _vi_plt
    _vi_fig = _vi_plt.gcf()
    _vi_has_plot = False
    for _vi_ax in _vi_fig.axes:
        if (_vi_ax.collections or _vi_ax.lines or _vi_ax.patches
                or _vi_ax.images or _vi_ax.texts):
            _vi_has_plot = True
            break
    _vi_result["__plot_exists__"] = _vi_has_plot
except Exception:
    _vi_result["__plot_exists__"] = False
"""

    code += f"""
print("{_VARIABLE_START}")
print(_vi_json.dumps(_vi_result, default=str))
print("{_VARIABLE_END}")
"""
    return code


def build_notebook(task: dict, agent_source: str) -> "nbformat.NotebookNode":
    """Construct an in-memory notebook with agent + inspector cells."""
    output_format = task.get("constraints", {}).get(
        "output_format", "matplotlib_figure"
    )

    if output_format == "notebook_variables":
        checks = task.get("scoring_logic", {}).get("checks", [])
        var_names = [c["variable"] for c in checks if "variable" in c]
        check_plot = any(c.get("assertion") == "plot_exists" for c in checks)
        inspector = _build_variable_inspector_cell(var_names, check_plot)
    else:
        inspector = _build_inspector_cell()

    nb = nbformat.v4.new_notebook()
    preamble = _build_preamble(task)
    nb.cells = [
        nbformat.v4.new_code_cell(source=preamble + "\n" + agent_source),
        nbformat.v4.new_code_cell(source=inspector),
    ]
    return nb


# ---------------------------------------------------------------------------
# Notebook execution & output parsing
# ---------------------------------------------------------------------------


def execute_notebook(nb: "nbformat.NotebookNode", timeout: int) -> tuple[bool, str]:
    """Execute *nb* in a fresh kernel. Return (success, error_message)."""
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": "."}},
    )
    try:
        client.execute()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def parse_inspector_output(nb: "nbformat.NotebookNode") -> dict | None:
    """Extract figure inspector JSON from the last cell's outputs."""
    if len(nb.cells) < 2:
        return None
    inspector_cell = nb.cells[-1]
    for output in inspector_cell.get("outputs", []):
        text = output.get("text", "")
        if _INSPECTOR_START in text:
            start = text.index(_INSPECTOR_START) + len(_INSPECTOR_START)
            end = text.index(_INSPECTOR_END)
            return json.loads(text[start:end].strip())
    return None


def parse_variable_inspector_output(nb: "nbformat.NotebookNode") -> dict | None:
    """Extract variable inspector JSON from the last cell's outputs."""
    if len(nb.cells) < 2:
        return None
    inspector_cell = nb.cells[-1]
    for output in inspector_cell.get("outputs", []):
        text = output.get("text", "")
        if _VARIABLE_START in text:
            start = text.index(_VARIABLE_START) + len(_VARIABLE_START)
            end = text.index(_VARIABLE_END)
            return json.loads(text[start:end].strip())
    return None


# ---------------------------------------------------------------------------
# Waterfall scoring
# ---------------------------------------------------------------------------


def score(task: dict, exec_ok: bool, exec_error: str, inspector: dict | None) -> dict:
    """Apply waterfall scoring for interactive/notebook tasks."""
    result = {
        "task_id": task["id"],
        "executability": 0.0,
        "validity": 0.0,
        "correctness": 0.0,
        "strategic_success": 0.0,
        "details": {},
    }

    # Gate 1: Executability
    if not exec_ok:
        result["details"]["error"] = exec_error[:2000]
        return result

    result["executability"] = 1.0

    # Gate 2: Inspector produced output?
    if inspector is None:
        result["details"]["error"] = (
            "Inspector cell produced no parseable output. "
            "Agent may not have created the expected output."
        )
        return result

    result["validity"] = 1.0

    # Gate 3: Score via task scorer
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
        scorer_result = scorer_fn(inspector, task)
    except Exception as exc:
        result["details"]["error"] = f"Scorer raised an exception: {exc}"
        return result

    result["correctness"] = scorer_result.get("correctness", 0.0)
    result["strategic_success"] = scorer_result.get("strategic_success", 0.0)
    result["details"] = scorer_result.get("details", {})

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    if not HAS_NB:
        print(
            "ERROR: nbformat and nbclient are required for notebook_runner.\n"
            "Install them with: conda install nbformat nbclient ipykernel",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Run an agent script in a notebook and score the output.",
    )
    parser.add_argument("task_json", help="Path to the task JSON file")
    parser.add_argument("agent_py", help="Path to the agent .py or .ipynb file")
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print the score JSON"
    )
    args = parser.parse_args(argv)

    # Load task
    with open(args.task_json) as f:
        task = json.load(f)

    # Load agent source (support .py and .ipynb)
    agent_path = Path(args.agent_py)
    if agent_path.suffix == ".ipynb":
        nb_agent = nbformat.read(str(agent_path), as_version=4)
        agent_source = "\n\n".join(
            c.source for c in nb_agent.cells if c.cell_type == "code"
        )
    else:
        agent_source = agent_path.read_text()

    timeout = task.get("constraints", {}).get("timeout_seconds", 60)

    # Build and execute
    start = time.monotonic()
    nb = build_notebook(task, agent_source)
    exec_ok, exec_error = execute_notebook(nb, timeout)
    wall = round(time.monotonic() - start, 3)

    # Parse inspector output (branch on output_format)
    output_format = task.get("constraints", {}).get(
        "output_format", "matplotlib_figure"
    )
    if exec_ok:
        if output_format == "notebook_variables":
            inspector = parse_variable_inspector_output(nb)
        else:
            inspector = parse_inspector_output(nb)
    else:
        inspector = None

    # Score
    result = score(task, exec_ok, exec_error, inspector)
    result["wall_time_seconds"] = wall

    print(json.dumps(result, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
