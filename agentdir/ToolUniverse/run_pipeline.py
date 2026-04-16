#!/usr/bin/env python3
"""Run a multi-step pipeline prompt and build a single ``onestep.py``.

The prompt file is split by lines matching ``# TASK <label>`` (or
``## TASK <label>``); everything until the next ``# TASK`` header becomes
the prompt for that step. Each step is sent to ChatGPT + ToolUniverse
independently and appended to ``<name>_onestep.py`` with a section header.

Output files in ``--out-dir`` (default ``outputs/``):

- ``<name>_onestep.py``     — concatenated Python script for all steps.
- ``<name>_output.txt``     — per-step raw reply + extracted code.
- ``<name>_tools_used.txt`` — compact tool-call summary per step.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _agent_core import (  # noqa: E402
    build_llm_client,
    chat_with_tools,
    extract_code,
    find_relevant_tools,
    load_dotenv_if_present,
    load_tool_universe,
)


TASK_HEADER_RE = re.compile(r"^\s*#{1,2}\s*TASK\s+([^\s#].*?)\s*$")

SYSTEM_MSG = (
    "You are an expert bioinformatics coding assistant. You have access to "
    "ToolUniverse tools. For each step you MUST call at least one relevant "
    "tool before writing the final code. Final answer must be a single "
    "fenced Python code block (```python ...```) with brief comments; "
    "do NOT re-emit code from earlier steps."
)

TOOL_INSTR = (
    "\n\nToolUniverse (IMPORTANT): You MUST call at least one tool before "
    "writing the code for this step. Output only executable Python with "
    "brief comments. Use import scanpy as sc where needed."
)

ONESTEP_RULES = (
    "\n\nonestep.py (CRITICAL): The pipeline is ONE script built "
    "incrementally. Output ONLY the Python for THIS step. Do NOT repeat "
    "code from earlier steps; assume they already ran. Start your snippet "
    "with one brief comment line naming the step."
)


def parse_tasks(text: str) -> List[Tuple[str, str]]:
    """Split a multi-step prompt file into ``(label, body)`` pairs."""
    tasks: List[Tuple[str, List[str]]] = []
    current_label: str | None = None
    current_lines: List[str] = []
    preamble: List[str] = []
    for raw in text.splitlines():
        m = TASK_HEADER_RE.match(raw)
        if m:
            if current_label is not None:
                tasks.append((current_label, current_lines))
            current_label = m.group(1).strip()
            current_lines = []
            continue
        if current_label is None:
            preamble.append(raw)
        else:
            current_lines.append(raw)
    if current_label is not None:
        tasks.append((current_label, current_lines))
    if not tasks:
        raise SystemExit(
            "ERROR: no '# TASK <label>' headers found in prompt file."
        )
    pre = "\n".join(preamble).strip()
    result: List[Tuple[str, str]] = []
    for label, body_lines in tasks:
        body = "\n".join(body_lines).strip()
        if pre and "# TASK" not in body:
            body = f"Context for the whole pipeline:\n{pre}\n\n{body}"
        body = f"{body}{TOOL_INSTR}{ONESTEP_RULES}"
        result.append((label, body))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run a multi-step prompt file with ChatGPT + ToolUniverse and "
            "build one onestep.py + per-step logs."
        )
    )
    ap.add_argument(
        "--prompt-file",
        required=True,
        metavar="FILE",
        help="UTF-8 file containing one or more '# TASK <label>' sections.",
    )
    ap.add_argument(
        "--name",
        default=None,
        metavar="STEM",
        help="Output prefix (default: timestamped).",
    )
    ap.add_argument(
        "--out-dir",
        default=str(_SCRIPT_DIR / "outputs"),
        metavar="DIR",
        help="Directory to write outputs into (default: outputs/).",
    )
    ap.add_argument(
        "--tool-limit",
        type=int,
        default=12,
        metavar="N",
        help="Number of tools to inject per step (default: 12).",
    )
    ap.add_argument(
        "--tool-hint",
        default="",
        metavar="TEXT",
        help="Extra keywords passed to the tool finder.",
    )
    ap.add_argument(
        "--max-rounds",
        type=int,
        default=15,
        metavar="N",
        help="Maximum tool-calling rounds per step (default: 15).",
    )
    ap.add_argument(
        "--env-file",
        default=str(_SCRIPT_DIR / ".env"),
        metavar="FILE",
        help="Path to a .env file to load (default: .env next to the script).",
    )
    args = ap.parse_args()

    load_dotenv_if_present(args.env_file)

    text = Path(args.prompt_file).read_text(encoding="utf-8")
    tasks = parse_tasks(text)
    name = args.name or time.strftime("pipeline_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    onestep_path = out_dir / f"{name}_onestep.py"
    output_txt = out_dir / f"{name}_output.txt"
    tools_used_txt = out_dir / f"{name}_tools_used.txt"

    llm = build_llm_client()
    print(f"[provider] {llm.provider} model={llm.model}", file=sys.stderr)

    tu = load_tool_universe()
    # One shared tool shortlist for the whole pipeline (stable across steps).
    functions = find_relevant_tools(
        tu, text, limit=args.tool_limit, hint=args.tool_hint
    )
    print(
        f"[ToolUniverse] {len(functions)} tools shared across steps",
        file=sys.stderr,
    )

    with open(onestep_path, "w", encoding="utf-8") as f:
        f.write('"""\n')
        f.write(
            "onestep.py — pipeline script generated by ToolUniverse Agent.\n"
        )
        f.write(f"name  : {name}\n")
        f.write(f"model : {llm.provider}/{llm.model}\n")
        f.write('"""\n')

    with open(tools_used_txt, "w", encoding="utf-8") as f:
        f.write(
            f"# Per-task ToolUniverse calls ({llm.provider}/{llm.model})\n"
        )
        f.write("# " + "=" * 50 + "\n")

    with open(output_txt, "w", encoding="utf-8") as out_f:
        out_f.write(
            f"# ToolUniverse pipeline log — {llm.provider}/{llm.model}\n"
        )
        out_f.write(f"# steps: {len(tasks)}\n\n")

        for label, body in tasks:
            print(f"Generating {label}...", file=sys.stderr)
            final_text, used = chat_with_tools(
                llm,
                tu,
                body,
                functions,
                system_msg=SYSTEM_MSG,
                max_rounds=args.max_rounds,
                task_label=label,
            )
            code = extract_code(final_text)

            with open(onestep_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n# ========== {label} ==========\n\n")
                f.write(code)
                if code and not code.endswith("\n"):
                    f.write("\n")

            with open(tools_used_txt, "a", encoding="utf-8") as f:
                f.write(
                    f"{label}: {', '.join(used) if used else '(none)'}\n"
                )

            out_f.write("\n" + "=" * 60 + "\n")
            out_f.write(f"# -------- {label} --------\n")
            out_f.write("=" * 60 + "\n\n")
            out_f.write("## Tools called (in call order)\n\n")
            if used:
                for i, n in enumerate(used, 1):
                    out_f.write(f"{i}. {n}\n")
            else:
                out_f.write("(none)\n")
            out_f.write("\n## Raw reply\n\n")
            out_f.write(final_text.rstrip() + "\n\n")
            out_f.write("## Extracted code\n\n")
            out_f.write(code.rstrip() + "\n")
            out_f.flush()

    print(f"[saved] {onestep_path}", file=sys.stderr)
    print(f"[saved] {output_txt}", file=sys.stderr)
    print(f"[saved] {tools_used_txt}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
