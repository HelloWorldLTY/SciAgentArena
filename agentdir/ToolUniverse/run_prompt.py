#!/usr/bin/env python3
"""Run a single prompt through ChatGPT + ToolUniverse.

Writes two files under ``--out-dir`` (default ``outputs/``):

- ``<name>.py``  — extracted executable Python.
- ``<name>.txt`` — full log: prompt, tool calls in call order, raw reply.

Provide the prompt via ``--prompt-file FILE``, ``--prompt "TEXT"``, or stdin.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _agent_core import (  # noqa: E402
    append_format_suffix,
    build_llm_client,
    chat_with_tools,
    extract_code,
    find_relevant_tools,
    load_dotenv_if_present,
    load_tool_universe,
)


SYSTEM_MSG = (
    "You are an expert bioinformatics coding assistant. You have access to "
    "ToolUniverse tools (ChatGPT function calls) that can look up APIs for "
    "Scanpy, AnnData, Scrublet, Harmony, SCVI, Scanorama, Squidpy, and "
    "related packages. You should call at least one relevant tool before "
    "writing your final code. Final answer must be a single fenced Python "
    "code block (```python ...```) with brief comments; nothing else."
)


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt.strip()
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return data
    raise SystemExit(
        "ERROR: no prompt provided. Use --prompt-file, --prompt, or pipe via stdin."
    )


def _default_name() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run a single prompt through ChatGPT + ToolUniverse and save "
            "the generated Python code and a run log."
        )
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--prompt", metavar="TEXT", help="Inline prompt (single line)."
    )
    src.add_argument(
        "--prompt-file",
        metavar="FILE",
        help="Path to a UTF-8 file containing the prompt.",
    )
    ap.add_argument(
        "--name",
        metavar="STEM",
        default=None,
        help="Output basename (shared by .py and .txt). Default: timestamped.",
    )
    ap.add_argument(
        "--out-dir",
        metavar="DIR",
        default=str(_SCRIPT_DIR / "outputs"),
        help="Directory to write outputs into (default: outputs/).",
    )
    ap.add_argument(
        "--tool-limit",
        type=int,
        default=12,
        metavar="N",
        help="Number of tools to inject as function calls (default: 12).",
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
        help="Maximum tool-calling rounds (default: 15).",
    )
    ap.add_argument(
        "--no-format-suffix",
        action="store_true",
        help="Do not append the default 'output only code' suffix.",
    )
    ap.add_argument(
        "--env-file",
        default=str(_SCRIPT_DIR / ".env"),
        metavar="FILE",
        help="Path to a .env file to load (default: .env next to the script).",
    )
    args = ap.parse_args()

    load_dotenv_if_present(args.env_file)

    prompt = _read_prompt(args)
    if not args.no_format_suffix:
        prompt = append_format_suffix(prompt)

    name = args.name or _default_name()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    py_path = out_dir / f"{name}.py"
    log_path = out_dir / f"{name}.txt"

    llm = build_llm_client()
    print(f"[provider] {llm.provider} model={llm.model}", file=sys.stderr)

    tu = load_tool_universe()
    functions = find_relevant_tools(
        tu, prompt, limit=args.tool_limit, hint=args.tool_hint
    )
    tool_names = [f["name"] for f in functions]
    print(f"[ToolUniverse] {len(functions)} tools available", file=sys.stderr)

    final_text, used_tools = chat_with_tools(
        llm,
        tu,
        prompt,
        functions,
        system_msg=SYSTEM_MSG,
        max_rounds=args.max_rounds,
    )
    code = extract_code(final_text)

    with open(py_path, "w", encoding="utf-8") as f:
        f.write(code)
        if code and not code.endswith("\n"):
            f.write("\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# ToolUniverse Agent — run log\n")
        f.write(f"# provider : {llm.provider}\n")
        f.write(f"# model    : {llm.model}\n")
        f.write(f"# name     : {name}\n")
        f.write(f"# py_path  : {py_path}\n\n")
        f.write("## Prompt\n\n")
        f.write(prompt.rstrip() + "\n\n")
        f.write("## Tools available (from Tool_Finder_Keyword)\n\n")
        if tool_names:
            for i, n in enumerate(tool_names, 1):
                f.write(f"{i}. {n}\n")
        else:
            f.write("(none)\n")
        f.write("\n## Tools called (in call order)\n\n")
        if used_tools:
            for i, n in enumerate(used_tools, 1):
                f.write(f"{i}. {n}\n")
        else:
            f.write("(none)\n")
        f.write("\n## Raw model reply\n\n")
        f.write(final_text.rstrip() + "\n\n")
        f.write("## Extracted code\n\n")
        f.write(code.rstrip() + "\n")

    print(f"[saved] {py_path}", file=sys.stderr)
    print(f"[saved] {log_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
