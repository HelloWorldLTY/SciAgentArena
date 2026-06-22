"""evaluate.py — one CLI to evaluate agents across all five categories.

The five categories were historically scored by separate per-branch scripts.
This is their unification: a single entry point that discovers any task, routes
it to the correct runner (batch vs. notebook) based on its declared
``output_format``, executes it through the shared waterfall scorer, and reports
results in one uniform schema for every category:

    executability  ->  validity  ->  correctness  ->  strategic_success

Commands
--------
    python evaluate.py list      [--category C1] [--mode batch]
    python evaluate.py doctor                      # check deps + scorer imports
    python evaluate.py run <task_id> <agent.py>    [--out results/run.json]
    python evaluate.py aggregate <results_glob>    # mean metrics by category

All paths inside task JSONs are repo-relative, so runners are always invoked
with cwd = this directory (ROOT).
"""

from __future__ import annotations

import argparse
import glob
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from agents.dotenv import load_dotenv
from registry import ROOT, build_registry, registry_by_id

# Load API keys from a local .env (if present) so `generate` can reach hosted
# LLMs. setdefault semantics — never overrides variables already in the shell.
load_dotenv(ROOT / ".env")

# Use the very interpreter running this CLI (the project venv) for subprocess
# runners and agents, so everything shares one environment.
PYTHON = sys.executable

# Uniform per-run result schema keys emitted by both runners.
SCORE_KEYS = ("executability", "validity", "correctness", "strategic_success")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> None:
    reg = build_registry()
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in reg:
        groups[e.get("category", "C?")].append(e)

    total_shown = 0
    for comp in sorted(groups):
        if args.category and args.category.upper() != comp:
            continue
        es = groups[comp]
        label = es[0].get("category_label", "")
        print(f"\n=== {comp}  {label}  ({len(es)} tasks) ===")
        for e in es:
            if args.mode and args.mode != e.get("mode"):
                continue
            tdc = "  [needs PyTDC]" if e.get("needs_tdc") else ""
            print(f"  {e['task_id']:<42s} {e['mode']:<8s} " f"{e['scorer']}{tdc}")
            total_shown += 1
    print(f"\n{total_shown} task(s) shown.")


# ---------------------------------------------------------------------------
# doctor — static smoke test of the framework wiring (no agents needed)
# ---------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> None:
    print(f"Python : {sys.version.split()[0]}")
    print(f"ROOT   : {ROOT}")

    print("\nDependencies:")
    for dep in [
        "rdkit",
        "pandas",
        "numpy",
        "matplotlib",
        "nbformat",
        "nbclient",
        "ipykernel",
        "scipy",
        "sklearn",
        "networkx",
        "tdc",
    ]:
        ok = importlib.util.find_spec(dep) is not None
        print(f"  {dep:12s}: {'OK' if ok else 'MISSING'}")

    print("\nLLM providers (for `generate`):")
    from agents.providers import DEFAULT_MODELS, provider_available

    _pkey = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    for prov in ("openai", "anthropic", "google"):
        sdk = "ok" if provider_available(prov) else "missing-sdk"
        key = "key-set" if os.environ.get(_pkey[prov]) else "no-key"
        azure = (
            " (+Azure endpoint)"
            if prov == "openai" and os.environ.get("AZURE_OPENAI_ENDPOINT")
            else ""
        )
        print(
            f"  {prov:10s}: sdk={sdk:11s} {key:8s} "
            f"default={DEFAULT_MODELS[prov]}{azure}"
        )

    print("\nRunner imports:")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    for runner in [
        "runners.batch_runner",
        "runners.notebook_runner",
        "runners.design_runner",
    ]:
        try:
            importlib.import_module(runner)
            print(f"  {runner:28s}: OK")
        except Exception as exc:
            print(f"  {runner:28s}: FAIL ({type(exc).__name__}: {str(exc)[:70]})")

    print("\nScorer module imports (referenced by tasks):")
    reg = build_registry()
    modules = sorted({e["scorer_module"] for e in reg if e.get("scorer_module")})
    ok_count = 0
    for mod in modules:
        try:
            importlib.import_module(f"scorers.{mod}")
            print(f"  scorers.{mod:30s}: OK")
            ok_count += 1
        except Exception as exc:
            print(f"  scorers.{mod:30s}: FAIL ({type(exc).__name__}: {str(exc)[:60]})")
    print(f"\n{ok_count}/{len(modules)} scorer modules import cleanly.")


# ---------------------------------------------------------------------------
# run — dispatch one task to its runner and normalize the result
# ---------------------------------------------------------------------------


def _have_tdc() -> bool:
    return importlib.util.find_spec("tdc") is not None


def run_task(task_id: str, agent_py: str) -> dict:
    """Run a single task through its runner; return a normalized record."""
    reg = registry_by_id()
    entry = reg.get(task_id)
    if entry is None:
        raise KeyError(f"Unknown task_id '{task_id}'. Run `list` to see options.")

    if entry["needs_tdc"] and not _have_tdc():
        print(
            f"WARNING: '{task_id}' is a C3 design task whose oracle needs PyTDC, "
            f"which is not installed. It will likely fail at scorer import. "
            f"Install with `uv pip install pytdc`.",
            file=sys.stderr,
        )

    agent_abs = os.path.abspath(agent_py)
    cmd = [
        PYTHON,
        "-m",
        f"runners.{entry['runner']}",
        entry["task_path"],
        agent_abs,
        "--pretty",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)

    try:
        score = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        score = {
            "error": "runner did not emit parseable JSON",
            "runner_stdout_tail": proc.stdout[-1500:],
            "runner_stderr_tail": proc.stderr[-1500:],
        }

    return {
        "task_id": entry["task_id"],
        "category": entry["category"],
        "category_label": entry["category_label"],
        "type": entry["type"],
        "mode": entry["mode"],
        "runner": entry["runner"],
        "scorer": entry["scorer"],
        "agent": agent_abs,
        "score": score,
    }


def cmd_run(args: argparse.Namespace) -> None:
    try:
        record = run_task(args.task_id, args.agent_py)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)

    print(json.dumps(record, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2))
        print(f"\nwrote {out}", file=sys.stderr)


# ---------------------------------------------------------------------------
# aggregate — mean metrics by category over a directory of run records
# ---------------------------------------------------------------------------


def cmd_aggregate(args: argparse.Namespace) -> None:
    paths = glob.glob(args.results_glob)
    if not paths:
        print(f"No result files matched {args.results_glob!r}", file=sys.stderr)
        sys.exit(2)

    by_comp: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        try:
            rec = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        by_comp[rec.get("category", "C?")].append(rec)

    print(f"{'comp':<6} {'n':>4}  " + "  ".join(f"{k[:5]:>6}" for k in SCORE_KEYS))
    for comp in sorted(by_comp):
        recs = by_comp[comp]
        means = []
        for k in SCORE_KEYS:
            vals = [
                r["score"].get(k, 0.0) for r in recs if isinstance(r.get("score"), dict)
            ]
            means.append(sum(vals) / len(vals) if vals else 0.0)
        print(f"{comp:<6} {len(recs):>4}  " + "  ".join(f"{m:6.3f}" for m in means))


# ---------------------------------------------------------------------------
# generate — produce an agent solution with a hosted LLM, optionally run it
# ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> None:
    import agents

    reg = registry_by_id()
    entry = reg.get(args.task_id)
    if entry is None:
        print(
            f"Unknown task_id '{args.task_id}'. Run `list` to see options.",
            file=sys.stderr,
        )
        sys.exit(2)
    task = json.load(open(ROOT / entry["task_path"], encoding="utf-8"))

    if args.dry_run:
        from agents.prompts import build_system_instruction, build_user_prompt

        print("=== SYSTEM INSTRUCTION ===")
        print(build_system_instruction(task))
        print("\n=== USER PROMPT ===")
        print(build_user_prompt(task))
        return

    code, _raw, model = agents.generate_solution(task, args.provider, args.model)
    out = (
        Path(args.out)
        if args.out
        else ROOT / "results" / "generated" / f"{args.task_id}.py"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(code, encoding="utf-8")
    print(
        f"wrote {out}  (provider={args.provider}, model={model}, {len(code)} chars)",
        file=sys.stderr,
    )

    if args.run:
        print(json.dumps(run_task(args.task_id, str(out)), indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list tasks grouped by category")
    p_list.add_argument("--category", help="filter to one category (e.g. C1)")
    p_list.add_argument("--mode", choices=["batch", "notebook"], help="filter by mode")
    p_list.set_defaults(func=cmd_list)

    p_doc = sub.add_parser("doctor", help="check deps and scorer imports")
    p_doc.set_defaults(func=cmd_doctor)

    p_run = sub.add_parser("run", help="run one task with an agent script")
    p_run.add_argument("task_id")
    p_run.add_argument("agent_py")
    p_run.add_argument("--out", help="also write the result record to this path")
    p_run.set_defaults(func=cmd_run)

    p_agg = sub.add_parser("aggregate", help="mean metrics by category over results")
    p_agg.add_argument("results_glob", help="glob for *.json run records")
    p_agg.set_defaults(func=cmd_aggregate)

    p_gen = sub.add_parser(
        "generate", help="generate an agent solution with a hosted LLM"
    )
    p_gen.add_argument("task_id")
    p_gen.add_argument(
        "--provider", choices=["openai", "anthropic", "google"], required=True
    )
    p_gen.add_argument("--model", help="override the provider's default model")
    p_gen.add_argument("--out", help="write the generated script here")
    p_gen.add_argument(
        "--run", action="store_true", help="run + score the generated script"
    )
    p_gen.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="print the prompts without calling the API",
    )
    p_gen.set_defaults(func=cmd_generate)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
