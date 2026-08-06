#!/usr/bin/env python3
"""
ToolUniverse Agent runner for all MedAgentBench task groups.
Backend model: gpt-5.2 (OpenAI)
ToolUniverse tools: rxnorm, dailymed, openfda, pubmed, icd, medlineplus,
                    drug_properties, omim, orphanet, clinical_guidelines

Usage:
  OPENAI_API_KEY=sk-... python3 run_tooluniverse_all_groups.py [--groups all]

Supports resuming: skips tasks already written to the output .jsonl file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai

# ── Add ToolUniverse src to path ──────────────────────────────────────────────
# Set TOOLUNIVERSE_SRC to your local ToolUniverse checkout's src/ directory;
# defaults to a sibling EHR_bench/agent_deployments/ToolUniverse/src directory.
TU_SRC = Path(os.environ.get("TOOLUNIVERSE_SRC", "./EHR_bench/agent_deployments/ToolUniverse/src")).resolve()
if str(TU_SRC) not in sys.path:
    sys.path.insert(0, str(TU_SRC))

# Suppress ToolUniverse noisy logging
os.environ.setdefault("TOOLUNIVERSE_LAZY_LOADING", "true")
os.environ.setdefault("TOOLUNIVERSE_QUIET", "1")

from tooluniverse import ToolUniverse  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()
BENCH_DIR  = SCRIPT_DIR.parent.parent / "MedAgentBench"

TASK_FILES = {
    "stepwise_v2_20":    BENCH_DIR / "synthea_stepwise_benchmark/output_v2/medagentbench_stepwise_v2_tasks.json",
    "stepwise_curated20": BENCH_DIR / "synthea_stepwise_benchmark/output/medagentbench_stepwise_curated20_tasks.json",
    "stepwise_0404":      BENCH_DIR / "synthea_stepwise_benchmark/output/medagentbench_0404_instantiated_tasks.json",
    "stepwise_all_435":   BENCH_DIR / "synthea_stepwise_benchmark/output/medagentbench_stepwise_tasks.json",
    "rare_disease_20":    BENCH_DIR / "synthea_stepwise_benchmark/output_rare_disease/medagentbench_rare_disease_tasks.json",
    "mimic_drug":         BENCH_DIR / "mimic_drug_recommendation_tasks.json",
}

GT_FILES = {
    "stepwise_v2_20":    BENCH_DIR / "synthea_stepwise_benchmark/output_v2/medagentbench_stepwise_v2_groundtruth.json",
    "stepwise_curated20": BENCH_DIR / "synthea_stepwise_benchmark/output/medagentbench_stepwise_curated20_groundtruth.json",
    "stepwise_all_435":   BENCH_DIR / "synthea_stepwise_benchmark/output/groundtruth.json",
}

MODEL = "gpt-5.2"
MAX_TOOL_ROUNDS = 8   # max agentic tool call rounds per task

ALL_GROUPS = [
    "stepwise_v2_20",
    "stepwise_curated20",
    "stepwise_0404",
    "stepwise_all_435",
    "rare_disease_20",
    "mimic_drug",
]

# ──────────────────────────────────────────────────────────────────────────────
# ToolUniverse initialization (done once globally)
# ──────────────────────────────────────────────────────────────────────────────

CLINICAL_CATEGORIES = [
    "rxnorm",
    "dailymed",
    "openfda",
    "pubmed",
    "icd",
    "medlineplus",
    "drug_properties",
    "omim",
    "orphanet",
    "clinical_guidelines",
]

print("Initializing ToolUniverse...", flush=True)
TU = ToolUniverse()
TU.load_tools(categories=CLINICAL_CATEGORIES)
print(f"  Loaded {len(TU.all_tool_dict)} tools: {list(TU.all_tool_dict.keys())[:10]}...", flush=True)

# Build OpenAI function-calling schemas for all loaded tools
OPENAI_TOOLS: List[Dict] = []
for name in TU.all_tool_dict:
    spec = TU.tool_specification(name, format="openai")
    if spec:
        OPENAI_TOOLS.append({"type": "function", "function": spec})

print(f"  Built {len(OPENAI_TOOLS)} OpenAI tool specs.", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

STEPWISE_SYSTEM = (
    "You are an EHR clinical workflow assistant with access to biomedical tools. "
    "Given the clinical scenario below, produce a concise step-by-step action plan "
    "as strict JSON. You may call tools to look up drug information, clinical "
    "guidelines, disease info, etc., but you MUST end with a complete JSON answer.\n\n"
    "Final output format (JSON only, no markdown):\n"
    "{\n"
    '  "steps": [\n'
    '    {"step": 1, "action": "fhir_verb_resource", "detail": "..."}\n'
    "  ],\n"
    '  "final_answer": "..."\n'
    "}\n\n"
    "Rules:\n"
    "- Final output must be JSON only, no markdown, no extra text.\n"
    "- Each action should be a concrete FHIR operation such as fhir_search_observation, "
    "fhir_create_medication_request, fhir_create_service_request, fhir_stop_medication, "
    "fhir_search_procedure, fhir_create_condition, verify_marker_condition, branch_decision, "
    "calculator, etc.\n"
    "- Include only clinically necessary steps.\n"
)

DRUG_SYSTEM = (
    "You are an expert clinical pharmacist and physician with access to biomedical tools. "
    "Given patient information, diagnoses, and lab results, provide a numbered list of "
    "recommended medications. For each drug, give its name and a brief one-line "
    "justification. Be specific with drug names. List the most important/primary drugs first. "
    "You may call tools to verify drug information, guidelines, or drug interactions."
)


def build_stepwise_prompt(case: Dict[str, Any]) -> str:
    task_id     = case.get("id") or case.get("task_id", "")
    difficulty  = case.get("difficulty", "")
    context     = case.get("context", "")
    instruction = case.get("instruction", case.get("scenario", ""))
    return (
        f"Task ID: {task_id}\n"
        f"Difficulty: {difficulty}\n"
        f"Context: {context}\n"
        f"Instruction: {instruction}"
    )


def build_drug_prompt(case: Dict[str, Any]) -> str:
    pi = case.get("patient_info", "")
    if isinstance(pi, dict):
        pi = json.dumps(pi, indent=2)
    dx = case.get("diagnoses", "")
    if isinstance(dx, list):
        dx = "\n".join(f"- {d}" for d in dx)
    labs = case.get("key_labs", "")
    if isinstance(labs, list):
        labs = "\n".join(f"- {l}" for l in labs)
    elif isinstance(labs, dict):
        labs = json.dumps(labs, indent=2)
    return (
        f"Question: {case.get('question', '')}\n\n"
        f"Patient Information:\n{pi}\n\n"
        f"Diagnoses:\n{dx}\n\n"
        f"Key Labs:\n{labs}\n"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────────────────────────────────────────────

_JSON_BLOCK   = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
_STEP_BRACKET = re.compile(
    r"\[(?:Step|STEP|HA|FHIR)\s*\d*\]\s*(?:Step\s+\d+\s*[:\-]\s*)?([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)


def _parse_json(text: str) -> Any:
    text = text.strip()
    m = _JSON_BLOCK.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m2 = re.search(r"\{[\s\S]*\}", text)
    if m2:
        try:
            return json.loads(m2.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _extract_steps(parsed: Any) -> List[Dict]:
    if isinstance(parsed, dict) and isinstance(parsed.get("steps"), list):
        return [x for x in parsed["steps"] if isinstance(x, dict)]
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []


def _get_predicted_actions(response: str) -> List[str]:
    parsed = _parse_json(response)
    steps  = _extract_steps(parsed)
    if steps:
        return [str(s.get("action", "")).lower().strip() for s in steps]
    bracket = _STEP_BRACKET.findall(response)
    if bracket:
        return [m.lower().strip() for m in bracket if m]
    return []


def score_stepwise(expected_steps: List[Dict], model_text: str) -> Dict[str, Any]:
    expected_actions = [str(s.get("action", "")).lower() for s in expected_steps if s.get("action")]
    produced_actions = _get_predicted_actions(model_text)

    exp_count  = len(expected_actions)
    pred_count = len(produced_actions)

    if exp_count == 0:
        return {"has_gt": False, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "exact_match": 0, "expected_count": 0, "produced_count": pred_count}

    hit     = sum(1 for a in expected_actions if any(a in c or c in a for c in produced_actions))
    matched = sum(1 for c in produced_actions if any(a in c or c in a for a in expected_actions)) if produced_actions else 0
    recall    = hit / exp_count
    precision = (matched / pred_count) if pred_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    exact_match = int(
        exp_count > 0
        and (set(expected_actions) == set(produced_actions[:exp_count])
             or (recall == 1.0 and precision == 1.0))
    )
    return {
        "has_gt":         True,
        "precision":      round(precision, 4),
        "recall":         round(recall, 4),
        "f1":             round(f1, 4),
        "exact_match":    exact_match,
        "expected_count": exp_count,
        "produced_count": pred_count,
    }


def _norm_drug(t: str) -> str:
    t = (t or "").lower().strip()
    t = re.sub(r"\([^)]*\)", " ", t).replace("-", " ")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _drug_match(g: str, p: str) -> bool:
    ng, np_ = _norm_drug(g), _norm_drug(p)
    if not ng or not np_:
        return False
    if ng == np_ or ng in np_ or np_ in ng:
        return True
    return SequenceMatcher(None, ng, np_).ratio() >= 0.82


def _gt_drug_names(meds: list) -> List[str]:
    return [re.split(r"\s+[—–-]\s+", e, 1)[0].strip() for e in meds if isinstance(e, str)]


def _pred_drugs(text: str) -> List[str]:
    if not text:
        return []
    text = re.sub(r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.I)
    numbered = re.findall(r"(?m)^\s*\d{1,2}[\.)]\s*(.+)", text)
    cands = [x.strip() for x in numbered] if numbered else (
        [x.strip() for x in re.findall(r"(?m)^[-*•]\s*(.+)", text)]
        or [l.strip() for l in text.split("\n") if l.strip()]
    )
    drugs = []
    for c in cands:
        c = re.sub(r"\*\*(.*?)\*\*", r"\1", c)
        c = re.split(r"\s+[—–-]\s+", c, 1)[0]
        c = re.split(r"\s*:\s+", c, 1)[0]
        c = re.sub(r"^\d+[\.)]\s*", "", c).strip(" \t:-.")
        if not c or len(c) > 100:
            continue
        low = c.lower()
        if any(low.startswith(k) for k in ["medication", "drug", "prescription", "recommendation",
                "based on", "the patient", "given the", "note:", "here", "primary", "key",
                "these", "following"]):
            continue
        if re.search(r"\b(should|would|could|patient|history|given)\b", low):
            continue
        drugs.append(c)
    return drugs


def score_drug(gt_medications: list, model_text: str) -> Dict[str, Any]:
    gt_names   = _gt_drug_names(gt_medications)
    pred_names = _pred_drugs(model_text)
    if not gt_names:
        return {"has_gt": False, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "gt_count": 0, "pred_count": len(pred_names)}
    tp = sum(1 for g in gt_names if any(_drug_match(g, p) for p in pred_names))
    recall    = tp / len(gt_names)
    precision = (tp / len(pred_names)) if pred_names else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "has_gt":    True,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "exact_match": int(tp == len(gt_names) and len(pred_names) == len(gt_names)),
        "gt_count":  len(gt_names),
        "pred_count": len(pred_names),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_group(group: str) -> List[Dict]:
    tasks = json.loads(TASK_FILES[group].read_text())

    if group == "stepwise_v2_20":
        gt_list = json.loads(GT_FILES["stepwise_v2_20"].read_text())
        gt_map  = {g["task_id"]: g["expected_steps"] for g in gt_list}
        for t in tasks:
            t["_expected_steps"] = gt_map.get(t.get("id", t.get("task_id", "")), [])
            t["_type"] = "stepwise"
            if "id" not in t:
                t["id"] = t.get("task_id", "")

    elif group == "stepwise_curated20":
        gt_list = json.loads(GT_FILES["stepwise_curated20"].read_text())
        gt_map  = {g["task_id"]: g["expected_steps"] for g in gt_list}
        for t in tasks:
            t["_expected_steps"] = gt_map.get(t["id"], [])
            t["_type"] = "stepwise"

    elif group == "stepwise_0404":
        for t in tasks:
            es = t.get("expected_steps", [])
            t["_expected_steps"] = es if isinstance(es, list) else []
            t["_type"] = "stepwise"
            if "id" not in t:
                t["id"] = t.get("task_id", "")

    elif group == "stepwise_all_435":
        gt_list = json.loads(GT_FILES["stepwise_all_435"].read_text())
        gt_map  = {g["task_id"]: g["expected_steps"] for g in gt_list}
        for t in tasks:
            t["_expected_steps"] = gt_map.get(t["id"], [])
            t["_type"] = "stepwise"

    elif group == "rare_disease_20":
        for t in tasks:
            t["_expected_steps"] = t.get("expected_steps", [])
            t["_type"] = "stepwise"
            if "id" not in t:
                t["id"] = t.get("task_id", "")

    elif group == "mimic_drug":
        for t in tasks:
            t["_gt_medications"] = t.get("ground_truth_medications", [])
            t["_type"] = "drug"
            if "id" not in t:
                t["id"] = t.get("task_id", "")

    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# Agent loop with ToolUniverse
# ──────────────────────────────────────────────────────────────────────────────

def call_tool_safe(tool_name: str, arguments: dict) -> str:
    """Execute a ToolUniverse tool and return result as string."""
    try:
        result = TU.run_one_function({"name": tool_name, "arguments": arguments})
        if result is None:
            return "No result returned."
        if isinstance(result, (dict, list)):
            text = json.dumps(result, ensure_ascii=False)
        else:
            text = str(result)
        # Truncate very long results
        if len(text) > 4000:
            text = text[:4000] + "...[truncated]"
        return text
    except Exception as e:
        return f"Tool error: {e}"


def run_agent(
    client: openai.OpenAI,
    system_prompt: str,
    user_message: str,
    task_type: str,
) -> tuple[str, dict]:
    """
    Run the ToolUniverse agent loop.
    Returns (final_text, usage_summary).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                   "tool_calls_made": 0}
    final_text = ""

    for _round in range(MAX_TOOL_ROUNDS + 1):
        # Last round: no tools allowed — must produce final answer
        tools_arg = OPENAI_TOOLS if _round < MAX_TOOL_ROUNDS else None
        tool_choice = "auto" if tools_arg else None

        for attempt in range(5):
            try:
                kwargs: Dict[str, Any] = {
                    "model": MODEL,
                    "messages": messages,
                    "max_completion_tokens": 2048,
                }
                if tools_arg:
                    kwargs["tools"] = tools_arg
                    kwargs["tool_choice"] = tool_choice
                resp = client.chat.completions.create(**kwargs)
                break
            except openai.RateLimitError:
                wait = 2 ** attempt * 5
                time.sleep(wait)
            except openai.APIError as e:
                if attempt == 4:
                    raise
                time.sleep(3)

        # Accumulate usage
        if resp.usage:
            total_usage["prompt_tokens"]     += resp.usage.prompt_tokens or 0
            total_usage["completion_tokens"] += resp.usage.completion_tokens or 0
            total_usage["total_tokens"]      += resp.usage.total_tokens or 0

        msg = resp.choices[0].message

        # No tool calls → final answer
        if not msg.tool_calls:
            final_text = msg.content or ""
            break

        # Process tool calls
        messages.append(msg.model_dump())  # assistant message with tool_calls
        total_usage["tool_calls_made"] += len(msg.tool_calls)

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = {}

            result_str = call_tool_safe(fn_name, fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    return final_text, total_usage


# ──────────────────────────────────────────────────────────────────────────────
# Runner for one group
# ──────────────────────────────────────────────────────────────────────────────

def run_group(client: openai.OpenAI, group: str) -> None:
    out_file = SCRIPT_DIR / f"tooluniverse_{group}.jsonl"

    # Load already-done task IDs for resuming
    done_ids: set = set()
    if out_file.exists():
        with out_file.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(r["task_id"])
                except Exception:
                    pass

    tasks = load_group(group)
    remaining = [t for t in tasks if (t.get("id") or t.get("task_id", "")) not in done_ids]

    print(f"\n{'='*60}")
    print(f"Group: {group}  total={len(tasks)}  done={len(done_ids)}  remaining={len(remaining)}")
    print(f"Output: {out_file}")
    print(f"{'='*60}")

    if not remaining:
        print("  All tasks already done, skipping.")
        return

    with out_file.open("a") as fout:
        for i, task in enumerate(remaining):
            task_id    = task.get("id") or task.get("task_id", f"{group}_{i}")
            difficulty = task.get("difficulty", "unknown")
            task_type  = task.get("_type", "stepwise")
            global_idx = len(done_ids) + i + 1

            if task_type == "drug":
                system_prompt = DRUG_SYSTEM
                user_message  = build_drug_prompt(task)
            else:
                system_prompt = STEPWISE_SYSTEM
                user_message  = build_stepwise_prompt(task)

            print(f"[{global_idx:03d}/{len(tasks):03d}] {task_id} ({difficulty}) ... ", end="", flush=True)

            response, usage = run_agent(client, system_prompt, user_message, task_type)

            if task_type == "drug":
                sc = score_drug(task.get("_gt_medications", []), response)
            else:
                sc = score_stepwise(task.get("_expected_steps", []), response)

            f1_str   = f"{sc['f1']:.3f}" if sc.get("has_gt", True) else "N/A"
            exp_c    = sc.get("expected_count", sc.get("gt_count", "?"))
            pred_c   = sc.get("produced_count", sc.get("pred_count", "?"))
            calls    = usage.get("tool_calls_made", 0)
            print(f"F1={f1_str}  exp={exp_c} pred={pred_c}  tools_used={calls}  tokens={usage.get('total_tokens','?')}")

            row = {
                "task_id":    task_id,
                "group":      group,
                "difficulty": difficulty,
                "response":   response,
                "score":      sc,
                "usage":      usage,
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()

    # Summary
    all_rows: List[Dict] = []
    with out_file.open() as f:
        for line in f:
            try:
                all_rows.append(json.loads(line))
            except Exception:
                pass

    scored = [r for r in all_rows if r["score"].get("has_gt", True)]
    if scored:
        avg_f1   = sum(r["score"]["f1"] for r in scored) / len(scored)
        accuracy = sum(1 for r in scored if r["score"]["f1"] >= 0.5) / len(scored)
        print(f"\n  Summary {group}: N={len(scored)} accuracy={accuracy:.1%} avg_f1={avg_f1:.4f}")
    else:
        print(f"\n  Summary {group}: no scoreable rows")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", default=["all"],
                        help="Groups to run (default: all)")
    args = parser.parse_args()

    groups = ALL_GROUPS if args.groups == ["all"] else args.groups

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)

    for group in groups:
        if group not in TASK_FILES:
            print(f"Unknown group: {group}", file=sys.stderr)
            continue
        run_group(client, group)

    print("\nAll done.")


if __name__ == "__main__":
    main()
