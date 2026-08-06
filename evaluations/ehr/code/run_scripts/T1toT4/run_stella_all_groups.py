#!/usr/bin/env python3
"""
STELLA multi-agent runner for all MedAgentBench task groups.
Ensemble: manager=grok-4, dev=claude-sonnet-4-6, critic=gemini-3.1-pro-preview

Usage:
  OPENROUTER_API_KEY=sk-or-v1-... python3 run_stella_all_groups.py [--groups all]

Supports resuming: skips tasks already written to the output .jsonl file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

# ── Must chdir to STELLA dir before importing stella_core ────────────────────
# Set STELLA_DIR to your local STELLA checkout; defaults to a sibling
# EHR_bench/agent_deployments/STELLA directory relative to this repo.
STELLA_DIR = Path(os.environ.get("STELLA_DIR", "./EHR_bench/agent_deployments/STELLA")).resolve()
os.chdir(STELLA_DIR)
if str(STELLA_DIR) not in sys.path:
    sys.path.insert(0, str(STELLA_DIR))

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
BENCH_DIR  = SCRIPT_DIR.parent.parent / "MedAgentBench"

TASK_FILES = {
    "stepwise_v2_20":     BENCH_DIR / "synthea_stepwise_benchmark/output_v2/medagentbench_stepwise_v2_tasks.json",
    "stepwise_curated20": BENCH_DIR / "synthea_stepwise_benchmark/output/medagentbench_stepwise_curated20_tasks.json",
    "stepwise_0404":      BENCH_DIR / "synthea_stepwise_benchmark/output/medagentbench_0404_instantiated_tasks.json",
    "stepwise_all_435":   BENCH_DIR / "synthea_stepwise_benchmark/output/medagentbench_stepwise_tasks.json",
    "rare_disease_20":    BENCH_DIR / "synthea_stepwise_benchmark/output_rare_disease/medagentbench_rare_disease_tasks.json",
    "mimic_drug":         BENCH_DIR / "mimic_drug_recommendation_tasks.json",
}

GT_FILES = {
    "stepwise_v2_20":     BENCH_DIR / "synthea_stepwise_benchmark/output_v2/medagentbench_stepwise_v2_groundtruth.json",
    "stepwise_curated20": BENCH_DIR / "synthea_stepwise_benchmark/output/medagentbench_stepwise_curated20_groundtruth.json",
    "stepwise_all_435":   BENCH_DIR / "synthea_stepwise_benchmark/output/groundtruth.json",
}

ALL_GROUPS = [
    "stepwise_v2_20",
    "stepwise_curated20",
    "stepwise_0404",
    "stepwise_all_435",
    "rare_disease_20",
    "mimic_drug",
]

# ── Prompts ──────────────────────────────────────────────────────────────────

# For v2_20 tasks: GT uses fhir_* action names
STEPWISE_SYSTEM_FHIR = (
    "You are an EHR clinical workflow assistant. "
    "Given the clinical scenario below, produce a concise step-by-step action plan "
    "as strict JSON.\n\n"
    "Final output format (JSON only, no markdown):\n"
    '{"steps": [{"step": 1, "action": "fhir_search_observation", "detail": "..."}], '
    '"final_answer": "..."}\n\n'
    "Action naming rules — use ONLY these snake_case action names:\n"
    "- fhir_search_observation  (retrieve labs, vitals, or observations)\n"
    "- fhir_create_medication_request  (order a medication)\n"
    "- fhir_create_service_request  (order a lab, imaging, or follow-up)\n"
    "- fhir_search_procedure  (look up past procedures)\n"
    "- fhir_search_condition  (look up diagnoses or conditions)\n"
    "- fhir_search_medication_request  (retrieve existing medication orders)\n"
    "- fhir_create_condition  (record a diagnosis or condition)\n"
    "Do NOT use REST-style names like GET_Observation or POST_MedicationRequest.\n\n"
    "Other rules:\n"
    "- Final output must be valid JSON only, no markdown.\n"
    "- Include only clinically necessary steps.\n"
)

# For all_435 / curated20 tasks: GT uses verify_marker_condition, branch_decision,
# verify_medication_order, verify_medication_dose — do NOT constrain action names
STEPWISE_SYSTEM = (
    "You are an EHR clinical workflow assistant. "
    "Given the clinical scenario below, produce a concise step-by-step action plan "
    "as strict JSON.\n\n"
    "Final output format (JSON only, no markdown):\n"
    '{"steps": [{"step": 1, "action": "action_name", "detail": "..."}], '
    '"final_answer": "..."}\n\n'
    "Action naming rules:\n"
    "- Use snake_case descriptive action names that match the task vocabulary.\n"
    "- For marker/condition checks use: verify_marker_condition\n"
    "- For branch decisions use: branch_decision\n"
    "- For medication order checks use: verify_medication_order\n"
    "- For dose verification use: verify_medication_dose\n"
    "- For observation queries use: query_observation\n\n"
    "Other rules:\n"
    "- Final output must be valid JSON only, no markdown.\n"
    "- Include only clinically necessary steps.\n"
)

DRUG_SYSTEM = (
    "You are an expert clinical pharmacist and physician. "
    "Given patient information, diagnoses, and lab results, provide a numbered list of "
    "recommended medications with brief one-line justifications. "
    "Be specific with drug names. List the most important drugs first."
)


def build_stepwise_prompt(case: Dict[str, Any], group: str = "") -> str:
    task_id     = case.get("id") or case.get("task_id", "")
    difficulty  = case.get("difficulty", "")
    context     = case.get("context", "")
    instruction = case.get("instruction", case.get("scenario", ""))
    return (
        f"Task ID: {task_id}\n"
        f"Difficulty: {difficulty}\n"
        f"Context: {context}\n"
        f"Instruction: {instruction}\n\n"
        "Respond with a JSON object containing 'steps' and 'final_answer'."
    )


def get_stepwise_system(group: str) -> str:
    """Return the appropriate system prompt for this task group."""
    if group in ("stepwise_v2_20", "rare_disease_20"):
        return STEPWISE_SYSTEM_FHIR
    return STEPWISE_SYSTEM


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


# ── Scoring helpers ──────────────────────────────────────────────────────────

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
    exact_match = int(exp_count > 0 and (
        set(expected_actions) == set(produced_actions[:exp_count])
        or (recall == 1.0 and precision == 1.0)
    ))
    return {
        "has_gt": True,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": exact_match,
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
        "has_gt": True,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": int(tp == len(gt_names) and len(pred_names) == len(gt_names)),
        "gt_count":  len(gt_names),
        "pred_count": len(pred_names),
    }


# ── Data loaders (identical to tooluniverse runner) ──────────────────────────

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


# ── STELLA agent call ────────────────────────────────────────────────────────

def _try_once_stella(manager_agent, prompt: str) -> tuple[str, dict]:
    """Single attempt to run STELLA. Returns (response_text, meta)."""
    import ast
    start = time.time()
    meta = {"elapsed_sec": 0.0, "error": None}
    try:
        result = manager_agent.run(prompt)
        if result is None:
            response = ""
        elif isinstance(result, str):
            response = result
        elif isinstance(result, dict):
            response = json.dumps(result)
        else:
            raw = str(result)
            try:
                obj = ast.literal_eval(raw)
                response = json.dumps(obj)
            except Exception:
                response = raw
    except Exception as e:
        meta["error"] = traceback.format_exc()
        response = f"[ERROR] {e}"
        print(f"\n  ⚠️  Exception: {e}")
    meta["elapsed_sec"] = round(time.time() - start, 1)
    return response, meta


def run_stella_task(manager_agent, prompt: str, task_type: str) -> tuple[str, dict]:
    """Run one task through STELLA manager_agent with up to 2 retries on error."""
    last_response, last_meta = "", {}
    for attempt in range(3):
        response, meta = _try_once_stella(manager_agent, prompt)
        last_response, last_meta = response, meta
        if response and not response.startswith("[ERROR]"):
            return response, meta
        if attempt < 2:
            print(f"\n  🔄 Retry {attempt+1}/2 after error...")
            time.sleep(5)
    return last_response, last_meta


# ── Runner for one group ─────────────────────────────────────────────────────

def run_group(manager_agent, group: str) -> None:
    out_file = SCRIPT_DIR / f"stella_{group}.jsonl"

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
                system_ctx   = DRUG_SYSTEM
                user_message = build_drug_prompt(task)
            else:
                system_ctx   = get_stepwise_system(group)
                user_message = build_stepwise_prompt(task, group)

            # Combine system context + user message into a single prompt for smolagents
            full_prompt = f"{system_ctx}\n\n{user_message}"

            print(f"[{global_idx:03d}/{len(tasks):03d}] {task_id} ({difficulty}) ... ", end="", flush=True)

            response, meta = run_stella_task(manager_agent, full_prompt, task_type)

            if task_type == "drug":
                sc = score_drug(task.get("_gt_medications", []), response)
            else:
                sc = score_stepwise(task.get("_expected_steps", []), response)

            f1_str = f"{sc['f1']:.3f}" if sc.get("has_gt", True) else "N/A"
            exp_c  = sc.get("expected_count", sc.get("gt_count", "?"))
            pred_c = sc.get("produced_count", sc.get("pred_count", "?"))
            print(f"F1={f1_str}  exp={exp_c} pred={pred_c}  elapsed={meta['elapsed_sec']}s")

            row = {
                "task_id":    task_id,
                "group":      group,
                "difficulty": difficulty,
                "response":   response,
                "score":      sc,
                "meta":       meta,
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", default=["all"],
                        help="Groups to run (default: all)")
    args = parser.parse_args()

    groups = ALL_GROUPS if args.groups == ["all"] else args.groups

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Validate task files
    for g in groups:
        if g not in TASK_FILES:
            print(f"Unknown group: {g}", file=sys.stderr)
            sys.exit(1)
        if not TASK_FILES[g].exists():
            print(f"Task file not found: {TASK_FILES[g]}", file=sys.stderr)
            sys.exit(1)

    print("🚀 Initializing STELLA (ensemble: grok-4 + claude-sonnet-4-6 + gemini-3.1-pro-preview)...")
    from stella_core import initialize_stella
    manager_agent = initialize_stella(use_template=False, use_mem0=False, enable_tool_creation=False)

    if manager_agent is None or manager_agent is False:
        print("ERROR: STELLA initialization failed", file=sys.stderr)
        sys.exit(1)

    print(f"✅ STELLA ready. Running {len(groups)} group(s): {groups}\n")

    for group in groups:
        run_group(manager_agent, group)

    print("\nAll done.")


if __name__ == "__main__":
    main()
