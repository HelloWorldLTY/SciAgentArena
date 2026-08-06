#!/usr/bin/env python3
"""
Claude Sonnet 4.6 runner for all MedAgentBench task groups.
Groups: stepwise_v2_20, stepwise_curated20, stepwise_0404, stepwise_all_435,
        rare_disease_20, mimic_drug

Usage:
  OPENROUTER_API_KEY=sk-or-... python3 run_claude46_all_groups.py
  OPENROUTER_API_KEY=sk-or-... python3 run_claude46_all_groups.py --groups mimic_drug rare_disease_20

Supports resuming: skips tasks already written to the output .jsonl file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

import openai

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

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

MODEL = "anthropic/claude-sonnet-4-6"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

ALL_GROUPS = [
    "stepwise_v2_20",
    "stepwise_curated20",
    "stepwise_0404",
    "stepwise_all_435",
    "rare_disease_20",
    "mimic_drug",
]

# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

STEPWISE_SYSTEM = (
    "You are an EHR clinical workflow assistant. Given the clinical scenario below, "
    "produce a concise step-by-step action plan as strict JSON.\n\n"
    "Output format:\n"
    "{\n"
    '  "steps": [\n'
    '    {"step": 1, "action": "fhir_verb_resource", "detail": "..."}\n'
    "  ],\n"
    '  "final_answer": "..."\n'
    "}\n\n"
    "Rules:\n"
    "- Output JSON only, no markdown and no extra text.\n"
    "- Each action should be a concrete FHIR operation such as fhir_search_observation, "
    "fhir_create_medication_request, fhir_create_service_request, fhir_stop_medication, "
    "fhir_search_procedure, fhir_create_condition, verify_marker_condition, branch_decision, "
    "calculator, etc.\n"
    "- Use English only.\n"
    "- Include only clinically necessary steps.\n\n"
)

DRUG_SYSTEM = (
    "You are an expert clinical pharmacist and physician. "
    "Given patient information, diagnoses, and lab results, "
    "provide a numbered list of recommended medications. "
    "For each drug, give its name and a brief one-line justification. "
    "Be specific with drug names. List the most important/primary drugs first."
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
        and set(expected_actions) == set(produced_actions[:exp_count])
        or (exp_count > 0 and recall == 1.0 and precision == 1.0)
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
    ng, np = _norm_drug(g), _norm_drug(p)
    if not ng or not np:
        return False
    if ng == np or ng in np or np in ng:
        return True
    return SequenceMatcher(None, ng, np).ratio() >= 0.82


def _gt_drug_names(meds: list) -> List[str]:
    return [re.split(r"\s+[—–-]\s+", e, 1)[0].strip() for e in meds if isinstance(e, str)]


def _pred_drugs(text: str) -> List[str]:
    if not text:
        return []
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
            t["_expected_steps"] = gt_map.get(t["id"], [])
            t["_type"] = "stepwise"

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
# Anthropic API call with retry
# ──────────────────────────────────────────────────────────────────────────────

def call_claude(client: openai.OpenAI, system: str, user: str, retries: int = 6) -> tuple[str, dict]:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = resp.choices[0].message.content or ""
            usage = {
                "prompt_tokens":     resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens":      resp.usage.total_tokens if resp.usage else 0,
            }
            return text, usage
        except openai.RateLimitError as e:
            wait = 2 ** attempt * 10
            print(f"\n  [RateLimit attempt {attempt+1}] waiting {wait}s... ({e})")
            time.sleep(wait)
        except openai.APIStatusError as e:
            err = str(e).lower()
            if "overload" in err or "529" in err or "503" in err:
                wait = 2 ** attempt * 5
                print(f"\n  [Overload attempt {attempt+1}] waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n  [APIError attempt {attempt+1}] {e}")
                if attempt == retries - 1:
                    return "", {}
                time.sleep(3)
        except Exception as e:
            print(f"\n  [Error attempt {attempt+1}] {e}")
            if attempt == retries - 1:
                return "", {}
            time.sleep(3)
    return "", {}


# ──────────────────────────────────────────────────────────────────────────────
# Runner for one group
# ──────────────────────────────────────────────────────────────────────────────

def run_group(client: openai.OpenAI, group: str) -> None:
    out_file = SCRIPT_DIR / f"claude46_{group}.jsonl"

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

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    rows: List[Dict] = []

    with out_file.open("a") as fout:
        for i, task in enumerate(remaining):
            task_id    = task.get("id") or task.get("task_id", f"{group}_{i}")
            difficulty = task.get("difficulty", "unknown")
            task_type  = task.get("_type", "stepwise")
            global_idx = len(done_ids) + i + 1

            if task_type == "drug":
                system = DRUG_SYSTEM
                user   = build_drug_prompt(task)
            else:
                system = STEPWISE_SYSTEM
                user   = build_stepwise_prompt(task)

            print(f"[{global_idx:03d}/{len(tasks):03d}] {task_id} ({difficulty}) ... ", end="", flush=True)

            response, usage = call_claude(client, system, user)

            if task_type == "drug":
                sc = score_drug(task.get("_gt_medications", []), response)
            else:
                sc = score_stepwise(task.get("_expected_steps", []), response)

            f1_str = f"{sc['f1']:.3f}" if sc.get("has_gt", True) else "N/A"
            exp_c  = sc.get("expected_count", sc.get("gt_count", "?"))
            pred_c = sc.get("produced_count", sc.get("pred_count", "?"))
            print(f"F1={f1_str}  exp={exp_c} pred={pred_c}  tokens={usage.get('total_tokens','?')}")

            for k, v in usage.items():
                total_usage[k] = total_usage.get(k, 0) + v

            row = {
                "task_id":    task_id,
                "group":      group,
                "difficulty": difficulty,
                "response":   response,
                "score":      sc,
                "usage":      usage,
            }
            rows.append(row)
            fout.write(json.dumps(row) + "\n")
            fout.flush()

    # Summary for this group
    evaluable = [r for r in rows if r["score"].get("has_gt", True)
                 and r["score"].get("expected_count", r["score"].get("gt_count", 1)) > 0]
    n_eval = len(evaluable)
    if n_eval == 0:
        print(f"\n  No evaluable tasks in this run.")
        return

    avg_f1   = sum(r["score"]["f1"] for r in evaluable) / n_eval
    avg_rec  = sum(r["score"]["recall"] for r in evaluable) / n_eval
    avg_prec = sum(r["score"]["precision"] for r in evaluable) / n_eval
    exact_pct    = sum(r["score"].get("exact_match", 0) for r in evaluable) / n_eval * 100
    any_correct  = sum(1 for r in evaluable if r["score"].get("recall", 0) > 0) / n_eval * 100
    accuracy     = sum(1 for r in evaluable if r["score"]["f1"] >= 0.5) / n_eval * 100

    by_diff: Dict[str, list] = {}
    for r in evaluable:
        by_diff.setdefault(r["difficulty"], []).append(r["score"]["f1"])

    print(f"\n  --- {group} (this run: {n_eval} evaluable tasks) ---")
    print(f"  Accuracy (F1≥0.5) : {accuracy:.1f}%  ({sum(1 for r in evaluable if r['score']['f1']>=0.5)}/{n_eval})")
    print(f"  Avg F1            : {avg_f1:.4f}")
    print(f"  Avg Precision     : {avg_prec:.4f}")
    print(f"  Avg Recall        : {avg_rec:.4f}")
    print(f"  Exact Match %     : {exact_pct:.1f}%")
    print(f"  AnyCorrect %      : {any_correct:.1f}%")
    print(f"\n  By difficulty:")
    for diff, scores in sorted(by_diff.items()):
        acc_d = sum(1 for s in scores if s >= 0.5)
        print(f"    {diff:12s}: Acc={acc_d}/{len(scores)}={acc_d/len(scores)*100:.0f}%  F1={sum(scores)/len(scores):.4f}")
    print(f"\n  Token usage (this run): {total_usage}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Claude Sonnet 4.6 on MedAgentBench groups")
    parser.add_argument(
        "--groups", nargs="+", default=["all"],
        help="Groups to run. Use 'all' for all groups, or list specific groups."
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("ERROR: OPENROUTER_API_KEY environment variable not set.")

    client = openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    groups = ALL_GROUPS if args.groups == ["all"] else args.groups
    invalid = [g for g in groups if g not in ALL_GROUPS]
    if invalid:
        raise SystemExit(f"Unknown groups: {invalid}. Valid: {ALL_GROUPS}")

    print(f"Model : {MODEL}")
    print(f"Groups: {groups}")

    for group in groups:
        run_group(client, group)

    print("\n\nAll groups done.")


if __name__ == "__main__":
    main()
