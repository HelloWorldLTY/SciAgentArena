"""oracle_chem — Generic RDKit-based chemical scoring oracles.

Scorer protocol
~~~~~~~~~~~~~~~
Every public scorer accepts ``(agent_stdout: str, task: dict)`` and returns::
    {
        "validity": float,      # fraction of outputs that are chemically valid/parsable
        "correctness": float,   # fraction that match ground truth within tolerance
        "details": dict,        # per-molecule breakdown or summary stats
    }
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# ---------------------------------------------------------------------------
# 1. Generic Numeric Scorer (MW, LogP, TPSA, QED, etc.)
# ---------------------------------------------------------------------------


def score_numeric_property(agent_output: str, task: dict) -> dict:
    """
    Generic scorer for any numeric property (MW, LogP, TPSA, QED).

    Configuration (in Task JSON 'scoring_logic'):
        ground_truth_file (str): Path to CSV containing SMILES and correct values.
        target_column (str): The column name in GT and Agent output to compare (e.g., "MW").
        tolerance (float): Maximum allowed difference for a correct answer.
    """
    # 1. Load Configuration
    config = task.get("scoring_logic", {})
    gt_path = config.get("ground_truth_file")
    col_name = config.get("target_column", "Value").upper()  # Normalize to uppercase
    merge_key = config.get("merge_key", "SMILES").upper()
    tolerance = config.get("tolerance", 0.01)

    if not gt_path or not os.path.exists(gt_path):
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": f"GT file not found: {gt_path}"},
        }

    # 2. Load Ground Truth
    try:
        df_gt = pd.read_csv(gt_path)
        df_gt.columns = [
            c.upper() for c in df_gt.columns
        ]  # Uppercase cols for matching
    except Exception as e:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": f"GT Load Error: {str(e)}"},
        }

    # 3. Parse Agent Output
    try:
        agent_data = json.loads(agent_output.strip())
        # Handle simple dict format {smiles: value} -> list of dicts
        if isinstance(agent_data, dict):
            agent_data = [{merge_key: k, col_name: v} for k, v in agent_data.items()]

        if not isinstance(agent_data, list):
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {"error": "Output is not a list or dict"},
            }

        df_agent = pd.DataFrame(agent_data)
        if not df_agent.empty:
            df_agent.columns = [c.upper() for c in df_agent.columns]
    except Exception as e:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": f"JSON Parse Error: {str(e)}"},
        }

    # 4. Data Alignment (Merge on SMILES)
    if merge_key not in df_agent.columns:
        return {
            "validity": 0.5,
            "correctness": 0.0,
            "details": {"error": f"Agent output missing '{merge_key}' column"},
        }

    # Deduplicate agent output
    df_agent = df_agent.drop_duplicates(subset=[merge_key])

    # Merge (Left join ensures we grade against all GT rows)
    merged = pd.merge(
        df_gt, df_agent, on=merge_key, how="left", suffixes=("_gt", "_agent")
    )

    # 5. Compute Metrics
    # Check if 'Valid' column exists in GT (from our 90-5-5 generation script), default to all True if missing
    if "VALID" in merged.columns:
        valid_rows = merged[merged["VALID"] == True].copy()
        invalid_rows = merged[merged["VALID"] == False].copy()
    else:
        valid_rows = merged.copy()
        invalid_rows = pd.DataFrame()

    # Determine column names in merged DF
    agent_col = (
        f"{col_name}_agent" if f"{col_name}_agent" in merged.columns else col_name
    )
    gt_col = f"{col_name}_gt" if f"{col_name}_gt" in merged.columns else col_name

    if agent_col not in valid_rows.columns:
        return {
            "validity": 1.0,
            "correctness": 0.0,
            "details": {"error": f"Column '{col_name}' missing in agent output"},
        }

    # Accuracy Calculation (on Valid molecules)
    if len(valid_rows) > 0:
        valid_rows[agent_col] = pd.to_numeric(valid_rows[agent_col], errors="coerce")
        diff = np.abs(valid_rows[gt_col] - valid_rows[agent_col])
        # Success: diff <= tolerance. NaNs (crashes/nulls) are failures.
        n_correct = (diff <= tolerance).sum()
        accuracy = n_correct / len(valid_rows)
        rmse = np.sqrt(np.nanmean(diff**2)) if len(diff) > 0 else 0.0
    else:
        accuracy = 0.0
        rmse = 0.0

    # Robustness Calculation (on Invalid molecules)
    # Goal: Agent should return NULL/NaN or omit the row. It should NOT return a number.
    robustness = 1.0
    if len(invalid_rows) > 0:
        if agent_col in invalid_rows.columns:
            # Count how many have a valid number (Hallucination = Bad)
            bad_values = (
                pd.to_numeric(invalid_rows[agent_col], errors="coerce").notna().sum()
            )
            robustness = 1.0 - (bad_values / len(invalid_rows))

    return {
        "validity": 1.0,  # JSON parsed successfully
        "correctness": round(accuracy, 4),
        "details": {
            "metric": col_name,
            "merge_key": merge_key,
            "rmse": round(rmse, 4),
            "robustness_score": round(robustness, 4),
            "n_samples": len(df_gt),
            "n_correct": int(n_correct) if len(valid_rows) > 0 else 0,
        },
    }


# ---------------------------------------------------------------------------
# 1b. Strict Formal Charge Scorer (schema + exact ID alignment)
# ---------------------------------------------------------------------------


def score_formal_charge_strict(agent_output: str, task: dict) -> dict:
    """
    Strict scorer for the formal-charge sanity-check tasks.

    Expected agent output:
      [
        {"ID": "FC_EASY_001", "Formal_Charge": 1},
        {"ID": "FC_EASY_002", "Formal_Charge": -1},
        ...
      ]

    Rules:
    - Top-level must be a JSON list.
    - Each item must be an object containing EXACTLY keys: ID, Formal_Charge.
    - ID must be a string.
    - Formal_Charge must be integer or null.
    - Duplicate IDs are invalid.
    - Correctness is exact match on integer charge against GT by ID.
    """
    scoring_conf = task.get("scoring_logic", {})
    gt_path = scoring_conf.get("ground_truth_file")
    if not gt_path or not os.path.exists(gt_path):
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {
                "error": f"GT file not found: {gt_path}",
                "error_type": "gt_missing",
                "comment": "Scoring failed: ground-truth file missing.",
            },
        }

    try:
        df_gt = pd.read_csv(gt_path)
    except Exception as e:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {
                "error": f"GT load error: {str(e)}",
                "error_type": "gt_load_error",
                "comment": "Scoring failed: could not read ground-truth file.",
            },
        }

    required_gt_cols = {"ID", "Formal_Charge"}
    if not required_gt_cols.issubset(set(df_gt.columns)):
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {
                "error": f"GT must contain columns {sorted(required_gt_cols)}",
                "error_type": "gt_schema_error",
                "comment": "Scoring failed: GT schema mismatch.",
            },
        }

    try:
        data = json.loads(agent_output.strip())
    except Exception as e:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {
                "error": f"JSON parse error: {str(e)}",
                "error_type": "json_parse_error",
                "comment": "Invalid output: not parseable JSON.",
            },
        }

    if not isinstance(data, list):
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {
                "error": "Output must be a JSON list",
                "error_type": "format_error",
                "comment": "Invalid format: top-level must be a list.",
            },
        }

    expected_keys = {"ID", "Formal_Charge"}
    parsed_rows = []
    seen_ids = set()
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {
                    "error": f"Item at index {idx} is not a JSON object",
                    "error_type": "format_error",
                    "comment": f"Invalid format at item {idx}: object expected.",
                },
            }

        if set(item.keys()) != expected_keys:
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {
                    "error": f"Item at index {idx} must contain exactly keys {sorted(expected_keys)}",
                    "error_type": "schema_error",
                    "comment": f"Invalid keys at item {idx}: expected only ID and Formal_Charge.",
                },
            }

        row_id = item["ID"]
        if not isinstance(row_id, str):
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {
                    "error": f"ID at index {idx} must be a string",
                    "error_type": "schema_error",
                    "comment": f"Invalid ID type at item {idx}.",
                },
            }
        if row_id in seen_ids:
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {
                    "error": f"Duplicate ID found: {row_id}",
                    "error_type": "duplicate_id",
                    "comment": f"Duplicate ID: {row_id}.",
                },
            }
        seen_ids.add(row_id)

        raw_charge = item["Formal_Charge"]
        if raw_charge is None:
            charge = None
        elif isinstance(raw_charge, bool):
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {
                    "error": f"Formal_Charge at index {idx} must be integer or null",
                    "error_type": "schema_error",
                    "comment": f"Invalid Formal_Charge type at item {idx}.",
                },
            }
        elif isinstance(raw_charge, int):
            charge = raw_charge
        elif isinstance(raw_charge, float) and raw_charge.is_integer():
            charge = int(raw_charge)
        else:
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {
                    "error": f"Formal_Charge at index {idx} must be integer or null",
                    "error_type": "schema_error",
                    "comment": f"Invalid Formal_Charge type at item {idx}.",
                },
            }

        parsed_rows.append({"ID": row_id, "Formal_Charge": charge})

    df_agent = pd.DataFrame(parsed_rows, columns=["ID", "Formal_Charge"])
    df_merge = pd.merge(
        df_gt, df_agent, on="ID", how="left", suffixes=("_gt", "_agent")
    )

    gt_col = (
        "Formal_Charge_gt"
        if "Formal_Charge_gt" in df_merge.columns
        else "Formal_Charge"
    )
    agent_col = (
        "Formal_Charge_agent"
        if "Formal_Charge_agent" in df_merge.columns
        else "Formal_Charge"
    )

    if agent_col not in df_merge.columns:
        return {
            "validity": 1.0,
            "correctness": 0.0,
            "details": {
                "error": "Missing Formal_Charge values in output after merge",
                "error_type": "missing_values",
                "comment": "No Formal_Charge values found after ID alignment.",
            },
        }

    gt_vals = pd.to_numeric(df_merge[gt_col], errors="coerce")
    agent_vals = pd.to_numeric(df_merge[agent_col], errors="coerce")
    correct_mask = (gt_vals == agent_vals) & gt_vals.notna() & agent_vals.notna()
    n_correct = int(correct_mask.sum())
    n_total = int(len(df_merge))
    correctness = (n_correct / n_total) if n_total > 0 else 0.0

    gt_ids = set(df_gt["ID"].astype(str))
    agent_ids = set(df_agent["ID"].astype(str))
    missing_ids = sorted(gt_ids - agent_ids)
    extra_ids = sorted(agent_ids - gt_ids)

    # Build concise but specific self-reflection comment.
    null_count = int(agent_vals.isna().sum())
    mismatch_count = n_total - n_correct
    wrong_charge_count = int(
        ((gt_vals != agent_vals) & gt_vals.notna() & agent_vals.notna()).sum()
    )
    mismatch_rows = df_merge[(gt_vals != agent_vals) | agent_vals.isna()].copy()
    mismatch_examples = []
    for _, r in mismatch_rows.head(3).iterrows():
        rid = str(r["ID"])
        p = r.get(agent_col, None)
        g = r.get(gt_col, None)
        p_str = "null" if pd.isna(p) else str(int(p) if float(p).is_integer() else p)
        g_str = "null" if pd.isna(g) else str(int(g) if float(g).is_integer() else g)
        mismatch_examples.append(f"{rid}: pred={p_str}, gt={g_str}")

    # Weak false-positive signal: constant-output strategy.
    unique_preds = set(
        [int(x) for x in agent_vals.dropna().tolist() if float(x).is_integer()]
    )
    gt_mode = None
    if len(gt_vals.dropna()) > 0:
        gt_mode = int(gt_vals.dropna().mode().iloc[0])
    constant_hit = (
        len(unique_preds) == 1
        and gt_mode is not None
        and next(iter(unique_preds)) == gt_mode
    )

    if correctness == 1.0:
        if n_total <= 1:
            comment = "Correct, but single-sample result may be a false positive."
            error_type = "possible_false_positive"
        elif constant_hit:
            comment = f"Correct, but all predictions are constant ({gt_mode}); possible false positive."
            error_type = "possible_false_positive"
        else:
            comment = "Correct and likely genuine (multi-sample exact match)."
            error_type = "none"
    else:
        if missing_ids or extra_ids:
            comment = f"Incorrect: ID alignment issue (missing={len(missing_ids)}, extra={len(extra_ids)})."
            error_type = "id_alignment"
        elif null_count > 0:
            ex = "; ".join(mismatch_examples) if mismatch_examples else ""
            comment = f"Incorrect: {null_count}/{n_total} predictions were null/unparsable. {ex}".strip()
            error_type = "null_prediction"
        elif wrong_charge_count > 0:
            ex = "; ".join(mismatch_examples) if mismatch_examples else ""
            comment = f"Incorrect: wrong formal charge for {wrong_charge_count}/{n_total} samples. {ex}".strip()
            error_type = "wrong_charge"
        else:
            ex = "; ".join(mismatch_examples) if mismatch_examples else ""
            comment = f"Incorrect: mismatch on {mismatch_count}/{n_total} samples. {ex}".strip()
            error_type = "mismatch"

    return {
        "validity": 1.0,
        "correctness": round(correctness, 4),
        "details": {
            "n_samples": n_total,
            "n_correct": n_correct,
            "missing_ids": missing_ids,
            "extra_ids": extra_ids,
            "error_type": error_type,
            "comment": comment,
        },
    }


# ---------------------------------------------------------------------------
# 1c. Hallucination / Corrupt-Input Scorer
# ---------------------------------------------------------------------------


def score_hallucination_stealth(agent_output: str, task: dict) -> dict:
    """
    Robustness / hallucination-detection scorer (stealth mode).

    Expected agent output:
      [
        {
          "ID": "...",
          "Status": "Success|Error",
          "Formal_Charge": int|null,
          "Error_Message": "Briefly explain why if Error, else null"
        }
      ]
    """
    # 1. Get stealth ground-truth rules from the task JSON.
    scoring_conf = task.get("scoring_logic", {})
    gt_conf = scoring_conf.get("ground_truth", {})
    expected_status = gt_conf.get("Expected_Status", "Error")
    expected_charge = gt_conf.get("Formal_Charge", None)
    required_keywords = gt_conf.get("Required_Keywords", [])

    try:
        data = json.loads(agent_output.strip())
    except Exception as e:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {
                "error": f"JSON parse error: {str(e)}",
                "error_type": "json_parse_error",
                "comment": "Invalid output: not parseable JSON.",
            },
        }

    if not isinstance(data, list) or len(data) == 0:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {
                "error": "Output must be a non-empty JSON list",
                "error_type": "format_error",
                "comment": "Invalid format: top-level must be a list.",
            },
        }

    expected_keys = {"ID", "Status", "Formal_Charge", "Error_Message"}
    seen_ids = set()
    n_total = len(data)
    n_correct = 0
    mismatch_examples = []

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {
                    "error": f"Item at index {idx} is not a JSON object",
                    "error_type": "format_error",
                },
            }

        if set(item.keys()) != expected_keys:
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {
                    "error": f"Item at index {idx} must contain exactly keys {sorted(expected_keys)}",
                    "error_type": "schema_error",
                },
            }

        row_id = item["ID"]
        if not isinstance(row_id, str):
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {
                    "error": f"ID at index {idx} must be a string",
                    "error_type": "schema_error",
                },
            }

        if row_id in seen_ids:
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {
                    "error": f"Duplicate ID found: {row_id}",
                    "error_type": "duplicate_id",
                },
            }
        seen_ids.add(row_id)

        status = item["Status"]
        charge = item["Formal_Charge"]
        error_msg = item["Error_Message"] if item["Error_Message"] is not None else ""

        item_is_correct = True
        error_comment = ""

        if status != expected_status:
            item_is_correct = False
            error_comment = (
                f"Status mismatch: expected '{expected_status}', got '{status}'."
            )
        elif charge is not expected_charge:
            item_is_correct = False
            error_comment = (
                f"Hallucinated a charge value ({charge}) when it should be null."
            )
        elif expected_status == "Error" and required_keywords:
            has_keyword = any(
                kw.lower() in str(error_msg).lower() for kw in required_keywords
            )
            if not has_keyword:
                item_is_correct = False
                error_comment = f"Missed scientific reasons. Agent wrote: '{error_msg}'"

        if item_is_correct:
            n_correct += 1
        else:
            mismatch_examples.append(f"[{row_id}]: {error_comment}")

    correctness = (n_correct / n_total) if n_total > 0 else 0.0

    if correctness == 1.0:
        comment = "Perfect hallucination detection. Agent correctly rejected invalid data with sound scientific reasoning."
        error_type = "none"
    else:
        ex = "; ".join(mismatch_examples[:3])
        if len(mismatch_examples) > 3:
            ex += " ... (truncated)"
        comment = f"Hallucination detected or incorrect reasoning: {ex}"
        error_type = "hallucination_or_poor_reasoning"

    return {
        "validity": 1.0,
        "correctness": round(correctness, 4),
        "details": {
            "n_samples": n_total,
            "n_correct": n_correct,
            "error_type": error_type,
            "comment": comment,
        },
    }


# ---------------------------------------------------------------------------
# 2. Generic Classification Scorer (Substructure Filtering)
# ---------------------------------------------------------------------------


def score_classification(agent_output: str, task: dict) -> dict:
    """
    Generic scorer for classification (e.g., Substructure Filtering).
    Compares a list of 'matches' against a boolean column in GT.
    """
    config = task.get("scoring_logic", {})
    gt_path = config.get("ground_truth_file")
    target_col = config.get("target_column").upper()  # e.g., "HAS_BENZO"

    if not gt_path or not os.path.exists(gt_path):
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": "GT file missing"},
        }

    df_gt = pd.read_csv(gt_path)
    df_gt.columns = [c.upper() for c in df_gt.columns]

    if target_col not in df_gt.columns:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": f"Column {target_col} not in GT"},
        }

    # Extract True Positives from GT
    true_smiles = set(df_gt[df_gt[target_col] == True]["SMILES"])

    # Parse Agent Output
    try:
        data = json.loads(agent_output.strip())
        # Agent usually returns {"matches": [...]} or just [...]
        if isinstance(data, dict):
            pred_smiles = set(data.get("matches", []) or data.get("MATCHES", []))
        elif isinstance(data, list):
            pred_smiles = set(data)
        else:
            pred_smiles = set()
    except:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": "JSON parse failed"},
        }

    # Metrics
    tp = len(pred_smiles.intersection(true_smiles))
    fp = len(pred_smiles - true_smiles)
    fn = len(true_smiles - pred_smiles)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "validity": 1.0,
        "correctness": round(f1, 4),
        "details": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
        },
    }


# ---------------------------------------------------------------------------
# 3. Generic Ranking Scorer (Similarity Search)
# ---------------------------------------------------------------------------


def score_ranking(agent_stdout: str, task: dict) -> dict:
    """
    Score agent ranking results against a ground truth CSV.

    Metrics:
    - Validity: Is the output a valid JSON list?
    - Correctness: Top-K overlap is the primary signal.
    - Secondary penalties: mild deductions for poor rank alignment or score
      calibration on overlapping molecules.
    """
    scoring_conf = task.get("scoring_logic", {})
    gt_path = scoring_conf.get("ground_truth_file")
    score_col = scoring_conf.get("score_column", "Similarity")
    smiles_col = scoring_conf.get("smiles_column", "SMILES")

    # 1. Load Ground Truth
    try:
        # Load the pre-calculated GT file which is already sorted by rank
        df_gt = pd.read_csv(gt_path)
        # We only care about the top K from the GT where K is the length of agent output or fixed 10
        # Assuming the GT file is already sorted descending by similarity as generated by your script
        gt_top_10 = df_gt.head(10)
        gt_smiles_set = set(gt_top_10[smiles_col].astype(str))
    except Exception as e:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": f"Failed to load ground truth: {str(e)}"},
        }

    # 2. Parse Agent Output
    try:
        agent_data = json.loads(agent_stdout.strip())

        # Handle case where agent outputs a dict {"matches": [...]} vs just a list [...]
        if isinstance(agent_data, dict):
            # Try to find a list value
            if "results" in agent_data:
                agent_data = agent_data["results"]
            elif "matches" in agent_data:
                agent_data = agent_data["matches"]
            elif "top_hits" in agent_data:
                agent_data = agent_data["top_hits"]
            else:
                # If just a single dict, wrap it? No, likely wrong format.
                pass

        if not isinstance(agent_data, list):
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {"error": "Output must be a JSON list of dictionaries."},
            }

        # Extract SMILES, Scores, and Ranks from Agent Output
        agent_smiles = []
        agent_scores = {}
        agent_ranks = {}

        for item in agent_data:
            if not isinstance(item, dict):
                continue
            # Case insensitive key lookup
            item_lower = {k.lower(): v for k, v in item.items()}
            smi = item_lower.get("smiles")
            score = item_lower.get("similarity") or item_lower.get("score")
            rank = item_lower.get("rank")

            if smi:
                smi_str = str(smi)
                agent_smiles.append(smi_str)
                if score is not None:
                    agent_scores[smi_str] = float(score)
                if rank is not None:
                    try:
                        agent_ranks[smi_str] = int(rank)
                    except Exception:
                        pass

    except json.JSONDecodeError:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": "Output is not valid JSON."},
        }
    except Exception as e:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": f"Error parsing agent output: {str(e)}"},
        }

    # 3. Calculate Scores

    # Validity: Did they find at least 1 molecule?
    if not agent_smiles:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": "No SMILES found in output"},
        }

    validity = 1.0  # JSON parsed and contained data

    # Correctness: Jaccard Intersection of Top K
    # We compare the set of SMILES strings.
    # Note: Canonicalization might be needed if agent outputs different SMILES strings for same mol.
    # But since they are selecting FROM the file, strings should match exactly.

    agent_set = set(agent_smiles[:10])  # Only look at top 10

    intersection = agent_set.intersection(gt_smiles_set)
    overlap_count = len(intersection)

    # Primary signal: overlap of top-k sets.
    overlap_score = overlap_count / 10.0
    correctness = overlap_score

    details = {
        "agent_top_k_count": len(agent_set),
        "gt_top_k_count": len(gt_smiles_set),
        "intersection_count": overlap_count,
        "overlap_score": overlap_score,
        "missed_molecules": list(gt_smiles_set - agent_set),
        "extra_molecules": list(agent_set - gt_smiles_set),
    }

    gt_rank_map = {}
    gt_score_map = {}
    for idx, row in gt_top_10.reset_index(drop=True).iterrows():
        smi = str(row[smiles_col])
        gt_rank_map[smi] = idx + 1
        try:
            gt_score_map[smi] = float(row[score_col])
        except Exception:
            pass

    # Secondary signal: numerical accuracy of scores for overlapping molecules.
    score_diffs = []
    for smi in intersection:
        if smi in agent_scores:
            gt_score_val = gt_score_map.get(smi, float("nan"))
            diff = abs(agent_scores[smi] - gt_score_val)
            score_diffs.append(diff)

    score_penalty = 0.0
    if score_diffs:
        avg_diff = sum(score_diffs) / len(score_diffs)
        details["avg_score_error"] = avg_diff

        # Similarity values are typically in [0, 1]. Penalize gently and cap the
        # effect so overlap remains the dominant signal.
        score_penalty = min(max(avg_diff, 0.0), 1.0) * 0.05
        details["score_penalty"] = score_penalty

    # Secondary signal: rank alignment for the overlapping molecules.
    rank_diffs = []
    for smi in intersection:
        if smi in agent_ranks and smi in gt_rank_map:
            rank_diffs.append(abs(agent_ranks[smi] - gt_rank_map[smi]))

    rank_penalty = 0.0
    if rank_diffs:
        avg_rank_diff = sum(rank_diffs) / len(rank_diffs)
        details["avg_rank_error"] = avg_rank_diff
        normalized_rank_error = avg_rank_diff / 9.0  # top-10 ranks => max diff 9
        rank_penalty = min(max(normalized_rank_error, 0.0), 1.0) * 0.05
        details["rank_penalty"] = rank_penalty

    total_penalty = score_penalty + rank_penalty
    if total_penalty > 0:
        details["total_penalty"] = total_penalty
        correctness = max(0.0, overlap_score - total_penalty)

    return {
        "validity": validity,
        "correctness": round(correctness, 2),
        "details": details,
    }


# ---------------------------------------------------------------------------
# 3. Generic 3d conformer Scorer (Format Conversion)
# ---------------------------------------------------------------------------


def score_3d_conformer(agent_stdout: str, task: dict) -> dict:
    """
    Score Format Conversion task: Checks if the output is a valid 3D SDF
    of the expected largest molecule.
    """
    scoring_conf = task.get("scoring_logic", {})
    gt_path = scoring_conf.get("ground_truth_file")

    # 1. Load Ground Truth (The Target Molecule)
    try:
        df_gt = pd.read_csv(gt_path)
        # GT file stores scalar values; first row holds Target_SMILES/Target_MW
        target_smiles = df_gt["Target_SMILES"].iloc[0]
        target_mw = df_gt["Target_MW"].iloc[0]
    except Exception as e:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": f"GT Load Failed: {e}"},
        }

    # 2. Parse Agent Output
    try:
        data = json.loads(agent_stdout.strip())
        # Expecting: {"largest_molecule": {"id": "...", "sdf_content": "..."}}
        # Or flattened: {"id": "...", "sdf_content": "..."}

        if "largest_molecule" in data:
            data = data["largest_molecule"]

        agent_sdf = data.get("sdf_content", "")
        agent_id = data.get("id", "")

    except Exception as e:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": f"JSON Parse Failed: {e}"},
        }

    if not agent_sdf:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": "No SDF content found"},
        }

    # 3. Validate SDF with RDKit
    try:
        mol = Chem.MolFromMolBlock(agent_sdf)
        if not mol:
            return {
                "validity": 0.5,
                "correctness": 0.0,
                "details": {"error": "SDF text could not be parsed by RDKit"},
            }
    except Exception as e:
        return {
            "validity": 0.5,
            "correctness": 0.0,
            "details": {"error": f"SDF Parse Error: {e}"},
        }

    # 4. Check 3D Coordinates (Z-axis should not be all 0)
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    z_coords = positions[:, 2]
    is_3d = not np.allclose(z_coords, 0.0)

    if not is_3d:
        return {
            "validity": 1.0,
            "correctness": 0.0,
            "details": {
                "error": "Molecule is flat (2D). EmbedMolecule failed or wasn't called."
            },
        }

    # 5. Check Identity (Is it the largest molecule?)
    # We check by MW or Canonical SMILES
    agent_mw = Descriptors.ExactMolWt(mol)

    # Tolerance for MW (0.1 Da)
    if abs(agent_mw - target_mw) > 0.1:
        return {
            "validity": 1.0,
            "correctness": 0.0,
            "details": {
                "error": f"Wrong molecule selected. Expected MW ~{target_mw}, got {agent_mw}",
                "agent_id": agent_id,
            },
        }

    return {
        "validity": 1.0,
        "correctness": 1.0,
        "details": {"msg": "Success! 3D conformer generated for largest molecule."},
    }


# ---------------------------------------------------------------------------
# 4. Lipinski Classification Scorer
# ---------------------------------------------------------------------------
def score_lipinski(agent_stdout: str, task: dict) -> dict:
    """
    Score Lipinski Classification task.
    Metric: Classification Accuracy (Fraction of correctly labeled molecules).

    Expected Agent Output:
    {
      "results": [
         {"id": "MOL_TEST_0001", "label": "Drug-Like", ...},
         ...
      ]
    }
    """
    import pandas as pd  # Ensure pandas is imported if not already at top

    scoring_conf = task.get("scoring_logic", {})
    gt_path = scoring_conf.get("ground_truth_file")

    # 1. Load Ground Truth
    try:
        df_gt = pd.read_csv(gt_path)
        # Create a map: ID -> Label
        # We normalize the label to lowercase to be lenient
        gt_map = {row["ID"]: row["Label"].lower() for _, row in df_gt.iterrows()}
    except Exception as e:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": f"Failed to load Ground Truth: {str(e)}"},
        }

    # 2. Parse Agent Output
    try:
        data = json.loads(agent_stdout.strip())
        results = data.get("results", [])

        if not isinstance(results, list) or not results:
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {"error": "JSON must contain a 'results' list"},
            }

    except json.JSONDecodeError:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": "Agent output is not valid JSON"},
        }
    except Exception as e:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": f"JSON parsing error: {str(e)}"},
        }

    # 3. Calculate Accuracy
    correct_count = 0
    total_checked = 0
    errors = []

    for item in results:
        mol_id = item.get("id")
        agent_label = str(item.get("label", "")).lower()

        # Only score IDs that exist in our Ground Truth
        if mol_id in gt_map:
            total_checked += 1
            true_label = gt_map[mol_id]

            # Flexible matching: "drug-like" == "drug-like"
            # Also handle simple "pass"/"fail" if agent diverges slightly, though prompt was specific.
            if agent_label == true_label:
                correct_count += 1
            else:
                if len(errors) < 5:  # Limit error log size
                    errors.append(
                        f"{mol_id}: Agent '{agent_label}' != GT '{true_label}'"
                    )

    if total_checked == 0:
        return {
            "validity": 1.0,
            "correctness": 0.0,
            "details": {
                "error": "No matching IDs found between Agent output and Ground Truth"
            },
        }

    accuracy = correct_count / total_checked

    return {
        "validity": 1.0,
        "correctness": round(accuracy, 2),
        "details": {
            "total_evaluated": total_checked,
            "correct_labels": correct_count,
            "sample_errors": errors,
        },
    }


# ---------------------------------------------------------------------------
# 5. Target Identification / Retrieval Scorer
# ---------------------------------------------------------------------------
def score_target_retrieval(agent_stdout: str, task: dict) -> dict:
    """
    Score Target Identification.
    Metrics:
    1. Correct MONDO ID (Exact String Match)
    2. Correct UniProt List (Jaccard Index of sets)
    """
    import pandas as pd

    scoring_conf = task.get("scoring_logic", {})
    gt_path = scoring_conf.get("ground_truth_file")

    # 1. Load Ground Truth
    try:
        df_gt = pd.read_csv(gt_path)
        # We assume the GT file contains ONLY the rows that should be returned
        gt_mondo = df_gt["MONDO_ID"].iloc[0] if not df_gt.empty else ""
        gt_uniprots = set(df_gt["UniProt"].astype(str))
    except Exception as e:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": f"GT Load Error: {e}"},
        }

    # 2. Parse Agent Output
    try:
        data = json.loads(agent_stdout.strip())

        agent_mondo = str(data.get("mondo_id", "")).strip()
        agent_targets_list = data.get("targets", [])

        if not isinstance(agent_targets_list, list):
            raise ValueError("Targets must be a list")

        agent_uniprots = set(str(x).strip() for x in agent_targets_list)

    except Exception as e:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": f"JSON Parse Error: {e}"},
        }

    # 3. Score
    # Part A: MONDO ID check (Binary)
    mondo_score = 1.0 if agent_mondo == gt_mondo else 0.0

    # Part B: UniProt Set check (Jaccard)
    if not agent_uniprots and not gt_uniprots:
        jaccard = 1.0
    elif not agent_uniprots or not gt_uniprots:
        jaccard = 0.0
    else:
        intersection = len(agent_uniprots.intersection(gt_uniprots))
        union = len(agent_uniprots.union(gt_uniprots))
        jaccard = intersection / union

    # Combined Score (50% ID accuracy, 50% Retrieval accuracy)
    final_score = (mondo_score * 0.5) + (jaccard * 0.5)

    return {
        "validity": 1.0,
        "correctness": round(final_score, 2),
        "details": {
            "expected_mondo": gt_mondo,
            "agent_mondo": agent_mondo,
            "expected_count": len(gt_uniprots),
            "agent_count": len(agent_uniprots),
            "missed": list(gt_uniprots - agent_uniprots),
            "extra": list(agent_uniprots - gt_uniprots),
        },
    }


def score_string_property(agent_output: str, task: dict) -> dict:
    """
    Generic scorer for string properties (e.g., SMILES or TARGETS).
    Expects agent output to be a JSON list of dicts with an ID column.
    """
    config = task.get("scoring_logic", {})
    gt_path = config.get("ground_truth_file")
    target_col = (config.get("target_column") or "VALUE").upper()

    if not gt_path or not os.path.exists(gt_path):
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": f"GT file not found: {gt_path}"},
        }

    try:
        df_gt = pd.read_csv(gt_path)
        df_gt.columns = [c.upper() for c in df_gt.columns]
    except Exception as e:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": f"GT Load Error: {str(e)}"},
        }

    try:
        agent_data = json.loads(agent_output.strip())
        if isinstance(agent_data, dict):
            # Try common wrapper keys
            if "results" in agent_data:
                agent_data = agent_data["results"]
            elif "matches" in agent_data:
                agent_data = agent_data["matches"]
            elif "top_hits" in agent_data:
                agent_data = agent_data["top_hits"]
            else:
                # If it's a dict of ID->value, convert
                agent_data = [{"ID": k, target_col: v} for k, v in agent_data.items()]
        if not isinstance(agent_data, list):
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {"error": "Output must be a JSON list"},
            }
        df_agent = pd.DataFrame(agent_data)
        if not df_agent.empty:
            df_agent.columns = [c.upper() for c in df_agent.columns]
    except Exception as e:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": f"JSON Parse Error: {str(e)}"},
        }

    if "ID" not in df_agent.columns:
        return {
            "validity": 0.5,
            "correctness": 0.0,
            "details": {"error": "Agent output missing 'ID' column"},
        }
    if target_col not in df_agent.columns:
        return {
            "validity": 0.5,
            "correctness": 0.0,
            "details": {"error": f"Agent output missing '{target_col}' column"},
        }

    df_agent = df_agent.drop_duplicates(subset=["ID"])
    merged = pd.merge(df_gt, df_agent, on="ID", how="left", suffixes=("_gt", "_agent"))
    merged.columns = [c.upper() for c in merged.columns]

    gt_col = f"{target_col}_GT" if f"{target_col}_GT" in merged.columns else target_col
    agent_col = (
        f"{target_col}_AGENT" if f"{target_col}_AGENT" in merged.columns else target_col
    )

    if gt_col not in merged.columns or agent_col not in merged.columns:
        return {
            "validity": 1.0,
            "correctness": 0.0,
            "details": {"error": f"Missing columns after merge: {gt_col}, {agent_col}"},
        }

    def _norm(x):
        if pd.isna(x):
            return ""
        return str(x).strip()

    def _canon_smiles(s: str) -> str:
        if not s:
            return ""
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                return s.strip()
            return Chem.MolToSmiles(m, canonical=True)
        except Exception:
            return s.strip()

    correct = 0
    total = len(merged)
    mismatches = []

    for _, row in merged.iterrows():
        gt_val = _norm(row[gt_col])
        agent_val = _norm(row[agent_col])
        if not gt_val and not agent_val:
            correct += 1
            continue
        if target_col == "TARGETS":
            gt_set = {s.strip() for s in gt_val.split(",") if s.strip()}
            ag_set = {s.strip() for s in agent_val.split(",") if s.strip()}
            if gt_set == ag_set:
                correct += 1
            else:
                mismatches.append(
                    {"ID": row["ID"], "expected": gt_val, "got": agent_val}
                )
        else:
            if target_col == "SMILES":
                gt_cmp = _canon_smiles(gt_val)
                ag_cmp = _canon_smiles(agent_val)
                if gt_cmp == ag_cmp:
                    correct += 1
                else:
                    mismatches.append(
                        {"ID": row["ID"], "expected": gt_val, "got": agent_val}
                    )
            else:
                if gt_val == agent_val:
                    correct += 1
                else:
                    mismatches.append(
                        {"ID": row["ID"], "expected": gt_val, "got": agent_val}
                    )

    score = float(correct / total) if total > 0 else 0.0
    return {
        "validity": 1.0,
        "correctness": round(score, 4),
        "details": {
            "n_correct": correct,
            "n_total": total,
            "mismatches": mismatches[:20],
        },
    }
