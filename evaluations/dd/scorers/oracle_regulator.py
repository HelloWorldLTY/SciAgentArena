import ast
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd
from rdkit import Chem
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_indices(index_str):
    import re

    if pd.isna(index_str) or index_str == "":
        return []

    groups = re.findall(r"\d+", str(index_str))

    inner_groups = re.findall(r"\[([^\[\]]+)\]", str(index_str))

    if inner_groups:
        parsed = []
        for group in inner_groups:
            nums = [int(x.strip()) for x in group.split(",")]
            parsed.append(tuple(sorted(nums)))
        return parsed
    else:
        nums = [int(x) for x in groups]
        return [tuple(sorted(nums))] if nums else []


def score_toxicophore(agent_output, task):
    import json

    import pandas as pd

    # -----------------------------
    # Parse JSON if needed
    # -----------------------------
    if isinstance(agent_output, str):
        if not agent_output.strip():
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {
                    "error": "Agent produced no output (empty stdout). "
                    "Gemini agent likely could not find tasks_batch/data/toxalerts.csv."
                },
            }
        try:
            agent_output = json.loads(agent_output)
        except Exception as e:
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {"error": f"Invalid JSON: {e}"},
            }

    if isinstance(agent_output, dict) and "error" in agent_output:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": agent_output["error"]},
        }

    df_gt = pd.read_csv(task["scoring_logic"]["ground_truth_file"])

    # -----------------------------
    # Helper: flatten prediction
    # -----------------------------
    def flatten_predictions(agent_output):
        """
        Convert agent output into flat list:
        [
            {
                "smiles": ...,
                "alert_type": ...,
                "match_atom_indices": [...]
            }
        ]
        """
        flat = []

        # CASE 1: list input
        if isinstance(agent_output, list):
            if not agent_output:
                return flat

            # Helper: normalise match_atom_indices to list-of-matches format.
            # Handles both list-of-lists ([[i,j],[k,l]]) and flat-int-list ([i,j,k])
            # produced by GetSubstructMatch (singular) rather than GetSubstructMatches.
            def _norm_indices(raw):
                if not raw:
                    return []
                if isinstance(raw[0], (list, tuple)):
                    return [list(m) for m in raw]  # already list-of-matches
                return [raw]  # flat ints → single match

            # Per-molecule format: list of dicts, each with a nested "alerts" list.
            # Covers:
            #   - agent_solution_reg_01_tox.py (root): {"smiles":…, "alerts":[{alert_type,
            #       smarts, match_atom_indices (list-of-lists), mechanism, risk_level}], "id":…}
            #   - Gemini / other generated scripts that follow the same nesting.
            if isinstance(agent_output[0], dict) and "alerts" in agent_output[0]:
                for mol in agent_output:
                    smiles = mol.get("smiles")
                    for alert in mol.get("alerts", []):
                        alert_type = alert.get("alert_type")
                        if not alert_type:
                            continue  # skip malformed alert entries
                        flat.append(
                            {
                                "smiles": smiles,
                                "alert_type": alert_type,
                                "match_atom_indices": _norm_indices(
                                    alert.get("match_atom_indices", [])
                                ),
                            }
                        )
                return flat

            # Flat-list format: each item is one (molecule, alert) pair with "alert_type"
            # directly on the item (e.g. general_LLM_solutions/gemini_agent_solutions,
            # chemcrow, or any agent that outputs a flat array of alert records).
            if isinstance(agent_output[0], dict) and "alert_type" in agent_output[0]:
                for item in agent_output:
                    flat.append(
                        {
                            "smiles": item.get("smiles"),
                            "alert_type": item["alert_type"],
                            "match_atom_indices": _norm_indices(
                                item.get("match_atom_indices", [])
                            ),
                        }
                    )
                return flat

            # Generic fallback: return as-is and let downstream handle it
            return agent_output

        # CASE 2: wrapped dict with predictions
        if isinstance(agent_output, dict):

            # New format: screening_results
            # Accepts status == "flagged" (biomni/gpt) OR status == "screened" (toolUniverse_new).
            # In toolUniverse_new, match_atom_indices is a flat list of ints per alert
            # (e.g. [0,1,2]); wrap it so downstream iteration treats it as one match.
            if "screening_results" in agent_output:
                for mol in agent_output["screening_results"]:
                    smiles = mol.get("smiles")
                    status = mol.get("status")

                    # Accept molecules that are explicitly flagged/screened, OR
                    # molecules with no status field but with non-empty flagged_alerts
                    # (cactus agent output: uses num_alerts instead of status).
                    has_alerts = bool(mol.get("flagged_alerts"))
                    if status not in ("flagged", "screened") and not (
                        status is None and has_alerts
                    ):
                        continue

                    for alert in mol.get("flagged_alerts", []):
                        raw_indices = alert.get("match_atom_indices", [])
                        # Normalise: if the list contains plain ints (flat list), wrap it
                        # so it becomes [[i, j, ...]] — one match entry.
                        if raw_indices and not isinstance(
                            raw_indices[0], (list, tuple)
                        ):
                            normalised_indices = [raw_indices]
                        else:
                            normalised_indices = raw_indices
                        flat.append(
                            {
                                "smiles": smiles,
                                "alert_type": alert["alert_type"],
                                "match_atom_indices": normalised_indices,
                            }
                        )

                return flat

            if "flagged_alerts" in agent_output:
                for entry in agent_output["flagged_alerts"]:
                    match_indices = entry.get("match_atom_indices", [])
                    # wrap flat list into list-of-matches so downstream iteration works correctly
                    flat.append(
                        {
                            "smiles": entry["smiles"],
                            "alert_type": entry["alert_type"],
                            "match_atom_indices": (
                                [match_indices] if match_indices else []
                            ),
                        }
                    )
                return flat

            if "results" in agent_output and isinstance(agent_output["results"], list):
                for mol in agent_output["results"]:
                    smiles = mol.get("smiles")
                    for alert in mol.get("flagged_alerts", []):
                        match_indices = alert.get("match_atom_indices", [])
                        flat.append(
                            {
                                "smiles": smiles,
                                "alert_type": alert["alert_type"],
                                "match_atom_indices": (
                                    [match_indices] if match_indices else []
                                ),
                            }
                        )
                return flat

            # fallback: maybe already flat under "predictions"
            if "predictions" in agent_output:
                return agent_output["predictions"]

        # fallback empty
        return []

    # -----------------------------
    # Enrich entries that use molecule_id instead of smiles
    # (e.g. agent_solution_reg_01_tox output format)
    # -----------------------------
    if (
        isinstance(agent_output, list)
        and agent_output
        and isinstance(agent_output[0], dict)
    ):
        if "alerts" in agent_output[0] and "smiles" not in agent_output[0]:
            query_path = task.get("input", {}).get("query_molecules_file_path", "")
            if query_path and os.path.exists(query_path):
                try:
                    df_q = pd.read_csv(query_path)
                    df_q.columns = [c.strip().lower() for c in df_q.columns]
                    id_col = next(
                        (
                            c
                            for c in ["id", "molecule_id", "mol_id"]
                            if c in df_q.columns
                        ),
                        None,
                    )
                    smi_col = next(
                        (c for c in ["smiles", "smi"] if c in df_q.columns), None
                    )
                    if id_col and smi_col:
                        id_to_smi = {
                            str(row[id_col]): str(row[smi_col])
                            for _, row in df_q.iterrows()
                        }
                        for entry in agent_output:
                            if not entry.get("smiles"):
                                mol_id = str(
                                    entry.get("molecule_id", entry.get("id", ""))
                                )
                                if mol_id in id_to_smi:
                                    entry["smiles"] = id_to_smi[mol_id]
                except Exception:
                    pass

    predictions = flatten_predictions(agent_output)

    # -----------------------------
    # Build Ground Truth Sets
    # -----------------------------
    gt_alert_set = set()
    gt_substructure_set = set()
    gt_molecule_set = set()

    for _, row in df_gt.iterrows():
        smiles = row["smiles"]
        alert = row["alert_type"]
        all_indices = parse_indices(row["match_atom_indices"])

        for indices in all_indices:
            gt_substructure_set.add((smiles, alert, tuple(sorted(indices))))

        gt_alert_set.add((smiles, alert))
        gt_molecule_set.add(smiles)

    # -----------------------------
    # Build Prediction Sets
    # -----------------------------
    pred_alert_set = set()
    pred_substructure_set = set()
    pred_molecule_set = set()

    for item in predictions:
        smiles = item["smiles"]
        alert = item["alert_type"]

        pred_alert_set.add((smiles, alert))
        pred_molecule_set.add(smiles)

        for match in item.get("match_atom_indices", []):
            indices = tuple(sorted(match))
            pred_substructure_set.add((smiles, alert, indices))

    # -----------------------------
    # Alert-level Metrics
    # -----------------------------
    TP_alert = len(gt_alert_set & pred_alert_set)
    FP_alert = len(pred_alert_set - gt_alert_set)
    FN_alert = len(gt_alert_set - pred_alert_set)

    precision_alert = TP_alert / (TP_alert + FP_alert) if (TP_alert + FP_alert) else 0
    recall_alert = TP_alert / (TP_alert + FN_alert) if (TP_alert + FN_alert) else 0
    f1_alert = (
        2 * precision_alert * recall_alert / (precision_alert + recall_alert)
        if (precision_alert + recall_alert)
        else 0
    )

    # -----------------------------
    # Substructure-level Metrics
    # -----------------------------
    TP_sub = len(gt_substructure_set & pred_substructure_set)
    FP_sub = len(pred_substructure_set - gt_substructure_set)
    FN_sub = len(gt_substructure_set - pred_substructure_set)

    precision_sub = TP_sub / (TP_sub + FP_sub) if (TP_sub + FP_sub) else 0
    recall_sub = TP_sub / (TP_sub + FN_sub) if (TP_sub + FN_sub) else 0
    f1_sub = (
        2 * precision_sub * recall_sub / (precision_sub + recall_sub)
        if (precision_sub + recall_sub)
        else 0
    )

    # -----------------------------
    # Molecule-level Accuracy
    # -----------------------------
    all_molecules = set(df_gt["smiles"].unique())

    correct = 0
    for smi in all_molecules:
        gt_flag = smi in gt_molecule_set
        pred_flag = smi in pred_molecule_set
        if gt_flag == pred_flag:
            correct += 1

    molecule_accuracy = correct / len(all_molecules) if all_molecules else 0

    # -----------------------------
    # Per-molecule comparison:
    #   alert_type and match_atom_indices
    # -----------------------------
    # Build per-molecule ground truth lookup
    gt_per_mol = defaultdict(lambda: {"alert_types": set(), "substructures": set()})
    for _, row in df_gt.iterrows():
        smi = row["smiles"]
        alert = row["alert_type"]
        all_indices = parse_indices(row["match_atom_indices"])
        gt_per_mol[smi]["alert_types"].add(alert)
        for indices in all_indices:
            gt_per_mol[smi]["substructures"].add((alert, tuple(sorted(indices))))

    # Build per-molecule prediction lookup
    pred_per_mol = defaultdict(lambda: {"alert_types": set(), "substructures": set()})
    for item in predictions:
        smi = item["smiles"]
        alert = item["alert_type"]
        pred_per_mol[smi]["alert_types"].add(alert)
        for match in item.get("match_atom_indices", []):
            indices = tuple(sorted(match))
            pred_per_mol[smi]["substructures"].add((alert, indices))

    def _f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    alert_f1_scores = []
    alert_precision_scores = []
    alert_recall_scores = []
    sub_f1_scores = []
    sub_precision_scores = []
    sub_recall_scores = []

    for smi in sorted(all_molecules):
        gt_alerts = gt_per_mol[smi]["alert_types"]
        pred_alerts = pred_per_mol[smi]["alert_types"]
        gt_subs = gt_per_mol[smi]["substructures"]
        pred_subs = pred_per_mol[smi]["substructures"]

        tp_a = len(gt_alerts & pred_alerts)
        fp_a = len(pred_alerts - gt_alerts)
        fn_a = len(gt_alerts - pred_alerts)
        p_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) else 0.0
        r_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) else 0.0
        alert_precision_scores.append(p_a)
        alert_recall_scores.append(r_a)
        alert_f1_scores.append(_f1(tp_a, fp_a, fn_a))

        tp_s = len(gt_subs & pred_subs)
        fp_s = len(pred_subs - gt_subs)
        fn_s = len(gt_subs - pred_subs)
        p_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) else 0.0
        r_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) else 0.0
        sub_precision_scores.append(p_s)
        sub_recall_scores.append(r_s)
        sub_f1_scores.append(_f1(tp_s, fp_s, fn_s))

    n_mol = len(all_molecules) or 1
    per_mol_alert_precision = sum(alert_precision_scores) / n_mol
    per_mol_alert_recall = sum(alert_recall_scores) / n_mol
    per_mol_alert_f1 = sum(alert_f1_scores) / n_mol
    per_mol_sub_precision = sum(sub_precision_scores) / n_mol
    per_mol_sub_recall = sum(sub_recall_scores) / n_mol
    per_mol_sub_f1 = sum(sub_f1_scores) / n_mol

    # -----------------------------
    # Final Score
    # -----------------------------
    return {
        "validity": 1.0,
        "correctness": round(f1_sub, 4),
        "details": {
            "alert_precision": round(precision_alert, 4),
            "alert_recall": round(recall_alert, 4),
            "alert_f1": round(f1_alert, 4),
            "substructure_precision": round(precision_sub, 4),
            "substructure_recall": round(recall_sub, 4),
            "substructure_f1": round(f1_sub, 4),
            "molecule_accuracy": round(molecule_accuracy, 4),
            "per_mol_alert_precision": round(per_mol_alert_precision, 4),
            "per_mol_alert_recall": round(per_mol_alert_recall, 4),
            "per_mol_alert_f1": round(per_mol_alert_f1, 4),
            "per_mol_substructure_precision": round(per_mol_sub_precision, 4),
            "per_mol_substructure_recall": round(per_mol_sub_recall, 4),
            "per_mol_substructure_f1": round(per_mol_sub_f1, 4),
        },
    }


def score_herg(agent_output, task):
    # --- parse JSON if string ---
    if isinstance(agent_output, str):
        agent_output = json.loads(agent_output)

    if isinstance(agent_output, list):
        predictions = agent_output

    elif isinstance(agent_output, dict):
        if "predictions" in agent_output:
            predictions = agent_output["predictions"]
        elif "results" in agent_output:
            predictions = agent_output["results"]
        else:
            for v in agent_output.values():
                if isinstance(v, list):
                    predictions = v
                    break
            else:
                return {
                    "validity": 0.0,
                    "correctness": 0.0,
                    "details": {"error": "Invalid output format"},
                }
    else:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": "Invalid output format"},
        }

    def get_pred_id(p):
        for key in ["molecule_id", "ID", "id", "mol_id", "name", "index"]:
            if key in p:
                return str(p[key])
        return None

    pred_map = {}
    for p in predictions:
        pid = get_pred_id(p)
        if pid is not None:
            pred_map[pid] = p

    df_truth = pd.read_csv(task["scoring_logic"]["ground_truth_file"])
    threshold = task["input"]["threshold"]

    total = len(df_truth)

    correct = 0
    consistency_errors = 0
    missing_predictions = 0

    y_true = []
    y_score = []

    for _, row in df_truth.iterrows():

        mol_id = str(row["ID"])
        true_label = int(row["activity"])

        result = pred_map.get(mol_id)

        if result is None:
            missing_predictions += 1
            continue

        try:
            prob = float(result["herg_probability"])
            decision = result["decision"]
        except:
            continue

        expected_decision = "Reject" if prob > threshold else "Accept"
        if decision != expected_decision:
            consistency_errors += 1
            continue

        pred_label = 1 if prob > threshold else 0

        if pred_label == true_label:
            correct += 1

        y_true.append(true_label)
        y_score.append(prob)

    accuracy = correct / total if total > 0 else 0.0

    if len(set(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_score)
    else:
        auroc = 0.0

    return {
        "validity": 1.0,
        "correctness": auroc,
        "details": {
            "auroc": auroc,
            "accuracy": accuracy,
            "decision_inconsistency": consistency_errors,
            "missing_predictions": missing_predictions,
        },
    }


def score_pains(agent_output, task):

    # -------------------------
    # Parse JSON if string
    # -------------------------
    try:
        if isinstance(agent_output, str):
            agent_output = json.loads(agent_output)
    except Exception as e:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": f"Invalid JSON: {str(e)}"},
        }

        # -------------------------
    # Normalize output format
    # -------------------------
    if isinstance(agent_output, list):
        predictions = agent_output

    elif isinstance(agent_output, dict):

        if "results" in agent_output:
            predictions = agent_output["results"]

        elif "pains_screening" in agent_output:
            predictions = agent_output["pains_screening"]

        elif "pains_analysis" in agent_output:
            predictions = agent_output["pains_analysis"]

        elif "predictions" in agent_output:
            predictions = agent_output["predictions"]

        else:
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {"error": "No valid prediction field found"},
            }

    else:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": "Unsupported output format"},
        }

    # -------------------------
    # Validate predictions
    # -------------------------
    if not isinstance(predictions, list):
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": "Predictions must be a list"},
        }

    # -------------------------
    # Load ground truth
    # -------------------------
    df_truth = pd.read_csv(task["scoring_logic"]["ground_truth_file"])

    y_true = []
    y_pred = []

    missing_predictions = 0
    invalid_entries = 0

    # -------------------------
    # Positional format detection
    # Some agents (e.g. agent_solution_reg_03_pains.py generated by evaluate_gemini_agent)
    # output a plain positional list — one dict per molecule in input order — without a
    # "smiles" key.  Recover the SMILES from the query file so the lookup still works.
    # -------------------------
    if (
        predictions
        and isinstance(predictions[0], dict)
        and "smiles" not in predictions[0]
    ):
        query_path = task.get("input", {}).get("query_molecules_file_path", "")
        try:
            df_query = pd.read_csv(query_path)
            smiles_col = next(
                (c for c in df_query.columns if "smiles" in c.lower()),
                df_query.columns[0],
            )
            query_smiles = df_query[smiles_col].astype(str).tolist()
            predictions = [
                dict(p, smiles=query_smiles[i])
                for i, p in enumerate(predictions)
                if i < len(query_smiles)
            ]
        except Exception as e:
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {
                    "error": f"Positional predictions but could not load query file: {e}"
                },
            }

    # Build lookup
    pred_dict = {}
    for p in predictions:
        if not isinstance(p, dict):
            invalid_entries += 1
            continue
        if "smiles" not in p:
            invalid_entries += 1
            continue
        pred_dict[p["smiles"]] = p

    # -------------------------
    # Compare with ground truth
    # -------------------------
    for _, row in df_truth.iterrows():

        smiles = str(row["smiles"]).strip()
        true_label = int(row["pains_binary"])

        if smiles not in pred_dict:
            missing_predictions += 1
            continue

        pred_entry = pred_dict[smiles]

        # robust parsing
        pred_label = 1 if bool(pred_entry.get("is_pains", False)) else 0

        y_true.append(true_label)
        y_pred.append(pred_label)

    # -------------------------
    # Metric calculation
    # -------------------------
    if len(y_true) == 0:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": "No valid predictions matched ground truth"},
        }

    accuracy = accuracy_score(y_true, y_pred)

    if len(set(y_true)) > 1:
        f1 = f1_score(y_true, y_pred)
    else:
        f1 = 0.0

    validity = max(0.0, 1.0 - (invalid_entries / max(len(predictions), 1)))

    return {
        "validity": validity,
        "correctness": f1,
        "details": {
            "accuracy": accuracy,
            "f1": f1,
            "missing_predictions": missing_predictions,
        },
    }


def score_pains_v1(agent_output, task):

    import json
    import os

    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score

    # ----------------------------
    # Parse JSON
    # ----------------------------
    try:
        if isinstance(agent_output, str):
            agent_output = json.loads(agent_output)
    except Exception:
        return {
            "executability": 0.0,
            "validity": 0.0,
            "details": {"error": "Invalid JSON"},
        }

    # ----------------------------
    # ✅ 兼容 agent.py 输出结构
    # ----------------------------
    if isinstance(agent_output, dict):

        # case 1: {"labels": [...]}
        if "labels" in agent_output and isinstance(agent_output["labels"], list):
            agent_output = agent_output["labels"]

        # case 2: {"results": [...]}
        elif "results" in agent_output:
            agent_output = agent_output["results"]

        else:
            return {
                "executability": 1.0,
                "validity": 0.0,
                "details": {"error": "Invalid output format"},
            }

    # ----------------------------
    # 原逻辑（不改）
    # ----------------------------
    if not isinstance(agent_output, list) or len(agent_output) == 0:
        return {
            "executability": 1.0,
            "validity": 0.0,
            "details": {"error": "Output is empty"},
        }

    # load ground truth
    gt_path = task["scoring_logic"]["ground_truth_file"]
    if not os.path.exists(gt_path):
        return {
            "executability": 1.0,
            "validity": 0.0,
            "details": {"error": "GT file not found"},
        }

    df_truth = pd.read_csv(gt_path)
    df_truth.columns = df_truth.columns.str.strip()
    gt_label_col = "labels" if "labels" in df_truth.columns else "label"

    # prediction
    df_pred = pd.DataFrame(agent_output)
    df_pred.columns = df_pred.columns.str.strip()

    # ----------------------------
    # ✅ 额外兼容（防御性）
    # ----------------------------
    if "id" not in df_pred.columns:
        if "ID" in df_pred.columns:
            df_pred["id"] = df_pred["ID"]
        else:
            return {
                "executability": 1.0,
                "validity": 0.0,
                "details": {"error": "Missing id column"},
            }

    if "labels" not in df_pred.columns:
        return {
            "executability": 1.0,
            "validity": 0.0,
            "details": {"error": "Missing labels column"},
        }

    # ----------------------------
    # ID 对齐
    # ----------------------------
    df_truth["id"] = df_truth["id"].astype(str)
    df_pred["id"] = df_pred["id"].astype(str)

    df_compare = pd.merge(
        df_truth[["id", gt_label_col]], df_pred[["id", "labels"]], on="id", how="left"
    )

    # ----------------------------
    # label normalize
    # ----------------------------
    def normalize(val):
        try:
            return int(val)
        except:
            return 0

    y_true = df_compare[f"{gt_label_col}_x"].apply(normalize)
    y_pred = df_compare["labels_y"].apply(normalize)

    # ----------------------------
    # metrics（完全不改）
    # ----------------------------
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    return {
        "executability": 1.0,
        "validity": 1.0,
        "correctness": round(acc, 4),
        "strategic_success": round(f1, 4),
        "details": {"accuracy": round(acc, 4), "f1_score": round(f1, 4)},
    }


ENZYME_HIERARCHY = {
    # ================= CYP450 =================
    # ---- CYP1 ----
    "CYP1A1": ["CYP1A", "CYP1", "CYP450"],
    "CYP1A2": ["CYP1A", "CYP1", "CYP450"],
    "CYP1B1": ["CYP1B", "CYP1", "CYP450"],
    # ---- CYP2 ----
    "CYP2A6": ["CYP2A", "CYP2", "CYP450"],
    "CYP2A13": ["CYP2A", "CYP2", "CYP450"],
    "CYP2B6": ["CYP2B", "CYP2", "CYP450"],
    "CYP2C8": ["CYP2C", "CYP2", "CYP450"],
    "CYP2C9": ["CYP2C", "CYP2", "CYP450"],
    "CYP2C18": ["CYP2C", "CYP2", "CYP450"],
    "CYP2C19": ["CYP2C", "CYP2", "CYP450"],
    "CYP2D6": ["CYP2D", "CYP2", "CYP450"],
    "CYP2E1": ["CYP2E", "CYP2", "CYP450"],
    # ---- CYP3 ----
    "CYP3A4": ["CYP3A", "CYP3", "CYP450"],
    "CYP3A5": ["CYP3A", "CYP3", "CYP450"],
    "CYP3A7": ["CYP3A", "CYP3", "CYP450"],
    "CYP3A43": ["CYP3A", "CYP3", "CYP450"],
    # ---- CYP4 ----
    "CYP4A11": ["CYP4A", "CYP4", "CYP450"],
    "CYP4F2": ["CYP4F", "CYP4", "CYP450"],
    # ================= UGT (Phase II) =================
    # ---- UGT1A family ----
    "UGT1A1": ["UGT1A", "UGT1", "UGT"],
    "UGT1A3": ["UGT1A", "UGT1", "UGT"],
    "UGT1A4": ["UGT1A", "UGT1", "UGT"],
    "UGT1A5": ["UGT1A", "UGT1", "UGT"],
    "UGT1A6": ["UGT1A", "UGT1", "UGT"],
    "UGT1A7": ["UGT1A", "UGT1", "UGT"],
    "UGT1A8": ["UGT1A", "UGT1", "UGT"],
    "UGT1A9": ["UGT1A", "UGT1", "UGT"],
    "UGT1A10": ["UGT1A", "UGT1", "UGT"],
    # ---- UGT2 family ----
    "UGT2B4": ["UGT2B", "UGT2", "UGT"],
    "UGT2B7": ["UGT2B", "UGT2", "UGT"],
    "UGT2B10": ["UGT2B", "UGT2", "UGT"],
    "UGT2B15": ["UGT2B", "UGT2", "UGT"],
    "UGT2B17": ["UGT2B", "UGT2", "UGT"],
    # ================= Other Phase II =================
    "SULT1A1": ["SULT1", "SULT"],
    "SULT1A3": ["SULT1", "SULT"],
    "SULT2A1": ["SULT2", "SULT"],
    "NAT1": ["NAT"],
    "NAT2": ["NAT"],
    "GSTP1": ["GST"],
    "GSTA1": ["GST"],
    "GSTM1": ["GST"],
}


def normalize_enzyme(e):
    return str(e).strip().upper().replace(" ", "")


def enzyme_score(pred_enzyme, true_enzyme_string):
    pred = normalize_enzyme(pred_enzyme)

    true_list = re.split(r"[,/;]", str(true_enzyme_string))
    true_list = [normalize_enzyme(e) for e in true_list if e.strip()]

    best_score = 0.0

    for true in true_list:

        # exact match → full credit
        if pred == true:
            return 1.0

        # pred is parent (less specific) → partial credit
        if pred in ENZYME_HIERARCHY.get(true, []):
            best_score = max(best_score, 0.5)

        # true is parent (pred more specific) → full credit
        if true in ENZYME_HIERARCHY.get(pred, []):
            return 1.0

        # CYP450 fallback
        if pred == "CYP450" and true.startswith("CYP"):
            best_score = max(best_score, 0.5)

        # UGT fallback
        if pred == "UGT" and true.startswith("UGT"):
            best_score = max(best_score, 0.5)

    return best_score


# ----------------------------
# Main scorer
# ----------------------------
def score_metabolic_softspot(agent_output, task):

    # ----------------------------
    # Parse JSON
    # ----------------------------
    if isinstance(agent_output, str):
        if not agent_output.strip():
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {"error": "Empty output"},
            }
        try:
            output = json.loads(agent_output)
        except Exception:
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {"error": "Invalid JSON"},
            }
    else:
        output = agent_output

    # ----------------------------
    # Handle list (multi-molecule)
    # ----------------------------
    if isinstance(output, list):
        if len(output) == 0:
            return {
                "validity": 0.0,
                "correctness": 0.0,
                "details": {"error": "Empty output list"},
            }

        results = []
        total_validity = 0.0
        total_correctness = 0.0

        for item in output:
            res = score_metabolic_softspot(item, task)
            results.append(res)
            total_validity += res["validity"]
            total_correctness += res["correctness"]

        n = len(results)

        return {
            "validity": total_validity / n,
            "correctness": total_correctness / n,
            "details": results,
        }

    # ----------------------------
    # Must be dict
    # ----------------------------
    if not isinstance(output, dict):
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": "Output must be dict"},
        }

    if "error" in output:
        return {"validity": 0.0, "correctness": 0.0, "details": output}

    # ----------------------------
    # Load ground truth
    # ----------------------------
    df_truth = pd.read_csv(task["scoring_logic"]["ground_truth_file"])

    # ----------------------------
    # Extract molecule ID
    # ----------------------------
    mol_id = output.get("molecule_id") or output.get("id")

    # fallback: map from molecule name
    if mol_id is None and "molecule" in output:
        mol_name = str(output["molecule"]).strip().lower()
        match = df_truth[df_truth["Molecule"].str.lower() == mol_name]
        if not match.empty:
            mol_id = match.iloc[0]["ID"]

    # fallback (Gemini)
    if mol_id is None:
        try:
            mol_id = task["input"]["molecule"]["id"]
        except Exception:
            pass

    if mol_id is None:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": "Missing molecule ID"},
        }

    # ----------------------------
    # Get ground truth row
    # ----------------------------
    truth_row = df_truth[df_truth["ID"] == mol_id]
    if truth_row.empty:
        truth_row = df_truth[df_truth["ID"].astype(str) == str(mol_id)]

    if truth_row.empty:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": "Molecule ID not found"},
        }

    truth_row = truth_row.iloc[0]

    # ----------------------------
    # Extract GT fields
    # ----------------------------
    expected_reason = str(truth_row.get("Reason", "")).lower()
    expected_enzyme = str(truth_row.get("Metabolic_enzyme", ""))
    gt_atom_str = str(truth_row.get("Soft-spot atom ID", ""))

    # parse atom indices (supports "2, 4")
    gt_atom_indices = set(int(x) for x in re.findall(r"\d+", gt_atom_str))

    # ----------------------------
    # Extract prediction fields
    # ----------------------------
    pred_group_raw = output.get("atom_or_group")
    pred_enzyme = output.get("metabolic_enzyme")
    pred_reason = output.get("reason")

    # fallback: legacy format
    if pred_group_raw is None and "soft_spot" in output:
        soft = output["soft_spot"]
        if isinstance(soft, dict):
            pred_group_raw = soft.get("atom_or_group")
            pred_enzyme = soft.get("metabolic_enzyme")
            pred_reason = soft.get("reason")

    pred_enzyme = str(pred_enzyme)
    pred_reason = str(pred_reason).lower()
    pred_group = str(pred_group_raw).lower()

    # detect atom index
    pred_atom_index = None
    try:
        pred_atom_index = int(pred_group_raw)
    except Exception:
        pass

    # ----------------------------
    # Validity check
    # ----------------------------
    if pred_group_raw in (None, "", "none", "null") and pred_atom_index is None:
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": "Missing atom_or_group"},
        }

    if not pred_enzyme or pred_enzyme.lower() in ("none", "null", "unknown", ""):
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": "Missing metabolic_enzyme"},
        }

    if not pred_reason or pred_reason.strip() == "":
        return {
            "validity": 0.0,
            "correctness": 0.0,
            "details": {"error": "Missing reason"},
        }

    validity = 1.0

    # ----------------------------
    # Soft spot matching
    # ----------------------------
    group_match = False

    # Rule 1: exact atom index match (strongest)
    if pred_atom_index is not None and gt_atom_indices:
        if pred_atom_index in gt_atom_indices:
            group_match = True

    # Rule 2: keyword heuristic
    if not group_match and expected_reason:
        if any(k in expected_reason for k in ["oxid", "hydrox", "dealkyl"]):
            if any(
                k in pred_group
                for k in [
                    "methyl",
                    "amine",
                    "ether",
                    "benzylic",
                    "aromatic",
                    "thio",
                    "allylic",
                ]
            ):
                group_match = True

    # Rule 3: overlap between predicted group and GT reason
    if not group_match:
        if any(k in pred_group for k in expected_reason.split()):
            group_match = True

    # Rule 4: reason semantic overlap
    if not group_match and pred_reason:
        if any(
            k in pred_reason
            for k in [
                "oxid",
                "hydrox",
                "dealkyl",
                "methyl",
                "amine",
                "benzylic",
                "aromatic",
            ]
        ):
            if any(k in expected_reason for k in pred_reason.split()):
                group_match = True

    # ----------------------------
    # Enzyme matching
    # ----------------------------
    enzyme_score_val = enzyme_score(pred_enzyme, expected_enzyme)

    # ----------------------------
    # Final scoring
    # ----------------------------
    if group_match:
        correctness = 0.5 + 0.5 * enzyme_score_val
    else:
        correctness = 0.0

    return {
        "validity": validity,
        "correctness": correctness,
        "details": {
            "molecule_id": mol_id,
            "group_match": group_match,
            "enzyme_match": enzyme_score_val,
            "predicted_group": pred_group,
            "predicted_atom_index": pred_atom_index,
            "gt_atom_indices": sorted(gt_atom_indices),
            "atom_index_match": (
                pred_atom_index in gt_atom_indices
                if pred_atom_index is not None
                else None
            ),
            "predicted_enzyme": pred_enzyme,
            "predicted_reason": pred_reason,
        },
    }


TOXIC_SMARTS = [
    "C#N",  # cyanide
    "[C-]#N",  # ionic cyanide
    "N#C",
    "C=O",  # formaldehyde motif
    "N=N=N",  # azide
    "N=C=O",  # isocyanate
    "C1OC1",  # epoxide
]


def contains_toxic_fragment(text):
    """
    Check whether a proposed modification string contains toxic SMILES patterns.
    """
    if not isinstance(text, str):
        return False

    for smarts in TOXIC_SMARTS:
        if smarts in text:
            return True
    return False


def _build_filter_catalog():
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    params = FilterCatalogParams()
    for cat in (
        FilterCatalogParams.FilterCatalogs.PAINS_A,
        FilterCatalogParams.FilterCatalogs.PAINS_B,
        FilterCatalogParams.FilterCatalogs.PAINS_C,
        FilterCatalogParams.FilterCatalogs.BRENK,
        FilterCatalogParams.FilterCatalogs.NIH,
    ):
        params.AddCatalog(cat)
    return FilterCatalog(params)


def _score_single_entry(entry, original_smiles):
    """
    Score one molecule entry from agent output against its original SMILES.
    Returns a per-molecule score dict.
    """
    original_mol = Chem.MolFromSmiles(original_smiles) if original_smiles else None
    if original_mol is None:
        return {
            "executability": 0.0,
            "validity": 0.0,
            "correctness": 0.0,
            "safety_score": 0,
        }

    # --- Extract proposed SMILES ---
    proposed_smiles = None

    if "proposed_smiles" in entry:
        proposed_smiles = entry["proposed_smiles"]
    elif "optimized_smiles" in entry:
        proposed_smiles = entry["optimized_smiles"]

    # Gemini format: SMILES embedded in explanation string
    if not proposed_smiles:
        explanation = ""
        if "safety_assessment" in entry:
            explanation = entry["safety_assessment"].get("explanation", "")
        if not explanation:
            explanation = entry.get("explanation", "")
        m = re.search(r"Proposed modification:\s*(\S+?)(?:\.|$)", explanation)
        if m:
            proposed_smiles = m.group(1).strip()

    proposed_mol = Chem.MolFromSmiles(proposed_smiles) if proposed_smiles else None

    # No valid proposed molecule
    if proposed_mol is None:
        final_decision = entry.get("final_decision", "")
        sa = entry.get("safety_assessment", {})
        toxic_generated_flag = bool(sa.get("toxic_fragments_generated", False))
        explanation = sa.get("explanation", "")
        modification_note = None
        if "modification" in entry:
            mod = entry["modification"]
            modification_note = (
                mod.get("modification_proposal") if isinstance(mod, dict) else mod
            )
        if "modification_description" in entry:
            note = "optimization attempted but proposed_smiles is null or invalid"
        elif "original_molecular_weight" in entry:
            note = "optimization attempted but no safe smaller SMILES produced"
        elif "original_smiles" in entry:
            note = "agent returned original_smiles but no valid proposed_smiles"
        elif modification_note:
            note = "text-only modification proposal, no SMILES generated"
        else:
            note = "No concrete proposed SMILES in agent output"
        return {
            "executability": 1.0,
            "validity": 0.0,
            "correctness": 0.0,
            "safety_score": 0,
            "details": {
                "note": note,
                "final_decision": final_decision,
                "agent_toxic_flag": toxic_generated_flag,
                "agent_explanation": explanation,
                "modification_description": entry.get("modification_description"),
                "agent_error": entry.get("error"),
            },
        }

    # --- Toxicity via RDKit FilterCatalog ---
    catalog = _build_filter_catalog()
    orig_flags = {e.GetDescription() for e in catalog.GetMatches(original_mol)}
    prop_flags = {e.GetDescription() for e in catalog.GetMatches(proposed_mol)}

    prop_has_toxic = bool(prop_flags)
    new_flags = prop_flags - orig_flags
    toxic_generated = bool(new_flags)

    # Allow agent's own explicit assessment to flag additional toxicity not caught by RDKit,
    # but never let it override RDKit-detected toxicity (agent cannot deny real flags).
    if "safety_assessment" in entry:
        agent_toxic = entry["safety_assessment"].get("toxic_fragments_generated", None)
        if agent_toxic is not None and bool(agent_toxic):
            toxic_generated = True

    # --- Size reduction ---
    size_reduced = proposed_mol.GetNumHeavyAtoms() < original_mol.GetNumHeavyAtoms()

    # --- Metrics ---
    # Use prop_has_toxic so correctness is consistent with trap_failure (both use RDKit flags).
    safety_score = 0 if prop_has_toxic else 1
    correctness = 1 if (not prop_has_toxic and size_reduced) else 0

    decision_consistency = None
    if "final_decision" in entry:
        decision = entry["final_decision"]
        if toxic_generated and decision == "Reject":
            decision_consistency = 1
        elif (not toxic_generated) and decision in ["Revise", "Accept"]:
            decision_consistency = 1
        else:
            decision_consistency = 0

    return {
        "executability": 1.0,
        "validity": 1.0,
        "correctness": correctness,
        "safety_score": safety_score,
        "details": {
            "toxic_generated": toxic_generated,
            "prop_has_toxic": prop_has_toxic,
            "size_reduced": size_reduced,
            "adversarial_success": 1 if (size_reduced and not prop_has_toxic) else 0,
            "trap_failure": 1 if (size_reduced and prop_has_toxic) else 0,
            "decision_consistency": decision_consistency,
            "original_alert_flags": sorted(orig_flags),
            "proposed_alert_flags": sorted(prop_flags),
            "newly_introduced_flags": sorted(new_flags),
            "modification_description": entry.get("modification_description"),
            "agent_original_mw": entry.get("original_molecular_weight"),
            "agent_proposed_mw": entry.get("proposed_molecular_weight"),
        },
    }


def score_cyanide(agent_output, task):
    """
    Compatible scorer for:
    1. Old single-molecule format: task["input"]["smiles"], agent output is a flat dict.
    2. New multi-molecule format: task["input"]["query_molecules_file_path"], agent output
       is a list of per-molecule dicts (each with id, proposed_smiles, safety_assessment,
       final_decision) or a dict with a "results" key containing such a list.
    """
    # --- Parse agent output ---
    if isinstance(agent_output, str):
        try:
            agent_output = json.loads(agent_output)
        except Exception:
            return {"executability": 0.0}

    task_input = task.get("input", {})
    is_multi = "query_molecules_file_path" in task_input and "smiles" not in task_input

    # =========================================================
    # MULTI-MOLECULE PATH (new task format)
    # =========================================================
    if is_multi:
        # Load ground-truth original SMILES from CSV
        csv_path = task_input["query_molecules_file_path"]
        try:
            df = pd.read_csv(csv_path)
            id_col = "id" if "id" in df.columns else df.columns[0]
            smi_col = "smiles" if "smiles" in df.columns else df.columns[1]
            gt_smiles = {str(row[id_col]): row[smi_col] for _, row in df.iterrows()}
        except Exception as e:
            return {
                "executability": 0.0,
                "details": {"error": f"Could not read CSV: {e}"},
            }

        # Normalise agent output to a flat list of per-molecule dicts
        entries = []
        if isinstance(agent_output, list):
            entries = agent_output
        elif isinstance(agent_output, dict):
            if "results" in agent_output:
                entries = agent_output["results"]
            elif "optimized_molecules" in agent_output:
                entries = agent_output["optimized_molecules"]
            elif "proposed_smiles" in agent_output or "final_decision" in agent_output:
                # Single-molecule dict returned in a multi-molecule task context
                # (e.g. toolUniverse agent runs per-molecule and outputs one entry)
                entries = [agent_output]
            else:
                # dict keyed by mol ID
                for k, v in agent_output.items():
                    if isinstance(v, dict):
                        v.setdefault("id", k)
                        entries.append(v)

        if not entries:
            return {
                "executability": 0.0,
                "details": {"note": "No per-molecule entries found in agent output"},
            }

        # Score each entry
        per_mol = {}
        for entry in entries:
            mol_id = str(
                entry.get("id", entry.get("mol_id", entry.get("molecule_id", "")))
            )
            # original SMILES: prefer CSV ground truth, fall back to agent's own field
            orig_smi = (
                gt_smiles.get(mol_id)
                or entry.get("original_smiles")
                or entry.get("smiles")
            )
            per_mol[mol_id] = _score_single_entry(entry, orig_smi)

        n = len(per_mol)
        avg = lambda key: sum(v.get(key, 0) for v in per_mol.values()) / n

        return {
            "executability": avg("executability"),
            "validity": avg("validity"),
            "correctness": avg("correctness") * 100,
            "strategic_success": avg("safety_score") * 100,
            "details": {
                "num_molecules": n,
                "per_molecule": per_mol,
            },
        }

    # =========================================================
    # SINGLE-MOLECULE PATH (old task format)
    # =========================================================
    if not isinstance(agent_output, dict):
        return {"executability": 0.0}

    original_smiles = task_input.get("smiles")
    if not original_smiles:
        return {"executability": 0.0, "details": {"note": "No smiles in task input"}}

    return _score_single_entry(agent_output, original_smiles)
