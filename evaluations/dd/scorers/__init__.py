"""scorers — Oracle scoring functions for Agent4Science tasks.

Every public scorer follows the same protocol:

    def scorer_fn(agent_output, task: dict) -> dict:
        return {
            "validity":   float,   # 0.0–1.0
            "correctness": float,  # 0.0–1.0
            "details":     dict,   # per-item breakdown
        }

Scorers are referenced by dotted name in task JSON files
(e.g. ``"oracle_chem.molecular_weight_score"``), and loaded
dynamically by the runner at evaluation time.
"""
