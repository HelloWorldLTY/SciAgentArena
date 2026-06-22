"""Smoke QED optimizer (NOT a real search): scores the lead plus a small fixed
panel of drug-like candidates under the budgeted QED oracle, to demonstrate the
C3 oracle-stream design path end to end. Hyperparameters: none (fixed panel of
~6 candidates, single pass). A real submission would run a genetic algorithm or
fragment search here.
"""

import json
import os
import sys

from scorers.oracle_budget_qed import BudgetedQEDOracle, OracleBudgetExceededError

payload = json.loads(os.environ["AGENT4S_INPUT_JSON"])
oracle = BudgetedQEDOracle(
    payload["lead_smiles"], budget=payload.get("oracle_budget", 100)
)

candidates = [
    payload["lead_smiles"],
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ibuprofen
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
    "O=C(O)c1ccccc1OC(C)=O",  # aspirin
    "Cc1ccc(cc1)S(=O)(=O)N",  # toluenesulfonamide
]

for smiles in candidates:
    if oracle.calls_remaining <= 0:
        break
    try:
        oracle.score(smiles)
    except OracleBudgetExceededError:
        break

print(f"scored up to {len(candidates)} candidates", file=sys.stderr)
