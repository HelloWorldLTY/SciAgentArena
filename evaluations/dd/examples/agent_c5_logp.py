"""Example agent for a C5 (Chemical Claim Validation) task: val_C1_01.

Demonstrates the notebook-variables contract for epistemic-validity tasks: the
runner injects ``_TASK_INPUT`` and the agent must leave named variables in the
namespace (both the task-specific ones and the shared diagnostic enums). The
variable inspector captures them and ``oracle_validity.score_validity_task``
grades them against the task's assertion checks.

The scientific point of val_C1_01: several rows are *multi-component* SMILES
(salts/mixtures, containing '.'), which do not map onto a single reportable
structure for whole-molecule MolLogP. A good agent detects this, sets those
records aside, and reports an inconclusive (data-quality) stance.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen

df = pd.read_csv(_TASK_INPUT["file_path"])  # noqa: F821  (injected by the runner)
n_total = len(df)

reported_values: dict[str, float] = {}
flagged: list[str] = []
for _, row in df.iterrows():
    cid, smiles = str(row["Compound_ID"]), str(row["SMILES"])
    if "." in smiles:  # multi-component: not a single structure
        flagged.append(cid)
        continue
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        flagged.append(cid)
        continue
    reported_values[cid] = round(Crippen.MolLogP(mol), 4)

n_flagged_records = len(flagged)
flagged_fraction = round(n_flagged_records / n_total, 4) if n_total else 0.0
summary_statistic = (
    round(sum(reported_values.values()) / len(reported_values), 4)
    if reported_values
    else 0.0
)
primary_result_id = (
    max(reported_values, key=reported_values.get) if reported_values else ""
)
is_output_ready = False

# Shared diagnostic enums (allowed values: tasks_interactive/data/c5_allowed_values.json)
conclusion_status = "inconclusive_data_quality"
failure_mode = "multi_component_smiles"
recommended_action = "reject_invalid_inputs"
diagnostic_evidence = (
    f"{n_flagged_records}/{n_total} records are multi-component SMILES "
    f"(contain '.'), which do not map onto a single reportable structure "
    f"for whole-molecule MolLogP ranking."
)
