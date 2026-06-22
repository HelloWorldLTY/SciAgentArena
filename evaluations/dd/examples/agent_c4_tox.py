"""Example agent for a C4 (Chemical Safety Assessment) batch task: reg_01_toxicophore.

Demonstrates the batch contract for a safety-screening task: read the query
molecules, screen each for structural alerts via RDKit SMARTS, and print the
strict per-molecule JSON schema the task requires. Scored by
oracle_regulator.score_toxicophore. (A minimal alert set — a real submission
would use a fuller toxicophore library.)
"""

import json
import os

import pandas as pd
from rdkit import Chem

# (standardized name, SMARTS, default risk level)
ALERTS = [
    ("nitroaromatic", "[a][N+](=O)[O-]", "High"),
    ("aniline", "[NX3;H2,H1][c]", "Medium"),
    ("michael_acceptor", "[CX3]=[CX3][CX3]=[OX1]", "High"),
    ("aldehyde", "[CX3H1](=O)[#6]", "Medium"),
    ("epoxide", "[OX2r3]1[#6r3][#6r3]1", "High"),
    ("alkyl_halide", "[CX4][Cl,Br,I]", "Low"),
    ("quinone", "O=C1C=CC(=O)C=C1", "High"),
]
_COMPILED = [(name, Chem.MolFromSmarts(s), s, lvl) for name, s, lvl in ALERTS]

payload = json.loads(os.environ["AGENT4S_INPUT_JSON"])
df = pd.read_csv(payload["query_molecules_file_path"])

results = []
for _, row in df.iterrows():
    mol_id, smiles = str(row["id"]), str(row["smiles"])
    mol = Chem.MolFromSmiles(smiles)
    alerts = []
    if mol is not None:
        for name, patt, smarts, level in _COMPILED:
            if patt is not None and mol.HasSubstructMatch(patt):
                idxs = sorted(
                    {i for match in mol.GetSubstructMatches(patt) for i in match}
                )
                alerts.append(
                    {
                        "alert_type": name,
                        "smarts": smarts,
                        "match_atom_indices": idxs,
                        "mechanism": f"{name.replace('_', ' ')} reactive/structural alert",
                        "risk_level": level,
                    }
                )
    results.append({"molecule_id": mol_id, "smiles": smiles, "alerts": alerts})

print(json.dumps(results))
