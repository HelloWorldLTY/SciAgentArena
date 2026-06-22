"""Example agent for a C1 (Chemical Data Preprocessing) batch task: molecular weight.

Demonstrates the batch contract: read the input payload from the
``AGENT4S_INPUT_JSON`` env var, do the chemistry, print a JSON list of
``{"SMILES": ..., "<PROP>": ...}`` records to stdout. Used to smoke-test the
unified batch path (batch_runner -> oracle_chem.score_numeric_property).
"""

import json
import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

payload = json.loads(os.environ["AGENT4S_INPUT_JSON"])
df = pd.read_csv(payload["file_path"])

results = []
for smiles in df["SMILES"]:
    mol = Chem.MolFromSmiles(smiles)
    mw = round(Descriptors.MolWt(mol), 3) if mol is not None else None
    results.append({"SMILES": smiles, "MW": mw})

print(json.dumps(results))
