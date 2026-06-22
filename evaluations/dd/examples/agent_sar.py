"""Example agent for a C2 (Chemical Data Analysis) interactive task: SAR scatter (ana_01_sar).

Demonstrates the notebook contract: the runner injects the task input as the
``_TASK_INPUT`` variable and provides ``plt``; the agent must leave a finished
matplotlib figure in a variable named ``fig``. Used to smoke-test the unified
notebook path (notebook_runner -> oracle_vis.check_plot_attributes).
"""

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors

molecules = _TASK_INPUT["molecules"]  # noqa: F821  (injected by the runner)

mws, logps, pic50s = [], [], []
for m in molecules:
    mol = Chem.MolFromSmiles(m["smiles"])
    mws.append(Descriptors.MolWt(mol))
    logps.append(Crippen.MolLogP(mol))
    pic50s.append(m["pIC50"])

fig, ax = plt.subplots()
scatter = ax.scatter(mws, logps, c=pic50s, cmap="viridis")
ax.set_xlabel("Molecular Weight (Da)")
ax.set_ylabel("LogP")
ax.set_title("SAR Landscape")
colorbar = fig.colorbar(scatter, ax=ax)
colorbar.set_label("pIC50")
