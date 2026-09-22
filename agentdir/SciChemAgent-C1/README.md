# SciChemAgent-C1

**Category:** Chemistry / Drug Discovery

**Repository:** https://github.com/skeleton2024/SciChemAgent-C1

## Overview

SciChemAgent-C1 is a deterministic, offline RDKit baseline for all 18
SciAgentArena Chemical Data Preprocessing (C1) tasks. It implements molecular
standardization and descriptors, exact mass, strict indole matching, Morgan
similarity ranking, XYZ-to-SMILES reconstruction, disease-target resolution,
and physiological formal-charge assignment.

The agent reads the public batch payload from `AGENT4S_INPUT_JSON`, prints the
task-specific JSON schema to stdout, and makes no network or hosted-LLM calls
during evaluation. The repository includes pinned dependencies, an index-driven
18-task runner, aggregate results, and an independent result verifier.

## Requirements

- Python 3.12
- RDKit 2026.3.5
- SciAgentArena commit `c413f660304bf5def1c54a23619267e3ee2ef6ad`

## Installation and evaluation

```powershell
git clone https://github.com/skeleton2024/SciChemAgent-C1.git
cd SciChemAgent-C1
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-c1.txt
```

Follow the repository README to obtain the pinned SciAgentArena checkout and
apply its documented exact-MW scorer compatibility patch, then run:

```powershell
.\.venv\Scripts\python.exe scripts\run_c1.py
.\.venv\Scripts\python.exe scripts\verify_results.py
```

## Reproduced C1 result

| Metric | Result |
|---|---:|
| Tasks accounted for | 18/18 |
| Mean executability | 1.0000 |
| Mean validity | 1.0000 |
| Mean correctness | 0.9244444444 |
| Full-correctness tasks | 15/18 |

This is a community-submitted result, not an official leaderboard result until
the maintainers reproduce and accept it. The repository also documents one
scientific-consistency discrepancy in the pinned acetic-acid formal-charge
ground truth rather than hard-coding the conflicting value.
