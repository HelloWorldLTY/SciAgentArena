# CACTUS

**Category:** Chemistry  
**Repository:** https://github.com/pnnl/cactus

## Overview

CACTUS is a chemistry agent with ten cheminformatics tools for molecular analysis, including property calculations (molecular weight, LogP, TPSA) and drug-likeness filters (Brenk, PAINS). All tools currently require SMILES notation as input.

## Requirements

- Python 3.10–3.12
- CUDA 12.1 (default PyTorch build); CPU-only supported

## Installation

### From PyPI / GitHub (standard)

```bash
pip install git+https://github.com/pnnl/cactus.git
```

### Editable / Development

```bash
git clone https://github.com/pnnl/cactus.git
cd cactus
python -m pip install -e .
```

### Using Rye

```bash
git clone https://github.com/pnnl/cactus.git
cd cactus
rye sync
```

### Non-CUDA Systems

Clone the repository, edit `pyproject.toml` to select a CPU-only PyTorch index, then install:

```bash
python -m pip install -e .
```

## Usage

```python
from cactus.agent import Cactus

model = Cactus(model_name="google/gemma7b", model_type="vllm")
model.run("What is the molecular weight of the smiles: OCC1OC(O)C(C(C1O)O)O")
```

## Available Tools

| Tool | Description |
|---|---|
| Molecular weight | Compute MW from SMILES |
| LogP | Lipophilicity (Wildman-Crippen) |
| TPSA | Topological polar surface area |
| Brenk Filter | Flag problematic substructures |
| PAINS Filter | Pan-assay interference detection |
| … | (10 tools total) |
