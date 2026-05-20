# ChemCrow

**Category:** Chemistry  
**Repository:** https://github.com/ur-whitelab/chemcrow-public

## Overview

ChemCrow is an LLM-powered chemistry agent that integrates 18 expert-designed tools covering synthesis planning, safety analysis, property prediction, and more.

## Installation

```bash
pip install chemcrow
```

## Configuration

```bash
export OPENAI_API_KEY="your-openai-api-key"

# Optional: web search
export SERP_API_KEY="your-serpapi-api-key"
```

## Usage

```python
from chemcrow.agents import ChemCrow

chem_model = ChemCrow(model="gpt-4-0613", temp=0.1, streaming=False)
chem_model.run("What is the molecular weight of tylenol?")
```

## Optional: Self-Hosted Reaction Tools

For local reaction prediction and retrosynthetic planning (requires GPU):

```bash
docker run --gpus all -d -p 8051:5000 doncamilom/rxnpred:latest
docker run --gpus all -d -p 8052:5000 doncamilom/retrosynthesis:latest
```

Then enable local tools:

```python
chem_model = ChemCrow(model="gpt-4-0613", temp=0.1, streaming=False, local_rxn=True)
```
