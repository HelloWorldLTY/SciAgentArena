# LIDDiA

**Category:** Chemistry  
**Repository:** https://github.com/ninglab/LIDDiA

## Overview

LIDDiA is an LLM-driven iterative drug design agent that generates molecules targeting specific protein structures. It currently samples from TDC ZINC; Pocket2Mol integration is under development.

## Requirements

- Conda (environment defined in `environment.yml`)
- Anthropic API key
- Protein structures in the `dataset/pdb/` directory (target must be present)

## Installation

### 1. Create Conda Environment

```bash
git clone https://github.com/ninglab/LIDDiA.git
cd LIDDiA
conda env create -f environment.yml
conda activate <env_name_from_yml>
```

### 2. Configure API Key

```bash
echo "your-anthropic-api-key" > my-anthropic-key.txt
```

## Usage

```bash
python run.py \
  --target EGFR \
  --max_iter 10 \
  --model "claude-3-5-sonnet-20241022"
```

### Parameters

| Parameter | Description |
|---|---|
| `--target` | Protein target name (must exist in `dataset/pdb/`) |
| `--max_iter` | Maximum molecule generation iterations |
| `--model` | Claude model ID (see [Anthropic model list](https://docs.anthropic.com/en/docs/about-claude/models/overview)) |

## Notes

- The repository is still under active development.
- Currently uses random sampling from TDC ZINC instead of structure-based Pocket2Mol.
