# CellForge

**Category:** Biomedicine  
**Repository:** https://github.com/gersteinlab/CellForge

## Overview

CellForge is an AI agent for automated single-cell biology analysis, supporting multi-phase workflows including task analysis, method design, and code generation.

## System Requirements

- Python 3.8+ (3.9 recommended)
- RAM: 8 GB minimum, 16 GB+ preferred
- Storage: 10 GB available
- Docker (for code generation capabilities)
- At least one LLM API key (OpenAI, Anthropic, or similar)

## Installation

### Automated Setup (Recommended)

```bash
git clone https://github.com/gersteinlab/scAgents.git
cd CellForge

conda create -n cellforge python=3.9
conda activate cellforge

python install.py

cp env.example .env
# Edit .env with your API credentials

python start.py  # verify installation
```

### Manual Installation

```bash
pip install -r requirements.txt
pip install -e .
cp env.example .env
```

## Configuration

Edit `.env` with at least one LLM provider key:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GITHUB_TOKEN` | GitHub access token (optional) |
| `SERPAPI_KEY` | Web search key (optional) |
| `PUBMED_API_KEY` | PubMed API key (optional) |

Place datasets (`.h5ad` / AnnData format) in `cellforge/data/datasets/`. Recommended source: [scPerturb](https://projects.sanderlab.org/scperturb/).

## Usage

```bash
# Full workflow
python main.py

# Individual phases
python main.py --phase task_analysis
python main.py --phase method_design
python main.py --phase code_generation

# Initialize project
python main.py --init
```
