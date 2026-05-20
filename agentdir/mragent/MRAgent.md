# MRAgent

**Category:** Genetics  
**Repository:** https://github.com/xuwei1997/MRAgent

## Overview

MRAgent is an AI agent for Mendelian Randomization (MR) analysis. It integrates with the OpenGWAS API to retrieve GWAS summary statistics and orchestrates R-based MR pipelines (TwoSampleMR, MRlap, etc.) from a Python interface.

## Requirements

- Python > 3.9
- R > 4.3.4
- OpenGWAS API token
- OpenAI API key or local Ollama model

## Installation

### 1. Install R Dependencies

Open an R session and install the required packages:

```r
install.packages(c("jsonlite"))
# Install Bioconductor / GitHub packages:
install.packages("remotes")
remotes::install_github("MRCIEU/TwoSampleMR")
remotes::install_github("MRCIEU/ieugwasr")
install.packages("vcfR")
remotes::install_github("n-mounier/MRlap")
```

> Automatic installation via the agent may fail; manual installation is recommended.

### 2. Install MRAgent (Python)

```bash
pip install mragent
```

### 3. (Optional) Local LLM via Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
pip install ollama
```

## Usage

### Knowledge Discovery Mode (disease-centered)

```python
from mragent import MRAgent

agent = MRAgent(
    outcome='back pain',
    LLM_model='gpt-4o',
    AI_key='your-openai-key',
    gwas_token='your-opengwas-token'
)
agent.run()
```

### Causal Validation Mode (specific exposure–outcome pair)

```python
from mragent import MRAgentOE

agent = MRAgentOE(
    exposure='osteoarthritis',
    outcome='back pain',
    AI_key='your-openai-key',
    gwas_token='your-opengwas-token'
)
agent.run()
```

Results are saved in `output/Disease_Model/` as data tables and analysis reports.

## Obtaining Credentials

| Credential | Source |
|---|---|
| OpenAI API key | https://platform.openai.com/docs/overview |
| OpenGWAS token | https://api.opengwas.io/ |
