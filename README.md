# SciAgentArena

A benchmark platform for evaluating AI agents across diverse scientific and biomedical domains. This repository provides agent implementations, standardized evaluation benchmarks, and a web-based submission interface.

## Overview

SciAgentArena enables reproducible evaluation of AI agents on real-world scientific tasks spanning single-cell genomics, spatial transcriptomics, drug discovery, electronic health records, clinical genetics, and more. Each agent is evaluated against structured benchmark tasks with automated scoring.

## Repository Structure

```
SciAgentArena/
├── agentdir/          # Agent implementations and setup guides (16 agents)
├── evaluations/       # Benchmark datasets and evaluation notebooks
│   ├── sc/            # Single-cell RNA-seq
│   ├── sp/            # Spatial transcriptomics
│   ├── ehr/           # Electronic health records
│   ├── genetics/      # Statistical genetics
│   └── cross_domain/  # eQTL and multi-omics
└── front_web/         # Web-based evaluation platform (Node.js + Python)
```

## Benchmark Domains

The platform covers **7 benchmark families** with 50+ evaluation tasks:

| Domain | Tasks | Key Methods |
|--------|-------|-------------|
| Single-Cell RNA-seq | 11 | QC, filtering, doublet detection, normalization, HVG, batch correction, clustering, DE, trajectory, perturbation |
| Spatial Transcriptomics | 11 | Spatial neighbors, SVG detection, neighborhood enrichment |
| Electronic Health Records | 5 | Code normalization, event extraction, outcome prediction, treatment recommendation |
| Cross-Domain | 2 | eQTL mapping, multi-omics association |
| Drug Discovery | 5 | ADMET, binding affinity, lead optimization, drug-target interaction, synergy |
| Statistical Genetics | 11 | Mendelian randomization, GWAS QC, polygenic risk scores |

## Agents

### Biomedicine & Omics

| Agent | Category | Description |
|-------|----------|-------------|
| [ToolUniverse](agentdir/ToolUniverse/) | Biomedicine | ChatGPT + ToolUniverse for biomedical pipeline generation. Outputs executable `.py` and full run logs. |
| [AutoBA](agentdir/autoba/AutoBA.md) | Biomedicine | Fully automated multi-omic bioinformatics analysis; supports OpenAI, local LLMs via Ollama, and Docker. |
| [CellForge](agentdir/cellforge/CellForge.md) | Biomedicine | Automated single-cell biology analysis with multi-phase workflows (task analysis, method design, code generation). |
| [STELLA](agentdir/stella/STELLA.md) | Biomedicine | Biomedical research agent with web interface, skill retrieval, and dynamic tool creation. Also available at [stella-agent.com](https://stella-agent.com/). |
| [Biomni](agentdir/Biomni/) | Biomedicine | Biology + omics specialized agent environment. |

### Chemistry & Drug Discovery

| Agent | Category | Description |
|-------|----------|-------------|
| [DELTA](agentdir/delta/DELTA.md) | Chemistry | Multi-agent drug design pipeline coordinating AutoDock (molecular docking), PLIP (protein-ligand interaction), and LLM-driven design cycles. |
| [ChemCrow](agentdir/chemcrow/ChemCrow.md) | Chemistry | LLM-powered agent with 18 expert-designed tools covering synthesis planning, safety analysis, and property prediction. |
| [CACTUS](agentdir/cactus/CACTUS.md) | Chemistry | Cheminformatics agent with 10 tools for molecular property calculations (MW, LogP, TPSA) and drug-likeness filters (Brenk, PAINS). |
| [ChemToolAgent](agentdir/chemtoolagent/ChemToolAgent.md) | Chemistry | OSU agent combining GPT-4o and Claude-3.5-Sonnet with property prediction checkpoints. |
| [LIDDiA](agentdir/liddia/LIDDiA.md) | Chemistry | LLM-driven iterative drug design agent targeting specific protein structures. |
| [DrugAgent](agentdir/drugagent/DrugAgent.md) | Chemistry | Multi-agent drug discovery system (Planner + Instructor) built on MLAgentBench; specializes in ADMET prediction. |

### Clinical & Genetics

| Agent | Category | Description |
|-------|----------|-------------|
| [Medea](agentdir/medea/Medea.md) | Clinical | Harvard MIMS clinical research agent (AgentLite framework) supporting research planning, analysis, and literature synthesis. Uses MedeaDB on HuggingFace. |
| [TxAgent](agentdir/txagent/TxAgent.md) | Clinical | Harvard MIMS treatment recommendation agent using retrieval-augmented ToolUniverse over drug databases and clinical guidelines. |
| [MRAgent](agentdir/mragent/MRAgent.md) | Genetics | Mendelian Randomization agent integrating OpenGWAS API with R-based MR pipelines (TwoSampleMR, MRlap). |

### General Purpose

| Agent | Category | Description |
|-------|----------|-------------|
| [ClaudeCode](agentdir/ClaudeCode/) | General | Claude Code CLI integration for code generation tasks. |
| [Codex](agentdir/Codex/) | General | General-purpose code generation agent. |

## Web Evaluation Platform

The `front_web/` directory contains a production-oriented evaluation platform:

- **Submission modes**: Submit Python pipeline code or a pre-computed `.h5ad` file
- **Output**: Per-task scores, aggregate metrics, execution logs, and overall average score
- **Upload limit**: Up to 2 GB
- **Execution timeout**: 15 minutes per submission

### Starting the Platform

```bash
cd front_web
npm start
```

Then open:
- Home: `http://localhost:3000`
- Single-cell benchmark: `http://localhost:3000/benchmarks/single-cell`
- Spatial benchmark: `http://localhost:3000/benchmarks/spatial`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/config` | GET | Platform config, benchmark catalog, dataset presets |
| `/api/submissions` | POST | Create an evaluation job |
| `/api/submissions/:id` | GET | Fetch job status and results |

### Runtime Requirements

**Node.js** (web layer) + **Python 3.10+** (evaluation):

```bash
pip install anndata scanpy numpy pandas scikit-learn
# Optional (for advanced metrics):
pip install scib scib_metrics squidpy scipy
```

### Submission Contract

**Code mode** — your Python script is executed with these variables preloaded:
- `path` / `dataset_path` / `DATASET_PATH`: path to the selected `.h5ad` dataset
- `np`, `pd`, `sc`: numpy, pandas, scanpy

The script must assign its final output to an `adata` variable of type `AnnData`.

**h5ad mode** — upload a pre-generated `.h5ad` file containing the expected fields and embeddings.

## ToolUniverse Agent Quick Start

The ToolUniverse agent (`agentdir/ToolUniverse/`) is the reference agent template for the platform.

### Install

```bash
cd agentdir/ToolUniverse
pip install -r requirements.txt
cp .env.example .env   # fill in OpenAI or Azure OpenAI credentials
```

## Adding a New Agent

1. Create a directory under `agentdir/<agent-name>/`.
2. Add a markdown file (e.g., `README.md` or `<AgentName>.md`) describing:
   - Category, repository link, and overview
   - System requirements and installation steps
   - How to run the agent on a benchmark prompt
   - Expected input/output format
3. Ensure outputs are compatible with the platform's submission contract (an `AnnData` assigned to `adata`, or a `.h5ad` file).

## Adding a New Benchmark

1. Add a new benchmark entry in `front_web/judge.config.json`.
2. Add dataset presets pointing to the benchmark id.
3. Add evaluator logic in `front_web/templates/pipeline_evaluator.py`.
4. Add page copy, starter code, and override fields in `front_web/public/benchmark.js`.
5. Visit `/benchmarks/<your-benchmark-id>`.

## Acknowledgements:

- ToolUniverse: <https://zitniklab.hms.harvard.edu/ToolUniverse/>
- AutoBA: <https://github.com/JoshuaChou2018/AutoBA>
- CellForge: <https://github.com/gersteinlab/CellForge>
- STELLA: <https://github.com/zaixizhang/STELLA>
- ChemCrow: <https://github.com/ur-whitelab/chemcrow-public>
- CACTUS: <https://github.com/pnnl/cactus>
- ChemToolAgent: <https://github.com/OSU-NLP-Group/ChemToolAgent>
- DELTA: <https://github.com/deltawave-tech/delta>
- LIDDiA: <https://github.com/ninglab/LIDDiA>
- DrugAgent: <https://github.com/FermiQ/drugagent>
- Medea: <https://github.com/mims-harvard/MEDEA>
- TxAgent: <https://github.com/mims-harvard/TxAgent>
- MRAgent: <https://github.com/xuwei1997/MRAgent>
