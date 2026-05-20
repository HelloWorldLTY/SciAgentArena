# DrugAgent (Sizhe)

**Category:** Chemistry  
**Repository:** https://github.com/FermiQ/drugagent

## Overview

DrugAgent is a multi-agent system for drug discovery tasks (ADMET prediction, etc.) composed of a Planner and an Instructor agent. It is built on the MLAgentBench framework.

## Requirements

- Python 3.10
- Conda
- OpenAI API key (GPT-4o-mini or GPT-4o recommended)

## Installation

```bash
git clone https://github.com/FermiQ/drugagent.git
cd drugagent

conda create --name drugagent python=3.10
conda activate drugagent

pip install -r requirements.txt
```

## Usage

### Run an Experiment

```bash
python -u -m drugagent.runner \
  --task admet \
  --device 0 \
  --log-dir first_test \
  --work-dir workspace \
  --llm-name openai/gpt-4o-mini \
  --edit-script-llm-name openai/gpt-4o-mini \
  --fast-llm-name openai/gpt-4o-mini \
  > log 2>&1
```

### Evaluate Results

```bash
python -m MLAgentBench.eval \
  --log-folder <log_folder> \
  --task <task_name> \
  --output-file <output_name>
```

## Output Structure

| Directory | Contents |
|---|---|
| `agent_log/` | Planner & Instructor logs, saved agent states |
| `env_log/` | Tool logs, workspace snapshots, interaction traces |

Results from `eval` are written as JSON files.
