# Setting Up LIDDiA For `evaluate_official_liddia_agent4s.py`

## What This Script Actually Uses

`evaluate_official_liddia_agent4s.py` does all of the following:

- loads one Agent4Science task JSON
- imports the official `Claude` agent class from the LIDDiA repo
- asks that agent to generate one Python solver
- saves the generated code and logs
- runs the solver through `runners.batch_runner`

Important: this is not the full official LIDDiA runtime. It is an Agent4Science bridge around the official LIDDiA code.

## Which LIDDiA Repo It Expects

The script expects the repo root at:

- [`external/LIDDiA/`](external/LIDDiA)

It imports:

- `liddia.agent.Claude`

from that checkout directly.

## Prerequisites

- conda
- Python 3.11
- the official LIDDiA repo cloned under `external/LIDDiA`
- an Anthropic API key

LIDDiA's own README says its conda dependencies are described in its environment file, but for the Agent4Science bridge we only need the subset required for code generation plus RDKit-based evaluation.

## Recommended Setup

This repo already includes a helper script:

- [setup_liddia_env.sh](setup_liddia_env.sh)

Run:

```bash
cd <repo-root>
bash setup_liddia_env.sh liddia-a4s 3.11
```

That script will:

- create a conda env
- install `anthropic`, `fire`, `tqdm`, `pandas`, and `numpy<2`
- install RDKit from conda-forge
- run a smoke import check

## Manual Setup

```bash
conda create -n liddia-a4s python=3.11 -y
conda activate liddia-a4s
python -m pip install --upgrade pip setuptools wheel
python -m pip install anthropic fire tqdm pandas "numpy<2"
conda install -n liddia-a4s -c conda-forge rdkit -y
```

No editable install is required here. The wrapper imports the local repo by path.

## Configure API Keys

The bridge looks for Anthropic credentials in either:

- the `ANTHROPIC_API_KEY` environment variable
- or [`external/LIDDiA/my-anthropic-key.txt`](external/LIDDiA/my-anthropic-key.txt)

The easiest option is:

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Sanity Check

```bash
conda activate liddia-a4s
cd <repo-root>
python -c "import sys; sys.path.insert(0, 'external/LIDDiA'); from liddia.agent import Claude; print(Claude.__name__)"
```

Expected:

```text
Claude
```

## Run `evaluate_official_liddia_agent4s.py`

Example:

```bash
conda activate liddia-a4s
python evaluate_official_liddia_agent4s.py \
  tasks_batch/tech_02_hard_indole.json \
  --model claude-3-5-sonnet-20241022 \
  --repo-dir external/LIDDiA \
  --pretty
```

What happens next:

- the script imports the official LIDDiA Claude agent
- it asks LIDDiA to generate one solver script
- it saves generated code under `results/official_liddia_agent4s/generated_code/`
- it saves logs under `results/official_liddia_agent4s/logs/`
- it runs the solver with `runners.batch_runner`

## Output Locations

By default:

- generated code: [`results/official_liddia_agent4s/generated_code/`](results/official_liddia_agent4s/generated_code)
- logs: [`results/official_liddia_agent4s/logs/`](results/official_liddia_agent4s/logs)

## Troubleshooting

### `Missing Anthropic key`

Set `ANTHROPIC_API_KEY` or write the key to `external/LIDDiA/my-anthropic-key.txt`.

### Import errors from `liddia.*`

Make sure `external/LIDDiA` exists and points to the repo root.

### Model alias mismatch

The wrapper normalizes a few common Claude aliases, but it is still safest to use a valid Anthropic model name explicitly.

## References

- Bridge script: [evaluate_official_liddia_agent4s.py](evaluate_official_liddia_agent4s.py)
- Setup helper: [setup_liddia_env.sh](setup_liddia_env.sh)
- Official local checkout: [external/LIDDiA/](external/LIDDiA)
- Official README: [external/LIDDiA/README.md](external/LIDDiA/README.md)
