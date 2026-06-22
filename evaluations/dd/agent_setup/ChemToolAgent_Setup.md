# Setting Up ChemToolAgent For `evaluate_chemtool_agent.py`

## Is This Using The Official Repo?

Yes.

`evaluate_chemtool_agent.py` imports the native `ChemAgent` class directly from the local ChemToolAgent repo:

- [`ChemToolAgent/chemagent/agent/agent.py`](ChemToolAgent/chemagent/agent/agent.py)

So this is not a reimplementation of the agent. The wrapper is ours, but the agent implementation is the official ChemToolAgent codebase already present in this workspace:

- [`ChemToolAgent/`](ChemToolAgent)

## What This Script Actually Uses

`evaluate_chemtool_agent.py` does all of the following:

- loads one or more Agent4Science task JSON files
- imports the official `ChemAgent`
- optionally starts the repo's Jupyter-backed Python tool server
- runs ChemToolAgent on the task
- extracts the final JSON answer
- scores it with the local Agent4Science scorer

Important: this is a direct adapter around the official ChemToolAgent agent, not a code-generation bridge.

## Prerequisites

- conda
- Python 3.9
- the local official ChemToolAgent repo at [`ChemToolAgent/`](ChemToolAgent)
- API keys for the backbone LLM and any optional tools you intend to use

## Recommended Setup

This repo already includes a helper script:

- [setup_chemtoolagent_env.sh](setup_chemtoolagent_env.sh)

Run:

```bash
cd <repo-root>
bash setup_chemtoolagent_env.sh chemtoolagent-a4s
```

That script will:

- create a Python 3.9 conda environment
- install `ChemToolAgent/requirements.txt`
- install `jupyter_kernel_gateway`
- pin `httpx==0.27.2` to avoid client incompatibilities
- run import checks
- verify that the `jupyter` CLI exists

## Manual Setup

```bash
conda create -n chemtoolagent-a4s python=3.9 -y
conda activate chemtoolagent-a4s
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r ChemToolAgent/requirements.txt
python -m pip install jupyter_kernel_gateway "httpx==0.27.2"
```

## Configure API Keys

At minimum, export the key for your selected backbone model:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

Some optional ChemToolAgent tools may require extra service keys as described in:

- [`ChemToolAgent/README.md`](ChemToolAgent/README.md)

## Jupyter Server Requirement

ChemToolAgent's Python tooling often expects its Jupyter-backed execution server.

You can either start it manually:

```bash
conda activate chemtoolagent-a4s
cd ChemToolAgent/python_server
./start_jupyter_server.sh 8888
```

or let the wrapper start it for you with:

```bash
--start-jupyter-server
```

## Sanity Check

```bash
conda activate chemtoolagent-a4s
cd <repo-root>
python -c "import sys; sys.path.insert(0, 'ChemToolAgent'); from chemagent import ChemAgent; print(ChemAgent.__name__)"
```

Expected:

```text
ChemAgent
```

## Run `evaluate_chemtool_agent.py`

Example:

```bash
conda activate chemtoolagent-a4s
cd <repo-root>
python evaluate_chemtool_agent.py \
  tasks_batch/tech_07_easy_acetate.json \
  --model claude-sonnet-4-6 \
  --start-jupyter-server \
  --pretty
```

You can also use the wrapper's focused preset:

```bash
python evaluate_chemtool_agent.py --formal-charge-only --pretty
```

## Optional Uni-Core Note

The official ChemToolAgent README mentions manual Uni-Core installation if you want property-prediction tooling. The setup helper in this repo does not install Uni-Core automatically.

That is usually fine for many Agent4Science evaluations unless you specifically need those property prediction tools.

## Troubleshooting

### `ChemToolAgent not found`

Make sure the local repo exists at [`ChemToolAgent/`](ChemToolAgent).

### Python tool execution issues

Start the Jupyter server manually or pass `--start-jupyter-server`.

### Client errors related to `httpx`

Keep the pinned version:

```bash
python -m pip install "httpx==0.27.2"
```

### Missing optional checkpoints

The official README mentions additional checkpoint downloads for some property prediction tools. Those are not required for every Agent4Science task.

## References

- Adapter script: [evaluate_chemtool_agent.py](evaluate_chemtool_agent.py)
- Setup helper: [setup_chemtoolagent_env.sh](setup_chemtoolagent_env.sh)
- Official local repo: [ChemToolAgent/](ChemToolAgent)
- Official README: [ChemToolAgent/README.md](ChemToolAgent/README.md)
