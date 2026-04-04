# Frontier Omics Judge

A production-oriented web application for evaluating both single-cell RNA-seq and spatial transcriptomics pipeline submissions. The benchmark logic is derived from `C:\Users\75244\Downloads\agentbench_overallpipeline_evaluate.py` and `C:\Users\75244\Downloads\spatial_overallpipeline_evaluate.py`.

## What it does

- Accepts either Python pipeline code or a finished `.h5ad` file.
- Uses dedicated benchmark pages instead of one mixed page.
- Returns structured per-task outputs, aggregate metrics, logs, and overall average score.
- Supports dataset, marker-reference, trajectory-reference, and spatial SVG reference configuration.
- Is organized so new benchmark families can get their own page later.

## Pages

- Home: [http://localhost:3000](http://localhost:3000)
- Single-cell benchmark: [http://localhost:3000/benchmarks/single-cell](http://localhost:3000/benchmarks/single-cell)
- Spatial benchmark: [http://localhost:3000/benchmarks/spatial](http://localhost:3000/benchmarks/spatial)

## Architecture

- `public/index.html`: benchmark landing page.
- `public/benchmark.html`: reusable benchmark page shell.
- `public/home.js`: renders benchmark links.
- `public/benchmark.js`: renders any benchmark page based on the URL slug.
- `src/server.js`: HTTP server and clean benchmark routes.
- `src/lib/judge-service.js`: submission queue, job persistence, evaluator orchestration.
- `templates/pipeline_evaluator.py`: Python evaluator for both benchmark families.
- `judge.config.json`: benchmark/task catalog, dataset presets, timeouts, and upload limits.
- `var/jobs/`: per-job working directories and persisted job JSON.

## API

- `GET /api/config`: public platform config, benchmarks, and dataset presets.
- `POST /api/submissions`: create a job.
- `GET /api/submissions/:id`: fetch current job state and evaluation result.

## Runtime requirements

This app uses Node.js for the web layer and Python for execution/evaluation.

Required Python packages for the evaluator:

- `anndata`
- `scanpy`
- `numpy`
- `pandas`
- `scikit-learn`
- Optional: `scib`, `scib_metrics`, `squidpy`, `scipy`

Set the Python executable if `python` is not on PATH:

```powershell
$env:PYTHON_EXECUTABLE='C:\path\to\python.exe'
```

## Start the app

```powershell
npm start
```

Then open one of the pages above.

## Submission contract

### Code mode

Your Python code is executed with these variables preloaded:

- `path`: selected dataset `.h5ad` path
- `dataset_path`: same as `path`
- `DATASET_PATH`: same as `path`
- `np`, `pd`, `sc`

The script must assign the final output to an `adata` variable of type `AnnData`.

### h5ad mode

Upload an already-generated `.h5ad` containing the expected fields and embeddings.

## Dataset presets

Edit `C:\Users\75244\OneDrive\文档\New project\judge.config.json` to point at your real datasets, marker reference `.pkl` files, trajectory reference `.h5ad` files, and spatial SVG reference `.pkl` files.

## Adding more benchmark pages later

1. Add a new benchmark entry in `C:\Users\75244\OneDrive\文档\New project\judge.config.json`.
2. Add dataset presets that point to that benchmark id.
3. Add evaluator logic in `C:\Users\75244\OneDrive\文档\New project\templates\pipeline_evaluator.py`.
4. Add page copy, starter code, and override fields in `C:\Users\75244\OneDrive\文档\New project\public\benchmark.js`.
5. Visit `/benchmarks/<your-benchmark-id>`.

## Notes

- Spatial benchmark tasks 10 and 11 depend on SVG references and Squidpy-style neighborhood enrichment outputs.
- Task 8 and the advanced spatial tasks gracefully fall back to zero-style scores when optional scientific packages are unavailable.
- The current queue is in-process and single-worker by design for predictable execution on one host.
- Jobs are persisted to disk so results remain available after refresh.
