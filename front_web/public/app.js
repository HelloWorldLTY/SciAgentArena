const state = {
  config: null,
  currentBenchmarkId: 'single-cell',
  currentMode: 'code',
  activeJobId: null,
  pollTimer: null,
};

const benchmarkSelect = document.querySelector('#benchmark-select');
const datasetSelect = document.querySelector('#dataset-select');
const taskList = document.querySelector('#task-list');
const taskCopy = document.querySelector('#task-copy');
const modeSwitch = document.querySelector('#mode-switch');
const form = document.querySelector('#submission-form');
const codeInputGroup = document.querySelector('#code-input-group');
const fileInputGroup = document.querySelector('#file-input-group');
const sourceCode = document.querySelector('#source-code');
const fileInput = document.querySelector('#h5ad-file');
const datasetFileInput = document.querySelector('#dataset-file');
const resultShell = document.querySelector('#result-shell');
const submitFeedback = document.querySelector('#submit-feedback');
const taskCount = document.querySelector('#task-count');
const benchmarkMeta = document.querySelector('#benchmark-meta');
const platformTitle = document.querySelector('#platform-title');

const starterCodeByBenchmark = {
  'single-cell': `import numpy as np\nimport scanpy as sc\n\nadata = sc.read_h5ad(path)\n\nbatch_cols = [c for c in adata.obs.columns if "batch" in c.lower()]\nif "sample" not in adata.obs.columns and batch_cols:\n    adata.obs["sample"] = adata.obs[batch_cols[0]].astype(str)\n\nfor c in adata.obs.columns:\n    cl = c.lower()\n    if cl in {"cell_type", "celltype"} or "celltype" in cl:\n        adata.obs["cell_type"] = adata.obs[c].astype(str)\n        break\n\nvn = adata.var_names.astype(str)\nadata.var["mt"] = vn.str.startswith(("MT-", "Mt-"))\nadata.var["ribo"] = vn.str.startswith(("RPS", "RPL"))\nadata.var["hb"] = vn.str.match(r"^HB(?!P)", case=False)\nsc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], log1p=True, inplace=True)\n\nadata.layers["counts"] = adata.X.copy()\nsc.pp.normalize_total(adata, target_sum=float(np.median(adata.obs["total_counts"])))\nsc.pp.log1p(adata)\nsc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)\nsc.tl.pca(adata, svd_solver="arpack")\nsc.pp.neighbors(adata)\nsc.tl.leiden(adata, key_added="cluster")\nsc.tl.umap(adata)\nsc.tl.rank_genes_groups(adata, groupby="cluster", method="wilcoxon")`,
  spatial: `import numpy as np\nimport scanpy as sc\n\nadata = sc.read_h5ad(path)\n\nif "sample" not in adata.obs.columns:\n    adata.obs["sample"] = "sample_1"\n\nif "spatial" not in adata.obsm and {"x", "y"}.issubset(set(adata.obs.columns)):\n    adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy()\n\nvn = adata.var_names.astype(str)\nadata.var["mt"] = vn.str.startswith(("MT-", "Mt-"))\nadata.var["ribo"] = vn.str.startswith(("RPS", "RPL"))\nadata.var["hb"] = vn.str.match(r"^HB(?!P)", case=False)\nsc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], log1p=True, inplace=True)\n\nadata.layers["counts"] = adata.X.copy()\nsc.pp.normalize_total(adata, target_sum=float(np.median(adata.obs["total_counts"])))\nsc.pp.log1p(adata)\nsc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)\nsc.tl.pca(adata, svd_solver="arpack")\nsc.pp.neighbors(adata)\nsc.tl.leiden(adata, key_added="cluster")\nsc.tl.umap(adata)\n\n# Add spatial_connectivities, svg_global, and neighborhood enrichment outputs as needed.`,
  'perturbation-prediction': `import numpy as np\nimport scanpy as sc\nfrom sklearn.linear_model import Ridge\n\nadata = sc.read_h5ad(path)\n\n# Identify control cells and split by perturbation\nCONTROL = {"control", "ctrl", "nt", "non-targeting"}\nperts = adata.obs["pert"].astype(str)\nis_ctrl = perts.str.lower().isin(CONTROL)\nunique_perts = sorted(p for p in perts.unique() if p.lower() not in CONTROL)\nrng = np.random.default_rng(0)\ntest_perts = set(rng.choice(unique_perts, size=max(1, len(unique_perts)//5), replace=False))\nadata.obs["split"] = "train"\nadata.obs.loc[perts.isin(test_perts), "split"] = "test"\n\n# Train a baseline predictor and store predictions in layers["X_pred"]\n# ... your model here ...\nadata.layers["X_pred"] = np.zeros_like(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X)`,
  ehr: `import pandas as pd\nimport numpy as np\nfrom sklearn.ensemble import GradientBoostingClassifier\nfrom sklearn.preprocessing import LabelEncoder\n\n# Load EHR dataset\ndf = pd.read_csv(path)\n\n# Normalize clinical codes (ICD-10 mappings)\n# Extract temporal event sequences\n# Train outcome prediction model\n# Generate treatment recommendations\n\n# Output must be a DataFrame with columns:\n#   patient_id, predicted_outcome, recommended_treatments`,
  'cross-domain': `import pandas as pd\nimport numpy as np\n\n# Load your computed eQTL results\npred = pd.read_csv(pred_path)\n\n# Required columns: gene_id (or gene/gene_name), variant_id (or chrom+pos+ref+alt)\n# Optional: pvalue, beta/slope\n\n# Filter by significance threshold\npred_filtered = pred[pred['pvalue'] <= 1e-5].copy()\n\n# Output must be a CSV with at minimum:\n#   gene_id, variant_id\npred_filtered[['gene_id', 'variant_id', 'pvalue', 'beta']].to_csv(output_path, index=False)`,
  'drug-discovery': `import pandas as pd\nimport numpy as np\nfrom rdkit import Chem\nfrom rdkit.Chem import Descriptors, QED\n\n# Load compound dataset\ncompounds = pd.read_csv(path)\n\ndef compute_features(smiles):\n    mol = Chem.MolFromSmiles(smiles)\n    if mol is None:\n        return None\n    return {\n        'MW': Descriptors.MolWt(mol),\n        'LogP': Descriptors.MolLogP(mol),\n        'HBD': Descriptors.NumHDonors(mol),\n        'HBA': Descriptors.NumHAcceptors(mol),\n        'QED': QED.qed(mol),\n        'TPSA': Descriptors.TPSA(mol),\n    }\n\nfeatures = compounds['smiles'].apply(compute_features).apply(pd.Series)`,
  'statistical-genetics': `import json\nimport numpy as np\nimport pandas as pd\nfrom pathlib import Path\nfrom scipy.stats import pearsonr\n\n# Configure paths\nAGENT_OUT = Path("./output")\nDATASET   = "dataset1"\nDATASET_DIR = AGENT_OUT / DATASET\n\n# 16 MR methods required\nEXPECTED_METHODS = {\n    "IVW_fe", "IVW", "dIVW", "Egger", "RAPS",\n    "Weighted-median", "Weighted-mode", "MR-PRESSO", "MRMix",\n    "cML-MA", "MR-Robust", "MR-Lasso", "MR-ConMix", "MR-CUE",\n    "CAUSE", "MR-APSS"\n}\nEXPECTED_THRESHOLDS = {5e-08, 5e-07, 5e-06, 5e-05}\n\n# PRS traits\nTRAITS = ["BMI", "Height", "T2D"]`,
};

function setFeedback(message, isError = false) {
  submitFeedback.textContent = message;
  submitFeedback.style.color = isError ? 'var(--danger)' : 'var(--muted)';
}

function formatScore(value) {
  return Number.isFinite(value) ? value.toFixed(4) : '0.0000';
}

function currentBenchmark() {
  return state.config?.benchmarks?.[state.currentBenchmarkId] || null;
}

function benchmarkDatasets() {
  return (state.config?.datasets || []).filter((dataset) => dataset.benchmark === state.currentBenchmarkId);
}

function renderBenchmarks() {
  const benchmarks = Object.entries(state.config?.benchmarks || {});
  benchmarkSelect.innerHTML = benchmarks.map(([id, benchmark]) => `
    <option value="${id}" ${id === state.currentBenchmarkId ? 'selected' : ''}>${benchmark.label}</option>
  `).join('');
}

function renderDatasets() {
  const datasets = benchmarkDatasets();
  datasetSelect.innerHTML = datasets.map((dataset) => `
    <option value="${dataset.id}">${dataset.label} - ${dataset.datasetPath}</option>
  `).join('');
}

function renderTasks() {
  const benchmark = currentBenchmark();
  const tasks = benchmark?.tasks || [];
  taskCount.textContent = `${tasks.length} Tasks`;
  benchmarkMeta.textContent = benchmark?.label || 'Unknown benchmark';
  taskCopy.textContent = benchmark ? `${benchmark.label} task set.` : 'No benchmark selected.';
  taskList.innerHTML = tasks.map((task) => `
    <article class="task-card">
      <h3>${task.title}</h3>
      <p class="task-summary">${task.id.toUpperCase()}</p>
      <p>${task.summary}</p>
    </article>
  `).join('');
}

function renderModeSwitch(modes) {
  modeSwitch.innerHTML = modes.map((mode) => `
    <label class="mode-card ${mode.id === state.currentMode ? 'active' : ''}">
      <input type="radio" name="mode" value="${mode.id}" ${mode.id === state.currentMode ? 'checked' : ''} />
      <strong>${mode.label}</strong>
      <span>${mode.id === 'code' ? 'Run a Python pipeline and evaluate the produced AnnData object.' : 'Upload a finished AnnData output and score it directly.'}</span>
    </label>
  `).join('');
}

function syncModeUi() {
  codeInputGroup.classList.toggle('hidden', state.currentMode !== 'code');
  fileInputGroup.classList.toggle('hidden', state.currentMode !== 'h5ad');
  [...modeSwitch.querySelectorAll('.mode-card')].forEach((card) => {
    const radio = card.querySelector('input');
    card.classList.toggle('active', radio.value === state.currentMode);
  });
}

function syncBenchmarkUi() {
  sourceCode.value = starterCodeByBenchmark[state.currentBenchmarkId] || starterCodeByBenchmark['single-cell'];
  const noMarkersField = ['spatial', 'cross-domain', 'drug-discovery', 'statistical-genetics'].includes(state.currentBenchmarkId);
  const noTrajectoryField = ['spatial', 'ehr', 'cross-domain', 'drug-discovery', 'statistical-genetics'].includes(state.currentBenchmarkId);
  document.querySelector('#custom-markers-path').closest('label').classList.toggle('hidden', noMarkersField);
  document.querySelector('#custom-trajectory-path').closest('label').classList.toggle('hidden', noTrajectoryField);
  renderDatasets();
  renderTasks();
}

function renderStatus(job) {
  const status = job?.status || 'queued';
  const progress = job?.progress?.detail || '';
  const result = job?.result;
  const scoreSummary = result?.score_summary || {};
  const metricCards = Object.entries(scoreSummary).map(([key, value]) => `
    <article class="metric-card">
      <h3>${key}</h3>
      <strong>${formatScore(value)}</strong>
    </article>
  `).join('');

  const taskResults = result?.results || {};
  const detailCards = Object.entries(taskResults).map(([key, value]) => `
    <article class="detail-card">
      <h3>${key}</h3>
      <pre>${JSON.stringify(value, null, 2)}</pre>
    </article>
  `).join('');

  resultShell.classList.remove('empty-state');
  resultShell.innerHTML = `
    <div class="result-hero">
      <div>
        <span class="status-badge ${status}">${status}</span>
        <h3>${job.datasetLabel || 'Submission'} evaluation</h3>
        <p>${progress}</p>
      </div>
      <div class="score-pill">Avg ${formatScore(result?.overallAverage || 0)}</div>
    </div>
    <div class="result-meta">
      <article class="task-card">
        <p class="meta-label">Job ID</p>
        <p class="meta-value">${job.id}</p>
      </article>
      <article class="task-card">
        <p class="meta-label">Mode</p>
        <p class="meta-value">${job.mode}</p>
      </article>
      <article class="task-card">
        <p class="meta-label">Benchmark</p>
        <p class="meta-value">${result?.benchmark || state.currentBenchmarkId}</p>
      </article>
      <article class="task-card">
        <p class="meta-label">Updated</p>
        <p class="meta-value">${job.updatedAt}</p>
      </article>
    </div>
    ${job.error ? `<article class="detail-card"><h3>Error</h3><pre>${job.error}</pre></article>` : ''}
    ${metricCards ? `<div class="metric-grid">${metricCards}</div>` : ''}
    ${detailCards ? `<div class="detail-grid">${detailCards}</div>` : ''}
    ${job.logs?.length ? `<article class="detail-card"><h3>Execution Logs</h3><pre>${job.logs.join('\n\n')}</pre></article>` : ''}
  `;
}

async function pollJob(jobId) {
  state.activeJobId = jobId;
  if (state.pollTimer) {
    clearTimeout(state.pollTimer);
  }

  const response = await fetch(`/api/submissions/${jobId}`);
  const job = await response.json();
  renderStatus(job);

  if (job.status === 'queued' || job.status === 'running') {
    state.pollTimer = setTimeout(() => pollJob(jobId), 2000);
  }
}

benchmarkSelect.addEventListener('change', () => {
  state.currentBenchmarkId = benchmarkSelect.value;
  syncBenchmarkUi();
});

modeSwitch.addEventListener('change', (event) => {
  if (event.target.name !== 'mode') {
    return;
  }
  state.currentMode = event.target.value;
  syncModeUi();
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  setFeedback('Submitting evaluation job...');

  try {
    const formData = new FormData();
    formData.append('datasetId', datasetSelect.value);
    formData.append('mode', state.currentMode);
    formData.append('customDatasetPath', document.querySelector('#custom-dataset-path').value.trim());
    formData.append('customMarkersPath', document.querySelector('#custom-markers-path').value.trim());
    formData.append('customTrajectoryPath', document.querySelector('#custom-trajectory-path').value.trim());

    const datasetFile = datasetFileInput.files[0];
    if (datasetFile) {
      formData.append('datasetFile', datasetFile);
    }

    if (state.currentMode === 'code') {
      formData.append('sourceCode', sourceCode.value);
    } else {
      const file = fileInput.files[0];
      if (!file) {
        throw new Error('Please choose a .h5ad file before submitting.');
      }
      formData.append('h5adFile', file);
    }

    const response = await fetch('/api/submissions', {
      method: 'POST',
      body: formData,
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.error || 'Submission failed');
    }

    setFeedback('Submission accepted. Polling for results...');
    renderStatus(body);
    await pollJob(body.id);
  } catch (error) {
    setFeedback(error.message || 'Submission failed', true);
  }
});

async function init() {
  const response = await fetch('/api/config');
  const config = await response.json();
  state.config = config;
  platformTitle.textContent = config.platformName || 'BioMedArena';
  renderBenchmarks();
  renderModeSwitch(config.submissionModes || []);
  syncModeUi();
  syncBenchmarkUi();
  setFeedback(`Ready. Max upload size: ${config.maxUploadSizeMb} MB.`);
}

init().catch((error) => {
  setFeedback(error.message || 'Failed to initialize UI', true);
});
