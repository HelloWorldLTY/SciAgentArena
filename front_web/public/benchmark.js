const state = {
  config: null,
  benchmarkId: window.location.pathname.split('/').filter(Boolean).at(-1) || 'single-cell',
  benchmark: null,
  datasets: [],
  currentMode: 'code',
  activeJobId: null,
  pollTimer: null,
};

const pageLinks = document.querySelector('#page-links');
const benchmarkEyebrow = document.querySelector('#benchmark-eyebrow');
const benchmarkTitle = document.querySelector('#benchmark-title');
const benchmarkCopy = document.querySelector('#benchmark-copy');
const benchmarkMeta = document.querySelector('#benchmark-meta');
const datasetSelect = document.querySelector('#dataset-select');
const taskList = document.querySelector('#task-list');
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
const taskCheckboxes = document.querySelector('#task-checkboxes');
const overrideGrid = document.querySelector('#override-grid');
const submissionCopy = document.querySelector('#submission-copy');
const taskCopy = document.querySelector('#task-copy');

const benchmarkPageContent = {
  'single-cell': {
    eyebrow: 'Dedicated benchmark page for single-cell RNA-seq workflows',
    copy: 'Submit a preprocessing or analysis pipeline for single-cell data, then score integration, clustering, DEGs, and trajectory preservation in one place.',
    meta: 'QC, filtering, doublets, integration, clustering, DEGs, trajectory',
    submissionCopy: 'Pick a single-cell dataset preset or override the paths below, then run evaluation.',
    taskCopy: 'This page is scoped to the single-cell benchmark family.',
    starterCode: `import numpy as np\nimport scanpy as sc\n\nadata = sc.read_h5ad(path)\n\nbatch_cols = [c for c in adata.obs.columns if "batch" in c.lower()]\nif "sample" not in adata.obs.columns and batch_cols:\n    adata.obs["sample"] = adata.obs[batch_cols[0]].astype(str)\n\nfor c in adata.obs.columns:\n    cl = c.lower()\n    if cl in {"cell_type", "celltype"} or "celltype" in cl:\n        adata.obs["cell_type"] = adata.obs[c].astype(str)\n        break\n\nvn = adata.var_names.astype(str)\nadata.var["mt"] = vn.str.startswith(("MT-", "Mt-"))\nadata.var["ribo"] = vn.str.startswith(("RPS", "RPL"))\nadata.var["hb"] = vn.str.match(r"^HB(?!P)", case=False)\nsc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], log1p=True, inplace=True)\n\nadata.layers["counts"] = adata.X.copy()\nsc.pp.normalize_total(adata, target_sum=float(np.median(adata.obs["total_counts"])))\nsc.pp.log1p(adata)\nsc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)\nsc.tl.pca(adata, svd_solver="arpack")\nsc.pp.neighbors(adata)\nsc.tl.leiden(adata, key_added="cluster")\nsc.tl.umap(adata)\nsc.tl.rank_genes_groups(adata, groupby="cluster", method="wilcoxon")`,
    overrideFields: [
      { id: 'custom-dataset-path', key: 'customDatasetPath', label: 'Dataset path override', placeholder: 'Optional custom .h5ad input path' },
      { id: 'custom-markers-path', key: 'customMarkersPath', label: 'Marker path override', placeholder: 'Optional .pkl marker reference' },
      { id: 'custom-trajectory-path', key: 'customTrajectoryPath', label: 'Trajectory reference override', placeholder: 'Optional trajectory .h5ad path' },
    ],
  },
  spatial: {
    eyebrow: 'Dedicated benchmark page for spatial transcriptomics workflows',
    copy: 'Submit a spatial transcriptomics pipeline and score coordinate handling, spatial graph construction, SVG discovery, clustering, and neighborhood enrichment.',
    meta: 'Coordinates, spatial graph, SVGs, clustering, Moran\'s I, neighborhood enrichment',
    submissionCopy: 'Pick a spatial dataset preset or override the core dataset path, then run the spatial benchmark.',
    taskCopy: 'This page is scoped to the spatial transcriptomics benchmark family.',
    starterCode: `import numpy as np\nimport scanpy as sc\n\nadata = sc.read_h5ad(path)\n\nif "sample" not in adata.obs.columns:\n    adata.obs["sample"] = "sample_1"\n\nif "spatial" not in adata.obsm and {"x", "y"}.issubset(set(adata.obs.columns)):\n    adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy()\n\nvn = adata.var_names.astype(str)\nadata.var["mt"] = vn.str.startswith(("MT-", "Mt-"))\nadata.var["ribo"] = vn.str.startswith(("RPS", "RPL"))\nadata.var["hb"] = vn.str.match(r"^HB(?!P)", case=False)\nsc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], log1p=True, inplace=True)\n\nadata.layers["counts"] = adata.X.copy()\nsc.pp.normalize_total(adata, target_sum=float(np.median(adata.obs["total_counts"])))\nsc.pp.log1p(adata)\nsc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)\nsc.tl.pca(adata, svd_solver="arpack")\nsc.pp.neighbors(adata)\nsc.tl.leiden(adata, key_added="cluster")\nsc.tl.umap(adata)\n\n# Add your spatial graph / SVG / neighborhood enrichment outputs below.`,
    overrideFields: [
      { id: 'custom-dataset-path', key: 'customDatasetPath', label: 'Dataset path override', placeholder: 'Optional custom spatial .h5ad input path' },
    ],
  },
  'perturbation-prediction': {
    eyebrow: 'Dedicated benchmark page for perturbation prediction',
    copy: 'Submit a perturbation prediction pipeline. Train on observed perturbations, predict expression for held-out conditions, and score with Pearson correlation, R², and MSE.',
    meta: 'Perturbation response, gene expression prediction, top-K DE genes, Pearson, R²',
    submissionCopy: 'Pick a perturbation dataset preset or override the dataset path, then run evaluation.',
    taskCopy: 'This page is scoped to the perturbation prediction benchmark family.',
    starterCode: `import numpy as np\nimport scanpy as sc\nfrom sklearn.decomposition import TruncatedSVD\nfrom sklearn.preprocessing import OneHotEncoder`,
    overrideFields: [
      { id: 'custom-dataset-path', key: 'customDatasetPath', label: 'Dataset path override', placeholder: 'Optional custom perturbation .h5ad input path' },
    ],
  },
  ehr: {
    eyebrow: 'Dedicated benchmark page for Electronic Health Records',
    copy: 'Submit an EHR analysis pipeline. Load structured patient records, normalize clinical codes, extract temporal events, predict outcomes, and score treatment recommendations — all evaluated by F1 between predicted and reference solutions.',
    meta: 'Clinical codes, ICD-10, temporal events, readmission prediction, treatment F1',
    submissionCopy: 'Pick an EHR dataset preset or override the data path, then run the clinical evaluation pipeline.',
    taskCopy: 'This page is scoped to the Electronic Health Records benchmark family.',
    starterCode: `import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# Load EHR dataset
df = pd.read_csv(path)

# Normalize clinical codes (ICD-10 mappings)
# Extract temporal event sequences
# Train outcome prediction model
# Generate treatment recommendations

# Output must be a DataFrame with columns:
#   patient_id, predicted_outcome, recommended_treatments
`,
    overrideFields: [
      { id: 'custom-dataset-path', key: 'customDatasetPath', label: 'EHR data path override', placeholder: 'Optional custom EHR CSV/parquet input path' },
      { id: 'custom-markers-path', key: 'customMarkersPath', label: 'Reference labels path override', placeholder: 'Optional reference labels CSV path' },
    ],
  },
  'cross-domain': {
    eyebrow: 'Dedicated benchmark page for Cross Domain Analysis',
    copy: 'Submit a cross-domain genomics pipeline. Map eQTLs and compare predicted (gene, variant) pairs against a reference set using per-gene precision, recall, and F1. Multi-omics association tasks are scored by exact-match accuracy.',
    meta: 'eQTL mapping, gene-variant pairs, precision/recall/F1, multi-omics, exact-match accuracy',
    submissionCopy: 'Pick a cross-domain dataset preset or override the prediction and reference file paths, then run evaluation.',
    taskCopy: 'This page is scoped to the Cross Domain Analysis benchmark family.',
    starterCode: `import pandas as pd
import numpy as np

# Load your computed eQTL results
pred = pd.read_csv(pred_path)

# Required columns: gene_id (or gene/gene_name), variant_id (or chrom+pos+ref+alt)
# Optional: pvalue, beta/slope

# Filter by significance threshold
pred_filtered = pred[pred['pvalue'] <= 1e-5].copy()

# Output must be a CSV with at minimum:
#   gene_id, variant_id
# Optionally include: pvalue, beta
pred_filtered[['gene_id', 'variant_id', 'pvalue', 'beta']].to_csv(output_path, index=False)
`,
    overrideFields: [
      { id: 'custom-dataset-path', key: 'customDatasetPath', label: 'Prediction file path override', placeholder: 'Optional computed eQTL results CSV/TSV path' },
      { id: 'custom-markers-path', key: 'customMarkersPath', label: 'Reference file path override', placeholder: 'Optional reference eQTL set CSV/TSV path' },
    ],
  },
  'drug-discovery': {
    eyebrow: 'Dedicated benchmark page for Drug Discovery',
    copy: 'Submit a drug discovery pipeline. Predict ADMET properties, protein-ligand binding affinities, and drug-target interactions — scored against experimental reference values. Lead optimization and synergy prediction tasks use ranking and correlation metrics.',
    meta: 'ADMET, binding affinity, lead optimization, drug-target interaction, synergy',
    submissionCopy: 'Pick a drug discovery dataset preset or override the compound and reference file paths, then run evaluation.',
    taskCopy: 'This page is scoped to the Drug Discovery benchmark family.',
    starterCode: `import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

# Load compound dataset
compounds = pd.read_csv(path)

# Compute molecular descriptors
def compute_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'QED': QED.qed(mol),
        'TPSA': Descriptors.TPSA(mol),
    }

features = compounds['smiles'].apply(compute_features).apply(pd.Series)

# Train property prediction model
# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
`,
    overrideFields: [
      { id: 'custom-dataset-path', key: 'customDatasetPath', label: 'Compound data path override', placeholder: 'Optional compounds CSV path (must include SMILES column)' },
      { id: 'custom-markers-path', key: 'customMarkersPath', label: 'Reference data path override', placeholder: 'Optional experimental reference values CSV path' },
    ],
  },
  'statistical-genetics': {
    eyebrow: 'Dedicated benchmark page for Statistical Genetics',
    copy: 'Submit a statistical genetics pipeline covering Mendelian Randomization (16 methods across 6 tasks) and Polygenic Risk Score computation (5 tasks). Evaluated on real GWAS summary statistics with checks for method correctness, IV selection, and predictive accuracy.',
    meta: 'MR, IVW, Egger, MR-APSS, CAUSE, PRS, LDpred2, PRS-CS, incremental R², AUC',
    submissionCopy: 'Pick a genetics dataset preset or override the dataset path, then run the MR or PRS evaluation pipeline.',
    taskCopy: 'This page covers both Mendelian Randomization (Tasks 1-6) and Polygenic Risk Score (Tasks 7-11) benchmarks.',
    starterCode: `import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

# ── Mendelian Randomization pipeline ──────────────────────────────────────
# Configure paths
AGENT_OUT = Path("./output")
DATASET   = "dataset1"
DATASET_DIR = AGENT_OUT / DATASET

# Expected: 16 MR methods across 4 IV p-value thresholds
EXPECTED_METHODS = {
    "IVW_fe", "IVW", "dIVW", "Egger", "RAPS",
    "Weighted-median", "Weighted-mode", "MR-PRESSO", "MRMix",
    "cML-MA", "MR-Robust", "MR-Lasso", "MR-ConMix", "MR-CUE",
    "CAUSE", "MR-APSS"
}
EXPECTED_THRESHOLDS = {5e-08, 5e-07, 5e-06, 5e-05}

# Step 1: Format GWAS summary statistics (see Task 2)
# Step 2: Estimate background parameters for MR-APSS (see Task 3)
# Step 3: LD clumping for IV selection via PLINK (see Task 4)
# Step 4: Run all 16 MR methods (see Task 5)
# Step 5: Aggregate results and compute Type-I error rates (see Task 6)

# ── Polygenic Risk Score pipeline ─────────────────────────────────────────
# Supported methods: PRS-CS, LDpred2, SBayesR, lassosum2
# Output: weights_merged.txt + prs_scores.sscore per trait
TRAITS = ["BMI", "Height", "T2D"]
`,
    overrideFields: [
      { id: 'custom-dataset-path', key: 'customDatasetPath', label: 'Agent output directory override', placeholder: 'Optional path to agent output directory' },
      { id: 'custom-markers-path', key: 'customMarkersPath', label: 'Reference data path override', placeholder: 'Optional path to MR paper reference or PRS gold standard' },
      { id: 'custom-trajectory-path', key: 'customTrajectoryPath', label: 'Dataset ID override', placeholder: 'Optional dataset ID (e.g. dataset1 through dataset7)' },
    ],
  },
};

function setFeedback(message, isError = false) {
  submitFeedback.textContent = message;
  submitFeedback.style.color = isError ? 'var(--danger)' : 'var(--muted)';
}

function formatScore(value) {
  return Number.isFinite(value) ? value.toFixed(4) : '0.0000';
}

function renderPageLinks() {
  const benchmarks = Object.entries(state.config?.benchmarks || {});
  pageLinks.innerHTML = benchmarks.map(([id, benchmark]) => `
    <a class="nav-link ${id === state.benchmarkId ? 'active' : ''}" href="/benchmarks/${id}">${benchmark.label}</a>
  `).join('');
}

function renderDatasets() {
  datasetSelect.innerHTML = state.datasets.map((dataset) => `
    <option value="${dataset.id}">${dataset.label} - ${dataset.datasetPath}</option>
  `).join('');
}

function renderTasks(tasks) {
  taskCount.textContent = `${tasks.length} Tasks`;
  taskList.innerHTML = tasks.map((task) => `
    <article class="task-card">
      <h3>${task.title}</h3>
      <p class="task-summary">${task.id.toUpperCase()}</p>
      <p>${task.summary}</p>
    </article>
  `).join('');
  taskCheckboxes.innerHTML = tasks.map((task) => `
    <label>
      <input type="checkbox" name="task" value="${task.id}" />
      ${task.id.toUpperCase()}: ${task.title}
    </label>
  `).join('');
}

function getSelectedTasks() {
  const checked = [...taskCheckboxes.querySelectorAll('input[type="checkbox"]:checked')];
  return checked.map((cb) => cb.value);
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

function renderOverrides() {
  const content = benchmarkPageContent[state.benchmarkId] || benchmarkPageContent['single-cell'];
  overrideGrid.innerHTML = content.overrideFields.map((field) => `
    <label>
      <span>${field.label}</span>
      <input id="${field.id}" name="${field.key}" type="text" placeholder="${field.placeholder}" />
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

function groupScoresByTask(scoreSummary) {
  const grouped = {};
  for (const [key, value] of Object.entries(scoreSummary)) {
    const sep = key.indexOf('_');
    const taskId = sep === -1 ? key : key.slice(0, sep);
    const metric = sep === -1 ? 'score' : key.slice(sep + 1);
    if (!grouped[taskId]) grouped[taskId] = {};
    grouped[taskId][metric] = value;
  }
  return grouped;
}

function taskAggregate(metrics) {
  const vals = Object.values(metrics).filter((v) => Number.isFinite(v));
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
}

function renderStatus(job) {
  const status = job?.status || 'queued';
  const progress = job?.progress?.detail || '';
  const result = job?.result;
  const scoreSummary = result?.score_summary || {};
  const evaluatedTasks = result?.evaluated_tasks || null;

  // Per-task aggregate pills
  const grouped = groupScoresByTask(scoreSummary);
  const taskPills = Object.entries(grouped)
    .sort(([a], [b]) => {
      const na = parseInt(a.replace('task', ''), 10);
      const nb = parseInt(b.replace('task', ''), 10);
      return na - nb;
    })
    .map(([taskId, metrics]) => {
      const agg = taskAggregate(metrics);
      const cls = agg >= 0.5 ? 'pass' : 'fail';
      return `<div class="per-task-pill ${cls}"><span>${taskId.toUpperCase()}</span><strong>${formatScore(agg)}</strong></div>`;
    }).join('');

  // Detailed metric cards (grouped under each task)
  const taskResults = result?.results || {};
  const detailCards = Object.entries(taskResults).map(([key, value]) => `
    <article class="detail-card">
      <h3>${key.toUpperCase()}</h3>
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
        ${evaluatedTasks ? `<p style="font-size:0.85rem;color:var(--muted)">Tasks evaluated: ${evaluatedTasks.map((t) => t.toUpperCase()).join(', ')}</p>` : ''}
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
        <p class="meta-value">${result?.benchmark || state.benchmarkId}</p>
      </article>
      <article class="task-card">
        <p class="meta-label">Updated</p>
        <p class="meta-value">${job.updatedAt}</p>
      </article>
    </div>
    ${taskPills ? `<h3 style="margin:16px 0 4px">Per-Task Scores</h3><div class="per-task-summary">${taskPills}</div>` : ''}
    ${job.error ? `<article class="detail-card"><h3>Error</h3><pre>${job.error}</pre></article>` : ''}
    ${detailCards ? `<h3 style="margin:16px 0 4px">Task Details</h3><div class="detail-grid">${detailCards}</div>` : ''}
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

    for (const field of benchmarkPageContent[state.benchmarkId].overrideFields) {
      const input = document.querySelector(`#${field.id}`);
      formData.append(field.key, input?.value?.trim() || '');
    }

    const selectedTasks = getSelectedTasks();
    if (selectedTasks.length > 0) {
      formData.append('tasks', JSON.stringify(selectedTasks));
    }

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
  state.benchmark = config.benchmarks?.[state.benchmarkId];
  if (!state.benchmark) {
    window.location.href = '/';
    return;
  }
  state.datasets = (config.datasets || []).filter((dataset) => dataset.benchmark === state.benchmarkId);
  const content = benchmarkPageContent[state.benchmarkId] || benchmarkPageContent['single-cell'];
  benchmarkEyebrow.textContent = content.eyebrow;
  benchmarkTitle.textContent = state.benchmark.label;
  benchmarkCopy.textContent = content.copy;
  benchmarkMeta.textContent = content.meta;
  submissionCopy.textContent = content.submissionCopy;
  taskCopy.textContent = content.taskCopy;
  sourceCode.value = content.starterCode;
  renderPageLinks();
  renderDatasets();
  renderTasks(state.benchmark.tasks || []);
  renderOverrides();
  renderModeSwitch(config.submissionModes || []);
  syncModeUi();
  setFeedback(`Ready. Max upload size: ${config.maxUploadSizeMb} MB.`);
}

init().catch((error) => {
  setFeedback(error.message || 'Failed to initialize page', true);
});
