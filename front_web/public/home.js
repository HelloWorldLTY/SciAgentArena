const linkList = document.querySelector('#benchmark-link-list');

function descriptionForBenchmark(id) {
  if (id === 'spatial') {
    return 'Spatial graph construction, SVG scoring, and neighborhood enrichment on a dedicated page.';
  }
  if (id === 'single-cell') {
    return 'The original single-cell RNA-seq judge with integration, clustering, DEGs, and trajectory scoring.';
  }
  if (id === 'perturbation-prediction') {
    return 'Predict gene expression after perturbation. Score with Pearson correlation, R², and top-K DE gene accuracy.';
  }
  if (id === 'ehr') {
    return 'Electronic Health Record analysis — clinical code normalization, temporal event extraction, outcome prediction, and treatment recommendation scored by F1.';
  }
  if (id === 'cross-domain') {
    return 'Cross-domain genomics — eQTL mapping evaluated by per-gene precision/recall/F1 and multi-omics association by exact-match accuracy against reference calls.';
  }
  if (id === 'drug-discovery') {
    return 'Drug discovery pipeline — ADMET prediction, protein-ligand binding affinity, lead optimization, drug-target interaction, and combination synergy scoring.';
  }
  if (id === 'statistical-genetics') {
    return 'Statistical genetics — Mendelian Randomization (16 methods, 6 tasks) and Polygenic Risk Score computation (5 tasks) evaluated on real GWAS summary statistics.';
  }
  return 'A dedicated benchmark page for this task family.';
}

async function init() {
  const response = await fetch('/api/config');
  const config = await response.json();
  const benchmarks = Object.entries(config.benchmarks || {});
  linkList.innerHTML = benchmarks.map(([id, benchmark]) => `
    <a class="benchmark-card-link" href="/benchmarks/${id}">
      <article class="task-card benchmark-card">
        <p class="task-summary">${id}</p>
        <h3 class="benchmark-card-title">${benchmark.label}</h3>
        <p>${descriptionForBenchmark(id)}</p>
      </article>
    </a>
  `).join('');
}

init().catch((error) => {
  linkList.innerHTML = `<article class="detail-card"><h3>Failed to load benchmarks</h3><pre>${error.message}</pre></article>`;
});
