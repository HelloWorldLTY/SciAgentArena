from __future__ import annotations

import argparse
import json
import os
import pickle
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    import anndata as ad
    import scanpy as sc
except Exception as exc:  # pragma: no cover
    raise SystemExit("The evaluator requires Python packages 'anndata' and 'scanpy'.") from exc


@dataclass
class StepResult:
    step_id: str
    passed: Any
    checks: Dict[str, Any]
    errors: List[str]


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def _read_h5ad(path: str) -> 'ad.AnnData':
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f'Missing file: {path}')
    return ad.read_h5ad(path)


def _has_all(container: Any, keys: List[str]) -> Tuple[bool, List[str]]:
    missing = [k for k in keys if k not in container]
    return (len(missing) == 0, missing)


def _is_finite_array(x: Any) -> bool:
    if x is None:
        return False
    arr = np.asarray(x)
    return arr.size > 0 and np.isfinite(arr).all()


def _align_on_obs_names(pred: 'ad.AnnData', gold: 'ad.AnnData') -> Tuple['ad.AnnData', 'ad.AnnData']:
    common = pred.obs_names.intersection(gold.obs_names)
    if len(common) == 0:
        raise ValueError('No overlapping cells/spots between prediction and reference AnnData objects.')
    common = common.sort_values()
    return pred[common].copy(), gold[common].copy()


def _majority_mapping(cluster: np.ndarray, celltype: np.ndarray) -> Tuple[Dict[str, str], Dict[str, float]]:
    mapping: Dict[str, str] = {}
    purity: Dict[str, float] = {}
    for label in np.unique(cluster):
        mask = cluster == label
        values = celltype[mask]
        if values.size == 0:
            mapping[str(label)] = 'NA'
            purity[str(label)] = 0.0
            continue
        unique_values, counts = np.unique(values, return_counts=True)
        best_idx = int(np.argmax(counts))
        mapping[str(label)] = str(unique_values[best_idx])
        purity[str(label)] = float(counts[best_idx] / values.size)
    return mapping, purity


def _normalize_gene(gene: str) -> str:
    return str(gene).strip().upper()


def _normalize_gene_set(genes: Iterable[str]) -> Set[str]:
    return {_normalize_gene(g) for g in genes if g is not None and str(g).strip()}


def _overlap_metrics(a: Set[str], b: Set[str]) -> Dict[str, float]:
    inter = a & b
    a_len = len(a)
    b_len = len(b)
    inter_len = len(inter)
    union_len = a_len + b_len - inter_len
    precision = inter_len / a_len if a_len else 0.0
    recall = inter_len / b_len if b_len else 0.0
    return {
        'jaccard': inter_len / union_len if union_len else 1.0,
        'overlap_coef': inter_len / min(a_len, b_len) if min(a_len, b_len) else 1.0,
        'precision(A_in_B)': precision,
        'recall(B_covered_by_A)': recall,
        'f1': (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
    }


def _find_coords(adata: 'ad.AnnData') -> Tuple[str, Optional[Tuple[str, str]]]:
    obs = adata.obs
    for x_col, y_col in [
        ('x', 'y'),
        ('X', 'Y'),
        ('spatial_x', 'spatial_y'),
        ('x_coord', 'y_coord'),
        ('pxl_col_in_fullres', 'pxl_row_in_fullres'),
        ('array_col', 'array_row'),
    ]:
        if x_col in obs.columns and y_col in obs.columns:
            return 'obs', (x_col, y_col)
    if 'spatial' in adata.obsm:
        arr = adata.obsm['spatial']
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] == adata.n_obs and arr.shape[1] >= 2:
            return 'obsm', None
    return 'missing', None


def _add_coord_checks(adata: 'ad.AnnData', checks: Dict[str, Any], errors: List[str]) -> None:
    source, obs_cols = _find_coords(adata)
    checks['coords_source'] = source
    if source == 'missing':
        errors.append("Missing spatial coordinates (need obs x/y columns or obsm['spatial']).")
        return
    if source == 'obsm':
        xy = np.asarray(adata.obsm['spatial'][:, :2], dtype=float)
        n_bad = int(np.sum(~np.isfinite(xy)))
        checks['coords_n_nonfinite'] = n_bad
        if n_bad > 0:
            errors.append("Spatial coordinates in obsm['spatial'] contain non-finite values.")
        return
    x_col, y_col = obs_cols
    checks['coords_x_col'] = x_col
    checks['coords_y_col'] = y_col
    x = pd.to_numeric(adata.obs[x_col], errors='coerce').to_numpy(dtype=float)
    y = pd.to_numeric(adata.obs[y_col], errors='coerce').to_numpy(dtype=float)
    n_bad = int(np.sum(~np.isfinite(x)) + np.sum(~np.isfinite(y)))
    checks['coords_n_nonfinite'] = n_bad
    if n_bad > 0:
        errors.append(f'Spatial coordinate columns ({x_col},{y_col}) contain NaN/inf values.')


def _build_scib_like_fallback() -> Dict[str, Any]:
    columns = [
        'Isolated labels', 'KMeans NMI', 'KMeans ARI', 'Silhouette label',
        'cLISI', 'Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity',
        'PCR comparison', 'Batch correction', 'Bio conservation', 'Total',
    ]
    return {'table': {name: 0.0 for name in columns}, 'aggregate_total': 0.0, 'source': 'fallback'}


def _load_pickle(path: str) -> Any:
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def check_basic_load_single_cell(adata: 'ad.AnnData') -> StepResult:
    errors: List[str] = []
    checks: Dict[str, Any] = {'n_obs': int(adata.n_obs), 'n_vars': int(adata.n_vars)}
    if 'sample' not in adata.obs:
        errors.append("Missing obs['sample']")
    if 'cell_type' not in adata.obs:
        errors.append("Missing obs['cell_type']")
    if 'sample' in adata.obs:
        sample_n = int(adata.obs['sample'].nunique(dropna=False))
        checks['sample_nunique'] = sample_n
        if sample_n < 2:
            errors.append("obs['sample'] must contain at least two unique labels")
    if adata.n_obs <= 0 or adata.n_vars <= 0:
        errors.append('AnnData has non-positive dimensions')
    return StepResult('task1', len(errors) == 0, checks, errors)


def check_basic_load_spatial(adata: 'ad.AnnData') -> StepResult:
    errors: List[str] = []
    checks: Dict[str, Any] = {'n_obs': int(adata.n_obs), 'n_vars': int(adata.n_vars)}
    if adata.n_obs <= 0 or adata.n_vars <= 0:
        errors.append('AnnData has non-positive dimensions')
    if 'sample' not in adata.obs:
        errors.append("Missing obs['sample']")
    else:
        sample_n = int(adata.obs['sample'].nunique(dropna=False))
        checks['sample_nunique'] = sample_n
        if sample_n < 2:
            errors.append("obs['sample'] must contain at least two unique labels")
    _add_coord_checks(adata, checks, errors)
    checks['var_names_unique'] = bool(adata.var_names.is_unique)
    if not adata.var_names.is_unique:
        errors.append('var_names are not unique')
    return StepResult('task1', len(errors) == 0, checks, errors)


def check_qc_metrics(adata: 'ad.AnnData', step_id: str = 'task2') -> StepResult:
    errors: List[str] = []
    checks: Dict[str, Any] = {}
    ok, missing = _has_all(adata.var, ['mt', 'ribo', 'hb'])
    checks['var_flags_present'] = ok
    if not ok:
        errors.append(f'Missing var flags: {missing}')
    required_obs = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'log1p_total_counts', 'log1p_n_genes_by_counts']
    ok, missing = _has_all(adata.obs, required_obs)
    checks['qc_obs_keys_present'] = ok
    if not ok:
        errors.append(f'Missing qc obs keys: {missing}')
    if 'pct_counts_mt' in adata.obs:
        pct = adata.obs['pct_counts_mt'].to_numpy()
        checks['pct_counts_mt_range'] = [float(np.nanmin(pct)), float(np.nanmax(pct))]
        if np.nanmin(pct) < -1e-6 or np.nanmax(pct) > 100 + 1e-6:
            errors.append('pct_counts_mt fell outside [0,100]')
    return StepResult(step_id, len(errors) == 0, checks, errors)


def check_filter(adata: 'ad.AnnData', prev: Optional['ad.AnnData'], step_id: str) -> StepResult:
    errors: List[str] = []
    checks: Dict[str, Any] = {'n_obs': int(adata.n_obs), 'n_vars': int(adata.n_vars)}
    if prev is not None:
        checks['prev_n_obs'] = int(prev.n_obs)
        checks['prev_n_vars'] = int(prev.n_vars)
        if adata.n_obs > prev.n_obs:
            errors.append('n_obs increased after filtering')
        if adata.n_vars > prev.n_vars:
            errors.append('n_vars increased after filtering')
    return StepResult(step_id, len(errors) == 0, checks, errors)


def check_normalize_log1p(adata: 'ad.AnnData', step_id: str) -> StepResult:
    errors: List[str] = []
    checks: Dict[str, Any] = {'raw_present': bool(adata.raw is not None), 'layers_counts_present': bool('counts' in adata.layers)}
    if not checks['raw_present'] and not checks['layers_counts_present']:
        errors.append("Expected adata.raw or adata.layers['counts']")
    matrix = adata.X.data if hasattr(adata.X, 'data') else np.asarray(adata.X).ravel()
    checks['X_finite'] = bool(np.isfinite(matrix).all())
    checks['X_min'] = float(np.min(matrix)) if matrix.size else 0.0
    if not checks['X_finite']:
        errors.append('.X contains non-finite values')
    if checks['X_min'] < -1e-6:
        errors.append('.X contains negative values after normalization/log1p')
    return StepResult(step_id, len(errors) == 0, checks, errors)


def check_hvg_pca(adata: 'ad.AnnData', hvg_range: Tuple[int, int], step_id: str, require_nbatches: bool) -> StepResult:
    errors: List[str] = []
    checks: Dict[str, Any] = {}
    if 'highly_variable' not in adata.var:
        errors.append("Missing var['highly_variable']")
    else:
        count = int(np.sum(adata.var['highly_variable'].to_numpy().astype(bool)))
        checks['hvg_count'] = count
        if not (hvg_range[0] <= count <= hvg_range[1]):
            errors.append(f'HVG count {count} outside expected range {hvg_range}')
    if require_nbatches and 'highly_variable_nbatches' not in adata.var:
        errors.append("Missing var['highly_variable_nbatches']")
    if 'X_pca' not in adata.obsm:
        errors.append("Missing obsm['X_pca']")
    else:
        x_pca = np.asarray(adata.obsm['X_pca'])
        checks['X_pca_shape'] = list(x_pca.shape)
        if x_pca.ndim != 2 or x_pca.shape[0] != adata.n_obs:
            errors.append('X_pca has invalid shape')
        if not _is_finite_array(x_pca):
            errors.append('X_pca contains non-finite values')
    return StepResult(step_id, len(errors) == 0, checks, errors)


def check_neighbors_umap(adata: 'ad.AnnData', step_id: str) -> StepResult:
    errors: List[str] = []
    checks: Dict[str, Any] = {}
    if 'connectivities' not in adata.obsp:
        errors.append("Missing obsp['connectivities']")
    if 'X_umap' not in adata.obsm:
        errors.append("Missing obsm['X_umap']")
    else:
        x_umap = np.asarray(adata.obsm['X_umap'])
        checks['X_umap_shape'] = list(x_umap.shape)
        checks['X_umap_var_sum'] = float(np.var(x_umap, axis=0).sum())
        if x_umap.shape != (adata.n_obs, 2):
            errors.append('X_umap must have shape (n_obs, 2)')
        if not _is_finite_array(x_umap):
            errors.append('X_umap contains non-finite values')
        if checks['X_umap_var_sum'] <= 0.0:
            errors.append('X_umap appears constant')
    return StepResult(step_id, len(errors) == 0, checks, errors)


def check_doublet(adata: 'ad.AnnData') -> StepResult:
    errors: List[str] = []
    checks: Dict[str, Any] = {}
    ok, missing = _has_all(adata.obs, ['predicted_doublet', 'doublet_score'])
    checks['scrublet_keys_present'] = ok
    if not ok:
        errors.append(f'Missing scrublet keys: {missing}')
    if 'doublet_score' in adata.obs:
        scores = adata.obs['doublet_score'].to_numpy()
        checks['doublet_score_finite'] = bool(np.isfinite(scores).all())
        if not np.isfinite(scores).all():
            errors.append('doublet_score contains non-finite values')
    return StepResult('task4', len(errors) == 0, checks, errors)


def check_spatial_graph(adata: 'ad.AnnData') -> StepResult:
    errors: List[str] = []
    checks: Dict[str, Any] = {}
    graph = adata.obsp.get('spatial_connectivities')
    if graph is None:
        errors.append("Missing obsp['spatial_connectivities']")
    else:
        checks['spatial_connectivities_shape'] = list(graph.shape)
        checks['spatial_connectivities_nnz'] = int(getattr(graph, 'nnz', 0))
        if graph.shape != (adata.n_obs, adata.n_obs):
            errors.append('spatial_connectivities has wrong shape')
        try:
            if not (graph != graph.T).nnz == 0:
                errors.append('spatial_connectivities not symmetric')
        except Exception:
            errors.append('spatial_connectivities symmetry check failed')
        data = getattr(graph, 'data', np.asarray(graph).ravel())
        if not np.isfinite(data).all():
            errors.append('spatial_connectivities has non-finite values')
        if getattr(graph, 'nnz', 0) == 0:
            errors.append('spatial_connectivities appears constant (all zeros)')
    if not errors:
        score = 1.0
    elif len(errors) == 1 and 'spatial_connectivities not symmetric' in errors:
        score = 0.5
    else:
        score = 0.0
    return StepResult('task7', score, checks, errors)


def evaluate_batch_effect_correction(adata: 'ad.AnnData') -> Dict[str, Any]:
    try:
        from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
    except Exception:
        return _build_scib_like_fallback()
    use_rep = 'X_scVI' if 'X_scVI' in adata.obsm else 'X_pca_harmony' if 'X_pca_harmony' in adata.obsm else 'X_pca' if 'X_pca' in adata.obsm else None
    if use_rep is None or 'sample' not in adata.obs or 'cell_type' not in adata.obs:
        return _build_scib_like_fallback()
    try:
        benchmarker = Benchmarker(
            adata,
            batch_key='sample',
            label_key='cell_type',
            bio_conservation_metrics=BioConservation(),
            batch_correction_metrics=BatchCorrection(),
            embedding_obsm_keys=[use_rep],
            n_jobs=1,
        )
        benchmarker.benchmark()
        results = benchmarker.get_results(min_max_scale=False)
        total = float(results['Total'].values[0])
        table = {str(column): float(results[column].values[0]) for column in results.columns if column != 'Metric Type'}
        return {'table': table, 'aggregate_total': total, 'source': use_rep}
    except Exception:
        return _build_scib_like_fallback()


def evaluate_clustering(adata: 'ad.AnnData', include_spatial_moran: bool) -> Dict[str, Any]:
    try:
        from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, confusion_matrix, fowlkes_mallows_score, homogeneity_completeness_v_measure, normalized_mutual_info_score
    except Exception as exc:
        raise RuntimeError('scikit-learn is required for clustering evaluation') from exc
    cluster_key = 'cluster' if 'cluster' in adata.obs else 'leiden' if 'leiden' in adata.obs else None
    if cluster_key is None or 'cell_type' not in adata.obs:
        metric_names = ['ARI', 'AMI', 'NMI', 'Homogeneity', 'Completeness', 'V-measure', 'FMI', 'Purity']
        if include_spatial_moran:
            metric_names.append('Global_Moran_I')
        return {'metrics': {name: 0.0 for name in metric_names}, 'mapping': {}, 'purity_by_cluster': {}, 'confusion': {}}
    cluster = np.asarray(adata.obs[cluster_key]).astype(str)
    celltype = np.asarray(adata.obs['cell_type']).astype(str)
    mask = (celltype != 'nan') & (celltype != 'None') & (celltype != 'NA') & (celltype != '')
    cluster = cluster[mask]
    celltype = celltype[mask]
    if cluster.size == 0:
        raise ValueError('No valid cells left after filtering for clustering evaluation')
    ari = float(adjusted_rand_score(celltype, cluster))
    ami = float(adjusted_mutual_info_score(celltype, cluster))
    nmi = float(normalized_mutual_info_score(celltype, cluster))
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(celltype, cluster)
    fmi = float(fowlkes_mallows_score(celltype, cluster))
    mapping, purity_by_cluster = _majority_mapping(cluster, celltype)
    overall_purity = 0.0
    for cluster_label in np.unique(cluster):
        label_mask = cluster == cluster_label
        overall_purity += label_mask.sum() * purity_by_cluster[str(cluster_label)]
    overall_purity /= cluster.size
    celltype_order = sorted(np.unique(celltype).tolist())
    confusion = confusion_matrix(celltype, cluster, labels=celltype_order)
    metrics = {
        'ARI': ari,
        'AMI': ami,
        'NMI': nmi,
        'Homogeneity': float(homogeneity),
        'Completeness': float(completeness),
        'V-measure': float(v_measure),
        'FMI': fmi,
        'Purity': float(overall_purity),
    }
    if include_spatial_moran:
        metrics['Global_Moran_I'] = 0.0
        spatial_graph = adata.obsp.get('spatial_connectivities')
        if spatial_graph is not None:
            A = spatial_graph[mask][:, mask]
            labels = pd.Categorical(cluster)
            codes = labels.codes.astype(float)
            row_sums = np.asarray(A.sum(axis=1)).ravel().astype(float)
            row_sums[row_sums == 0] = 1.0
            W = A.multiply(1.0 / row_sums[:, None])
            z = codes - codes.mean()
            denom = float(np.dot(z, z))
            if denom != 0.0:
                metrics['Global_Moran_I'] = float(z @ (W @ z) / denom)
    return {
        'metrics': metrics,
        'mapping': mapping,
        'purity_by_cluster': purity_by_cluster,
        'confusion': {'celltype_order': celltype_order, 'matrix': confusion.tolist()},
    }


def get_top_markers_dict(adata: 'ad.AnnData', n_top: int = 10) -> Dict[str, List[str]]:
    try:
        ranking = adata.uns['rank_genes_groups']
        groups = ranking['names'].dtype.names
        return {group: list(ranking['names'][group][:n_top]) for group in groups}
    except Exception:
        return {}


def evaluate_marker_overlap_by_celltype(predicted: Dict[str, Iterable[str]], reference: Dict[str, Iterable[str]]) -> Dict[str, float]:
    shared = sorted(set(predicted.keys()) & set(reference.keys()))
    if not shared:
        return {key: 0.0 for key in ['jaccard', 'overlap_coef', 'precision(A_in_B)', 'recall(B_covered_by_A)', 'f1']}
    rows = [_overlap_metrics(_normalize_gene_set(predicted[cell_type]), _normalize_gene_set(reference[cell_type])) for cell_type in shared]
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def evaluate_svg_overlap(adata: 'ad.AnnData', svg_references: Dict[str, str], n_top: int = 50) -> Dict[str, float]:
    svg_global = adata.uns.get('svg_global')
    if svg_global is None:
        return {key: 0.0 for key in ['jaccard', 'overlap_coef', 'precision(A_in_B)', 'recall(B_covered_by_A)', 'f1']}
    predicted = list(svg_global)[:n_top]
    metrics_list = []
    for _, ref_path in (svg_references or {}).items():
        if ref_path and os.path.exists(ref_path):
            reference = list(_load_pickle(ref_path))[:n_top]
            metrics_list.append(_overlap_metrics(_normalize_gene_set(predicted), _normalize_gene_set(reference)))
    if not metrics_list:
        return {key: 0.0 for key in ['jaccard', 'overlap_coef', 'precision(A_in_B)', 'recall(B_covered_by_A)', 'f1']}
    return {key: float(np.mean([row[key] for row in metrics_list])) for key in metrics_list[0]}


def evaluate_trajectory(reference_path: str, adata: 'ad.AnnData') -> float:
    if not reference_path or not os.path.exists(reference_path):
        return 0.0
    try:
        import scib
    except Exception:
        return 0.0
    reference = _read_h5ad(reference_path)
    pred, gold = _align_on_obs_names(adata, reference)
    if 'X_pca' in pred.obsm and 'connectivities' not in pred.obsp:
        sc.pp.neighbors(pred, use_rep='X_pca')
    try:
        return float(scib.me.trajectory_conservation(gold, pred, label_key='cell_type'))
    except Exception:
        return 0.0


def evaluate_nhood_enrichment(adata: 'ad.AnnData', cluster_key: str = 'cell_type', pass_score: float = 0.70) -> StepResult:
    errors: List[str] = []
    checks: Dict[str, Any] = {}
    try:
        import squidpy as sq
        from scipy.stats import spearmanr
    except Exception:
        return StepResult('task11', False, {'score': 0.0}, ['squidpy and scipy are required for neighborhood enrichment evaluation'])
    ref_key = f'{cluster_key}_nhood_enrichment'
    blob = adata.uns.get(ref_key, {}) if isinstance(adata.uns.get(ref_key, {}), dict) else {}
    agent_zscore = blob.get('zscore')
    agent_count = blob.get('count')
    if agent_zscore is None:
        errors.append('Missing agent neighborhood enrichment zscore')
    if agent_count is None:
        errors.append('Missing agent neighborhood enrichment count')
    if cluster_key not in adata.obs:
        errors.append(f"Missing obs['{cluster_key}']")
        return StepResult('task11', False, {'score': 0.0}, errors)
    if agent_zscore is not None:
        agent_zscore = np.asarray(agent_zscore)
        checks['agent_zscore_shape'] = list(agent_zscore.shape)
    if agent_count is not None:
        agent_count = np.asarray(agent_count)
        checks['agent_count_shape'] = list(agent_count.shape)
    gold_z = None
    gold_c = None
    try:
        ad_ref = adata.copy()
        ad_ref.obs[cluster_key] = pd.Categorical(ad_ref.obs[cluster_key])
        sq.gr.spatial_neighbors(ad_ref, coord_type='generic', spatial_key='spatial', n_neighs=8)
        sq.gr.nhood_enrichment(ad_ref, cluster_key=cluster_key, n_perms=1000, n_jobs=None)
        gold_blob = ad_ref.uns[f'{cluster_key}_nhood_enrichment']
        gold_z = np.asarray(gold_blob['zscore'])
        gold_c = np.asarray(gold_blob['count'])
    except Exception as exc:
        errors.append(f'Reference computation failed: {type(exc).__name__}: {exc}')
    score = 0.0
    if gold_z is not None and gold_c is not None and agent_zscore is not None and agent_count is not None:
        def upper_tri_vec(matrix: np.ndarray) -> np.ndarray:
            idx = np.triu_indices(matrix.shape[0], k=0)
            return matrix[idx]
        def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
            corr = spearmanr(a, b).correlation
            return float(0.0 if corr is None or not np.isfinite(corr) else corr)
        az = upper_tri_vec(agent_zscore)
        gz = upper_tri_vec(gold_z)
        ac = upper_tri_vec(agent_count)
        gc = upper_tri_vec(gold_c)
        z_spear = safe_spearman(az, gz)
        c_spear = safe_spearman(ac, gc)
        z_rmse = float(np.sqrt(np.mean((az - gz) ** 2)))
        c_rmse = float(np.sqrt(np.mean((ac - gc) ** 2)))
        topk = min(20, az.size)
        ia = np.argpartition(np.abs(az), -topk)[-topk:]
        ib = np.argpartition(np.abs(gz), -topk)[-topk:]
        topk_jaccard = float(len(set(map(int, ia)) & set(map(int, ib))) / max(1, len(set(map(int, ia)) | set(map(int, ib)))))
        similarity = 0.35 * ((z_spear + 1.0) / 2.0) + 0.15 * (1.0 / (1.0 + z_rmse)) + 0.20 * ((c_spear + 1.0) / 2.0) + 0.10 * (1.0 / (1.0 + c_rmse)) + 0.20 * topk_jaccard
        penalty = 0.30 if any('Reference computation failed' in err for err in errors) else 0.0
        score = float(np.clip(similarity - penalty, 0.0, 1.0))
        checks.update({
            'metric_z_spearman': z_spear,
            'metric_count_spearman': c_spear,
            'metric_z_rmse': z_rmse,
            'metric_count_rmse': c_rmse,
            'metric_topk_jaccard_absz': topk_jaccard,
            'score': score,
        })
    else:
        checks['score'] = 0.0
    return StepResult('task11', bool(checks.get('score', 0.0) >= pass_score and not any('Reference computation failed' in err for err in errors)), checks, errors)


_CONTROL_LABELS = frozenset({'control', 'ctrl', 'nt', 'non-targeting', 'non_targeting'})


def _as_dense(x: Any) -> np.ndarray:
    if hasattr(x, 'toarray'):
        return x.toarray()
    return np.asarray(x)


def check_perturbation_data(adata: 'ad.AnnData') -> StepResult:
    """Validate perturbation data: obs['pert'], obs['cell_type'], control cells."""
    errors: List[str] = []
    checks: Dict[str, Any] = {'n_obs': int(adata.n_obs), 'n_vars': int(adata.n_vars)}
    if 'pert' not in adata.obs:
        errors.append("Missing obs['pert']")
    if 'cell_type' not in adata.obs:
        errors.append("Missing obs['cell_type']")
    if 'pert' in adata.obs:
        perts = adata.obs['pert'].astype(str)
        n_unique = int(perts.nunique())
        checks['n_perturbations'] = n_unique
        is_ctrl = perts.str.lower().isin(_CONTROL_LABELS)
        n_ctrl = int(is_ctrl.sum())
        checks['n_control_cells'] = n_ctrl
        n_pert = int((~is_ctrl).sum())
        checks['n_perturbed_cells'] = n_pert
        if n_ctrl == 0:
            errors.append("No control cells found (expected pert labels like 'ctrl', 'control', 'nt')")
        if n_pert == 0:
            errors.append('No perturbed cells found')
    if adata.n_obs <= 0 or adata.n_vars <= 0:
        errors.append('AnnData has non-positive dimensions')
    return StepResult('task1', len(errors) == 0, checks, errors)


def check_prediction_format(adata: 'ad.AnnData') -> StepResult:
    """Validate layers['X_pred'] exists with correct shape, and obs['split'] labels."""
    errors: List[str] = []
    checks: Dict[str, Any] = {}
    if 'X_pred' not in adata.layers:
        errors.append("Missing layers['X_pred'] — store predicted expression here")
        return StepResult('task2', False, checks, errors)
    X_pred = _as_dense(adata.layers['X_pred'])
    checks['X_pred_shape'] = list(X_pred.shape)
    checks['X_pred_finite'] = bool(np.isfinite(X_pred).all())
    if X_pred.shape != (adata.n_obs, adata.n_vars):
        errors.append(f"layers['X_pred'] shape {list(X_pred.shape)} != expected [{adata.n_obs}, {adata.n_vars}]")
    if not np.isfinite(X_pred).all():
        errors.append("layers['X_pred'] contains non-finite values")
    if 'split' not in adata.obs:
        errors.append("Missing obs['split'] — must contain 'train'/'test' labels to identify held-out perturbations")
    else:
        splits = sorted(adata.obs['split'].astype(str).unique().tolist())
        checks['split_labels'] = splits
        if 'test' not in splits:
            errors.append("obs['split'] must contain 'test' labels")
    return StepResult('task2', len(errors) == 0, checks, errors)


def evaluate_perturbation_metrics(adata: 'ad.AnnData', top_k: int = 20) -> Dict[str, Any]:
    """Compute per-condition perturbation prediction metrics.

    Metrics per condition:
        pearson_all, pearson_top{K}, r2, mse, mean_delta_err
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import r2_score

    perts = adata.obs['pert'].astype(str)
    is_ctrl = perts.str.lower().isin(_CONTROL_LABELS)

    ctrl_X = _as_dense(adata[is_ctrl].X)
    ctrl_mean = ctrl_X.mean(axis=0).ravel()

    # Identify test cells
    if 'split' in adata.obs:
        test_mask = adata.obs['split'].astype(str) == 'test'
    else:
        test_mask = ~is_ctrl

    test_adata = adata[test_mask]
    test_perts = test_adata.obs['pert'].astype(str)
    unique_test_perts = sorted(
        p for p in test_perts.unique() if p.lower() not in _CONTROL_LABELS
    )

    if len(unique_test_perts) == 0:
        return {
            'metrics': {f'pearson_top{top_k}': 0.0, 'pearson_all': 0.0, 'r2': 0.0, 'mse': 0.0, 'mean_delta_err': 0.0},
            'per_perturbation': {},
            'n_test_perturbations': 0,
        }

    X_pred = _as_dense(test_adata.layers['X_pred'])
    X_true = _as_dense(test_adata.X)

    rows: List[Dict[str, Any]] = []
    for pert in unique_test_perts:
        mask = (test_perts == pert).values
        pred = X_pred[mask].mean(axis=0).ravel()
        true = X_true[mask].mean(axis=0).ravel()

        # Top-K DE gene indices by |true - ctrl_mean|
        delta = np.abs(true - ctrl_mean)
        topk_idx = np.argsort(delta)[-top_k:][::-1]

        r_all, _ = pearsonr(pred, true)
        r_topk, _ = pearsonr(pred[topk_idx], true[topk_idx])

        rows.append({
            'perturbation': pert,
            'pearson_all': float(r_all),
            f'pearson_top{top_k}': float(r_topk),
            'r2': float(r2_score(true, pred)),
            'mse': float(np.mean((pred - true) ** 2)),
            'mean_delta_err': float(np.mean(np.abs((pred - ctrl_mean) - (true - ctrl_mean)))),
        })

    df = pd.DataFrame(rows).set_index('perturbation')
    avg_metrics = {col: float(df[col].mean()) for col in df.columns}

    return {
        'metrics': avg_metrics,
        'per_perturbation': {idx: row.to_dict() for idx, row in df.iterrows()},
        'n_test_perturbations': len(unique_test_perts),
    }


def evaluate_perturbation_prediction(metadata: Dict[str, Any], adata: 'ad.AnnData') -> Dict[str, Any]:
    """Orchestrate the perturbation-prediction benchmark (task1–task3)."""
    selected: Optional[List[str]] = metadata.get('tasks')
    top_k: int = int(metadata.get('topK', 20))
    results: Dict[str, Any] = {}

    if _task_enabled('task1', selected):
        results['task1'] = asdict(check_perturbation_data(adata))
    if _task_enabled('task2', selected):
        results['task2'] = asdict(check_prediction_format(adata))
    if _task_enabled('task3', selected):
        try:
            results['task3'] = evaluate_perturbation_metrics(adata, top_k=top_k)
        except Exception as exc:
            results['task3'] = {
                'metrics': {f'pearson_top{top_k}': 0.0, 'pearson_all': 0.0, 'r2': 0.0, 'mse': 0.0, 'mean_delta_err': 0.0},
                'per_perturbation': {},
                'n_test_perturbations': 0,
                'error': str(exc),
            }

    score_summary: Dict[str, float] = {}
    for task_id in ['task1', 'task2']:
        if task_id in results:
            score_summary[task_id] = 1.0 if results[task_id].get('passed', False) else 0.0
    if 'task3' in results:
        metrics = results['task3'].get('metrics', {})
        # Include correlation and R² in score_summary (interpretable as quality scores).
        # MSE and mean_delta_err are error metrics kept in detailed results only.
        for key in ['pearson_all', f'pearson_top{top_k}', 'r2']:
            if key in metrics:
                score_summary[f'task3_{key}'] = float(metrics[key])

    return {'results': results, 'score_summary': score_summary}


def load_submission(metadata: Dict[str, Any]) -> 'ad.AnnData':
    if metadata['mode'] == 'h5ad':
        return _read_h5ad(metadata['submissionPath'])
    namespace: Dict[str, Any] = {
        '__name__': '__main__',
        'path': metadata['datasetPath'],
        'dataset_path': metadata['datasetPath'],
        'DATASET_PATH': metadata['datasetPath'],
        'np': np,
        'pd': pd,
        'sc': sc,
    }
    with open(metadata['submissionPath'], 'r', encoding='utf-8') as handle:
        source = handle.read()
    exec(compile(source, metadata['submissionPath'], 'exec'), namespace, namespace)
    if 'adata' not in namespace or not isinstance(namespace['adata'], ad.AnnData):
        raise RuntimeError("Submission must define 'adata' as an AnnData object.")
    return namespace['adata']


def _task_enabled(task_id: str, selected: Optional[List[str]]) -> bool:
    """Return True if *task_id* should be evaluated.  None means 'all'."""
    return selected is None or task_id in selected


def evaluate_single_cell(metadata: Dict[str, Any], adata: 'ad.AnnData') -> Dict[str, Any]:
    selected: Optional[List[str]] = metadata.get('tasks')  # None ⇒ run all
    source_data = _read_h5ad(metadata['datasetPath'])
    reference_markers = pd.read_pickle(metadata['markersPath']) if metadata.get('markersPath') and os.path.exists(metadata['markersPath']) else {}
    results: Dict[str, Any] = {}
    step_checks = [
        ('task1', lambda: check_basic_load_single_cell(adata)),
        ('task2', lambda: check_qc_metrics(adata, 'task2')),
        ('task3', lambda: check_filter(adata, source_data, 'task3')),
        ('task4', lambda: check_doublet(adata)),
        ('task5', lambda: check_normalize_log1p(adata, 'task5')),
        ('task6', lambda: check_hvg_pca(adata, tuple(metadata.get('hvgRange', [1, 3000])), 'task6', True)),
        ('task7', lambda: check_neighbors_umap(adata, 'task7')),
    ]
    for task_id, fn in step_checks:
        if _task_enabled(task_id, selected):
            item = fn()
            results[item.step_id] = asdict(item)
    if _task_enabled('task8', selected):
        results['task8'] = evaluate_batch_effect_correction(adata)
    if _task_enabled('task9', selected):
        results['task9'] = evaluate_clustering(adata, include_spatial_moran=False)
    if _task_enabled('task10', selected):
        results['task10'] = evaluate_marker_overlap_by_celltype(get_top_markers_dict(adata), reference_markers) if reference_markers else {key: 0.0 for key in ['jaccard', 'overlap_coef', 'precision(A_in_B)', 'recall(B_covered_by_A)', 'f1']}
    if _task_enabled('task11', selected):
        results['task11'] = {'trajectory_conservation': evaluate_trajectory(metadata.get('trajectoryPath', ''), adata)}
    score_summary: Dict[str, float] = {}
    for task_id in ['task1', 'task2', 'task3', 'task4', 'task5', 'task6', 'task7']:
        if task_id in results:
            score_summary[task_id] = 1.0 if results[task_id]['passed'] else 0.0
    if 'task8' in results:
        score_summary['task8_total'] = float(results['task8'].get('aggregate_total', 0.0))
    if 'task9' in results:
        for metric_name, value in results['task9']['metrics'].items():
            score_summary[f'task9_{metric_name}'] = float(value)
    if 'task10' in results:
        for metric_name, value in results['task10'].items():
            score_summary[f'task10_{metric_name}'] = float(0.0 if pd.isna(value) else value)
    if 'task11' in results:
        score_summary['task11'] = float(results['task11']['trajectory_conservation'])
    return {'results': results, 'score_summary': score_summary}


def evaluate_spatial(metadata: Dict[str, Any], adata: 'ad.AnnData') -> Dict[str, Any]:
    selected: Optional[List[str]] = metadata.get('tasks')  # None ⇒ run all
    source_data = _read_h5ad(metadata['datasetPath'])
    results: Dict[str, Any] = {}
    step_checks = [
        ('task1', lambda: check_basic_load_spatial(adata)),
        ('task2', lambda: check_qc_metrics(adata, 'task2')),
        ('task3', lambda: check_filter(adata, source_data, 'task3')),
        ('task4', lambda: check_normalize_log1p(adata, 'task4')),
        ('task5', lambda: check_hvg_pca(adata, tuple(metadata.get('hvgRange', [1, 30000])), 'task5', False)),
        ('task6', lambda: check_neighbors_umap(adata, 'task6')),
        ('task7', lambda: check_spatial_graph(adata)),
    ]
    for task_id, fn in step_checks:
        if _task_enabled(task_id, selected):
            item = fn()
            results[item.step_id] = asdict(item)
    if _task_enabled('task8', selected):
        results['task8'] = evaluate_batch_effect_correction(adata)
    if _task_enabled('task9', selected):
        results['task9'] = evaluate_clustering(adata, include_spatial_moran=True)
    if _task_enabled('task10', selected):
        results['task10'] = evaluate_svg_overlap(adata, metadata.get('svgReferences', {}))
    if _task_enabled('task11', selected):
        results['task11'] = asdict(evaluate_nhood_enrichment(adata, cluster_key='cell_type'))
    score_summary: Dict[str, float] = {}
    for task_id in ['task1', 'task2', 'task3', 'task4', 'task5', 'task6']:
        if task_id in results:
            score_summary[task_id] = 1.0 if results[task_id]['passed'] else 0.0
    if 'task7' in results:
        score_summary['task7'] = float(results['task7']['passed'])
    if 'task8' in results:
        score_summary['task8_total'] = float(results['task8'].get('aggregate_total', 0.0))
    if 'task9' in results:
        for metric_name, value in results['task9']['metrics'].items():
            score_summary[f'task9_{metric_name}'] = float(value)
    if 'task10' in results:
        for metric_name, value in results['task10'].items():
            score_summary[f'task10_{metric_name}'] = float(0.0 if pd.isna(value) else value)
    if 'task11' in results:
        score_summary['task11'] = 1.0 if results['task11']['passed'] else 0.0
    return {'results': results, 'score_summary': score_summary}


def evaluate_submission(metadata: Dict[str, Any]) -> Dict[str, Any]:
    adata = load_submission(metadata)
    benchmark = metadata.get('benchmark', 'single-cell','perturbation-prediction')
    if benchmark == 'spatial':
        payload = evaluate_spatial(metadata, adata)
    elif benchmark == 'perturbation-prediction':
        payload = evaluate_perturbation_prediction(metadata, adata)
    elif benchmark == 'single-cell':
        payload = evaluate_single_cell(metadata, adata)
    overall_average = float(np.mean(list(payload['score_summary'].values()))) if payload['score_summary'] else 0.0
    result: Dict[str, Any] = {
        'status': 'completed',
        'message': 'Evaluation complete.',
        'dataset': metadata['datasetLabel'],
        'benchmark': benchmark,
        'results': payload['results'],
        'score_summary': payload['score_summary'],
        'overall_average': overall_average,
    }
    if metadata.get('tasks'):
        result['evaluated_tasks'] = metadata['tasks']
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate a code or h5ad submission against the benchmark pipeline.')
    parser.add_argument('--config', required=True, help='Path to metadata JSON')
    args = parser.parse_args()
    metadata = _read_json(args.config)
    try:
        result = evaluate_submission(metadata)
    except Exception as exc:
        result = {'status': 'failed', 'message': str(exc), 'traceback': traceback.format_exc(), 'score_summary': {}, 'overall_average': 0.0}
    _write_json(metadata['outputPath'], result)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
