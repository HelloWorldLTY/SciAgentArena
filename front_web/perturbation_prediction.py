import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# ==============================
# Perturbation prediction pipeline
# - adata.obs['pert'] contains perturbation target labels
# - test data contains perturbations not seen in training
# ==============================

def _as_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def parse_pert_label(pert_label):
    """
    Parse a perturbation label into a list of target genes.
    Supports labels like 'GENE', 'GENE1+GENE2', 'ctrl', 'control'.
    """
    if pert_label is None:
        return []
    s = str(pert_label).strip()
    if s == "":
        return []
    if s.lower() in {"control", "ctrl", "nt", "non-targeting", "non_targeting"}:
        return []
    if "+" in s:
        return [p.strip() for p in s.split("+") if p.strip()]
    return [s]


def build_gene_embedding(adata_ctrl, n_components=50):
    """
    Build gene embedding using control cells only.
    Returns dict: gene -> embedding vector.
    """
    # Use genes x cells matrix for gene-gene structure
    X = _as_dense(adata_ctrl.X)
    # Center per gene to emphasize coexpression structure
    X = X - X.mean(axis=0, keepdims=True)
    # SVD on cells x genes; use transpose to embed genes
    # We use TruncatedSVD for efficiency with sparse data.
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    # Fit on cells x genes to get components over genes
    svd.fit(X)
    # Components shape: n_components x n_genes
    gene_embed = svd.components_.T  # n_genes x n_components
    return {g: gene_embed[i, :] for i, g in enumerate(adata_ctrl.var_names)}


def make_pert_embedding(pert_label, gene_embed, n_components=50):
    genes = parse_pert_label(pert_label)
    if len(genes) == 0:
        return np.zeros(n_components, dtype=np.float32)
    vecs = []
    for g in genes:
        if g in gene_embed:
            vecs.append(gene_embed[g])
    if len(vecs) == 0:
        return np.zeros(n_components, dtype=np.float32)
    return np.mean(np.vstack(vecs), axis=0)


def build_features(adata, gene_embed, celltype_key="cell_type", n_components=50):
    # Cell type one-hot
    if celltype_key not in adata.obs.columns:
        raise ValueError(f"Missing cell type column: {celltype_key}")
    ct = adata.obs[celltype_key].astype(str).to_numpy().reshape(-1, 1)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ct_feat = ohe.fit_transform(ct)

    # Perturbation embedding
    pert_labels = adata.obs["pert"].astype(str).tolist()
    pert_feat = np.vstack([
        make_pert_embedding(p, gene_embed, n_components=n_components)
        for p in pert_labels
    ])

    X_feat = np.hstack([ct_feat, pert_feat])
    return X_feat, ohe


def build_features_with_ohe(adata, gene_embed, ohe, celltype_key="cell_type", n_components=50):
    ct = adata.obs[celltype_key].astype(str).to_numpy().reshape(-1, 1)
    ct_feat = ohe.transform(ct)
    pert_labels = adata.obs["pert"].astype(str).tolist()
    pert_feat = np.vstack([
        make_pert_embedding(p, gene_embed, n_components=n_components)
        for p in pert_labels
    ])
    return np.hstack([ct_feat, pert_feat])


def compute_baseline_by_celltype(adata, celltype_key="cell_type"):
    """
    Mean control expression per cell type.
    Uses control cells where pert parses to empty.
    """
    perts = adata.obs["pert"].astype(str).tolist()
    is_ctrl = [len(parse_pert_label(p)) == 0 for p in perts]
    ctrl = adata[is_ctrl].copy()
    if ctrl.n_obs == 0:
        raise ValueError("No control cells found. Use pert labels like 'control' or 'ctrl'.")
    X_ctrl = _as_dense(ctrl.X)
    ctrl.obs["_ct"] = ctrl.obs[celltype_key].astype(str)
    baseline = {}
    for ct in ctrl.obs["_ct"].unique():
        idx = np.where(ctrl.obs["_ct"] == ct)[0]
        baseline[ct] = X_ctrl[idx].mean(axis=0)
    return baseline


def make_targets(adata, baseline, celltype_key="cell_type"):
    X = _as_dense(adata.X)
    cts = adata.obs[celltype_key].astype(str).tolist()
    base = np.vstack([baseline[ct] for ct in cts])
    # Predict delta expression from baseline
    return X - base, base


def split_by_pert(adata, test_perts=None, test_frac=0.2, random_state=0):
    perts = adata.obs["pert"].astype(str)
    unique_perts = sorted(perts.unique())
    if test_perts is None:
        rng = np.random.default_rng(random_state)
        n_test = max(1, int(len(unique_perts) * test_frac))
        test_perts = rng.choice(unique_perts, size=n_test, replace=False).tolist()
    test_mask = perts.isin(test_perts).to_numpy()
    train = adata[~test_mask].copy()
    test = adata[test_mask].copy()
    return train, test, test_perts


def train_model(adata, n_components=50, celltype_key="cell_type"):
    # Restrict to control cells for gene embedding
    perts = adata.obs["pert"].astype(str).tolist()
    is_ctrl = [len(parse_pert_label(p)) == 0 for p in perts]
    adata_ctrl = adata[is_ctrl].copy()
    gene_embed = build_gene_embedding(adata_ctrl, n_components=n_components)

    baseline = compute_baseline_by_celltype(adata, celltype_key=celltype_key)
    X_feat, ohe = build_features(adata, gene_embed, celltype_key=celltype_key, n_components=n_components)
    Y_delta, _ = make_targets(adata, baseline, celltype_key=celltype_key)

    # Multi-output ridge regression
    model = Ridge(alpha=1.0, random_state=0)
    model.fit(X_feat, Y_delta)

    return {
        "model": model,
        "gene_embed": gene_embed,
        "ohe": ohe,
        "baseline": baseline,
        "celltype_key": celltype_key,
        "n_components": n_components,
    }


def predict(model_pack, adata):
    model = model_pack["model"]
    gene_embed = model_pack["gene_embed"]
    ohe = model_pack["ohe"]
    baseline = model_pack["baseline"]
    celltype_key = model_pack["celltype_key"]
    n_components = model_pack["n_components"]

    X_feat = build_features_with_ohe(adata, gene_embed, ohe, celltype_key=celltype_key, n_components=n_components)
    Y_delta_pred = model.predict(X_feat)
    _, base = make_targets(adata, baseline, celltype_key=celltype_key)
    X_pred = base + Y_delta_pred
    return X_pred


def evaluate(model_pack, adata):
    X_pred = predict(model_pack, adata)
    X_true = _as_dense(adata.X)
    # Per-cell R^2 and overall R^2
    r2_cells = [r2_score(X_true[i], X_pred[i]) for i in range(X_true.shape[0])]
    r2_overall = r2_score(X_true, X_pred, multioutput="variance_weighted")
    return {
        "r2_overall": float(r2_overall),
        "r2_cells_mean": float(np.mean(r2_cells)),
    }


# ==============================
# Example usage
# ==============================
if __name__ == "__main__":
    # Load adata (expects preprocessed log1p data)
    # Update path to your dataset
    path = "your_data.h5ad"
    adata = sc.read_h5ad(path)

    # Ensure required columns exist
    if "pert" not in adata.obs.columns:
        raise ValueError("Expected adata.obs['pert']")
    if "cell_type" not in adata.obs.columns:
        raise ValueError("Expected adata.obs['cell_type']")

    # Split by perturbation targets (test has different targets)
    train_adata, test_adata, test_perts = split_by_pert(adata, test_perts=None, test_frac=0.2)
    print(f"Test perturbations: {test_perts}")

    # Train
    model_pack = train_model(train_adata, n_components=50, celltype_key="cell_type")

    # Test
    metrics = evaluate(model_pack, test_adata)
    print(metrics)

    # Predict full expression for test cells
    X_pred = predict(model_pack, test_adata)
    # Optionally store predictions
    test_adata.layers["X_pred"] = X_pred
    test_adata.write_h5ad("test_with_predictions.h5ad")
