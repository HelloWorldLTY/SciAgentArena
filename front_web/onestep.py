import numpy as np
import scanpy as sc

# Optional tools
try:
    import scrublet as scr
except Exception:  # optional dependency
    scr = None

# T1: Read dataset from `path` and map labels
adata = sc.read_h5ad(path)
batch_cols = [c for c in adata.obs.columns if "batch" in c.lower()]
if "sample" not in adata.obs.columns and batch_cols:
    adata.obs["sample"] = adata.obs[batch_cols[0]].astype(str)
celltype_col = None
for c in adata.obs.columns:
    cl = c.lower()
    if cl in {"cell_type", "celltype"} or "celltype" in cl:
        celltype_col = c
        break
if "cell_type" not in adata.obs.columns and celltype_col:
    adata.obs["cell_type"] = adata.obs[celltype_col].astype(str)

# T2: Gene category annotation + QC metrics
vn = adata.var_names.astype(str)
adata.var["mt"] = vn.str.startswith(("MT-", "Mt-"))
adata.var["ribo"] = vn.str.startswith(("RPS", "RPL"))
adata.var["hb"] = vn.str.match(r"^HB(?!P)", case=False)
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], log1p=True, inplace=True)

# T3: Basic scRNA-seq filtering
if "n_genes_by_counts" in adata.obs:
    adata = adata[adata.obs["n_genes_by_counts"] >= 200, :].copy()
if "pct_counts_mt" in adata.obs:
    adata = adata[adata.obs["pct_counts_mt"] < 20, :].copy()
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=200)

# T4: Doublet detection with Scrublet
if scr is not None:
    counts = adata.X
    scrub = scr.Scrublet(counts)
    dbl_score, dbl_pred = scrub.scrub_doublets()
    adata.obs["doublet_score"] = dbl_score
    adata.obs["predicted_doublet"] = dbl_pred

# T5: Preprocess counts
adata.layers["counts"] = adata.X.copy()
if "total_counts" not in adata.obs:
    adata.obs["total_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
target_sum = float(np.median(adata.obs["total_counts"]))
sc.pp.normalize_total(adata, target_sum=target_sum)
sc.pp.log1p(adata)

# T6: HVGs and PCA
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
adata = adata[:, adata.var["highly_variable"]].copy()
sc.tl.pca(adata, svd_solver="arpack")

# T7: Neighbors, Leiden, UMAP
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.leiden(adata, key_added="leiden")
sc.tl.umap(adata)

# T8: Batch correction (Harmony preferred)
if "sample" in adata.obs.columns:
    try:
        sc.external.pp.harmony_integrate(adata, key="sample")
        if "X_pca_harmony" in adata.obsm:
            adata.obsm["X_pca"] = adata.obsm["X_pca_harmony"]
    except Exception:
        try:
            sc.pp.combat(adata, key="sample")
            sc.tl.pca(adata, svd_solver="arpack")
        except Exception:
            pass

# T9: Best clustering method
sc.tl.leiden(adata, key_added="cluster")

# T10: Differential expression
if "cluster" in adata.obs.columns:
    sc.tl.rank_genes_groups(adata, groupby="cluster", method="wilcoxon")
