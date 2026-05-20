#!/usr/bin/env python3
"""
Evaluate computed eQTL calls against a reference eQTL set *per gene*.

This script is intentionally simple: it treats eQTLs as sets of (gene_id, variant_id)
pairs after optional filtering (e.g., p-value threshold or top-k per gene), then
computes per-gene classification metrics.

Expected inputs (CSV/TSV):
  - "pred" (your computed eQTLs): must include gene and variant columns.
  - "ref" (reference eQTL set): must include gene and variant columns.

Minimum required columns:
  - gene: one of {gene_id, gene, gene_name, Name, id} (or specify --gene-col)
  - variant: either variant_id (or specify --variant-col),
    OR chrom+pos+ref+alt so we can build a variant_id.

Outputs:
  - per-gene metrics CSV (default: eqtl_quality_per_gene.csv)
  - prints a small global summary to stdout

Examples:
  python evaluate_eqtl_quality.py --pred blood_eqtl_results.csv --ref gtex_blood_ref.csv
  python evaluate_eqtl_quality.py --pred blood_eqtl_results.csv --ref gtex_blood_ref.csv --pred-p 1e-5 --ref-p 1e-5
  python evaluate_eqtl_quality.py --pred blood_eqtl_results.csv --ref gtex_blood_ref.csv --pred-topk 100 --ref-topk 100
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional, Sequence

import pandas as pd


GENE_ID_CANDIDATES = ("gene_id", "gene", "gene_name", "Name", "id")
VARIANT_ID_CANDIDATES = ("variant_id", "variant", "snp", "rsid", "rs_id", "ID")
PVALUE_CANDIDATES = ("pvalue", "p_value", "p", "pval", "P")
BETA_CANDIDATES = ("beta", "slope", "effect", "effect_size", "es", "b")
CHR_CANDIDATES = ("chr", "chrom", "chromosome", "seqnames")
POS_CANDIDATES = ("pos", "position", "bp", "bp_position")
REF_CANDIDATES = ("ref", "reference", "allele0")
ALT_CANDIDATES = ("alt", "alternate", "allele1")


def _first_existing(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    by_lower = {str(c).lower(): str(c) for c in columns}
    for cand in candidates:
        hit = by_lower.get(cand.lower())
        if hit is not None:
            return hit
    return None


def _normalize_chrom(value: object) -> str:
    chrom = str(value).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    # common alternate encodings
    if chrom == "23":
        chrom = "X"
    elif chrom == "24":
        chrom = "Y"
    elif chrom in {"25", "M"}:
        chrom = "MT"
    return chrom


def _make_variant_id(df: pd.DataFrame) -> pd.Series:
    chr_col = _first_existing(df.columns, CHR_CANDIDATES)
    pos_col = _first_existing(df.columns, POS_CANDIDATES)
    ref_col = _first_existing(df.columns, REF_CANDIDATES)
    alt_col = _first_existing(df.columns, ALT_CANDIDATES)
    missing = [name for name, col in [("chr", chr_col), ("pos", pos_col), ("ref", ref_col), ("alt", alt_col)] if col is None]
    if missing:
        raise ValueError(
            "Could not determine variant identifiers. Provide --variant-col with a variant_id column, "
            f"or include columns for {', '.join(missing)} (chrom/pos/ref/alt)."
        )
    chrom = df[chr_col].map(_normalize_chrom)
    pos = pd.to_numeric(df[pos_col], errors="coerce").astype("Int64")
    ref = df[ref_col].astype(str)
    alt = df[alt_col].astype(str)
    return chrom.astype(str) + ":" + pos.astype(str) + ":" + ref + ":" + alt


def _load_table(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} file not found: {path}")
    # pandas can infer separators poorly if commas appear in quoted strings; assume CSV unless user supplies TSV.
    sep = "\t" if path.lower().endswith((".tsv", ".tab")) else ","
    df = pd.read_csv(path, sep=sep)
    if df.empty:
        raise ValueError(f"{name} table is empty: {path}")
    return df


def _filter_calls(
    df: pd.DataFrame,
    gene_col: str,
    variant_col: str,
    p_col: Optional[str],
    beta_col: Optional[str],
    p_thresh: Optional[float],
    topk: Optional[int],
) -> pd.DataFrame:
    cols = [gene_col, variant_col]
    if p_col:
        cols.append(p_col)
    if beta_col and beta_col not in cols:
        cols.append(beta_col)

    keep = df[cols].copy()
    keep[gene_col] = keep[gene_col].astype(str)
    keep[variant_col] = keep[variant_col].astype(str)

    if p_col is not None:
        keep[p_col] = pd.to_numeric(keep[p_col], errors="coerce")
    if beta_col is not None:
        keep[beta_col] = pd.to_numeric(keep[beta_col], errors="coerce")

    if p_thresh is not None:
        if p_col is None:
            raise ValueError("p-value threshold requested but no p-value column was found/provided.")
        keep = keep.loc[keep[p_col].notna() & (keep[p_col] <= float(p_thresh))].copy()

    if topk is not None:
        if topk <= 0:
            raise ValueError("--*-topk must be a positive integer.")
        if p_col is None:
            # no p-values: just take first k rows per gene after dropping duplicates
            keep = keep.drop_duplicates([gene_col, variant_col])
            keep["_rank"] = keep.groupby(gene_col).cumcount() + 1
            keep = keep.loc[keep["_rank"] <= int(topk)].drop(columns=["_rank"])
        else:
            # smallest p-values are "best"
            keep = keep.sort_values([gene_col, p_col, variant_col], ascending=[True, True, True])
            keep = keep.drop_duplicates([gene_col, variant_col])
            keep["_rank"] = keep.groupby(gene_col).cumcount() + 1
            keep = keep.loc[keep["_rank"] <= int(topk)].drop(columns=["_rank"])

    return keep.drop_duplicates([gene_col, variant_col])


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate eQTL quality against a reference set (per gene).")
    p.add_argument("--pred", required=True, help="Computed eQTL result file (CSV/TSV).")
    p.add_argument("--ref", required=True, help="Reference eQTL set file (CSV/TSV).")
    p.add_argument("--out", default="eqtl_quality_per_gene.csv", help="Output per-gene metrics CSV.")

    p.add_argument("--gene-col", default=None, help="Gene column name to use (overrides auto-detect).")
    p.add_argument("--variant-col", default=None, help="Variant column name to use (overrides auto-detect).")

    p.add_argument("--pred-p-col", default=None, help="P-value column in pred (overrides auto-detect).")
    p.add_argument("--ref-p-col", default=None, help="P-value column in ref (overrides auto-detect).")

    p.add_argument("--pred-beta-col", default=None, help="Beta/effect-size column in pred (overrides auto-detect).")
    p.add_argument("--ref-beta-col", default=None, help="Beta/effect-size column in ref (overrides auto-detect).")

    p.add_argument("--pred-p", type=float, default=None, help="Keep pred calls with p <= this value.")
    p.add_argument("--ref-p", type=float, default=None, help="Keep ref calls with p <= this value.")

    p.add_argument("--pred-topk", type=int, default=None, help="Keep only top-K pred variants per gene (by p if available).")
    p.add_argument("--ref-topk", type=int, default=None, help="Keep only top-K ref variants per gene (by p if available).")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    pred = _load_table(args.pred, "pred")
    ref = _load_table(args.ref, "ref")

    gene_col_pred = args.gene_col or _first_existing(pred.columns, GENE_ID_CANDIDATES)
    gene_col_ref = args.gene_col or _first_existing(ref.columns, GENE_ID_CANDIDATES)
    if gene_col_pred is None or gene_col_ref is None:
        raise ValueError("Could not detect gene column in pred/ref. Use --gene-col.")

    variant_col_pred = args.variant_col or _first_existing(pred.columns, VARIANT_ID_CANDIDATES)
    variant_col_ref = args.variant_col or _first_existing(ref.columns, VARIANT_ID_CANDIDATES)

    if variant_col_pred is None:
        pred = pred.copy()
        pred["variant_id"] = _make_variant_id(pred)
        variant_col_pred = "variant_id"
    if variant_col_ref is None:
        ref = ref.copy()
        ref["variant_id"] = _make_variant_id(ref)
        variant_col_ref = "variant_id"

    pcol_pred = args.pred_p_col or _first_existing(pred.columns, PVALUE_CANDIDATES)
    pcol_ref = args.ref_p_col or _first_existing(ref.columns, PVALUE_CANDIDATES)

    bcol_pred = args.pred_beta_col or _first_existing(pred.columns, BETA_CANDIDATES)
    bcol_ref = args.ref_beta_col or _first_existing(ref.columns, BETA_CANDIDATES)

    pred_calls = _filter_calls(pred, gene_col_pred, variant_col_pred, pcol_pred, bcol_pred, args.pred_p, args.pred_topk)
    ref_calls = _filter_calls(ref, gene_col_ref, variant_col_ref, pcol_ref, bcol_ref, args.ref_p, args.ref_topk)

    # Normalize column names for merge logic
    pred_calls = pred_calls.rename(columns={gene_col_pred: "gene_id", variant_col_pred: "variant_id"})
    ref_calls = ref_calls.rename(columns={gene_col_ref: "gene_id", variant_col_ref: "variant_id"})
    if bcol_pred is not None and bcol_pred in pred_calls.columns:
        pred_calls = pred_calls.rename(columns={bcol_pred: "beta_pred"})
    if bcol_ref is not None and bcol_ref in ref_calls.columns:
        ref_calls = ref_calls.rename(columns={bcol_ref: "beta_ref"})

    pred_by_gene = pred_calls.groupby("gene_id")["variant_id"].apply(lambda s: set(s.astype(str))).to_dict()
    ref_by_gene = ref_calls.groupby("gene_id")["variant_id"].apply(lambda s: set(s.astype(str))).to_dict()

    all_genes = sorted(set(pred_by_gene) | set(ref_by_gene))
    rows = []
    tp_total = fp_total = fn_total = 0

    beta_overlap = None
    if "beta_pred" in pred_calls.columns and "beta_ref" in ref_calls.columns:
        beta_overlap = pred_calls.merge(ref_calls, on=["gene_id", "variant_id"], how="inner")

    for gene in all_genes:
        pred_set = pred_by_gene.get(gene, set())
        ref_set = ref_by_gene.get(gene, set())
        tp = len(pred_set & ref_set)
        fp = len(pred_set - ref_set)
        fn = len(ref_set - pred_set)

        tp_total += tp
        fp_total += fp
        fn_total += fn

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
        jaccard = _safe_div(tp, tp + fp + fn)

        n_beta = float("nan")
        pearson_r = float("nan")
        spearman_rho = float("nan")
        if beta_overlap is not None:
            sub = beta_overlap.loc[beta_overlap["gene_id"] == gene, ["beta_pred", "beta_ref"]].dropna()
            n_beta = int(len(sub))
            if n_beta >= 2:
                pearson_r = float(sub["beta_pred"].corr(sub["beta_ref"], method="pearson"))
                spearman_rho = float(sub["beta_pred"].corr(sub["beta_ref"], method="spearman"))

        rows.append(
            {
                "gene_id": gene,
                "n_pred": len(pred_set),
                "n_ref": len(ref_set),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "jaccard": jaccard,
                "n_beta_overlap": n_beta,
                "beta_pearson_r": pearson_r,
                "beta_spearman_rho": spearman_rho,
            }
        )

    out = pd.DataFrame(rows).sort_values(["f1", "gene_id"], ascending=[False, True])
    out.to_csv(args.out, index=False)

    precision_micro = _safe_div(tp_total, tp_total + fp_total)
    recall_micro = _safe_div(tp_total, tp_total + fn_total)
    f1_micro = _safe_div(2 * tp_total, 2 * tp_total + fp_total + fn_total)

    genes_with_ref = sum(1 for g in all_genes if len(ref_by_gene.get(g, set())) > 0)
    genes_with_pred = sum(1 for g in all_genes if len(pred_by_gene.get(g, set())) > 0)

    print(f"Genes: {len(all_genes)} (with_ref={genes_with_ref}, with_pred={genes_with_pred})")
    print(f"Pairs: pred={len(pred_calls)}, ref={len(ref_calls)}")
    print(f"Micro precision={precision_micro:.4g} recall={recall_micro:.4g} f1={f1_micro:.4g}")
    if beta_overlap is not None:
        sub = beta_overlap[["beta_pred", "beta_ref"]].dropna()
        if len(sub) >= 2:
            r_all = float(sub["beta_pred"].corr(sub["beta_ref"], method="pearson"))
            rho_all = float(sub["beta_pred"].corr(sub["beta_ref"], method="spearman"))
            print(f"Beta correlation on overlaps (all genes): pearson_r={r_all:.4g} spearman_rho={rho_all:.4g} n={len(sub)}")
    print(f"Output: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
