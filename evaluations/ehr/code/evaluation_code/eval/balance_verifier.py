"""Programmatic balance verification for causal analysis cohorts.

Computes standardized mean difference (SMD) between treatment and control
groups from the agent's saved cohort file, both unweighted (pre-adjustment)
and IPW-weighted (post-adjustment) if weights are available.

This verifies the agent's self-reported balance diagnostics against
actual data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CovariateSMD:
    """SMD result for a single covariate."""

    name: str
    smd_before: float = 0.0  # unweighted SMD
    smd_after: float | None = None  # IPW-weighted SMD (None if no weights)
    n_treated: int = 0
    n_control: int = 0
    n_missing: int = 0


@dataclass
class BalanceReport:
    """Full balance verification report."""

    covariates: list[CovariateSMD] = field(default_factory=list)
    n_treated: int = 0
    n_control: int = 0
    has_weights: bool = False

    @property
    def max_smd_before(self) -> float:
        if not self.covariates:
            return 0.0
        return max(c.smd_before for c in self.covariates)

    @property
    def max_smd_after(self) -> float | None:
        if not self.has_weights or not self.covariates:
            return None
        smds = [c.smd_after for c in self.covariates if c.smd_after is not None]
        return max(smds) if smds else None

    @property
    def imbalanced_before(self) -> list[str]:
        """Covariates with unweighted SMD > 0.1."""
        return [c.name for c in self.covariates if c.smd_before > 0.1]

    @property
    def imbalanced_after(self) -> list[str]:
        """Covariates with weighted SMD > 0.1 (if weights available)."""
        if not self.has_weights:
            return []
        return [
            c.name for c in self.covariates
            if c.smd_after is not None and c.smd_after > 0.1
        ]

    def summary(self) -> str:
        lines = [
            f"Balance Report: {self.n_treated} treated, {self.n_control} control",
            f"Covariates checked: {len(self.covariates)}",
            f"Max unweighted SMD: {self.max_smd_before:.3f}",
        ]
        if self.has_weights:
            lines.append(f"Max weighted SMD: {self.max_smd_after:.3f}")
            lines.append(f"Imbalanced after weighting: {self.imbalanced_after}")
        lines.append(f"Imbalanced before: {self.imbalanced_before}")
        return "\n".join(lines)


def compute_smd(
    treated: pd.Series,
    control: pd.Series,
    weights_treated: pd.Series | None = None,
    weights_control: pd.Series | None = None,
) -> float:
    """Compute standardized mean difference between two groups.

    For continuous variables:
        SMD = |mean_treated - mean_control| / sqrt((var_treated + var_control) / 2)

    For binary variables (0/1):
        SMD = |p_treated - p_control| / sqrt((p_treated*(1-p_treated) + p_control*(1-p_control)) / 2)

    If weights are provided, uses weighted means and variances.
    """
    treated = treated.dropna()
    control = control.dropna()

    if len(treated) == 0 or len(control) == 0:
        return 0.0

    is_binary = set(treated.unique()) | set(control.unique()) <= {0, 1, 0.0, 1.0}

    if weights_treated is not None and weights_control is not None:
        wt = weights_treated.reindex(treated.index).fillna(1.0)
        wc = weights_control.reindex(control.index).fillna(1.0)
        mean_t = np.average(treated, weights=wt)
        mean_c = np.average(control, weights=wc)
        if is_binary:
            var_t = mean_t * (1 - mean_t)
            var_c = mean_c * (1 - mean_c)
        else:
            var_t = np.average((treated - mean_t) ** 2, weights=wt)
            var_c = np.average((control - mean_c) ** 2, weights=wc)
    else:
        mean_t = treated.mean()
        mean_c = control.mean()
        if is_binary:
            var_t = mean_t * (1 - mean_t)
            var_c = mean_c * (1 - mean_c)
        else:
            var_t = treated.var(ddof=1)
            var_c = control.var(ddof=1)

    pooled_sd = np.sqrt((var_t + var_c) / 2)
    if pooled_sd < 1e-10:
        return 0.0

    return abs(mean_t - mean_c) / pooled_sd


def verify_balance(
    cohort_df: pd.DataFrame,
    treatment_col: str = "treatment",
    weight_col: str | None = None,
    exclude_cols: set[str] | None = None,
) -> BalanceReport:
    """Compute SMD for all numeric covariates in the cohort.

    Args:
        cohort_df: Analysis-ready DataFrame with treatment column and covariates.
        treatment_col: Name of the binary treatment column (0/1).
        weight_col: Name of IPW weight column. If present, computes weighted SMD.
        exclude_cols: Columns to skip (e.g. IDs, outcome).

    Returns:
        BalanceReport with per-covariate SMD results.
    """
    if exclude_cols is None:
        exclude_cols = set()

    # Standard columns to exclude from covariate checking
    skip = {"hadm_id", "subject_id", treatment_col, "outcome", "propensity_score"} | exclude_cols
    if weight_col:
        skip.add(weight_col)

    treated_mask = cohort_df[treatment_col] == 1
    control_mask = cohort_df[treatment_col] == 0

    report = BalanceReport(
        n_treated=int(treated_mask.sum()),
        n_control=int(control_mask.sum()),
        has_weights=weight_col is not None and weight_col in cohort_df.columns,
    )

    # Get weights if available
    wt_treated = None
    wt_control = None
    if report.has_weights:
        wt_treated = cohort_df.loc[treated_mask, weight_col]
        wt_control = cohort_df.loc[control_mask, weight_col]

    for col in cohort_df.columns:
        if col in skip:
            continue
        if not pd.api.types.is_numeric_dtype(cohort_df[col]):
            continue

        treated_vals = cohort_df.loc[treated_mask, col]
        control_vals = cohort_df.loc[control_mask, col]

        cov_result = CovariateSMD(
            name=col,
            n_treated=int(treated_vals.notna().sum()),
            n_control=int(control_vals.notna().sum()),
            n_missing=int(cohort_df[col].isna().sum()),
        )

        # Unweighted SMD
        cov_result.smd_before = compute_smd(treated_vals, control_vals)

        # Weighted SMD
        if report.has_weights:
            cov_result.smd_after = compute_smd(
                treated_vals, control_vals, wt_treated, wt_control
            )

        report.covariates.append(cov_result)

    return report


def load_and_verify(
    cohort_path: str | Path,
    treatment_col: str = "treatment",
) -> BalanceReport | None:
    """Load a cohort file and run balance verification.

    Auto-detects weight column (ipw_weight, weight, weights).
    Auto-detects treatment column (treatment, treated, treat).
    """
    path = Path(cohort_path)
    if not path.exists():
        return None

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Auto-detect treatment column
    treat_candidates = [treatment_col, "treatment", "treated", "treat"]
    for tc in treat_candidates:
        if tc in df.columns:
            treatment_col = tc
            break
    else:
        return None

    # Auto-detect weight column
    weight_col = None
    for wc in ["ipw_weight", "weight", "weights", "iptw", "iptw_weight"]:
        if wc in df.columns:
            weight_col = wc
            break

    return verify_balance(df, treatment_col, weight_col)
