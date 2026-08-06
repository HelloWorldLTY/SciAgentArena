"""Recompute causal effect estimates from the agent's saved cohort file.

Provides an independent ATE recomputation using IPW (Horvitz-Thompson)
from the cohort's treatment, outcome, and ipw_weight columns. This is
used by D7 to verify that the agent's reported estimate is internally
consistent with its own data — not that it matches a golden value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class EstimateReport:
    """Result of an independent ATE recomputation."""

    ate_ipw: float  # IPW (Horvitz-Thompson) ATE
    ate_naive: float  # unadjusted difference in means
    n_treated: int
    n_control: int
    outcome_rate_treated: float
    outcome_rate_control: float
    method: str = "ipw_horvitz_thompson"
    has_weights: bool = True


def recompute_ate(
    cohort_df: pd.DataFrame,
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    weight_col: str | None = None,
) -> EstimateReport | None:
    """Recompute ATE from the cohort DataFrame.

    Uses IPW (Horvitz-Thompson) estimator when weights are available:
        ATE = E[Y*W*T] / E[W*T]  -  E[Y*W*(1-T)] / E[W*(1-T)]

    Falls back to naive difference in means when no weights.

    Returns None if required columns are missing or data is degenerate.
    """
    # Check required columns
    if treatment_col not in cohort_df.columns:
        return None
    if outcome_col not in cohort_df.columns:
        return None

    df = cohort_df[[treatment_col, outcome_col]].copy()
    if weight_col and weight_col in cohort_df.columns:
        df["_w"] = cohort_df[weight_col].fillna(1.0)
        has_weights = True
    else:
        df["_w"] = 1.0
        has_weights = False

    # Drop rows with missing treatment or outcome
    df = df.dropna(subset=[treatment_col, outcome_col])
    if len(df) == 0:
        return None

    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    n_treated = len(treated)
    n_control = len(control)

    if n_treated == 0 or n_control == 0:
        return None

    # Naive (unadjusted) difference in means
    outcome_rate_treated = float(treated[outcome_col].mean())
    outcome_rate_control = float(control[outcome_col].mean())
    ate_naive = outcome_rate_treated - outcome_rate_control

    # IPW (Horvitz-Thompson) estimate
    if has_weights:
        w_t = treated["_w"].values
        y_t = treated[outcome_col].values
        w_c = control["_w"].values
        y_c = control[outcome_col].values

        # Weighted mean outcome in treated
        mu_1 = np.sum(y_t * w_t) / np.sum(w_t) if np.sum(w_t) > 0 else 0.0
        # Weighted mean outcome in control
        mu_0 = np.sum(y_c * w_c) / np.sum(w_c) if np.sum(w_c) > 0 else 0.0
        ate_ipw = float(mu_1 - mu_0)
    else:
        ate_ipw = ate_naive

    return EstimateReport(
        ate_ipw=ate_ipw,
        ate_naive=ate_naive,
        n_treated=n_treated,
        n_control=n_control,
        outcome_rate_treated=outcome_rate_treated,
        outcome_rate_control=outcome_rate_control,
        has_weights=has_weights,
    )


def load_and_recompute(
    cohort_path: str | Path,
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
) -> EstimateReport | None:
    """Load cohort file and recompute ATE.

    Auto-detects weight column (ipw_weight, weight, weights, iptw, iptw_weight).
    Auto-detects treatment/outcome columns with fallback names.
    """
    path = Path(cohort_path)
    if not path.exists():
        return None

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Auto-detect treatment column
    for tc in [treatment_col, "treatment", "treated", "treat"]:
        if tc in df.columns:
            treatment_col = tc
            break
    else:
        return None

    # Auto-detect outcome column
    for oc in [outcome_col, "outcome", "y", "result"]:
        if oc in df.columns:
            outcome_col = oc
            break
    else:
        return None

    # Auto-detect weight column
    weight_col = None
    for wc in ["ipw_weight", "weight", "weights", "iptw", "iptw_weight"]:
        if wc in df.columns:
            weight_col = wc
            break

    return recompute_ate(df, treatment_col, outcome_col, weight_col)
