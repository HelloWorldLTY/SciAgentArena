"""Post-hoc temporal leakage verifier.

Verifies that agent-extracted covariates come from pre-TREATMENT measurements
by re-querying the source MIMIC tables and checking measurement timestamps.

Temporal boundary:
  - Treated patients: covariate must be measured before first treatment dose.
    The treatment hasn't altered the patient's physiology yet, so the value
    reflects baseline state.
  - Control patients: covariate must be measured before admittime (time zero).
    Since no treatment exists, admittime is the study entry point. Any
    measurement at admission reflects baseline.

Approach:
  For each covariate with a timestamped source (chartevents, labevents, etc.):
  1. Load source measurements for cohort patients
  2. Determine per-patient temporal cutoff (treatment time or admittime)
  3. Split into pre-cutoff and post-cutoff
  4. If agent has a non-null value but only post-cutoff measurements exist → leakage
  5. Report per-covariate and per-patient violations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class CovariateSource:
    """Defines where a covariate comes from in MIMIC tables."""

    name: str
    table_path: str  # relative to data root, e.g. "hosp/labevents.csv.gz"
    time_col: str  # timestamp column, e.g. "charttime"
    id_col: str = "hadm_id"
    itemid_filter: list[int] = field(default_factory=list)  # filter by itemid
    itemid_col: str = "itemid"
    value_col: str = "valuenum"  # column containing the measurement value


@dataclass
class LeakageResult:
    """Result of temporal verification for one covariate."""

    covariate: str
    n_cohort: int = 0  # total patients in cohort
    n_has_measurements: int = 0  # patients with any measurements
    n_pre_only: int = 0  # patients with only pre-cutoff data
    n_post_only: int = 0  # patients with only post-cutoff data (LEAKAGE if agent has value)
    n_both: int = 0  # patients with both pre and post data
    n_no_measurements: int = 0  # patients with no measurements at all
    # Agent-specific (only available if agent cohort file has this column)
    n_agent_has_value: int = 0  # patients where agent has a non-null value
    n_leaked: int = 0  # patients where agent has value but only post-cutoff data exists
    # Value-matched leakage: agent's value matches a post-treatment measurement
    n_value_matched_pre: int = 0  # agent value matches a pre-treatment measurement
    n_value_matched_post: int = 0  # agent value matches a post-treatment measurement (LEAKAGE)
    n_value_matched_ambiguous: int = 0  # matches both or aggregated

    @property
    def leak_rate(self) -> float:
        total = self.n_leaked + self.n_value_matched_post
        denom = self.n_agent_has_value if self.n_agent_has_value > 0 else 1
        return total / denom

    @property
    def is_clean(self) -> bool:
        return self.n_leaked == 0 and self.n_value_matched_post == 0

    def summary(self) -> str:
        parts = []
        if self.n_leaked > 0:
            parts.append(f"no_pre_data={self.n_leaked}")
        if self.n_value_matched_post > 0:
            parts.append(f"value_matches_post={self.n_value_matched_post}")
        if self.n_agent_has_value > 0:
            leak_detail = ", ".join(parts) if parts else "clean"
            return (
                f"{self.covariate}: {leak_detail} "
                f"(of {self.n_agent_has_value} treated w/ values), "
                f"pre_only={self.n_pre_only}, post_only={self.n_post_only}, "
                f"both={self.n_both}, no_data={self.n_no_measurements}"
            )
        return (
            f"{self.covariate}: pre_only={self.n_pre_only}, "
            f"post_only={self.n_post_only}, both={self.n_both}, "
            f"no_data={self.n_no_measurements} [no agent values to check]"
        )


@dataclass
class VerificationReport:
    """Full verification report across all covariates."""

    results: list[LeakageResult] = field(default_factory=list)

    @property
    def total_leaked(self) -> int:
        return sum(r.n_leaked for r in self.results)

    @property
    def any_leakage(self) -> bool:
        return self.total_leaked > 0

    @property
    def covariates_with_leakage(self) -> list[str]:
        return [r.covariate for r in self.results if not r.is_clean]

    def summary(self) -> str:
        lines = ["Temporal Leakage Verification Report", "=" * 40]
        for r in self.results:
            flag = "LEAK" if not r.is_clean else "OK"
            lines.append(f"  [{flag}] {r.summary()}")
        lines.append(f"\nTotal leaked: {self.total_leaked}")
        return "\n".join(lines)


class TemporalVerifier:
    """Verifies covariates are extracted from pre-treatment measurements.

    Uses per-patient treatment time as the temporal boundary for treated
    patients, and admittime for controls.
    """

    def __init__(self, data_root: str | Path, admissions_path: str | None = None):
        self.data_root = Path(data_root)
        self._admissions: pd.DataFrame | None = None
        self._admissions_path = admissions_path

    def _load_admissions(self) -> pd.DataFrame:
        if self._admissions is not None:
            return self._admissions
        path = self._admissions_path or str(
            self.data_root / "physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz"
        )
        self._admissions = pd.read_csv(
            path,
            usecols=["hadm_id", "admittime"],
            parse_dates=["admittime"],
        )
        return self._admissions

    def _load_source_measurements(
        self,
        source: CovariateSource,
        cohort_hadm_ids: set[int],
        load_values: bool = False,
    ) -> pd.DataFrame:
        """Load timestamped measurements from source table for cohort patients."""
        table_path = self.data_root / "physionet.org/files/mimiciv/3.1" / source.table_path
        if not table_path.exists():
            raise FileNotFoundError(f"Source table not found: {table_path}")

        usecols = [source.id_col, source.time_col]
        if source.itemid_filter:
            usecols.append(source.itemid_col)
        if load_values and source.value_col:
            usecols.append(source.value_col)

        chunks = []
        for chunk in pd.read_csv(
            table_path,
            usecols=usecols,
            parse_dates=[source.time_col],
            chunksize=500_000,
        ):
            chunk = chunk[chunk[source.id_col].isin(cohort_hadm_ids)]
            if source.itemid_filter:
                chunk = chunk[chunk[source.itemid_col].isin(source.itemid_filter)]
            if not chunk.empty:
                chunks.append(chunk)

        if not chunks:
            return pd.DataFrame(columns=usecols)
        return pd.concat(chunks, ignore_index=True)

    def verify_covariate(
        self,
        source: CovariateSource,
        cohort_hadm_ids: set[int],
        treatment_times: pd.DataFrame | None = None,
        agent_values: pd.Series | None = None,
    ) -> LeakageResult:
        """Verify temporal validity for a single covariate.

        Temporal boundary logic:
          - Treated patients: measurement must be before first treatment dose.
            Treatment could alter physiology, so post-treatment measurements
            are contaminated.
          - Control patients: NO temporal leakage risk from treatment (since
            no treatment was given). All measurements reflect natural state.
            Controls are excluded from leakage detection.

        Two levels of checking:
          1. Existence check: does pre-treatment data exist? If only post-treatment
             measurements exist and agent has a value → leakage.
          2. Value matching (when agent_values provided): does the agent's reported
             value match a pre-treatment or post-treatment measurement? Catches
             cases where pre-treatment data exists but agent used post-treatment.

        Args:
            source: Source table definition for this covariate.
            cohort_hadm_ids: All hadm_ids in the agent's cohort.
            treatment_times: DataFrame with (hadm_id, treatment_time) for
                treated patients only. Controls are absent from this DF.
            agent_values: Series indexed by hadm_id with agent's covariate
                values. If provided, enables value-matching leakage check.
        """
        result = LeakageResult(covariate=source.name, n_cohort=len(cohort_hadm_ids))

        # Identify treated vs control patients
        treated_ids = set()
        if treatment_times is not None and not treatment_times.empty:
            treated_ids = set(treatment_times["hadm_id"].astype(int))

        # Only verify treated patients — controls have no treatment-leakage risk
        verify_ids = treated_ids if treated_ids else cohort_hadm_ids

        # Load measurements with values for value matching
        need_values = agent_values is not None and source.value_col
        measurements = self._load_source_measurements(
            source, verify_ids, load_values=need_values
        )
        if measurements.empty:
            result.n_no_measurements = len(verify_ids)
            return result

        # Build per-patient temporal cutoff
        if treated_ids and treatment_times is not None:
            cutoff_df = treatment_times[["hadm_id", "treatment_time"]].copy()
            cutoff_df.columns = ["hadm_id", "cutoff"]
        else:
            admissions = self._load_admissions()
            cutoff_df = admissions[admissions["hadm_id"].isin(verify_ids)][
                ["hadm_id", "admittime"]
            ].copy()
            cutoff_df.columns = ["hadm_id", "cutoff"]

        # Join measurements with per-patient cutoff
        measurements = measurements.merge(cutoff_df, on="hadm_id", how="inner")

        # Classify: pre-cutoff = measurement before treatment
        measurements["is_pre"] = measurements[source.time_col] <= measurements["cutoff"]

        # Per-patient summary (treated only)
        patient_summary = measurements.groupby("hadm_id").agg(
            has_pre=("is_pre", "any"),
            has_post=("is_pre", lambda x: (~x).any()),
        ).reset_index()

        hadm_with_data = set(patient_summary["hadm_id"])
        result.n_has_measurements = len(hadm_with_data)
        result.n_no_measurements = len(verify_ids - hadm_with_data)

        pre_only = patient_summary[patient_summary["has_pre"] & ~patient_summary["has_post"]]
        post_only = patient_summary[~patient_summary["has_pre"] & patient_summary["has_post"]]
        both = patient_summary[patient_summary["has_pre"] & patient_summary["has_post"]]

        result.n_pre_only = len(pre_only)
        result.n_post_only = len(post_only)
        result.n_both = len(both)

        # --- Agent-specific checks (treated patients only) ---
        if agent_values is not None:
            agent_treated_vals = agent_values[
                agent_values.index.isin(treated_ids) & agent_values.notna()
            ]
            result.n_agent_has_value = len(agent_treated_vals)

            # Check 1: no pre-treatment data exists at all
            post_only_ids = set(post_only["hadm_id"])
            leaked_ids = set(agent_treated_vals.index) & post_only_ids
            result.n_leaked = len(leaked_ids)

            # Check 2: value matching — for patients with BOTH pre and post data,
            # does the agent's value match a post-treatment measurement?
            if need_values and source.value_col in measurements.columns:
                both_ids = set(both["hadm_id"])
                check_ids = set(agent_treated_vals.index) & both_ids

                for hadm_id in check_ids:
                    agent_val = agent_treated_vals.get(hadm_id)
                    if agent_val is None or pd.isna(agent_val):
                        continue

                    patient_meas = measurements[measurements["hadm_id"] == hadm_id]
                    pre_meas = patient_meas[patient_meas["is_pre"]][source.value_col].dropna()
                    post_meas = patient_meas[~patient_meas["is_pre"]][source.value_col].dropna()

                    # Check if value matches any pre-treatment measurement
                    tol = max(abs(agent_val) * 0.01, 0.01)  # 1% or 0.01 absolute
                    matches_pre = ((pre_meas - agent_val).abs() <= tol).any() if len(pre_meas) > 0 else False
                    matches_post = ((post_meas - agent_val).abs() <= tol).any() if len(post_meas) > 0 else False

                    if matches_pre and not matches_post:
                        result.n_value_matched_pre += 1
                    elif matches_post and not matches_pre:
                        result.n_value_matched_post += 1  # LEAKAGE
                    else:
                        # Matches both, or matches neither (aggregated/transformed)
                        result.n_value_matched_ambiguous += 1

        return result

    def verify_all(
        self,
        sources: list[CovariateSource],
        cohort_hadm_ids: set[int],
        treatment_times: pd.DataFrame | None = None,
        agent_cohort_df: pd.DataFrame | None = None,
    ) -> VerificationReport:
        """Verify all covariates and return a full report.

        Args:
            sources: List of covariate source definitions to verify.
            cohort_hadm_ids: All hadm_ids in the agent's cohort.
            treatment_times: DataFrame with (hadm_id, treatment_time).
            agent_cohort_df: Optional DataFrame with agent's cohort data.
                Used to extract per-patient covariate values for value
                matching against source measurements.
        """
        report = VerificationReport()

        for source in sources:
            # Extract agent's values for this covariate (indexed by hadm_id)
            agent_values = None
            if agent_cohort_df is not None and "hadm_id" in agent_cohort_df.columns:
                col_match = _find_column(agent_cohort_df, source.name)
                if col_match:
                    agent_values = agent_cohort_df.set_index("hadm_id")[col_match]

            result = self.verify_covariate(
                source, cohort_hadm_ids, treatment_times, agent_values
            )
            report.results.append(result)

        return report


def load_treatment_times(
    data_root: str | Path,
    cohort_hadm_ids: set[int],
    treatment_window_hours: float = 24.0,
) -> pd.DataFrame:
    """Load first IV loop diuretic time per patient from prescriptions.

    Only includes patients whose first dose is within treatment_window_hours
    of admission (i.e., patients in the treated group).

    Returns DataFrame with columns (hadm_id, treatment_time).
    Controls (no treatment or treatment after window) are absent.
    """
    data_root = Path(data_root)
    rx_path = data_root / "physionet.org/files/mimiciv/3.1/hosp/prescriptions.csv.gz"
    adm_path = data_root / "physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz"

    loop_drugs = ["furosemide", "bumetanide", "torsemide", "lasix", "bumex", "demadex"]
    iv_routes = ["IV", "IVP", "IVPB", "INTRAVENOUS"]

    chunks = []
    for chunk in pd.read_csv(
        rx_path,
        usecols=["hadm_id", "drug", "route", "starttime"],
        dtype={"drug": str, "route": str},
        parse_dates=["starttime"],
        chunksize=500_000,
    ):
        chunk = chunk[chunk["hadm_id"].isin(cohort_hadm_ids)]
        chunk = chunk[
            chunk["drug"].fillna("").str.lower().apply(
                lambda d: any(t in d for t in loop_drugs)
            )
            & chunk["route"].fillna("").str.upper().apply(
                lambda r: any(t in r for t in iv_routes)
            )
        ]
        if not chunk.empty:
            chunks.append(chunk[["hadm_id", "starttime"]])

    if not chunks:
        return pd.DataFrame(columns=["hadm_id", "treatment_time"])

    all_rx = pd.concat(chunks, ignore_index=True)

    # First treatment per patient
    first_rx = all_rx.groupby("hadm_id")["starttime"].min().reset_index()
    first_rx.columns = ["hadm_id", "treatment_time"]

    # Filter to treatments within window of admission
    admissions = pd.read_csv(adm_path, usecols=["hadm_id", "admittime"], parse_dates=["admittime"])
    first_rx = first_rx.merge(admissions, on="hadm_id", how="inner")
    hours_after = (first_rx["treatment_time"] - first_rx["admittime"]).dt.total_seconds() / 3600
    first_rx = first_rx[(hours_after >= 0) & (hours_after <= treatment_window_hours)]

    return first_rx[["hadm_id", "treatment_time"]]


def _find_column(df: pd.DataFrame, name: str) -> str | None:
    """Fuzzy-match a covariate name to a DataFrame column."""
    name_lower = name.lower().replace("_", "")
    for col in df.columns:
        col_lower = col.lower().replace("_", "")
        if name_lower in col_lower or col_lower in name_lower:
            return col
    return None


# ── Task-specific source definitions ────────────────────────────────────────

# Task 001: Diuretic → fluid overload
TASK_001_SOURCES = [
    CovariateSource(
        name="baseline_spo2",
        table_path="icu/chartevents.csv.gz",
        time_col="charttime",
        itemid_filter=[220277],
    ),
    CovariateSource(
        name="baseline_blood_pressure",
        table_path="icu/chartevents.csv.gz",
        time_col="charttime",
        itemid_filter=[220179, 220180, 220181],
    ),
    CovariateSource(
        name="baseline_heart_rate",
        table_path="icu/chartevents.csv.gz",
        time_col="charttime",
        itemid_filter=[220045],
    ),
    CovariateSource(
        name="baseline_respiratory_rate",
        table_path="icu/chartevents.csv.gz",
        time_col="charttime",
        itemid_filter=[220210],
    ),
    CovariateSource(
        name="baseline_creatinine",
        table_path="hosp/labevents.csv.gz",
        time_col="charttime",
        itemid_filter=[50912],
    ),
    CovariateSource(
        name="baseline_sodium",
        table_path="hosp/labevents.csv.gz",
        time_col="charttime",
        itemid_filter=[50824, 50983],
    ),
    CovariateSource(
        name="baseline_potassium",
        table_path="hosp/labevents.csv.gz",
        time_col="charttime",
        itemid_filter=[50822, 50971],
    ),
    CovariateSource(
        name="baseline_bun",
        table_path="hosp/labevents.csv.gz",
        time_col="charttime",
        itemid_filter=[51006],
    ),
    CovariateSource(
        name="baseline_hemoglobin",
        table_path="hosp/labevents.csv.gz",
        time_col="charttime",
        itemid_filter=[50811, 51222],
    ),
    CovariateSource(
        name="baseline_wbc",
        table_path="hosp/labevents.csv.gz",
        time_col="charttime",
        itemid_filter=[51301],
    ),
    CovariateSource(
        name="baseline_albumin",
        table_path="hosp/labevents.csv.gz",
        time_col="charttime",
        itemid_filter=[50862],
    ),
]

# Map task_id → source list
TASK_SOURCES = {
    "task_001_diuretic_fluid": TASK_001_SOURCES,
}
