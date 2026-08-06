"""Pydantic models for answer.json and golden.json schemas."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


_NA_STRINGS = {"N/A", "NA", "NONE", ""}


def _coerce_float(v: object) -> float:
    if v is None or (isinstance(v, str) and v.strip().upper() in _NA_STRINGS):
        return 0.0
    return float(v)


def _coerce_int(v: object) -> int:
    if v is None or (isinstance(v, str) and v.strip().upper() in _NA_STRINGS):
        return 0
    return int(v)


class EffectType(str, Enum):
    ATE = "ATE"
    ATT = "ATT"
    CATE = "CATE"
    risk_difference = "risk_difference"
    risk_ratio = "risk_ratio"
    odds_ratio = "odds_ratio"
    hazard_ratio = "hazard_ratio"


# --- Agent answer.json models ---


class EstimandAnswer(BaseModel):
    population: str
    treatment: str
    comparator: str
    outcome: str
    time_zero: str
    time_horizon: str
    effect_type: str  # validated against EffectType in scoring, kept as str for flexibility


class AttritionStepAnswer(BaseModel):
    step: str
    n_remaining: int = 0


class CohortAnswer(BaseModel):
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    index_event: str = ""
    n_initial: int = 0
    n_final: int = 0
    attrition: list[AttritionStepAnswer] = Field(default_factory=list)


class TemporalAnswer(BaseModel):
    time_zero: str = ""
    allowed_data_end: str = ""
    post_t0_features_used: list[str] = Field(default_factory=list)


class VariableSpecAnswer(BaseModel):
    name: str = ""
    definition: str = ""
    source: str = ""
    time_window: str = "pre_t0"


class VariablesAnswer(BaseModel):
    treatment: Optional[VariableSpecAnswer] = None
    outcome: Optional[VariableSpecAnswer] = None
    confounders: list[VariableSpecAnswer] = Field(default_factory=list)
    effect_modifiers: list[VariableSpecAnswer] = Field(default_factory=list)
    imaging_features_used: list[str] = Field(default_factory=list)


class MethodAnswer(BaseModel):
    primary_estimator: str = ""
    estimation_family: Optional[str] = None
    propensity_model: str = ""
    outcome_model: str = ""
    adjustment_set: list[str] = Field(default_factory=list)


class BalanceDiagAnswer(BaseModel):
    max_smd_before: float = 0.0
    max_smd_after: float = 0.0

    @field_validator("max_smd_before", "max_smd_after", mode="before")
    @classmethod
    def _coerce(cls, v: object) -> float:
        return _coerce_float(v)


class OverlapDiagAnswer(BaseModel):
    min_propensity: float = 0.0
    max_propensity: float = 1.0

    @field_validator("min_propensity", "max_propensity", mode="before")
    @classmethod
    def _coerce(cls, v: object) -> float:
        return _coerce_float(v)


class MissingnessDiagAnswer(BaseModel):
    rows_dropped: int = 0
    imputation_used: bool = False

    @field_validator("rows_dropped", mode="before")
    @classmethod
    def _coerce(cls, v: object) -> int:
        return _coerce_int(v)


class DiagnosticsAnswer(BaseModel):
    balance: BalanceDiagAnswer = Field(default_factory=BalanceDiagAnswer)
    overlap: OverlapDiagAnswer = Field(default_factory=OverlapDiagAnswer)
    missingness: MissingnessDiagAnswer = Field(default_factory=MissingnessDiagAnswer)
    positivity_violations: list[str] = Field(default_factory=list)
    model_failures: list[str] = Field(default_factory=list)


class PrimaryEffectAnswer(BaseModel):
    measure: str = ""
    estimate: float = 0.0
    ci_95: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    p_value: float = 0.0
    direction: str = "null"

    @field_validator("estimate", "p_value", mode="before")
    @classmethod
    def _coerce_float(cls, v: object) -> float:
        return _coerce_float(v)

    @field_validator("ci_95", mode="before")
    @classmethod
    def _coerce_ci(cls, v: list | None) -> list[float]:
        if v is None:
            return [0.0, 0.0]
        return [_coerce_float(x) for x in v]

    @field_validator("direction", mode="before")
    @classmethod
    def _coerce_dir(cls, v: str | None) -> str:
        if v is None or (isinstance(v, str) and v.strip().upper() in _NA_STRINGS):
            return "null"
        return v


class ResultAnswer(BaseModel):
    primary_effect: PrimaryEffectAnswer = Field(default_factory=PrimaryEffectAnswer)
    heterogeneity: list = Field(default_factory=list)
    interpretation: str = ""


class AlternativeEstimatorAnswer(BaseModel):
    name: str = ""
    estimate: float = 0.0
    ci_95: list[float] = Field(default_factory=lambda: [0.0, 0.0])


class SensitivityAnalysisAnswer(BaseModel):
    name: str = ""
    result: str = ""


class RobustnessAnswer(BaseModel):
    alternative_estimators: list[AlternativeEstimatorAnswer] = Field(default_factory=list)
    sensitivity_analyses: list[SensitivityAnalysisAnswer] = Field(default_factory=list)


class ArtifactsAnswer(BaseModel):
    analysis_script: str = ""
    cohort_file: str = ""
    log_file: str = ""


class FinalDecisionAnswer(BaseModel):
    status: str = "error"
    confidence: str = "low"
    abstain_reason: str = ""


class AnswerSchema(BaseModel):
    """Full answer schema supporting D1-D8."""

    task_id: str
    run_id: Optional[str] = None
    estimand: EstimandAnswer
    cohort: CohortAnswer = Field(default_factory=CohortAnswer)
    temporal: TemporalAnswer = Field(default_factory=TemporalAnswer)
    variables: Optional[VariablesAnswer] = None
    method: Optional[MethodAnswer] = None
    diagnostics: DiagnosticsAnswer = Field(default_factory=DiagnosticsAnswer)
    result: Optional[ResultAnswer] = None
    robustness: RobustnessAnswer = Field(default_factory=RobustnessAnswer)
    artifacts: ArtifactsAnswer = Field(default_factory=ArtifactsAnswer)
    final_decision: FinalDecisionAnswer = Field(default_factory=FinalDecisionAnswer)


# --- Golden answer models ---


class GoldenFieldSpec(BaseModel):
    """Specification for how to score a single golden answer field."""

    canonical: str
    type: Optional[str] = None  # "categorical", "iso_duration", "alias_match", or None (freetext)
    required_components: list[str] = Field(default_factory=list)
    disqualifying_components: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    disqualifying_values: list[str] = Field(default_factory=list)
    allowed_values: list[str] = Field(default_factory=list)


class EstimandGolden(BaseModel):
    population: GoldenFieldSpec
    treatment: GoldenFieldSpec
    comparator: GoldenFieldSpec
    outcome: GoldenFieldSpec
    time_zero: GoldenFieldSpec
    time_horizon: GoldenFieldSpec
    effect_type: GoldenFieldSpec


class CohortGolden(BaseModel):
    inclusion_criteria_required: list[str] = Field(default_factory=list)
    exclusion_criteria_expected: list[str] = Field(default_factory=list)
    index_event_canonical: str = ""
    index_event_aliases: list[str] = Field(default_factory=list)
    golden_cohort_csv: str = ""
    n_final_tolerance: float = 0.20
    attrition_min_steps: int = 3


class TemporalGolden(BaseModel):
    time_zero_canonical: str = ""
    allowed_data_end: str = "time_zero"
    forbidden_post_t0_features: list[str] = Field(default_factory=list)
    valid_time_windows: list[str] = Field(
        default_factory=lambda: ["pre_t0", "pre_t0_or_at_t0", "at_t0"]
    )


class ConfounderSpec(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)


class VariablesGolden(BaseModel):
    treatment_name_pattern: str = ""
    outcome_name_pattern: str = ""
    confounders: dict[str, list[ConfounderSpec]] = Field(default_factory=dict)
    imaging_required: bool = False


class MethodGolden(BaseModel):
    allowed_estimators: list[str] = Field(default_factory=list)
    partial_credit_estimators: list[str] = Field(default_factory=list)
    zero_credit_estimators: list[str] = Field(default_factory=list)
    allowed_families: list[str] = Field(default_factory=list)


class DiagnosticsGolden(BaseModel):
    required_fields: list[str] = Field(default_factory=list)
    balance_smd_threshold: float = 0.1
    overlap_min_propensity_threshold: float = 0.01
    overlap_max_propensity_threshold: float = 0.99


class ResultGolden(BaseModel):
    valid_measures: list[str] = Field(default_factory=list)
    valid_directions: list[str] = Field(default_factory=list)
    requires_ci: bool = True
    requires_p_value: bool = False
    golden_analysis_cohort_csv: str = ""
    null_effect_threshold: float = 0.05


class RobustnessGolden(BaseModel):
    min_alternative_estimators: int = 1
    min_sensitivity_analyses: int = 0
    abstain_requires_reason: bool = True


class GoldenSchema(BaseModel):
    task_id: str
    estimand_golden: EstimandGolden
    cohort_golden: CohortGolden = Field(default_factory=CohortGolden)
    temporal_golden: TemporalGolden = Field(default_factory=TemporalGolden)
    variables_golden: VariablesGolden = Field(default_factory=VariablesGolden)
    method_golden: MethodGolden = Field(default_factory=MethodGolden)
    diagnostics_golden: DiagnosticsGolden = Field(default_factory=DiagnosticsGolden)
    result_golden: ResultGolden = Field(default_factory=ResultGolden)
    robustness_golden: RobustnessGolden = Field(default_factory=RobustnessGolden)


# --- Task specification models ---


class TaskHints(BaseModel):
    """Optional hints to nudge the agent toward the right direction."""

    treatment_route: Optional[str] = None
    imaging_modality: Optional[str] = None
    outcome_window: Optional[str] = None
    custom: dict[str, str] = Field(default_factory=dict)


class TaskConstraints(BaseModel):
    """Constraints the agent must follow."""

    allowed_effect_types: list[str] = Field(
        default_factory=lambda: [e.value for e in EffectType]
    )
    allowed_estimation_families: list[str] = Field(
        default_factory=lambda: ["matching", "ipw", "aipw", "tmle", "g_formula", "dml", "iv"]
    )
    time_format: str = "ISO 8601"
    duration_format: str = "ISO 8601 (e.g. P1D, PT24H)"


class TaskSpec(BaseModel):
    """Machine-readable task specification given to the agent."""

    task_id: str
    version: str = "1.0"
    question: str
    domain: Optional[str] = None
    difficulty: Optional[str] = None
    hints: TaskHints = Field(default_factory=TaskHints)
    constraints: TaskConstraints = Field(default_factory=TaskConstraints)
    data_manifest_path: str = "docs/dataset_manifest.json"


# --- Scoring result models ---


class FieldScore(BaseModel):
    """Result of scoring a single field."""

    score: float
    components: Optional[dict[str, bool]] = None
    disqualified: bool = False
    reason: str = ""


class DimensionResult(BaseModel):
    """Result for any scoring dimension (D1-D8)."""

    dimension: str
    grade: int  # 0, 1, or 2
    mean_field_score: float
    weighted_score: float
    field_scores: dict[str, FieldScore] = Field(default_factory=dict)
    hard_gate_failed: bool = False
    notes: str = ""


# Keep D1Result as alias for backward compat
D1Result = DimensionResult
