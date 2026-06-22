# Chemical Claim Validation (Category 5)

Tests whether an agent can detect when a scientific conclusion is unsupported by
the evidence, name the specific failure mode, and avoid overclaiming. Every task
in this suite is intentionally flawed, so a correct response reports an
inconclusive status with the right failure mode instead of forcing a numerical
answer. Tasks are JSON-defined and scored by `scorers/oracle_validity.py`
against per-task assertion checks.

Run them through the unified CLI like any other task:

```bash
python evaluate.py run val_C1_01_multicomponent_logp examples/agent_c5_logp.py
```

## Task-ID convention

The 19 validity tasks each pair with one of the other categories; the
`C1`/`C2`/`C4` token in the ID denotes the *paired* category (not a separate
benchmark):

- `val_C1_*` — paired with Chemical Data Preprocessing (5 tasks)
- `val_C2_*` — paired with Chemical Data Analysis (11 tasks)
- `val_C4_*` — paired with Chemical Safety Assessment (3 tasks)

## Files

```text
tasks_interactive/C5_chemical_claim_validation/   task JSON definitions (this folder)
tasks_interactive/data/                           CSV / PDB inputs (shared with C2)
scorers/data/answer_pool.json                     canonical enum + ground-truth schema
scorers/data/failure_modes.json                   documentation-only helper vocabulary
```

## Task inventory

| Task ID | Concepts | Ground-truth failure mode |
|---|---|---|
| `val_C1_01_multicomponent_logp` | descriptor_applicability, multicomponent_smiles, data_validation | `multi_component_smiles` |
| `val_C1_02_metal_out_of_domain_logp` | descriptor_applicability, metal_detection, domain_scope | `metal_out_of_domain` |
| `val_C1_03_species_mismatch_targets` | species_filtering, metadata_join, pipeline_scope | `species_mismatch` |
| `val_C1_04_invalid_geometry` | 3d_geometry_validation, pdb_parsing, distance_checks | `invalid_3d_geometry` |
| `val_C1_05_invalid_structure_chemistry` | pdb_validation, formal_charge, chemical_sanity | `invalid_structure_chemistry` |
| `val_C2_01_hidden_units` | unit_awareness, data_validation, safe_stopping | `mixed_units` |
| `val_C2_02_sparse_sar` | statistical_reasoning, sample_size, confounding | `insufficient_n` |
| `val_C2_03_alias_sar` | join_validation, data_integrity, scaffold_annotation | `shuffled_annotation` |
| `val_C2_04_false_cliff` | structure_deduplication, activity_cliff, fingerprint_analysis | `identity_duplicates` |
| `val_C2_05_confounded_trend` | batch_effects, reference_compounds, confounding | `batch_drift` |
| `val_C2_06_claim_verification` | claim_verification, chembl_lookup, epistemic_humility | `memory_conflict` |
| `val_C2_07_stale_vs_current` | stale_knowledge, chembl_current_value, epistemic_humility | `stale_knowledge` |
| `val_C2_08_identifier_masquerade` | identity_duplicates, inchikey, pubchem_verification | `identity_duplicates` |
| `val_C2_09_retracted_assay` | assay_confidence, chembl_metadata, epistemic_humility | `low_confidence_assay` |
| `val_C2_10_ontology_scope` | chebi_ontology, role_annotation, tool_use | `ontology_scope_mismatch` |
| `val_C2_11_prediction_vs_measurement` | admetai_prediction, model_vs_measurement, tool_use | `prediction_measurement_mismatch` |
| `val_C4_01_corrupt_data` | label_noise, data_quality, annotation_errors | `data_corruption` |
| `val_C4_02_mismatch` | identity_duplicates, label_conflict, data_integrity | `identity_duplicates` |
| `val_C4_03_optimize_conflict` | ontology_scope_mismatch, bioavailability_proxy, permeability_solubility_tradeoff | `conflicting_objectives` |

## Schema

`scorers/data/answer_pool.json` is the single source of truth; it defines
`conclusion_status.values`, `failure_mode.values`,
`failure_mode.ground_truth_labels`, and `recommended_action.values`.
`scorers/data/failure_modes.json` is a documentation-only helper vocabulary that
must not drift from `answer_pool.json`.

Every task asks the agent to set the shared diagnostic variables
`conclusion_status`, `failure_mode`, `recommended_action`, and
`diagnostic_evidence`, and is scored on at least `failure_mode` and
`conclusion_status`.
