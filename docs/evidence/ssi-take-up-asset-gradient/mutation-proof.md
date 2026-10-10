# SSI asset gradient mutation proof

Copied source function compiled into the actual function's code slot during pytest_runtest_call; original code restored after each call. Imports remain warm between pytest invocations. Original source files unchanged.

Baseline return code: 0. All 23 source mutations killed: True.

| Behavior assertion | Deliberate source mutation | Failed pytest node IDs | Result |
| --- | --- | --- | --- |
| `test_propensity_is_monotone_for_nonpositive_slopes` | Reverse the asset coefficient sign. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_propensity_is_monotone_for_nonpositive_slopes` | Pytest assertion failed (exit 1) |
| `test_finite_propensity_stays_strictly_inside_unit_interval` | Remove strict probability endpoint clipping. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_finite_propensity_stays_strictly_inside_unit_interval` | Pytest assertion failed (exit 1) |
| `test_intercept_solve_reproduces_weighted_target` | Bias the solved intercept by 0.1. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_intercept_solve_reproduces_weighted_target` | Pytest assertion failed (exit 1) |
| `test_zero_weight_assets_do_not_move_intercept` | Give zero-weight observations positive mass. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_zero_weight_assets_do_not_move_intercept` | Pytest assertion failed (exit 1) |
| `test_every_person_obeys_own_asset_law_and_reporters_stay_true` | Remove unconditional reporter pinning. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_every_person_obeys_own_asset_law_and_reporters_stay_true[observed_asset_range]` | Pytest assertion failed (exit 1) |
| `test_every_person_obeys_own_asset_law_and_reporters_stay_true[extreme_intercept_precision]` | Round-trip the exact intercept through sigmoid and logit, losing steep-slope precision. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_every_person_obeys_own_asset_law_and_reporters_stay_true[extreme_intercept_precision]` | Pytest assertion failed (exit 1) |
| `test_source_draws_survive_person_row_reordering` | Key the uniform draw by the first physical row index. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_source_draws_survive_person_row_reordering` | Pytest assertion failed (exit 1) |
| `test_weight_split_support_clones_preserve_original_flags` | Invert the extra support clone's flag. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_weight_split_support_clones_preserve_original_flags` | Pytest assertion failed (exit 1) |
| `test_zero_slope_flags_match_frozen_schema4_implementation_exactly` | Shift the historical constant-prior threshold. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_zero_slope_flags_match_frozen_schema4_implementation_exactly` | Pytest assertion failed (exit 1) |
| `test_nonzero_slopes_reject_scalar_only_legacy_prior_basis` | Silently default a scalar artifact to constant priors. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_nonzero_slopes_reject_scalar_only_legacy_prior_basis` | Pytest assertion failed (exit 1) |
| `test_schema5_delivered_basis_retains_asset_distribution_and_target_mass` | Seed the retry from assignment assets instead of delivered asset mass. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_schema5_delivered_basis_retains_asset_distribution_and_target_mass` | Pytest assertion failed (exit 1) |
| `test_gate_rejects_gradient_coefficient_corruption` | Ignore the integrity gate's detected coefficient failures. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_gate_rejects_gradient_coefficient_corruption` | Pytest assertion failed (exit 1) |
| `test_gate_rejects_candidate_asset_mass_corruption` | Ignore the integrity gate's detected asset mass failures. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_gate_rejects_candidate_asset_mass_corruption` | Pytest assertion failed (exit 1) |
| `test_physical_asec_assets_own_source_despite_support_reimputation` | Ignore the physical ASEC owner and retain first-row support assets. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_physical_asec_assets_own_source_despite_support_reimputation` | Pytest assertion failed (exit 1) |
| `test_asec_absent_source_assets_must_be_unambiguous` | Accept conflicting support assets when their physical ASEC owner is absent. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_asec_absent_source_assets_must_be_unambiguous` | Pytest assertion failed (exit 1) |
| `test_captured_source_assets_preserve_frozen_law_after_owner_pruning` | Ignore captured source holdings after their physical owner is pruned. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_captured_source_assets_preserve_frozen_law_after_owner_pruning` | Pytest assertion failed (exit 1) |
| `test_captured_source_asset_map_requires_valid_coverage_and_owner_match[missing]` | Accept a captured attribute map missing a retained source identity. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_captured_source_asset_map_requires_valid_coverage_and_owner_match[missing]` | Pytest assertion failed (exit 1) |
| `test_captured_source_asset_map_requires_valid_coverage_and_owner_match[negative]` | Accept a captured source attribute with a negative holding. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_captured_source_asset_map_requires_valid_coverage_and_owner_match[negative]` | Pytest assertion failed (exit 1) |
| `test_captured_source_asset_map_requires_valid_coverage_and_owner_match[nan]` | Accept a captured source attribute with a nonfinite holding. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_captured_source_asset_map_requires_valid_coverage_and_owner_match[nan]` | Pytest assertion failed (exit 1) |
| `test_captured_source_asset_map_requires_valid_coverage_and_owner_match[physical_mismatch]` | Accept captured holdings that disagree with a physical ASEC owner. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_captured_source_asset_map_requires_valid_coverage_and_owner_match[physical_mismatch]` | Pytest assertion failed (exit 1) |
| `test_missing_liquid_asset_inputs_are_rejected` | Silently fill a missing asset input with zero. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_missing_liquid_asset_inputs_are_rejected` | Pytest assertion failed (exit 1) |
| `test_asset_age_diagnostics_reconcile_all_source_people_and_weights` | Drop the asset-by-age diagnostics. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_asset_age_diagnostics_reconcile_all_source_people_and_weights` | Pytest assertion failed (exit 1) |
| `test_public_reseed_uses_existing_weights_and_candidate_probe` | Discard the engine candidate-basis probe in public reseeding. | `packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py::test_public_reseed_uses_existing_weights_and_candidate_probe` | Pytest assertion failed (exit 1) |

Verbatim baseline pytest tail:

```text
.......................                                                  [100%]
```

The JSON companion records exact replacement fragments, source SHA-256 values, and each mutation's verbatim pytest failure tail.
