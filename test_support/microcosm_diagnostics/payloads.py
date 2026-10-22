"""Valid calibration-diagnostics schema-8 documents shared by tests.

Each call returns a fresh, fully reconciled document, so a test can mutate one
field and know that any refusal comes from that change.
"""

from __future__ import annotations


def schema_8_payload() -> dict:
    """A minimal US-registry document with one target and no optional blocks."""

    digest = "a" * 64
    return {
        "schema_version": 8,
        "weight_entity": "household",
        "options": {"epochs": 1},
        "target_surface": {
            "schema_version": 1,
            "weight_entity": "household",
            "n_targets": 1,
            "n_records": 2,
            "constraint_matrix": {"rows": 1, "columns": 2, "nnz": 2},
            "sha256": digest,
            "names_sha256": digest,
            "values_sha256": digest,
        },
        "target_registry": {"country": "us", "version": digest, "n_specs": 1},
        "l0_lambda": 0.0,
        "n_nonzero": 2,
        "n_records": 2,
        "initial_loss": 0.1,
        "final_loss": 0.01,
        "fraction_within_10pct": 1.0,
        "effective_sample_size": 2.0,
        "realized_max_weight_ratio": 1.0,
        "top_1pct_weight_share": 0.5,
        "loss_trajectory": [0.1],
        "skipped": [],
        "past_cap_census": None,
        "diagnostic_warnings": [],
        "targets": [
            {
                "name": "population@2024",
                "target_name": "population",
                "period": 2024,
                "entity": "household",
                "measure": {"kind": "column", "name": "household_count"},
                "filter": None,
                "source": "Census fixture",
                "metadata": {},
                "target": 2.0,
                "compiled_target": 2.0,
                "initial_estimate": 1.8,
                "final_estimate": 2.0,
                "relative_error": 0.0,
                "within_tolerance": None,
                "hierarchy": {
                    "provider": {"id": "census", "label": "Census"},
                    "category": {
                        "id": "census.population",
                        "label": "Population",
                        "provider_id": "census",
                    },
                    "geography": {
                        "id": "0100000US",
                        "label": "United States",
                        "level": "country",
                    },
                    "dimensions": [],
                    "target": {"id": "population", "label": "Population"},
                },
            }
        ],
    }


def uk_schema_8_payload() -> dict:
    """The minimal document on a UK registry, carrying every UK extension block."""

    payload = schema_8_payload()
    payload["target_registry"]["country"] = "uk"
    payload["n_nonzero"] = 1
    payload["effective_sample_size"] = 1.0
    payload["top_1pct_weight_share"] = 1.0
    payload["uk_diagnostics"] = {
        "schema_version": 1,
        "weights": {
            "n_records": 2,
            "positive_weight_records": 1,
            "zero_weight_records": 1,
            "total_weight": 1.0,
            "effective_sample_size": 1.0,
            "ess_fraction": 0.5,
            "median_positive_weight": 1.0,
            "max_weight": 1.0,
            "max_to_median_positive_weight": 1.0,
            "top_1pct_weight_share": 1.0,
        },
        "zero_weight_rows_by_stratum": [
            {
                "stratum": {"source": "frs"},
                "rows": 2,
                "positive_weight_rows": 1,
                "zero_weight_rows": 1,
                "weight_sum": 1.0,
            }
        ],
        "target_pass_rates_by_geography_level": [
            {
                "geography_level": level,
                "n_targets": 1 if level == "national" else 0,
                "n_scored": 1 if level == "national" else 0,
                "n_skipped": 0,
                "n_within_10pct": 1 if level == "national" else 0,
                "pass_rate": 1.0 if level == "national" else None,
            }
            for level in (
                "national",
                "region",
                "country",
                "local_authority",
                "constituency",
            )
        ],
        "target_observation_basis": {"population@2024": "annual_stock"},
        "weakest_families": [
            {
                "family": "population",
                "n_targets": 1,
                "n_within_10pct": 1,
                "pass_rate": 1.0,
                "worst_target": "population@2024",
                "worst_abs_relative_error": 0.0,
                "loss_contribution": 0.0,
                "loss_share": 0.0,
            }
        ],
        "weakest_areas_by_fit": {
            "limit": 15,
            "n_areas_scored": 1,
            "bottom_by_fit": [
                {
                    "geography_level": "constituency",
                    "area_code": "E14000001",
                    "country": "England",
                    "n_targets": 1,
                    "n_within_10pct": 1,
                    "pass_rate": 1.0,
                    "worst_target": "population@2024",
                    "worst_abs_relative_error": 0.0,
                    "loss_contribution": 0.0,
                    "support_below_floor": True,
                }
            ],
            "countries": [
                {
                    "country": "England",
                    "geography_level": "constituency",
                    "n_areas": 1,
                    "n_targets": 1,
                    "n_within_10pct": 1,
                    "pass_rate": 1.0,
                    "worst_target": "population@2024",
                    "worst_abs_relative_error": 0.0,
                    "loss_contribution": 0.0,
                }
            ],
        },
        "rotated_holdout": {
            "report_only": True,
            "method": "rotated_folds",
            "target_loss_cap": 10.0,
            "loss_weight_scale": "held_local_grains_only",
            "target_weight_rule": "grain_equal",
            "population": "held_out_local_targets",
            "grains": ["constituency", "local_authority"],
            "n_folds": 2,
            "seed": 20260529,
            "solve_seed": 7,
            "mean_holdout_loss": 0.15,
            "worst_holdout_loss": 0.2,
            "fold_losses": [0.1, 0.2],
            "folds": [
                {
                    "fold": 0,
                    "n_train_targets": 1,
                    "n_holdout_targets": 1,
                    "holdout_target_indices": [0],
                    "training_national_rows": 1,
                    "holdout_loss": 0.1,
                },
                {
                    "fold": 1,
                    "n_train_targets": 1,
                    "n_holdout_targets": 1,
                    "holdout_target_indices": [1],
                    "training_national_rows": 1,
                    "holdout_loss": 0.2,
                },
            ],
        },
        "consumption_drift": {
            "report_only": True,
            "totals": {
                "petrol_spending": {
                    "design": 100.0,
                    "final": 102.0,
                    "change": 2.0,
                    "relative_change": 0.02,
                }
            },
            "shares": {
                "fuel_car_households_without_fuel": {
                    "design": 0.0,
                    "final": 0.0,
                    "change": 0.0,
                    "relative_change": None,
                }
            },
        },
    }
    return payload
