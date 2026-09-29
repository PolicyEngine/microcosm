"""Tests split from packages/microcosm-build/tests/test_validation_input_coverage.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.validation_input_coverage import *


class TestValidationInputLeafRegistry:
    def test_every_entry_names_rows_and_provisions(self) -> None:
        assert US_VALIDATION_PROVISION_INPUT_LEAVES
        for entry in US_VALIDATION_PROVISION_INPUT_LEAVES:
            assert entry.leaf
            assert entry.provision_variables
            assert entry.validation_rows

    def test_entry_requires_provision_variables(self) -> None:
        with pytest.raises(ValueError, match="provision_variables is required"):
            ValidationInputLeaf(
                leaf="x", provision_variables=(), validation_rows=("r",)
            )

    def test_entry_requires_validation_rows(self) -> None:
        with pytest.raises(ValueError, match="validation_rows is required"):
            ValidationInputLeaf(
                leaf="x", provision_variables=("v",), validation_rows=()
            )

    def test_registry_leaves_are_provision_inputs_when_configs_are_present(
        self,
    ) -> None:
        # Every registered leaf must be a validation-config row id that actually
        # exists in the shipped configs, so a failure names a real row.
        import json
        from importlib.resources import files

        row_ids: set[str] = set()
        for filename, key in (
            ("obbba_reforms.json", "reforms"),
            ("tax_expenditure_reforms.json", "reforms"),
            ("soi_baseline_levels.json", "levels"),
        ):
            payload = json.loads((files("microcosm.build.us") / filename).read_text())
            row_ids.update(row["id"] for row in payload.get(key, ()))
        for entry in US_VALIDATION_PROVISION_INPUT_LEAVES:
            for row in entry.validation_rows:
                assert row in row_ids, (
                    f"{entry.leaf} registered under unknown validation row {row!r}"
                )

    def test_registry_is_current_against_live_engine_graph(self) -> None:
        # The anti-rot check: each registered leaf is a pure input leaf and a
        # dependency of its provision variable per the live PolicyEngine-US
        # graph. Runs only where the [us] extra is installed.
        assert_validation_leaf_registry_current()
