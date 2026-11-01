"""Which UK stage transforms declare fit-weight records (microcosm#1428).

The release-cut weights audit reads an empty list of fit-weight records as a
fitting stage that recorded nothing, and fails (``uk_weights_audit``). The ISA
subscriptions stage declared an empty tuple although it fits no model, and
every national certified after #1421 failed. This roster pins the stage
transforms that declare the hook, read from the source so it runs without
the engine: a new one must be reviewed, and either record each fit it makes
or declare nothing (the #1428 review, finding 3).
"""

from __future__ import annotations

import ast
from pathlib import Path

from microcosm.build import uk_runtime

#: The stage transforms that fit a model and record its weights.
FITTING_STAGE_TRANSFORMS = frozenset(
    {
        "etb_services.UKETBServicesStageTransform",
        "etb_vat.UKETBVATStageTransform",
        "lcfs_consumption.UKLCFSConsumptionStageTransform",
        "nts_bus_travel.UKNTSBusTravelStageTransform",
        "property_components.UKPropertyComponentsStageTransform",
        "spi_spine.UKSPIIncomeSpineStageTransform",
        "was_lisa.UKWASLISAStageTransform",
        "was_wealth.UKWASWealthStageTransform",
    }
)


def _declares_hook(node: ast.ClassDef) -> bool:
    for item in node.body:
        if (
            isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
            and item.name == "fit_weight_records"
        ):
            return True
        if isinstance(item, ast.AnnAssign | ast.Assign) and "fit_weight_records" in {
            target.id
            for target in (
                [item.target] if isinstance(item, ast.AnnAssign) else item.targets
            )
            if isinstance(target, ast.Name)
        }:
            return True
    return False


def test_only_the_reviewed_stage_transforms_declare_fit_weight_records() -> None:
    declared = set()
    for path in sorted(Path(uk_runtime.__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ClassDef)
                and node.name.endswith("StageTransform")
                and _declares_hook(node)
            ):
                declared.add(f"{path.stem}.{node.name}")
    assert declared == FITTING_STAGE_TRANSFORMS


def test_the_isa_subscriptions_stage_declares_nothing() -> None:
    assert "isa_subscriptions.UKISASubscriptionsStageTransform" not in (
        FITTING_STAGE_TRANSFORMS
    )
