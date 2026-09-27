"""Assign shared geography after the complete combined-survey support clone.

This declaration has no country-specific geography kernel or source admission.
Callers provide normalized, qualified observed columns and declared support.
"""

from microcosm.build.atomic_geography import validate_assignment_spec
from microcosm.build.graph_atomic_geography import atomic_geography_nodes

from .graph_combined_clone import us_combined_survey_clone_nodes

ASSIGNMENT_IDENTITY = (
    "survey_geography_origin_key",
    "household_support_clone_index",
)


def atomic_survey_clone_nodes(
    definition,
    columns,
    *,
    base,
    geography_prefix="geography",
    clone_prefix="combined_survey_puf_support_clone",
):
    """Complete initial clones, then draw on each stable source/role identity.

    The base carries qualified observed constraints, not assigned geography.
    The clone-index Slice depends on its ownership claim; the population edge
    depends on EXPAND. Numeric remapped household IDs are only row coordinates.
    """
    columns = tuple(columns)
    definition = validate_assignment_spec(definition)
    if tuple(definition.get("identity", ())) != ASSIGNMENT_IDENTITY:
        raise ValueError(
            "US postclone geography requires its stable source/clone identity."
        )
    clones = us_combined_survey_clone_nodes(
        columns,
        base=base,
        source_channels=("acs", "asec"),
        prefix=clone_prefix,
    )
    inventory = {(o.entity, o.column): o for o in columns}
    inventory.update({(o.entity, o.column): o for o in clones[1].outputs})
    origin = inventory.get(("household", ASSIGNMENT_IDENTITY[0]))
    if origin is None or origin.dtype != "string":
        raise ValueError("US postclone geography requires a string source identity.")
    geography = atomic_geography_nodes(
        definition,
        tuple(inventory.values()),
        base=clones[0].id,
        prefix=geography_prefix,
        emit_validation_artifact=True,
    )
    return (*clones, *geography)
