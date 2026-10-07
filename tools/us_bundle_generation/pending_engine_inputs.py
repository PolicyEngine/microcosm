"""Reviewed unreleased input declarations used only by bundle generation.

The locked PolicyEngine-US version predates the two E19200 mortgage leaves.
Their declared Person/float/year contract lets this migration tool regenerate
the authored bundle before the follow-up engine repin. Runtime ownership and
consumer checks continue to use the installed engine without this fallback.
"""

from microcosm.build.us_runtime.puf_interest_components import (
    US_PUF_E19200_RESIDUAL_PERSON_OUTPUTS,
)
from microcosm.frame.adapters.policyengine_us import (
    PolicyEngineUSVariableMetadataIndex,
)
from microcosm.frame.schema import VariableMetadata


def generator_variable_metadata(
    index: PolicyEngineUSVariableMetadataIndex, name: str
) -> VariableMetadata:
    """Resolve installed metadata or the two reviewed pending declarations."""

    if name in US_PUF_E19200_RESIDUAL_PERSON_OUTPUTS[:2]:
        expected = VariableMetadata(
            name=name, entity="person", dtype="float", period="year"
        )
        try:
            installed = index.variable_metadata(name)
        except ValueError as error:
            if str(error) != f"Unknown PolicyEngine-US source variable {name!r}.":
                raise
            return expected
        if installed != expected:
            raise ValueError(
                f"Pending E19200 input {name!r} differs from its reviewed "
                f"Person/float/year declaration: {installed!r}."
            )
        return installed
    return index.variable_metadata(name)
