"""Mark the UK outputs that rest on the SPI Public Use Tape (SN 9422; microcosm#1095).

The tape's Special Conditions add two obligations to the End User Licence
(see :mod:`microcosm.build.uk_runtime.receipt_counts`): no attempt may be made
to attribute a record to a named individual, and material that includes
analyses of the tape goes to HMRC at least ten working days before it is
distributed or published. So a build records whether any stage it ran read the
tape, and which ones did. The spine's build record carries the marker; the
calibrated candidate inherits it with the spine's provenance, which the signed
diagnostics hold; and the release manifest restates it, on the microdata's own
entry and with the conditions, rederived from the stages the provenance names.
How SPI-dependent outputs are released is María's ruling: the marker records
the dependency and decides nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from microcosm.build.uk_runtime.spi_income import SPI_DONOR_UKDS_STUDY

#: The key the spine's build record and the spine provenance carry it under.
UK_SPI_PUBLIC_USE_TAPE_KEY = "spi_public_use_tape"
#: The tape's Special Conditions beyond the End User Licence (checked
#: 2026-10-16, microcosm#1278).
UK_SPI_PUBLIC_USE_TAPE_CONDITIONS = (
    "No attempt may be made to attribute a record to a named individual.",
    "Material that includes analyses of the tape goes to HMRC at least ten "
    "working days before it is distributed or published.",
)


def uk_ukds_study_number(value: object) -> str:
    """A declared UK Data Service study number as ``SN <number>``.

    The spec writes most numbers bare (9563) and the tape's as ``SN 9422``.
    """

    text = str(value).strip()
    number = text.removeprefix("SN").strip()
    if not number.isdigit():
        raise ValueError(f"Not a UK Data Service study number: {value!r}.")
    return f"SN {int(number)}"


def uk_spi_public_use_tape_marker(stages: Iterable[Any]) -> dict[str, object]:
    """Whether the given spec stages read the tape, and which of them do.

    ``stages`` are the stages the run selected (a run may select a subset),
    each with the ``stage`` name and the ``artifacts`` the spec declares.
    """

    reading = sorted(
        {
            str(stage.stage)
            for stage in stages
            for artifact in stage.artifacts
            if artifact.get("ukds_study_number") is not None
            and uk_ukds_study_number(artifact["ukds_study_number"])
            == SPI_DONOR_UKDS_STUDY
        }
    )
    return {
        "study": SPI_DONOR_UKDS_STUDY,
        "dependent": bool(reading),
        "stages": reading,
        "conditions": list(UK_SPI_PUBLIC_USE_TAPE_CONDITIONS) if reading else [],
    }


def uk_spi_public_use_tape_marker_for_stage_names(
    names: Iterable[str],
) -> dict[str, object]:
    """The marker for the named stages of the packaged UK spec.

    For readers that hold a run's stage names but not its spec objects (the
    release assemblers, reading the signed spine provenance). A name the
    packaged spec does not declare is refused: the marker would not know what
    that stage read.
    """

    from microcosm.build.country_spec import load_country_spec

    declared = {stage.stage: stage for stage in load_country_spec("uk").sources.stages}
    wanted = list(names)
    unknown = sorted(set(wanted) - set(declared))
    if unknown:
        raise ValueError(f"The packaged UK spec declares no stage {unknown}.")
    return uk_spi_public_use_tape_marker(declared[name] for name in wanted)


def uk_check_spi_public_use_tape_marker(
    recorded: Mapping[str, object] | None,
    derived: Mapping[str, object],
) -> None:
    """Refuse a recorded marker that disagrees with the one the stages give.

    A provenance from a spine built before the marker records none; that is
    allowed, and the derived marker stands.
    """

    if recorded and dict(recorded) != dict(derived):
        raise ValueError(
            "The spine's SPI Public Use Tape marker disagrees with the stages it "
            f"names: recorded {dict(recorded)!r}, derived {dict(derived)!r}."
        )
