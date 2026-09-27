"""Validate the measured person-signal summary shape before gate decisions."""

from collections.abc import Mapping


def validate_person_signal_summary(
    summary: Mapping[str, object],
    *,
    family: str,
    outputs: tuple[str, ...],
    share_bands: Mapping[str, tuple[str, tuple[float, float]]],
    invariants: tuple[str, ...] = (),
) -> None:
    """Require genuine-summary fields, numeric domains and registered bands.

    This validates evidence shape and the fixed decision policy, not the source
    identity or truth of the supplied measurements. Genuine summary functions
    emit Python integers/floats and list-valued bands; immutable tuple bands are
    also accepted for callers that freeze the same data in memory.
    """
    prefix = f"Malformed {family}-input summary"
    fields = {
        "unique_counts",
        *share_bands,
        *(name for name, _ in share_bands.values()),
        *invariants,
    }
    if not isinstance(summary, Mapping) or set(summary) != fields:
        raise ValueError(f"{prefix}: expected the exact registered field inventory.")
    counts = summary["unique_counts"]
    if not isinstance(counts, Mapping) or set(counts) != set(outputs):
        raise ValueError(f"{prefix}: expected a count for every registered output.")
    for name, value in (*counts.items(), *((key, summary[key]) for key in invariants)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{prefix}: {name!r} must be a nonnegative integer.")
    for share_name, (band_name, registered) in share_bands.items():
        share = summary[share_name]
        # The closed interval also refuses both infinities and NaN without
        # coercing strings, bools, or oversized integer evidence to float64.
        if type(share) not in (int, float) or not 0.0 <= share <= 1.0:
            raise ValueError(f"{prefix}: {share_name!r} must be finite in [0, 1].")
        band = summary[band_name]
        if (
            not isinstance(band, (list, tuple))
            or len(band) != 2
            or any(type(value) not in (int, float) for value in band)
            or tuple(band) != registered
        ):
            raise ValueError(
                f"{prefix}: {band_name!r} differs from its registered band."
            )
