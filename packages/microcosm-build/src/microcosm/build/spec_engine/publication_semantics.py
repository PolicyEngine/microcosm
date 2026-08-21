"""Pure compilation of publication rungs and release-id grammars."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy

from .errors import SpecValidationError

_APPROVED_RUNG_TOKENS = ("f001", "f004", "f010", "f025", "f100")
_RELEASE_FIELDS = (
    "line",
    "rung",
    "seed",
    "asec_households",
    "acs_households",
    "timestamp",
    "nonce",
)


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SpecValidationError(f"{location}: object required")
    return value


def _array(value: object, location: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise SpecValidationError(f"{location}: array required")
    return value


def publication_rung_rows(
    publication: Mapping[str, object],
) -> list[dict[str, int | float | str]]:
    """Validate and return the sole ordered five-rung declaration."""

    release = _mapping(publication.get("release"), "publication/release")
    rows = _array(release.get("rung_fractions"), "publication/release/rung_fractions")
    if not rows:
        raise SpecValidationError(
            "publication/release/rung_fractions: at least one rung required"
        )
    result: list[dict[str, int | float | str]] = []
    seen_tokens: set[str] = set()
    seen_fractions: set[float] = set()
    for index, value in enumerate(rows):
        row = _mapping(value, f"publication/release/rung_fractions/{index}")
        token = str(row.get("token"))
        try:
            fraction = float(row["fraction"])
            basis_points = int(row["percent_basis_points"])
        except (KeyError, TypeError, ValueError) as error:
            raise SpecValidationError(
                f"publication/release/rung_fractions/{index}: malformed rung"
            ) from error
        if token not in _APPROVED_RUNG_TOKENS:
            raise SpecValidationError(
                "publication/release/rung_fractions/"
                f"{index}/token: unsupported rung {token!r}"
            )
        if token in seen_tokens or fraction in seen_fractions:
            raise SpecValidationError(
                "publication/release/rung_fractions: tokens and fractions "
                "must be unique"
            )
        token_percent = int(token[1:])
        if fraction != token_percent / 100 or basis_points != token_percent * 100:
            raise SpecValidationError(
                "publication/release/rung_fractions/"
                f"{index}: fraction and basis points disagree with {token!r}"
            )
        seen_tokens.add(token)
        seen_fractions.add(fraction)
        result.append(
            {
                "fraction": fraction,
                "token": token,
                "percent_basis_points": basis_points,
            }
        )
    if tuple(row["token"] for row in result) != _APPROVED_RUNG_TOKENS:
        raise SpecValidationError(
            "publication/release/rung_fractions: expected ordered rungs "
            f"{list(_APPROVED_RUNG_TOKENS)!r}"
        )
    return result


def publication_rung_token(publication: Mapping[str, object], fraction: float) -> str:
    """Resolve one exact declared fraction to its release token."""

    matches = [
        str(row["token"])
        for row in publication_rung_rows(publication)
        if row["fraction"] == fraction
    ]
    if len(matches) != 1:
        raise SpecValidationError(
            f"publication/release/rung_fractions: no rung for fraction {fraction!r}"
        )
    return matches[0]


def compile_publication_regex(
    *,
    pattern: str,
    line: str,
    rung_tokens: Sequence[str],
) -> str:
    """Compile the closed release-id template into an anchored reader regex."""

    if re.fullmatch(r"[a-z0-9-]+", line) is None:
        raise SpecValidationError(
            f"publication/release/line: {line!r} is not literal-safe"
        )
    if tuple(rung_tokens) != _APPROVED_RUNG_TOKENS:
        raise SpecValidationError(
            "publication/release/rung_fractions: release grammar requires the "
            "approved ordered rung tokens"
        )
    fields = tuple(match.group(1) for match in re.finditer(r"\{([a-z_]+)\}", pattern))
    if fields != _RELEASE_FIELDS:
        raise SpecValidationError(
            "publication/release/pattern: expected placeholders "
            f"{list(_RELEASE_FIELDS)!r}, found {list(fields)!r}"
        )
    rung_pattern = (
        "f(?:"
        + "|".join(re.escape(token.removeprefix("f")) for token in rung_tokens)
        + ")"
    )
    try:
        body = pattern.format(
            line=line,
            rung=rung_pattern,
            seed="[0-9]+",
            asec_households="[0-9]+",
            acs_households="[0-9]+",
            timestamp="[0-9]{8}T[0-9]{6}Z",
            nonce="[0-9a-f]{8}",
        )
    except (IndexError, KeyError, ValueError) as error:
        raise SpecValidationError(
            "publication/release/pattern: unsupported placeholders"
        ) from error
    return f"^{body}$"


def project_publication_legacy_release(
    publication: Mapping[str, object],
) -> dict[str, object]:
    """Inflate constants-era rung tokens and reader regular expressions."""

    release = _mapping(publication.get("release"), "publication/release")
    line = _mapping(release.get("line"), "publication/release/line")
    line_value = str(line.get("value"))
    prefixes_value = line.get("legacy_prefixes")
    prefixes = _array(prefixes_value, "publication/release/line/legacy_prefixes")
    if not all(isinstance(value, str) for value in prefixes):
        raise SpecValidationError(
            "publication/release/line/legacy_prefixes: strings required"
        )
    pattern = str(release.get("pattern"))
    rung_rows = publication_rung_rows(publication)
    rung_tokens = [str(row["token"]) for row in rung_rows]
    result = deepcopy(dict(release))
    result["rung_fractions"] = rung_rows
    result["rungs"] = rung_tokens
    result["compiled_regex"] = compile_publication_regex(
        pattern=pattern,
        line=line_value,
        rung_tokens=rung_tokens,
    )
    result["legacy_compiled_regexes"] = [
        compile_publication_regex(
            pattern=pattern,
            line=str(prefix),
            rung_tokens=rung_tokens,
        )
        for prefix in prefixes
    ]
    return result


def project_spine_legacy_sampling(
    spine: Mapping[str, object],
    *,
    publication: Mapping[str, object],
) -> dict[str, object]:
    """Resolve the sampling rung reference for constants-era consumers."""

    sampling = _mapping(spine.get("sampling"), "spine/sampling")
    result = deepcopy(dict(sampling))
    fraction = _mapping(result.get("fraction"), "spine/sampling/fraction")
    fraction_result = dict(fraction)
    reference = fraction_result.pop("rungs_ref", None)
    expected = {"domain": "publication", "pointer": "/release/rung_fractions"}
    if reference != expected:
        raise SpecValidationError(
            "spine/sampling/fraction/rungs_ref: expected publication rung "
            f"reference, found {reference!r}"
        )
    default = fraction_result.get("default")
    if isinstance(default, bool) or not isinstance(default, (int, float)):
        raise SpecValidationError("spine/sampling/fraction/default: number required")
    fraction_result["default"] = float(default)
    fraction_result["rungs"] = publication_rung_rows(publication)
    result["fraction"] = fraction_result
    return result


__all__ = [
    "compile_publication_regex",
    "project_publication_legacy_release",
    "project_spine_legacy_sampling",
    "publication_rung_rows",
    "publication_rung_token",
]
