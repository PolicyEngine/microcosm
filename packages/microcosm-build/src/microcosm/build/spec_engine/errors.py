"""Stable public exceptions for the spec-engine front end."""

from __future__ import annotations


class SpecEngineError(ValueError):
    """Base class for deterministic, user-facing spec errors.

    ``line`` and ``column`` are one-based so the rendered location can be
    pasted directly into an editor.  The original parser exception is kept as
    the Python exception cause, never interpolated into this stable message.
    """

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        self.message = message
        self.source = source
        self.line = line
        self.column = column

        location = source
        if location is not None and line is not None:
            location = f"{location}:{line}"
            if column is not None:
                location = f"{location}:{column}"
        elif location is None and line is not None:
            location = str(line)
            if column is not None:
                location = f"{location}:{column}"

        rendered = f"{location}: {message}" if location else message
        super().__init__(rendered)


class SpecParseError(SpecEngineError):
    """The authored YAML is outside the accepted YAML 1.2 subset."""


class SpecSchemaError(SpecEngineError):
    """The immutable schema catalog itself is missing or inconsistent."""


class SpecValidationError(SpecEngineError):
    """A parsed resource does not conform to its closed-world schema."""
