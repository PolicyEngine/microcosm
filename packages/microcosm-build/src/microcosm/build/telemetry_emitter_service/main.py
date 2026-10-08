"""Backward-compatible entrypoint; the general service owns composition."""

from microcosm.build.emitter_service.main import build_parser, main

__all__ = ["build_parser", "main"]
