# ruff: noqa: F401
from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from referencing.exceptions import NoSuchResource

from microcosm.build.spec_engine.errors import (
    SpecSchemaError,
    SpecValidationError,
)
from microcosm.build.spec_engine.schemas import (
    DRAFT_2020_12,
    SCHEMA_FILENAMES,
    SCHEMA_IDS,
    SchemaRegistry,
    assert_schema_id_allowed,
    load_schema_registry,
)
from microcosm.build.spec_engine.seeds import LEGACY_V1_PROTOCOL

SchemaMutation = Callable[[dict[str, dict[str, Any]]], None]


def _mutated_catalog(
    tmp_path: Path,
    mutation: SchemaMutation,
) -> Path:
    source = Path(load_schema_registry().source)
    documents = {
        filename: json.loads((source / filename).read_text(encoding="utf-8"))
        for filename in SCHEMA_FILENAMES
    }
    mutation(documents)
    for filename, document in documents.items():
        (tmp_path / filename).write_text(
            json.dumps(document, sort_keys=True),
            encoding="utf-8",
        )
    return tmp_path


__all__ = [name for name in globals() if not name.startswith("__")]
