"""The input-surface refresh tool reads artifact formats 2 and 3 (G8-x2).

Engine-free. A stand-in CLI answers ``--version``, ``capabilities`` and
``compile`` the way ``tools/refresh_axiom_input_surface.py`` calls them, and
writes the artifact each test gives it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tools import refresh_axiom_input_surface as refresh

_STAND_IN = """\
import json
import sys

ADVERTISED = json.loads({advertised!r})
ARTIFACT = json.loads({artifact!r})
args = sys.argv[1:]
if args == ["--version"]:
    print("axiom-rules-engine 0.0.0-stand-in")
elif args == ["capabilities"]:
    print(json.dumps({{"artifact_format_version": ADVERTISED}}))
elif args[:1] == ["compile"]:
    with open(args[args.index("--output") + 1], "w", encoding="utf-8") as handle:
        json.dump(ARTIFACT, handle)
else:
    sys.exit(f"unexpected arguments {{args}}")
"""

#: One Person rule that reads input ``a`` and a Scalar rule that reads ``b``.
_PROGRAM = {
    "derived": [
        {
            "name": "x",
            "entity": "Person",
            "expr": {
                "kind": "add",
                "left": {"kind": "input", "name": "a"},
                "right": {"kind": "derived", "name": "s"},
            },
            "versions": [],
        },
        {
            "name": "s",
            "entity": "Scalar",
            "expr": {"kind": "input", "name": "b"},
            "versions": [],
        },
    ]
}
_FORMAT_2 = {
    "artifact_format_version": 2,
    "engine_version": "0.0.0",
    "program": _PROGRAM,
    "metadata": {
        "evaluation_order": [],
        "fast_path": {},
        "input_catalog": [{"slot": "a", "canonical_request_name": "nz:a"}],
    },
}


def _format_3() -> dict:
    """The same program as format 3, with the blocks format 3 adds.

    The stand-in ``interface`` entries carry ``kind`` values no expression
    has, so a walk that read the interface would refuse them.
    """
    artifact = copy.deepcopy(_FORMAT_2)
    artifact["artifact_format_version"] = 3
    artifact["engine_build"] = {"commit": "1" * 40, "dirty": False}
    artifact["source_closure"] = {
        "root": "nz:m",
        "modules": [{"target": "nz:m", "sha256": "0" * 64}],
        "digest": "sha256:" + "0" * 64,
    }
    artifact["program"]["interface"] = {
        "entities": [{"name": "Person", "kind": "not_an_expression_kind"}],
        "inputs": [{"slot": "a", "kind": "not_an_expression_kind"}],
    }
    return artifact


def _engine(tmp_path: Path, advertised, artifact: dict | None = None) -> Path:
    script = tmp_path / "stand_in_engine.py"
    script.write_text(
        _STAND_IN.format(
            advertised=json.dumps(advertised),
            artifact=json.dumps(_FORMAT_2 if artifact is None else artifact),
        ),
        encoding="utf-8",
    )
    engine = tmp_path / "axiom-rules-engine"
    engine.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
    )
    engine.chmod(engine.stat().st_mode | stat.S_IXUSR)
    return engine


_MODULE_TEXT = b"stand-in module; the stand-in CLI ignores it\n"


def _rulespec_root(tmp_path: Path) -> Path:
    module = tmp_path / "rulespec" / "nz" / "m.yaml"
    module.parent.mkdir(parents=True)
    module.write_bytes(_MODULE_TEXT)
    return tmp_path / "rulespec"


def _snapshot(tmp_path: Path, artifact: dict) -> dict:
    tmp_path.mkdir()
    engine = _engine(tmp_path, artifact["artifact_format_version"], artifact)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    snapshot, differential = refresh.build_snapshot(
        country="nz",
        rulespec_root=_rulespec_root(tmp_path),
        rulespec_commit="0" * 40,
        engine_bin=engine,
        engine_commit="1" * 40,
        dense_python=None,
        artifact_dir=artifacts,
    )
    assert differential is None
    return snapshot


# ---------------------------------------------------------------------------
# The format gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("artifact_format", [2, 3])
def test_the_walk_reads_formats_2_and_3(tmp_path, artifact_format) -> None:
    artifact = _FORMAT_2 if artifact_format == 2 else _format_3()
    engine = _engine(tmp_path, artifact_format, artifact)
    assert refresh.engine_identity(engine) == (
        "axiom-rules-engine 0.0.0-stand-in",
        artifact_format,
    )
    module = _rulespec_root(tmp_path) / "nz" / "m.yaml"
    compiled, error = refresh.compile_module(
        engine, module, module.parents[1], tmp_path / "m.json", artifact_format
    )
    assert error is None
    assert compiled == artifact


@pytest.mark.parametrize("advertised", [1, 4, 3.0, True, "3", None])
def test_an_engine_of_another_format_refuses(tmp_path, advertised) -> None:
    # 3.0 equals 3 and true equals 1 in Python; neither is a format.
    with pytest.raises(refresh.SurfaceError, match="emits artifact format"):
        refresh.engine_identity(_engine(tmp_path, advertised))


@pytest.mark.parametrize("written", [2, 3.0, None])
def test_an_artifact_off_the_engines_format_refuses(tmp_path, written) -> None:
    artifact = _format_3()
    artifact["artifact_format_version"] = written
    engine = _engine(tmp_path, 3, artifact)
    assert refresh.engine_identity(engine)[1] == 3
    module = _rulespec_root(tmp_path) / "nz" / "m.yaml"
    with pytest.raises(
        refresh.SurfaceError,
        match=f"has artifact format {written!r}; .* names format 3",
    ):
        refresh.compile_module(
            engine, module, module.parents[1], tmp_path / "m.json", 3
        )


def test_a_format_3_artifact_gives_the_format_2_snapshot(tmp_path) -> None:
    # The walk reads program.derived and metadata.input_catalog only, so the
    # blocks format 3 adds change nothing in the snapshot.
    two = _snapshot(tmp_path / "two", _FORMAT_2)
    three = _snapshot(tmp_path / "three", _format_3())
    assert two == three
    assert two["modules"] == [
        {
            "path": "nz/m.yaml",
            "sha256": hashlib.sha256(_MODULE_TEXT).hexdigest(),
            "status": "compiled",
            "inputs": {"Person": ["a", "b"], "Scalar": ["b"]},
            "canonical_inputs": {"a": "nz:a", "b": None},
        }
    ]


# ---------------------------------------------------------------------------
# Dense errors the walk explains
# ---------------------------------------------------------------------------

_BLOCKER = refresh._CROSS_ENTITY.format(
    dependency="hh_income", derived="person_benefit", entity="Household"
)
#: Advice in each of the three forms engine 753ce7d9 writes
#: (``cross_entity_advice``, src/dense.rs:560-590).
_ADVICE = (
    "This dense program's root is the entity-free `Scalar`, whose rows hold "
    "no `Household`: compile it for `Household` instead, or compile only "
    "outputs that do not reach `person_benefit`",
    "These rows' ids are `Household` ids, but dense reads a rule of another "
    "entity only through a relation, and none carries this read: compile only "
    "outputs that do not reach `person_benefit`",
    "Dense evaluates the rows of one entity and reads another entity's rule "
    "only through a relation, by design: read `hh_income` with a related "
    "aggregate (`only`, `sum`, `min`, `max`, `count_where`) over a relation "
    "between `Person` and `Household`, such as `member_of_household`, which "
    "this program declares, or compile only outputs that do not reach "
    "`person_benefit`",
)


def test_a_cross_entity_blocker_explains_its_message_with_or_without_advice() -> None:
    assert refresh.explained(_BLOCKER, [_BLOCKER])
    for advice in _ADVICE:
        assert refresh.explained(f"{_BLOCKER}. {advice}", [_BLOCKER])


@pytest.mark.parametrize(
    "message",
    [
        _BLOCKER + " " + _ADVICE[0],
        _BLOCKER + "." + _ADVICE[0],
        _BLOCKER.replace("`Household`", "`TaxUnit`") + ". " + _ADVICE[0],
        _BLOCKER.replace("`Household`", "`HouseholdX`") + ". " + _ADVICE[0],
        _BLOCKER.replace("`hh_income`", "`hh_wealth`"),
        _BLOCKER[:-1],
    ],
)
def test_a_cross_entity_blocker_explains_no_other_message(message) -> None:
    assert not refresh.explained(message, [_BLOCKER])


def test_any_other_blocker_explains_only_its_own_text() -> None:
    versioned = refresh._VERSIONED.format(name="rr_threshold")
    assert refresh.explained(versioned, [versioned])
    assert not refresh.explained(f"{versioned}. {_ADVICE[2]}", [versioned])
    assert not refresh.explained(_BLOCKER, [versioned])
    assert not refresh.explained(_BLOCKER, [])


_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",
    min_size=1,
    max_size=12,
)
_TRIPLE = st.tuples(_NAME, _NAME, _NAME)


@given(_TRIPLE, _TRIPLE, st.text(max_size=80))
def test_property_a_cross_entity_blocker_explains_exactly_its_names(
    blocked, read, advice
) -> None:
    # A message names (dependency, derived, entity); with or without advice,
    # a blocker explains it exactly when the blocker names the same three.
    def message(names):
        dependency, derived, entity = names
        return refresh._CROSS_ENTITY.format(
            dependency=dependency, derived=derived, entity=entity
        )

    blocker = message(blocked)
    same = blocked == read
    assert refresh.explained(message(read), [blocker]) is same
    assert refresh.explained(f"{message(read)}. {advice}", [blocker]) is same


@given(_NAME, st.text(max_size=120))
def test_property_a_versioned_blocker_explains_only_itself(name, message) -> None:
    versioned = refresh._VERSIONED.format(name=name)
    assert refresh.explained(message, [versioned]) is (message == versioned)
