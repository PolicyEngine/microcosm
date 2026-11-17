"""The input-surface refresh tool reads artifact formats 2 and 3 (G8-x2).

Engine-free. A stand-in CLI answers ``--version``, ``capabilities`` and
``compile`` the way ``tools/refresh_axiom_input_surface.py`` calls them, and
writes the artifact each test gives it. A stand-in ``--dense-python`` reads
the dense probe's jobs from stdin and answers each entity from the outcomes
its test gives it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
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


def _wrapper(path: Path, script: Path) -> Path:
    """An executable at ``path`` that runs ``script`` under this interpreter."""
    path.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _engine(tmp_path: Path, advertised, artifact: dict | None = None) -> Path:
    script = tmp_path / "stand_in_engine.py"
    script.write_text(
        _STAND_IN.format(
            advertised=json.dumps(advertised),
            artifact=json.dumps(_FORMAT_2 if artifact is None else artifact),
        ),
        encoding="utf-8",
    )
    return _wrapper(tmp_path / "axiom-rules-engine", script)


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


@pytest.mark.parametrize(
    ("advertised", "written"), [(3, 2), (3, 3.0), (3, None), (2, 3)]
)
def test_an_artifact_off_the_engines_format_refuses(
    tmp_path, advertised, written
) -> None:
    artifact = _format_3()
    artifact["artifact_format_version"] = written
    engine = _engine(tmp_path, advertised, artifact)
    assert refresh.engine_identity(engine)[1] == advertised
    module = _rulespec_root(tmp_path) / "nz" / "m.yaml"
    with pytest.raises(
        refresh.SurfaceError,
        match=f"has artifact format {written!r}; .* names format {advertised}",
    ):
        refresh.compile_module(
            engine, module, module.parents[1], tmp_path / "m.json", advertised
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
#: Advice in each form engine 753ce7d9 writes (``cross_entity_advice``,
#: src/dense.rs:560-590): for a Scalar root, for a filtered entity's rows, and
#: for a read through a relation, with one the program declares and with none.
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
    "Dense evaluates the rows of one entity and reads another entity's rule "
    "only through a relation, by design: read `hh_income` with a related "
    "aggregate (`only`, `sum`, `min`, `max`, `count_where`) over a relation "
    "between `Person` and `Household` (this program declares none; a "
    "`data_relation` whose `arguments` name both entities is one), or compile "
    "only outputs that do not reach `person_benefit`",
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


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
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
    assert refresh.explained(blocker, [blocker])
    assert refresh.explained(f"{blocker}. {advice}", [blocker])
    same = blocked == read
    assert refresh.explained(message(read), [blocker]) is same
    assert refresh.explained(f"{message(read)}. {advice}", [blocker]) is same


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_NAME, _NAME, st.text(max_size=120))
def test_property_a_versioned_blocker_explains_only_itself(name, other, text) -> None:
    # Its own text matches. Another rule's versioned text matches only under
    # the same name, and no text matches with advice after it.
    versioned = refresh._VERSIONED.format(name=name)
    assert refresh.explained(versioned, [versioned])
    assert refresh.explained(refresh._VERSIONED.format(name=other), [versioned]) is (
        other == name
    )
    assert refresh.explained(text, [versioned]) is (text == versioned)
    assert not refresh.explained(f"{versioned}. {text}", [versioned])


# ---------------------------------------------------------------------------
# Dense mode
# ---------------------------------------------------------------------------

#: Stands in for ``--dense-python``. It ignores the probe after ``-c``, writes
#: the jobs it reads from stdin to JOBS, and answers every entity a job names
#: from OUTCOMES.
_DENSE_STAND_IN = """\
import json
import sys

OUTCOMES = json.loads({outcomes!r})
if sys.argv[1:2] != ["-c"]:
    sys.exit(f"unexpected arguments {{sys.argv[1:]}}")
jobs = json.load(sys.stdin)
with open({jobs!r}, "w", encoding="utf-8") as handle:
    json.dump(jobs, handle)
results = []
for job in jobs:
    entities = {{entity: OUTCOMES[entity] for entity in job["entities"]}}
    results.append({{"module": job["module"], "entities": entities}})
json.dump(results, sys.stdout)
"""
_CATALOG = {"h": "nz:h", "p": "nz:p"}
#: Household's dense program as the walk sees it: root input ``h``.
_HOUSEHOLD = {
    "root_entity": "Household",
    "root_inputs": ["h"],
    "input_catalog": _CATALOG,
}


def _cross_entity_artifact() -> dict:
    """A Person rule that reads a Household rule, as format 3.

    The walk blocks Person's dense program with ``_BLOCKER`` and finds
    Household's unblocked.
    """
    artifact = _format_3()
    artifact["program"]["derived"] = [
        {
            "name": "person_benefit",
            "entity": "Person",
            "expr": {
                "kind": "add",
                "left": {"kind": "input", "name": "p"},
                "right": {"kind": "derived", "name": "hh_income"},
            },
            "versions": [],
        },
        {
            "name": "hh_income",
            "entity": "Household",
            "expr": {"kind": "input", "name": "h"},
            "versions": [],
        },
    ]
    artifact["metadata"]["input_catalog"] = [
        {"slot": slot, "canonical_request_name": name}
        for slot, name in _CATALOG.items()
    ]
    return artifact


def _dense_python(tmp_path: Path, outcomes: dict) -> Path:
    script = tmp_path / "stand_in_dense.py"
    script.write_text(
        _DENSE_STAND_IN.format(
            outcomes=json.dumps(outcomes), jobs=str(tmp_path / "dense_jobs.json")
        ),
        encoding="utf-8",
    )
    return _wrapper(tmp_path / "dense-python", script)


def _dense_run(tmp_path: Path, outcomes: dict) -> tuple[dict, dict | None]:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return refresh.build_snapshot(
        country="nz",
        rulespec_root=_rulespec_root(tmp_path),
        rulespec_commit="0" * 40,
        engine_bin=_engine(tmp_path, 3, _cross_entity_artifact()),
        engine_commit="1" * 40,
        dense_python=_dense_python(tmp_path, outcomes),
        artifact_dir=artifacts,
    )


@pytest.mark.parametrize("advice", ["", *(f". {advice}" for advice in _ADVICE)])
def test_dense_mode_records_the_walks_text_for_a_cross_entity_error(
    tmp_path, advice
) -> None:
    # Engines from 4d7ec14f on add advice to the error. The snapshot keeps
    # the walk's text, so a change in the engine's advice can't move it.
    snapshot, differential = _dense_run(
        tmp_path, {"Household": _HOUSEHOLD, "Person": {"error": _BLOCKER + advice}}
    )
    root = tmp_path / "rulespec"
    jobs = json.loads((tmp_path / "dense_jobs.json").read_text(encoding="utf-8"))
    assert jobs == [
        {
            "module": str(root / "nz" / "m.yaml"),
            "root": str(root),
            "entities": ["Household", "Person"],
        }
    ]
    assert snapshot["engine"]["surface"] == refresh.SURFACE_DENSE
    assert snapshot["modules"] == [
        {
            "path": "nz/m.yaml",
            "sha256": hashlib.sha256(_MODULE_TEXT).hexdigest(),
            "status": "dense_unsupported",
            "inputs": {"Household": ["h"]},
            "canonical_inputs": _CATALOG,
            "error": f"Person: {_BLOCKER}",
            "dense_unsupported_entities": ["Person"],
            "cli_walk_inputs": {"Person": ["p"]},
        }
    ]
    assert differential == {
        "entity_programs": 2,
        "dense_compiled": 1,
        "input_mismatches": [],
        "status_mismatches": [],
        "catalog_mismatches": [],
    }


@pytest.mark.parametrize(
    ("outcomes", "match"),
    [
        pytest.param(
            {
                "Household": _HOUSEHOLD,
                "Person": {
                    "error": _BLOCKER.replace("`hh_income`", "`hh_wealth`")
                    + f". {_ADVICE[2]}"
                },
            },
            r"nz/m\.yaml \[Person\]: dense error not explained by the artifact walk",
            id="another-cross-entity-read",
        ),
        pytest.param(
            {
                "Household": _HOUSEHOLD,
                "Person": {"error": "dense compilation does not yet support `y`"},
            },
            r"nz/m\.yaml \[Person\]: dense error not explained by the artifact walk",
            id="a-new-limitation",
        ),
        pytest.param(
            {
                "Household": {"error": f"{_BLOCKER}. {_ADVICE[0]}"},
                "Person": {"error": _BLOCKER},
            },
            r"nz/m\.yaml \[Household\]: dense error not explained by the "
            r"artifact walk",
            id="an-entity-the-walk-doesnt-block",
        ),
        pytest.param(
            {
                "Household": {"error": refresh._NO_PROGRAM.format("Household")},
                "Person": {"error": _BLOCKER},
            },
            r"nz/m\.yaml: dense reports no Household program",
            id="no-program",
        ),
        pytest.param(
            {
                "Household": {**_HOUSEHOLD, "root_entity": "Person"},
                "Person": {"error": _BLOCKER},
            },
            r"nz/m\.yaml: asked for Household, dense rooted at Person",
            id="another-root",
        ),
        pytest.param(
            {
                "Household": {**_HOUSEHOLD, "root_inputs": ["h", "h"]},
                "Person": {"error": _BLOCKER},
            },
            r"nz/m\.yaml \[Household\]: duplicate root inputs",
            id="duplicate-root-inputs",
        ),
    ],
)
def test_dense_mode_refuses_what_the_walk_cant_account_for(
    tmp_path, outcomes, match
) -> None:
    with pytest.raises(refresh.SurfaceError, match=match):
        _dense_run(tmp_path, outcomes)


def test_dense_mode_counts_where_dense_and_the_walk_disagree(tmp_path) -> None:
    # Dense compiles Person although the walk blocks it, reads an input the
    # walk doesn't, and reports another catalog for Household.
    snapshot, differential = _dense_run(
        tmp_path,
        {
            "Household": {
                **_HOUSEHOLD,
                "root_inputs": ["h", "q"],
                "input_catalog": {"h": "nz:h"},
            },
            "Person": {
                "root_entity": "Person",
                "root_inputs": ["p"],
                "input_catalog": _CATALOG,
            },
        },
    )
    assert differential == {
        "entity_programs": 2,
        "dense_compiled": 2,
        "input_mismatches": [("nz/m.yaml", "Household", ["q"], [])],
        "status_mismatches": [("nz/m.yaml", "Person", _BLOCKER)],
        "catalog_mismatches": [("nz/m.yaml", "Household")],
    }
    (module,) = snapshot["modules"]
    assert module["status"] == "compiled"
    assert module["inputs"] == {"Household": ["h", "q"], "Person": ["p"]}
    assert module["canonical_inputs"] == {"h": "nz:h", "p": "nz:p", "q": None}
