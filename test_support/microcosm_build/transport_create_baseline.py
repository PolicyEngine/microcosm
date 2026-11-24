"""Frozen CREATE baselines: canonical bytes with no donor profile set.

Each scenario builds a toy donor or a composed toy graph and runs CREATE on
its declared parameters and sources. Every scenario except ``toy_v2`` also
loads the same donor through the donor codec; ``toy_v2`` runs
``transport.create@2`` on a donor written by the same call as ``toy``'s, and
its ``codec`` field is null. :func:`scenario_bytes` returns the canonical
bytes of CREATE's frame and receipt and, where the scenario loads it, of the
codec's frame. The committed fixtures under
``packages/microcosm-build/tests/fixtures/transport_create_baseline/`` were
captured at the commit their provenance file names, before CREATE accepted a
donor profile, so a test that recomputes them shows that a graph setting no
profile still gets the same bytes. Write them with::

    python -m test_support.microcosm_build.transport_create_baseline <directory>
"""

from __future__ import annotations

import math
import sys
import tempfile
from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.frame import Frame
from microcosm.graph import KernelContext, Node
from microcosm.graph.canonical import canonical_json

#: The scenarios, in capture order.
SCENARIOS = ("toy", "toy_v2", "toy_profile_columns", "composed", "nz_toy")


def add_profile_columns(tables: dict[str, pd.DataFrame]) -> None:
    """Add the raw survey answers and SPM units a donor profile reads.

    For the six-person synthetic donor. Household 101 owns with a mortgage
    and carries rent; household 202 rents, with one assisted and one
    unassisted SPM unit, and carries mortgage interest. Person 4 (source
    year 2024) has blank enrollment answers. A donor without a profile
    ignores every one of these columns and the ``spm_unit`` table.
    """
    person = tables["person"]
    person["home_mortgage_interest"] = [12.0, 0.0, 0.0, 7.0, 0.0, 0.0]
    person["person_spm_unit_id"] = [11, 11, 11, 21, 22, 22]
    person["A_MARITL"] = [1, 1, 7, 1, 1, 7]
    person["A_ENRLW"] = [2.0, 1.0, 1.0, np.nan, 0.0, 1.0]
    person["A_FTPT"] = [0.0, 1.0, 1.0, np.nan, 0.0, 2.0]
    household = tables["household"]
    household["tenure_type"] = ["OWNED_WITH_MORTGAGE", "RENTED"]
    household["H_TENURE"] = [1, 2]
    tables["spm_unit"] = pd.DataFrame(
        {
            "spm_unit_id": [11, 21, 22],
            "spm_unit_tenure_type": ["OWNER_WITH_MORTGAGE", "RENTER", "RENTER"],
            "receives_housing_assistance": [False, False, True],
        }
    )


def _scalar(value: object) -> object:
    """One cell as a typed canonical JSON value; NA and NaN stay distinct."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else {"nonfinite": repr(value)}
    if isinstance(value, str):
        return value
    raise TypeError(f"No canonical form for {type(value).__name__}.")


def _index(index: pd.Index) -> dict:
    return {
        "name": index.name,
        "dtype": repr(index.dtype),
        "values": [_scalar(value) for value in index.tolist()],
    }


def _series(series: pd.Series) -> dict:
    return {
        "name": series.name,
        "dtype": repr(series.dtype),
        "index": _index(series.index),
        "values": [_scalar(value) for value in series.tolist()],
    }


def _table(table: pd.DataFrame) -> dict:
    return {
        "index": _index(table.index),
        "columns_name": table.columns.name,
        "columns": [
            {
                "name": column,
                "dtype": repr(table[column].dtype),
                "values": [_scalar(value) for value in table[column].tolist()],
            }
            for column in table.columns
        ],
    }


def _metadata(value: object) -> object:
    """Frame metadata with its order and container kinds kept."""
    if isinstance(value, Mapping):
        return {"mapping": [[key, _metadata(item)] for key, item in value.items()]}
    if isinstance(value, tuple):
        return {"tuple": [_metadata(item) for item in value]}
    if isinstance(value, list):
        return {"list": [_metadata(item) for item in value]}
    if isinstance(value, frozenset):
        return {
            "frozenset": sorted(
                (_metadata(item) for item in value),
                key=lambda item: canonical_json(item),
            )
        }
    return _scalar(value)


def frame_document(frame: Frame) -> dict:
    """Every table, dtype, index, weight, stratum and metadata item of a frame."""
    schema = frame.schema
    return {
        "schema": {
            "person_entity": schema.person_entity,
            "group_entities": list(schema.group_entities),
            "links": [
                [link.name, link.left_entity, link.right_entity]
                for link in schema.links
            ],
        },
        "tables": [[entity, _table(frame.table(entity))] for entity in frame.entities],
        "links": [[name, _table(frame.link(name))] for name in frame.links],
        "weights": [
            [
                entity,
                frame.weights_for(entity).kind.value,
                _series(pd.Series(frame.weights_for(entity).values)),
            ]
            for entity in frame.weighted_entities
        ],
        "strata": _series(frame.strata),
        "mass_log": [
            [
                record.entity,
                _scalar(record.old_total),
                _scalar(record.new_total),
                _scalar(record.declared_factor),
                record.reason,
            ]
            for record in frame.mass_log
        ],
        "metadata": _metadata(frame.metadata),
    }


def run_create(
    kernel_ref: str,
    params: Mapping[str, object],
    sources: Mapping[str, Path],
    *,
    node_id: str = "create",
):
    """Run CREATE on declared params and sources, as the CLI's inventory probe.

    CREATE reads no node id, so ``node_id`` names the probe node only.
    """
    from microcosm.build.transport.population_kernels import (
        TRANSPORT_CREATE,
        TRANSPORT_CREATE_DECLARED,
    )

    kernel = {
        TRANSPORT_CREATE.ref: TRANSPORT_CREATE,
        TRANSPORT_CREATE_DECLARED.ref: TRANSPORT_CREATE_DECLARED,
    }[kernel_ref]
    node = Node(node_id, kernel_ref, params=dict(params), sources=tuple(sources))
    return kernel.run(
        KernelContext(
            node=node,
            tables={},
            weights={},
            strata=pd.Series(dtype="float64"),
            params=node.params,
            rng=np.random.default_rng(0),
            sources={name: Path(path).resolve() for name, path in sources.items()},
        )
    )


def _document(
    name: str,
    kernel_ref: str,
    params,
    sources,
    donor: Path | None,
    *,
    node_id: str = "create",
) -> dict:
    from microcosm.build.transport.codecs import load_populace_us_h5

    result = run_create(kernel_ref, params, sources, node_id=node_id)
    return {
        "scenario": name,
        "create": {
            "kernel": kernel_ref,
            "frame": frame_document(result.frame),
            "receipt": result.receipt,
        },
        "codec": None if donor is None else frame_document(load_populace_us_h5(donor)),
    }


def _toy(root: Path, name: str) -> dict:
    from microcosm.frame.unit_construction import MultiUnitRule
    from test_support.microcosm_build.transport_graph import canonical_text
    from test_support.microcosm_build.transport_population import (
        UNIT_RULE,
        create_params,
        digest,
        write_population_sources,
    )

    edit = add_profile_columns if name == "toy_profile_columns" else None
    sources = write_population_sources(
        root / name,
        edit_tables=edit,
        time_period=2024 if edit is not None else None,
    )
    params = create_params(sources)
    kernel = "transport.create@1"
    if name == "toy_v2":
        declaration = MultiUnitRule.from_benefit_unit(UNIT_RULE).to_dict()
        kernel = "transport.create@2"
        params = {
            **params,
            "unit_rule": canonical_text(declaration),
            "unit_rule_sha256": digest(declaration),
        }
    donor = None if name == "toy_v2" else sources.donor
    return _document(
        name,
        kernel,
        params,
        {"donor": sources.donor, "facts": sources.facts},
        donor,
    )


def _composed(root: Path, name: str) -> dict:
    from microcosm.build.transport.compose import skeleton_create_declaration

    if name == "composed":
        from test_support.microcosm_build.transport_composed import (
            make_composed_fixture,
        )

        fixture = make_composed_fixture(root / name)
    else:
        from test_support.microcosm_build.transport_nz_toy import make_nz_toy_fixture

        fixture = make_nz_toy_fixture(root / name)
    declaration = skeleton_create_declaration(fixture.spec)
    sources = {source: fixture.sources[source] for source in declaration.sources}
    donor = sources[declaration.params["donor_source"]]
    # The composed graphs' own CREATE node id, as the CLI's probe uses it.
    return _document(
        name,
        declaration.kernel,
        declaration.params,
        sources,
        donor,
        node_id=declaration.id,
    )


_BUILDERS: Mapping[str, Callable[[Path, str], dict]] = {
    "toy": _toy,
    "toy_v2": _toy,
    "toy_profile_columns": _toy,
    "composed": _composed,
    "nz_toy": _composed,
}


def scenario_document(root: Path, name: str) -> dict:
    """The CREATE and codec document of one scenario, built under ``root``."""
    return _BUILDERS[name](root, name)


def scenario_bytes(root: Path, name: str) -> bytes:
    """The canonical bytes a scenario's fixture holds."""
    return canonical_json(scenario_document(root, name))


def write_fixtures(directory: Path) -> dict[str, str]:
    """Write every scenario's canonical bytes; return their SHA-256 digests."""
    directory.mkdir(parents=True, exist_ok=True)
    digests = {}
    with tempfile.TemporaryDirectory() as temporary:
        for name in SCENARIOS:
            payload = scenario_bytes(Path(temporary), name)
            (directory / f"{name}.json").write_bytes(payload)
            digests[name] = sha256(payload).hexdigest()
    return digests


if __name__ == "__main__":  # pragma: no cover - fixture capture entry point
    for scenario, value in write_fixtures(Path(sys.argv[1])).items():
        print(f"{scenario} {value}")
