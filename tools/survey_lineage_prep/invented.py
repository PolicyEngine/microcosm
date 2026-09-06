"""Small invented archives through the existing registered source codecs."""

import csv
import hashlib
import importlib.util
import io
import json
import sys
from dataclasses import asdict
from zipfile import ZIP_DEFLATED, ZipFile


def peer(checkout, name):
    path = checkout / "packages/microcosm-build/tests" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make(root, checkout):
    import pytest

    from microcosm.build.us_runtime import graph_composed_population as composed
    from microcosm.build.us_runtime import native_household_origin as origin
    from microcosm.build.us_runtime.graph_sources import frame_column_declarations

    with pytest.MonkeyPatch.context() as patch:
        peer(checkout, "test_us_asec_prepared_source").invented_prepared_directory(
            root,
            patch,
            households_per_cohort=1,
        )
        cp = peer(checkout, "test_us_graph_composed_population")
        cp.invented_acs_archives(root / "acs")
        # Both ACS readers consume these same archives. Native-origin authority
        # requires the original lexical Census namespace, not generated IDs.
        for filename in ("csv_hus.zip", "csv_pus.zip"):
            path = root / "acs" / filename
            with ZipFile(path) as archive:
                members = {name: archive.read(name) for name in archive.namelist()}
            with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
                for name, payload in members.items():
                    rows = list(csv.reader(io.StringIO(payload.decode())))
                    integer_columns = {
                        "TEN",
                        "NP",
                        "WGTP",
                        "TYPEHUGQ",
                        "SPORDER",
                        "PWGTP",
                    }
                    for row in rows[1:]:
                        for i, column in enumerate(rows[0]):
                            if column in integer_columns and row[i].endswith(".0"):
                                row[i] = row[i][:-2]
                    stream = io.StringIO()
                    csv.writer(stream, lineterminator="\n").writerows(rows)
                    payload = stream.getvalue().encode()
                    archive.writestr(
                        name,
                        payload.replace(b"0000000000123", b"2024HU0000123").replace(
                            b"0000000000045", b"2024HU0000045"
                        ),
                    )
        peer(checkout, "test_us_graph_composed_asec_binding")._write_pins(root)
        cp._lookup_files(root)
        direct = composed.compose_from_sources(
            root / "prepared-source",
            root / "acs",
            sample_fraction=1.0,
            sample_seed=17,
        )
        columns = [
            [c.entity, c.column, c.dtype]
            for c in frame_column_declarations(direct.frame)
        ]
        (root / "columns.json").write_text(json.dumps(columns))
        projection = peer(
            checkout, "test_us_graph_composed_asec_demographics"
        )._projection(
            root,
            patch,
            unknown=True,
        )
        pins = []
        for year in (2022, 2023, 2024):
            # The prepared fixture has H_SEQ=7 in each distinct income cohort.
            payload = b"UNUSED,H_SEQ\nignored,7\n"
            path = root / f"hhpub{str(year + 1)[-2:]}.csv"
            path.write_bytes(payload)
            pins.append(
                asdict(
                    origin.AsecNativeMemberPin(
                        income_year=year,
                        survey_year=year + 1,
                        canonical_member_id=f"invented/asec/{year + 1}/household",
                        member_name=path.name,
                        archive_sha256=origin.ASEC_EDUCATION_ASSISTANCE_ARCHIVES[
                            year
                        ].zip_sha256,
                        member_sha256=hashlib.sha256(payload).hexdigest(),
                        size_bytes=len(payload),
                        rows=1,
                    )
                )
            )
        acs_pins = []
        for role, filename in (("household", "csv_hus.zip"), ("person", "csv_pus.zip")):
            payload = (root / "acs" / filename).read_bytes()
            acs_pins.append(
                [role, filename, hashlib.sha256(payload).hexdigest(), len(payload)]
            )
        (root / "origin-pins.json").write_text(
            json.dumps({"asec": pins, "acs": acs_pins})
        )
        (root / "projection-identity.json").write_text(
            json.dumps(
                {
                    "projection_sha256": hashlib.sha256(
                        (root / "demographic.bin").read_bytes()
                    ).hexdigest(),
                    "rows": projection.receipt["rows"],
                    "cohort_rows": [[y, 2] for y in (2022, 2023, 2024)],
                }
            )
        )
    return {
        "households": direct.frame.n("household"),
        "persons": direct.frame.n("person"),
    }


def restore(root, checkout, *, baseline=False):
    peer(checkout, "test_us_graph_composed_population")._restore_test_pins(root)
    if not baseline:
        from microcosm.build.us_runtime import asec_demographic_source as demographic
        from microcosm.build.us_runtime import native_household_origin as origin

        demographic._MEMBER_PINS = tuple(
            tuple(row)
            for row in json.loads((root / "demographic-pins.json").read_text())
        )
        pins = json.loads((root / "origin-pins.json").read_text())
        origin._ASEC_MEMBER_PINS = tuple(
            origin.AsecNativeMemberPin(**pin) for pin in pins["asec"]
        )
        origin.acs._ARCHIVE_PINS = tuple(tuple(pin) for pin in pins["acs"])


def sources(root, *, baseline=False):
    from microcosm.build.us_runtime import graph_composed_population as composed
    from microcosm.build.us_runtime.graph_geography import LOOKUP_SOURCES

    result = {
        composed.ASEC_PREPARED_SOURCE_NAME: root / "prepared-source",
        composed.ACS_NATIVE_SOURCE_NAME: root / "acs",
        LOOKUP_SOURCES[0].name: root / "ladder.npz",
        LOOKUP_SOURCES[1].name: root / "crosswalk.csv",
    }
    if not baseline:
        from microcosm.build.us_runtime import (
            graph_composed_asec_demographics as demographic,
        )
        from microcosm.build.us_runtime import graph_native_household_origin as origin

        result[demographic.PROJECTION_SOURCE_NAME] = root / "demographic.bin"
        result[origin.ACS_SOURCE] = root / "acs"
        result.update(
            {
                name: root / f"hhpub{str(year + 1)[-2:]}.csv"
                for year, name in zip(
                    (2022, 2023, 2024), origin.ASEC_SOURCES, strict=True
                )
            }
        )
    return result
