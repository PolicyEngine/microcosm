#!/usr/bin/env python
"""Generate the UK local-area identity crosswalk from the OA ladder artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from microcosm.build.uk_runtime.geography_ladder import region_tier_by_area

DEFAULT_LADDER_ARTIFACT = Path("build/uk/uk_oa_ladder_2021.npz")
DEFAULT_LADDER_SUMMARY = Path("build/uk/ladder_summary.json")
DEFAULT_OUTPUT = Path(
    "packages/microcosm-build/src/microcosm/build/uk/local_area_crosswalk.json"
)

EXPECTED_COUNTS = {
    "constituency": 650,
    "local_authority": 361,
}
# Accept-sets, not single labels: publishers stamp the vintage of the LOOKUP
# they published against, and ONS products list every UK local authority on
# the 2023 LAD frame -- including Scottish council areas and Northern Irish
# districts, whose boundaries (and GSS codes) are unchanged since ca_2019 /
# lgd_2014. The devolved publishers stamp their own frames. Both labels name
# the same boundary set, so the compile accepts either; a vintage OUTSIDE the
# set is a genuinely different boundary frame and still refuses.
#: Roster areas a publisher recodes after the roster's boundary vintage. The
#: ladder (and so the roster) is on the April 2023 LAD frame; ONS recoded
#: Barnsley and Sheffield in the April 2025 LAD layer and MHCLG's 2025 council
#: taxbase return files them under the new codes (chronicle #264 review). A
#: consumer selects facts under either spelling and keeps the roster code as
#: the cell's identity; the alias vintage is accepted for the aliased facts
#: only. Nothing else about the two authorities changed (microcosm#929).
LOCAL_AUTHORITY_CODE_ALIASES: dict[str, dict[str, Any]] = {
    "E08000016": {
        "alias_codes": ["E08000038"],
        "alias_vintage": "lad_2025",
        "note": "Barnsley, recoded E08000038 in the April 2025 ONS LAD layer.",
    },
    "E08000019": {
        "alias_codes": ["E08000039"],
        "alias_vintage": "lad_2025",
        "note": "Sheffield, recoded E08000039 in the April 2025 ONS LAD layer.",
    },
}

EXPECTED_FACT_VINTAGE = {
    "constituency": ["pcon_2024"],
    "local_authority": {
        "E": ["lad_2023"],
        "W": ["lad_2023"],
        "S": ["ca_2019", "lad_2023"],
        "N": ["lgd_2014", "lad_2023"],
    },
}


def main() -> None:
    args = _parser().parse_args()
    resource = build_local_area_crosswalk(
        ladder_artifact=args.ladder_artifact,
        ladder_summary=args.ladder_summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(resource, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                level: len(payload["area_ids"])
                for level, payload in resource["levels"].items()
            },
            sort_keys=True,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder-artifact", type=Path, default=DEFAULT_LADDER_ARTIFACT)
    parser.add_argument("--ladder-summary", type=Path, default=DEFAULT_LADDER_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def build_local_area_crosswalk(
    *,
    ladder_artifact: Path,
    ladder_summary: Path,
) -> dict[str, Any]:
    """Return the packaged crosswalk resource for the supplied ladder artifact."""

    if not ladder_artifact.exists():
        raise FileNotFoundError(f"UK OA ladder artifact not found: {ladder_artifact}")
    if not ladder_summary.exists():
        raise FileNotFoundError(f"UK OA ladder summary not found: {ladder_summary}")
    summary = json.loads(ladder_summary.read_text(encoding="utf-8"))
    with np.load(ladder_artifact, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        levels = {
            "constituency": _level_payload(
                payload,
                metadata,
                level="constituency",
                code_column="constituency_code",
                ladder_layer="constituency",
            ),
            "local_authority": _level_payload(
                payload,
                metadata,
                level="local_authority",
                code_column="local_authority_code",
                ladder_layer="local_authority",
            ),
        }

    ladder_sha = _sha256(ladder_artifact)
    summary_sha = str(summary.get("output_sha256") or "")
    if summary_sha and ladder_sha != summary_sha:
        raise ValueError(
            f"UK OA ladder artifact sha256 {ladder_sha} does not match "
            f"ladder_summary.json output_sha256 {summary_sha}."
        )
    return {
        "schema_version": 1,
        "country": "uk",
        "kind": "uk_local_area_crosswalk",
        "description": (
            "Identity crosswalk for UK local Ledger target references. Area "
            "ids come from the packaged OA ladder roster; facts must carry "
            "the same geography ids and the declared fact-side vintages."
        ),
        "ladder_artifact": ladder_artifact.as_posix(),
        "ladder_artifact_sha256": ladder_sha,
        "ladder_summary": ladder_summary.as_posix(),
        "ladder_summary_sha256": _sha256(ladder_summary),
        "levels": levels,
    }


def _level_payload(
    payload: Any,
    metadata: dict[str, Any],
    *,
    level: str,
    code_column: str,
    ladder_layer: str,
) -> dict[str, Any]:
    if code_column not in payload.files:
        raise ValueError(f"UK OA ladder artifact is missing {code_column!r}.")
    if "region_code" not in payload.files:
        raise ValueError("UK OA ladder artifact is missing 'region_code'.")
    codes = np.asarray(payload[code_column]).astype(str)
    area_ids = sorted(set(codes))
    expected = EXPECTED_COUNTS[level]
    if len(area_ids) != expected:
        raise ValueError(
            f"UK OA ladder {level} roster has {len(area_ids)} area id(s), "
            f"expected {expected}."
        )
    region_code_by_area = region_tier_by_area(
        codes, np.asarray(payload["region_code"]).astype(str), level=level
    )
    layers = metadata.get("layers") or {}
    layer = layers.get(ladder_layer) or {}
    ladder_vintage = str(layer.get("vintage") or "")
    if not ladder_vintage:
        raise ValueError(f"UK OA ladder metadata is missing {ladder_layer} vintage.")
    countries = layer.get("countries")
    if not isinstance(countries, dict) or not countries:
        raise ValueError(
            f"UK OA ladder metadata is missing {ladder_layer} country sources."
        )
    ladder_layer_sources: dict[str, dict[str, str]] = {}
    required_source_fields = ("source", "url", "sha256", "vintage")
    for country, source in sorted(countries.items()):
        if not isinstance(source, dict):
            raise ValueError(
                f"UK OA ladder {ladder_layer} source {country!r} must be an object."
            )
        missing = [field for field in required_source_fields if not source.get(field)]
        if missing:
            raise ValueError(
                f"UK OA ladder {ladder_layer} source {country!r} is missing {missing}."
            )
        ladder_layer_sources[str(country)] = {
            field: str(source[field]) for field in required_source_fields
        }
    return {
        "ledger_geography_level": level,
        "ladder_layer": ladder_layer,
        "ladder_layer_sources": ladder_layer_sources,
        "ladder_code_column": code_column,
        "ladder_vintage": ladder_vintage,
        "expected_vintage": EXPECTED_FACT_VINTAGE[level],
        "area_count": len(area_ids),
        "area_ids": area_ids,
        "region_code_by_area": region_code_by_area,
        "code_aliases": (
            {
                code: dict(alias)
                for code, alias in sorted(LOCAL_AUTHORITY_CODE_ALIASES.items())
            }
            if level == "local_authority"
            else {}
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
