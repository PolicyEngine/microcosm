"""Write a UK local build's clone pool for the private-rent receipt (microcosm#1262).

``tools/uk_private_rent_receipt.py --pool`` reads the households of the whole
clone pool at the problem's design weights. They come from the frame of
``uk.full.pool``, the checkpoint taken after every column writer and before
calibration. Beside each row's rewritten rent and BRMA, it carries its spine
household's from ``uk.full.spine_checkpoint``: a matched before and after for
``uk.full.geography.brma`` and ``uk.full.private_rent``. Pool rows are matched
to their spine household by the key the rent node ranks with
(``property_localisation.spine_household_keys``).

The table is licensed output. It stays in the run's audit directory, outside
the repository. Loading the pool frame takes about 15 GB of memory, so run
this when no build is running.

Usage::

    uv run --no-sync python tools/uk_private_rent_pool.py \\
        --run <runs/head> --graph-store <store> [--out <pool.npz>]
"""

from __future__ import annotations

import argparse
import gc
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.property_localisation import spine_household_keys
from microcosm.graph import ContentStore, RunManifest

POOL_NODE = "uk.full.pool"
SPINE_NODE = "uk.full.spine_checkpoint"
#: The key's source labels only need to agree between the spine and the pool.
KEY_SOURCE = ("frs", "2024_25")


def _frame(store: ContentStore, manifest: RunManifest, node: str):
    record = manifest.nodes[node]
    return store.load_frame(record.frame_key, node_key=record.key)


def _keys(household: pd.DataFrame) -> np.ndarray:
    source, vintage = KEY_SOURCE
    return spine_household_keys(household, source=source, source_vintage=vintage)


def spine_values(store: ContentStore, manifest: RunManifest) -> pd.DataFrame:
    """Each spine household's rent and BRMA, by its spine key."""

    household = _frame(store, manifest, SPINE_NODE).table("household")
    keys = _keys(household)
    if not pd.Index(keys).is_unique:
        raise ValueError("Two spine households share a spine key.")
    return pd.DataFrame(
        {
            "spine_rent": household["rent"].to_numpy(dtype=float),
            "spine_brma": household["brma"].astype(str).to_numpy(),
        },
        index=pd.Index(keys, name="spine_key"),
    )


def pool_table(
    store: ContentStore, manifest: RunManifest, spine: pd.DataFrame
) -> dict[str, np.ndarray]:
    """The pool's household columns at design weights, with the spine's values."""

    frame = _frame(store, manifest, POOL_NODE)
    household = frame.table("household")
    weights = np.asarray(frame.weights_for("household").values, dtype=float)
    person = frame.table("person")[["person_household_id", "housing_benefit_reported"]]
    del frame
    gc.collect()
    benefit = person.groupby("person_household_id")["housing_benefit_reported"].sum()
    matched = spine.reindex(_keys(household))
    if matched["spine_rent"].isna().any():
        raise ValueError("Some pool rows have no spine household.")
    la_column = (
        "local_authority_code" if "local_authority_code" in household else "la_code_oa"
    )
    return {
        "household_id": household["household_id"].to_numpy(dtype=np.int64),
        "weight": weights,
        "region": household["region"].astype(str).to_numpy(dtype=str),
        "tenure_type": household["tenure_type"].astype(str).to_numpy(dtype=str),
        "rent": household["rent"].to_numpy(dtype=float),
        "subrent": household["subrent"].to_numpy(dtype=float),
        "num_bedrooms": household["num_bedrooms"].to_numpy(dtype=float),
        "brma": household["brma"].astype(str).to_numpy(dtype=str),
        "la_code": household[la_column].astype(str).to_numpy(dtype=str),
        "source_key": household["source_household_key"].astype(str).to_numpy(dtype=str),
        "channel": household["household_support_channel"]
        .astype(str)
        .to_numpy(dtype=str),
        "housing_benefit_reported": household["household_id"]
        .map(benefit)
        .fillna(0.0)
        .to_numpy(dtype=float),
        "spine_rent": matched["spine_rent"].to_numpy(dtype=float),
        "spine_brma": matched["spine_brma"].astype(str).to_numpy(dtype=str),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--graph-store", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    store = ContentStore(args.graph_store, create=False)
    manifest = RunManifest.load(args.run / "build.graph.json", store)
    spine = spine_values(store, manifest)
    table = pool_table(store, manifest, spine)
    out = args.out or args.run / "audit-1262" / "pool.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **table)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
