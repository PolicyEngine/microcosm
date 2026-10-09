"""Retry preserved graph uploads without registering a telemetry run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microcosm.build.telemetry_emitter import default_spool_path
from microcosm.build.telemetry_emitter_service.auth import (
    CollectorSession,
    validated_origin,
)
from microcosm.build.telemetry_emitter_service.constants import PRODUCTION_COLLECTOR_URL
from microcosm.build.telemetry_emitter_service.graph_publication import (
    DEFAULT_GRAPH_PUBLICATION_ORIGIN,
    GraphPublicationDelivery,
    GraphPublicationQueue,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spool", type=Path, default=default_spool_path())
    parser.add_argument("--publication-id", required=True)
    parser.add_argument(
        "--directory",
        type=Path,
        help="Optional preserved bundle containing orrery.upload.json; re-enqueues after moving machines.",
    )
    parser.add_argument("--origin", default=DEFAULT_GRAPH_PUBLICATION_ORIGIN)
    args = parser.parse_args(argv)
    queue = GraphPublicationQueue(args.spool)
    if args.directory is not None:
        inventory = json.loads((args.directory / "orrery.upload.json").read_text())
        if inventory["publication_id"] != args.publication_id:
            parser.error("The preserved inventory has a different publication ID.")
        queue.enqueue(args.directory, inventory)
    if queue.receipt(args.publication_id) is None:
        parser.error("Publication job not found; supply its preserved directory.")
    queue.retry(args.publication_id)
    session = CollectorSession(validated_origin(PRODUCTION_COLLECTOR_URL))
    delivery = GraphPublicationDelivery(
        queue,
        credential=session.credential,
        origin=args.origin,
        invalidate_credential=session.invalidate,
    )
    # Explicit retries process this job only, not unrelated pending jobs.
    delivery.flush_once(publication_id=args.publication_id)
    receipt = queue.receipt(args.publication_id)
    assert receipt is not None, "A queued graph publication must have a receipt."
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "published" else 1


if __name__ == "__main__":
    raise SystemExit(main())
