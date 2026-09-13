"""Pinned additive consumer status, separate from the immutable money recipe.

The source recipe's declarations and execution identity stay byte-for-byte
unchanged. This graph record reports the routes actually wired by this slice;
it grants no source authority and changes no source address or monetary rule.
"""

from hashlib import sha256
from importlib import resources

from .asec_current_money import RESOURCE_PINS, _parse

GRAPH_CONSUMERS_RESOURCE = "asec_current_money_graph_consumers_v1.json"
GRAPH_CONSUMERS_SHA256 = (
    "67432a10d05e11dd4b97a80cd4f04ed0eb6e56c4c8e21d5dffe87c5f67e27c66"
)


def load_graph_current_money_consumers() -> dict:
    """Verify graph status and its unchanged source-declaration parent."""
    package = resources.files(__package__)
    payload = package.joinpath(GRAPH_CONSUMERS_RESOURCE).read_bytes()
    if sha256(payload).hexdigest() != GRAPH_CONSUMERS_SHA256:
        raise ValueError("GRAPH_CONSUMERS_FINGERPRINT")
    document = _parse(payload)
    parent = document["source_declarations"]
    if (
        parent["resource"] != "asec_current_money_consumers_v1.json"
        or parent["sha256"] != RESOURCE_PINS[2]
        or sha256(package.joinpath(parent["resource"]).read_bytes()).hexdigest()
        != parent["sha256"]
    ):
        raise ValueError("GRAPH_CONSUMERS_PARENT_FINGERPRINT")
    return document
