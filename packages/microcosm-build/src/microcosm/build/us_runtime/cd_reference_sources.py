"""Pinned public ACS acquisition; credentials never enter stored provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

TABLES = ("S0101", "B01001", "B19001", "B25003")
MAX_RESPONSE_BYTES = 32 * 1024 * 1024


def census_requests() -> list[dict]:
    """The complete, fixed public reference request surface."""
    requests = []
    for table in TABLES:
        dataset = "acs/acs1/subject" if table == "S0101" else "acs/acs1"
        base = f"https://api.census.gov/data/2024/{dataset}"
        for kind, url in (
            ("metadata", f"{base}/groups/{table}.json"),
            (
                "data",
                base
                + "?"
                + urlencode(
                    {
                        "get": f"group({table})",
                        "for": "congressional district:*",
                        "in": "state:*",
                    }
                ),
            ),
        ):
            requests.append(
                {
                    "table": table,
                    "kind": kind,
                    "year": 2024,
                    "dataset": dataset,
                    "url": url,
                }
            )
    return requests


def strict_json(payload: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("nonfinite JSON number")

    try:
        return json.loads(payload, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("source is not valid UTF-8 JSON") from None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def fetch_census(url: str, api_key: str | None) -> bytes:
    """Fetch a declared request, suppressing URL-bearing transport errors."""
    if url not in {item["url"] for item in census_requests()}:
        raise ValueError("request is not declared")
    request_url = url
    if api_key:
        request_url += ("&" if "?" in url else "?") + urlencode({"key": api_key})
    try:
        with build_opener(_NoRedirect()).open(
            Request(request_url, headers={"User-Agent": "Microcosm-CD-reference/1"}),
            timeout=90,
        ) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        raise ValueError(f"Census request failed (HTTP {error.code})") from None
    except (URLError, OSError, ValueError):
        raise ValueError("Census request failed (transport error)") from None
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError("Census response exceeds size limit")
    return payload


def read_census_key(service: str, account: str) -> str:
    """Read only the explicitly requested agent-secret into process memory."""
    try:
        result = subprocess.run(
            ["agent-secret", "get", service, account],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        raise ValueError("Census credential helper failed") from None
    if result.returncode or not result.stdout.strip():
        raise ValueError("Census credential unavailable")
    try:
        return result.stdout.decode("utf-8").strip()
    except UnicodeError:
        raise ValueError("Census credential is not text") from None


def write_json(path: Path, value) -> None:
    payload = (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        + b"\n"
    )
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refusing to replace immutable artifact: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def capture_request(
    root: Path,
    request: dict,
    *,
    api_key: str | None = None,
    fetch=fetch_census,
    offline: bool = False,
) -> dict:
    """Fetch once, pin exact response bytes, then verify on every replay."""
    if request not in census_requests():
        raise ValueError("request metadata is not declared")
    root = Path(root)
    request_key = hashlib.sha256(request["url"].encode()).hexdigest()
    descriptor_path = root / "requests" / f"{request_key}.json"
    if descriptor_path.exists():
        descriptor = strict_json(descriptor_path.read_bytes())
        digest = descriptor.get("sha256", "")
        expected_path = f"raw/{digest}.json"
        if (
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or descriptor.get("path") != expected_path
            or any(descriptor.get(key) != value for key, value in request.items())
        ):
            raise ValueError("cached request descriptor mismatch")
        payload = (root / expected_path).read_bytes()
        if (
            hashlib.sha256(payload).hexdigest() != digest
            or len(payload) != descriptor["size_bytes"]
        ):
            raise ValueError("cached source digest mismatch")
        strict_json(payload)
        return descriptor
    if offline:
        raise ValueError("offline source is missing")
    payload = fetch(request["url"], api_key)
    if api_key and api_key.encode() in payload:
        raise ValueError("response contains credential; refusing to persist")
    strict_json(payload)
    digest = hashlib.sha256(payload).hexdigest()
    relative_path = f"raw/{digest}.json"
    raw_path = root / relative_path
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        if raw_path.read_bytes() != payload:
            raise ValueError("raw source digest collision")
    else:
        with raw_path.open("xb") as stream:
            stream.write(payload)
    descriptor = {
        **request,
        "sha256": digest,
        "path": relative_path,
        "size_bytes": len(payload),
        "retrieved_at": datetime.now(UTC).isoformat(),
    }
    write_json(descriptor_path, descriptor)
    return descriptor
