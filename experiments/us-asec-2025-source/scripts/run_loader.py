"""Run the archived US data package's CensusCPS loader for one income year.

    cd <policyengine-us-data checkout at the loader commit>
    python run_loader.py INCOME_YEAR OUT.h5

Writes to an explicit output path (never the package's storage folder) and
records, beside the output, the SHA-256 of the archive bytes the loader
downloaded, the output's SHA-256, the loader commit and the library versions.
The loader is not modified; ``requests.get`` is wrapped only to hash the
streamed bytes.
"""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def _hash_downloads(requests, record: dict) -> None:
    original_get = requests.get

    def hashed_get(url, *args, **kwargs):
        response = original_get(url, *args, **kwargs)
        original_iter = response.iter_content
        digest = hashlib.sha256()
        size = [0]

        def iter_content(chunk_size=1, *iter_args, **iter_kwargs):
            for chunk in original_iter(chunk_size, *iter_args, **iter_kwargs):
                digest.update(chunk)
                size[0] += len(chunk)
                yield chunk
            record.update(
                url=url,
                sha256=digest.hexdigest(),
                bytes=size[0],
                last_modified=response.headers.get("last-modified"),
                etag=response.headers.get("etag"),
            )

        response.iter_content = iter_content
        return response

    requests.get = hashed_get


def main(year: int, out: Path) -> dict:
    import numpy
    import pandas
    import requests
    import tables

    download: dict = {}
    _hash_downloads(requests, download)

    from policyengine_core.data import Dataset
    from policyengine_us_data.datasets.cps import census_cps

    class Run(census_cps.CensusCPS):
        time_period = year
        label = f"Census CPS ({year})"
        name = f"census_cps_{year}"
        file_path = out
        data_format = Dataset.TABLES

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    Run().generate()

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], text=True).strip()

    record = dict(
        income_year=year,
        output=str(out),
        output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        output_bytes=out.stat().st_size,
        download=download,
        loader_commit=git("rev-parse", "HEAD"),
        loader_dirty=git("status", "--porcelain", "--", "policyengine_us_data"),
        python=platform.python_version(),
        pandas=pandas.__version__,
        tables=tables.__version__,
        numpy=numpy.__version__,
    )
    (out.parent / f"run_{year}.json").write_text(json.dumps(record, indent=1))
    return record


if __name__ == "__main__":
    print(json.dumps(main(int(sys.argv[1]), Path(sys.argv[2])), indent=1))
