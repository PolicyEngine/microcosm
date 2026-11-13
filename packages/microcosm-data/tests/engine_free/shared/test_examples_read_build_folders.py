"""The example notebooks read published datasets the way the loader does.

The switch tool's ``delete-old-layout`` step deletes the dataset files at a
dataset repository's root, so an example that downloads one from ``main``
stops working once it has run. Every ``hf_hub_download`` call in an example
names ``releases.json`` or a file inside ``releases/``.
"""

from __future__ import annotations

import ast
import json

from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-data")


def _download_filenames(source: str) -> list[ast.expr]:
    """The ``filename`` argument of every ``hf_hub_download`` call in ``source``."""
    lines = [
        line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
    ]
    tree = ast.parse("\n".join(lines))
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", None))
            == "hf_hub_download"
        ):
            found.extend(item.value for item in node.keywords if item.arg == "filename")
    return found


def test_no_example_downloads_a_dataset_file_from_a_repository_root():
    notebooks = sorted((_TEST_PATHS.repository / "examples").glob("*.ipynb"))
    assert notebooks, "the examples directory holds no notebook"
    root_reads = []
    for notebook in notebooks:
        document = json.loads(notebook.read_text(encoding="utf-8"))
        for cell in document["cells"]:
            if cell["cell_type"] != "code":
                continue
            for filename in _download_filenames("".join(cell["source"])):
                if isinstance(filename, ast.Constant) and isinstance(
                    filename.value, str
                ):
                    name = filename.value
                    if name != "releases.json" and not name.startswith("releases/"):
                        root_reads.append(f"{notebook.name}: {name}")
    assert root_reads == []
