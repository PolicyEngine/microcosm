"""The UK raw-byte source codecs join the shared registry explicitly, never on import.

The wheels gate runs every package's suite in one process, and the shared codec
suite asserts the shipped raw-byte set is exactly ``raw-bytes-v1``; an import-time
registration from a country adapter broke that invariant. Registration therefore
happens with the UK target kernels and is idempotent.
"""

from __future__ import annotations

import subprocess
import sys

from microcosm.build.uk_runtime.full_targets import (
    CHRONICLE_SOURCE_CODEC,
    load_chronicle_source_bytes,
)
from microcosm.build.uk_runtime.graph_targets import (
    register_uk_source_codecs,
    register_uk_target_kernels,
)
from microcosm.graph import KernelRegistry
from microcosm.graph.codecs import SOURCE_CODECS, SourceCodecRegistry


def test_importing_the_uk_target_module_leaves_the_shared_registry_untouched():
    script = (
        "import microcosm.build.uk_runtime.graph_targets\n"
        "from microcosm.graph.codecs import SOURCE_CODECS\n"
        "assert SOURCE_CODECS.bytes_names() == ('raw-bytes-v1',), SOURCE_CODECS.bytes_names()\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_registration_is_explicit_and_idempotent():
    registry = SourceCodecRegistry()
    registry.register_bytes("raw-bytes-v1", SOURCE_CODECS.get("raw-bytes-v1"))
    register_uk_source_codecs(registry)
    register_uk_source_codecs(registry)
    assert set(registry.bytes_names()) == {"raw-bytes-v1", CHRONICLE_SOURCE_CODEC}
    assert registry.get(CHRONICLE_SOURCE_CODEC) is load_chronicle_source_bytes


def test_kernel_registration_registers_the_codec_on_the_shared_registry():
    register_uk_target_kernels(KernelRegistry())
    assert SOURCE_CODECS.get(CHRONICLE_SOURCE_CODEC) is load_chronicle_source_bytes
