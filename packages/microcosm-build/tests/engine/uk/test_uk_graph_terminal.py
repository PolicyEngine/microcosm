"""Terminal-node identity that binds the country reference resources."""


def test_gate_cache_identity_includes_country_reference_resource_bytes(monkeypatch):
    from types import SimpleNamespace

    from microcosm.build import country_spec
    from microcosm.build.uk_runtime.graph_terminal import UKFullGateKernel

    kernel = UKFullGateKernel(coverage_engine=object(), engine_identity="fixture")
    monkeypatch.setattr(
        country_spec,
        "load_country_spec",
        lambda country: SimpleNamespace(fingerprint="a" * 64),
    )
    first = kernel.implementation_hash()
    monkeypatch.setattr(
        country_spec,
        "load_country_spec",
        lambda country: SimpleNamespace(fingerprint="b" * 64),
    )
    assert kernel.implementation_hash() != first
