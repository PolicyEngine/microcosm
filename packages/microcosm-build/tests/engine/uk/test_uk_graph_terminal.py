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


def test_the_solve_side_gate_kernels_ignore_a_gate_only_sign_off(monkeypatch):
    """microcosm#1301: the preflight and the holdout feed the solve and key on
    the solve fingerprint; the calibrated gates key on every resource."""
    from dataclasses import replace
    from types import MappingProxyType

    from microcosm.build import country_spec
    from microcosm.build.uk_runtime.graph_terminal import (
        UKFullGateKernel,
        UKFullHoldoutKernel,
        UKFullPreflightGateKernel,
    )

    kernels = {
        "preflight": UKFullPreflightGateKernel(
            coverage_engine=object(), engine_identity="fixture"
        ),
        "holdout": UKFullHoldoutKernel(),
        "calibrated": UKFullGateKernel(
            coverage_engine=object(), engine_identity="fixture"
        ),
    }
    before = {name: kernel.implementation_hash() for name, kernel in kernels.items()}
    spec = country_spec.load_country_spec("uk")

    def hashes_with(resource):
        edited = replace(
            spec,
            resource_hashes=MappingProxyType(
                {**spec.resource_hashes, resource: "0" * 64}
            ),
        )
        with monkeypatch.context() as patch:
            patch.setattr(
                country_spec, "_load_packaged_country_spec", lambda _country: edited
            )
            return {
                name: kernel.implementation_hash() for name, kernel in kernels.items()
            }

    signed = hashes_with("local_target_fit_reviewed_exclusions.json")
    assert signed["preflight"] == before["preflight"]
    assert signed["holdout"] == before["holdout"]
    assert signed["calibrated"] != before["calibrated"]
    solve_scoped = hashes_with("calibration_measure_exclusions.json")
    assert all(solve_scoped[name] != before[name] for name in kernels)
