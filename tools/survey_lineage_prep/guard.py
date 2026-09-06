"""Durable calculation/fit refusal and genuine-path fences for invented workers."""

import os
import sys


def install():
    denials, wires = [], []

    def path_guard(raw):
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return
        path = os.fsdecode(raw)
        if any(
            part in path
            for part in (
                "/_buildo-runtime/",
                "/.cache/microcosm/",
                "/storage/",
            )
        ):
            denials.append("GENUINE_PATH")
            raise RuntimeError("GENUINE_PATH_FORBIDDEN")

    original_stat = os.stat
    original_lstat = os.lstat

    def stat(path, *args, **kwargs):
        path_guard(path)
        return original_stat(path, *args, **kwargs)

    def lstat(path, *args, **kwargs):
        path_guard(path)
        return original_lstat(path, *args, **kwargs)

    os.stat, os.lstat = stat, lstat

    def audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            denials.append("NETWORK")
            raise RuntimeError("NETWORK_FORBIDDEN")
        if event in ("open", "os.listdir", "os.scandir"):
            path_guard(args[0])

    sys.addaudithook(audit)

    def wire(holder, name):
        if hasattr(holder, name):

            def refuse(*args, **kwargs):
                denials.append("CALCULATION_FIT_DRAW")
                raise RuntimeError("CALCULATION_FIT_DRAW_FORBIDDEN")

            setattr(holder, name, refuse)
            wires.append((holder, name, refuse))

    import policyengine_core.simulations as core
    import policyengine_us as us

    for package in (core, us):
        for name in ("Simulation", "Microsimulation", "IndividualSim"):
            cls = getattr(package, name, None)
            if cls is not None:
                for method in (
                    "calculate",
                    "calculate_add",
                    "calculate_divide",
                    "calculate_output",
                    "calculate_dataframe",
                    "_calculate",
                ):
                    wire(cls, method)
    import microcosm.fit as fit
    from microcosm.fit import qrf, qrf_target

    wire(fit, "fit")
    wire(qrf_target, "fit_target")
    for cls in (qrf.RegimeGatedQRF, qrf.FittedRegimeGatedQRF, qrf._Forest):
        for method in (
            "fit",
            "fit_draw_next",
            "predict",
            "predict_from_uniforms",
            "draw",
        ):
            wire(cls, method)
    return denials, wires
