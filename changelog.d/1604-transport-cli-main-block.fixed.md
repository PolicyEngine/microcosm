Run the transport CLI when it is started as a module: `python -m
microcosm.build.transport.cli` imported the module and exited 0 without calling
`main`, because the module had no `__main__` block. Test both entry forms, the
module and `tools/build_transport.py`, in a subprocess.
