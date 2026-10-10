"""Container entry: stub the heavy package __init__s, then run a script.

Usage (inside the image, with the evidence scripts, ``stage_loader.py`` and a
checkout of the builder commit mounted under /work):
    python3 -B /work/container_replay.py /work/platform_stage_input.py replay ...
"""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, "/work")
import stage_loader  # noqa: E402

stage_loader.install(Path("/work/src"))
script = sys.argv[1]
sys.argv = [script, *sys.argv[2:]]
runpy.run_path(script, run_name="__main__")
