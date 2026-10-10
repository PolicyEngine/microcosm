"""Container entry: stub the heavy package __init__s, then run the replay."""

import sys
from pathlib import Path

sys.path.insert(0, "/work")
import stage_loader  # noqa: E402

stage_loader.install(Path("/work/src"))
sys.argv = ["platform_stage_input.py", *sys.argv[1:]]
import platform_stage_input  # noqa: E402

platform_stage_input.main()
