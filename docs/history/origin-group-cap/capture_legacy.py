"""Capture the invented prepatch fixture, never importing a mutable source tree."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
assert sys.flags.dont_write_bytecode and sys.pycache_prefix
assert Path(sys.pycache_prefix).is_absolute() and not Path(sys.pycache_prefix).exists()
def git(*args):
    return subprocess.check_output(['git', '-c', 'core.hooksPath=/dev/null', '-C', str(root), *args])
assert git('rev-parse','HEAD').decode().strip() == 'a4e69c0a1e7b5e953922d6d0a0d0283890c3c329'
paths = git('ls-files','packages/*/src/*.py').decode().splitlines()
before = {}
for rel in paths:
    data = (root/rel).read_bytes()
    expected = git('rev-parse', 'HEAD:'+rel).decode().strip()
    actual = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert actual == expected, rel
    before[rel] = {'git_blob': actual, 'sha256': hashlib.sha256(data).hexdigest()}
sys.path[:0] = [str(root/'packages'/('microcosm-'+s)/'src') for s in ('frame','calibrate','graph','fit','build','data')]
import numpy as np
import pandas as pd
import torch
from microcosm.calibrate import Target, TargetSet, calibrate
from microcosm.frame import Frame, EntitySchema, Weights, WeightKind
from microcosm.graph.keys import platform_fingerprint

torch.set_num_threads(1)
# Explicit data from test_solve.py::_exact_k_design_fixture, using its tiny
# design Frame construction without requesting the L0/conservation options.
inputs = {'weights': [100.0, 120.0, 80.0, 40.0], 'measure': [1.0, 2.0, 3.0, 0.5], 'target': 800.0}
options = {'epochs': 7, 'learning_rate': 0.02, 'seed': 0, 'mass': 'free', 'max_weight_ratio': 2.0}
def fixture():
    frame = Frame({'person': pd.DataFrame({'person_id': range(4),'person_household_id': range(4)}), 'household': pd.DataFrame({'household_id': range(4), 'adjudicated_measure': inputs['measure']})}, EntitySchema(group_entities=('household',)), {'household': Weights(values=np.array(inputs['weights']), kind=WeightKind.DESIGN)})
    return frame, TargetSet((Target(name='adjudicated',entity='household',value=inputs['target'],measure='adjudicated_measure'),))
settings = {'platform_fingerprint': platform_fingerprint(), 'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(), 'device': 'cpu', 'backend': 'Adam', 'parameter_dtype': 'float32', 'matrix_dtype': 'float32', 'output_dtype': 'float64', 'thread_environment': {k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}, 'python_binary_sha256': hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()}
frame, targets = fixture()
result = calibrate(frame, targets, **options)
loaded = {}
for name, module in sorted(sys.modules.items()):
    path = getattr(module, '__file__', None)
    if not path or not name.startswith('microcosm.'):
        continue
    path = Path(path).resolve()
    assert path.is_relative_to(root), (name, str(path))
    rel = str(path.relative_to(root))
    if rel not in before:
        continue
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == before[rel]['sha256']
    assert git('rev-parse','HEAD:'+rel).decode().strip() == before[rel]['git_blob']
    loaded[rel] = before[rel]
def array_evidence(value):
    arr = np.asarray(value,dtype='<f8')
    return {'shape': list(arr.shape), 'bytes_hex': arr.tobytes().hex(), 'sha256': hashlib.sha256(arr.tobytes()).hexdigest()}
payload = {'source_head':git('rev-parse','HEAD').decode().strip(), 'source_tree':str(root), 'fixture_source':'packages/microcosm-calibrate/tests/test_solve.py::_exact_k_design_fixture', 'inputs':inputs,'options':options,'settings':settings,'weights':array_evidence(result.weights),'trajectory':array_evidence(result.loss_trajectory),'loaded_source_before_after':loaded,'launch': {'dont_write_bytecode':sys.flags.dont_write_bytecode,'pycache_prefix':sys.pycache_prefix, 'isolated':sys.flags.isolated}}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps({'weights':payload['weights']['sha256'],'trajectory':payload['trajectory']['sha256'],'loaded_source_files':len(loaded),'output':str(out)}))
