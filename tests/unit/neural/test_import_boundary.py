"""The extracted package must import without any domain or AutoML runtime."""
import ast
from pathlib import Path
import subprocess
import sys

import fedot_ind.neural


def test_neural_core_has_only_internal_and_torch_dependencies():
    root = Path(fedot_ind.neural.__file__).parent
    violations = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.startswith("fedot_ind.") and not name.startswith("fedot_ind.neural"):
                    violations.append((str(path.relative_to(root)), node.lineno, name))
                if name.split(".")[0] in {"fedot", "golem", "FEDOT", "fastai", "benchmark"}:
                    violations.append((str(path.relative_to(root)), node.lineno, name))
    assert violations == []


def test_fresh_process_imports_and_runs_without_domain_modules():
    code = '''
import importlib.abc
import sys
class RejectDomain(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in ('fedot', 'golem', 'FEDOT', 'fastai', 'benchmark')
                or fullname.startswith('fedot_ind.core')):
            raise ImportError('Forbidden neural dependency: ' + fullname)
sys.meta_path.insert(0, RejectDomain())
import torch
from fedot_ind.neural.encoders import EncoderStack, stats_encoder_config
from fedot_ind.neural.fusion import MultiModalGatedFusion
from fedot_ind.neural.blocks.attention import MultiHeadAttention
from fedot_ind.neural.device import TensorDevicePolicy
model = EncoderStack({'sensor': stats_encoder_config(4, d_model=8, hidden_dims=(4,))})
result = model({'sensor': torch.ones(2, 4, requires_grad=True)})['sensor']
result.square().sum().backward()
assert result.shape == (2, 8)
assert not any(name.startswith('fedot_ind.core') for name in sys.modules)
'''
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr


def test_legacy_forecasting_helpers_do_not_import_forecasting_runtime_or_fedot():
    code = '''
import importlib.abc
import sys
class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in ('fedot', 'golem', 'FEDOT')
                or fullname.startswith('fedot_ind.core.models.ts_forecasting')):
            raise ImportError('Forbidden reverse dependency: ' + fullname)
sys.meta_path.insert(0, RejectRuntime())
from fedot_ind.core.models.nn.network_impl.forecasting_model.common import resolve_neural_patch_length
assert resolve_neural_patch_length([1.] * 20, 5, requested_patch_len=30) == 15
assert not any(name.startswith('fedot_ind.core.models.ts_forecasting') for name in sys.modules)
'''
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
