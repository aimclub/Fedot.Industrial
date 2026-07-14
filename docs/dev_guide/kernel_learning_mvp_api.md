# Kernel Learning MVP API

This note shows the minimal supervised TSC/TSER entry points added in
`fedot_ind.core.kernel_learning`. The MVP works with numpy-like time-series
inputs shaped as `(n_samples, n_timestamps)`, `(n_samples, n_channels,
n_timestamps)`, or already flattened tabular features.

## Direct estimators

```python
import numpy as np

from fedot_ind.core.kernel_learning import KernelEnsembleClassifier, KernelEnsembleRegressor


X_cls = np.array([[0.0, 0.1, 0.2], [0.1, 0.2, 0.3], [1.0, 1.1, 1.2], [1.1, 1.2, 1.3]])
y_cls = np.array(["normal", "normal", "fault", "fault"])

classifier = KernelEnsembleClassifier(
    generator_names=("statistical_summary",),
    kernel="linear",
    C=10.0,
)
classifier.fit(X_cls, y_cls)
labels = classifier.predict(X_cls)
probabilities = classifier.predict_proba(X_cls)


X_reg = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
y_reg = np.array([1.0, 3.0, 5.0, 7.0, 9.0])

regressor = KernelEnsembleRegressor(
    generator_names=("statistical_summary",),
    kernel="linear",
    alpha=1e-6,
)
regressor.fit(X_reg, y_reg)
values = regressor.predict(np.array([[5.0], [6.0]]))
```

After `fit`, both estimators expose diagnostics:

```python
classifier.selection_report_
classifier.kernel_bundles_
classifier.selected_generators_
classifier.selected_weights_
```

`statistical_summary` is a compatibility alias for the repository-native
`TorchQuantileExtractor`. The default kernel-learning generators are backed by
existing Industrial transformations: `quantile_extractor_torch`, `wavelet_basis`,
`fourier_basis`, `eigen_basis`, `recurrence_extractor`, `topological_extractor`,
and `tabular_extractor`.

## benchmark.industrial model specs

```python
from benchmark.industrial import ModelSpec


tsc_model = ModelSpec(
    adapter_name="kernel_ensemble_classifier",
    display_name="KernelEnsembleClassifier",
    params={
        "generator_names": ("statistical_summary",),
        "kernel": "linear",
        "C": 10.0,
    },
)

tser_model = ModelSpec(
    adapter_name="kernel_ensemble_regressor",
    display_name="KernelEnsembleRegressor",
    params={
        "generator_names": ("statistical_summary",),
        "kernel": "linear",
        "alpha": 1e-6,
    },
)
```

## Experiment scripts

The default scripts are declarative `benchmark.industrial` experiment suites.
They are grouped by task under `benchmark/experiments/kernel_learning`.

```powershell
python benchmark/experiments/kernel_learning/classification/run_ucr.py
python benchmark/experiments/kernel_learning/regression/run_tser.py
```

The UCR script uses `data/` as the local dataset root first. If a UCR dataset is
missing locally, `benchmark.industrial` falls back to the existing repository
UCR loader, which downloads the dataset from the UCR archive and saves it under
the same local root.

To change datasets, model grids, metrics, or output templates, update the package-local JSON defaults and let the typed config builders normalize them into `BenchmarkSuiteConfig`:

```text
benchmark/experiments/kernel_learning/defaults.json
benchmark/industrial/experiments/preset_defaults.json
examples/utils/current_api/manifests/*.json
```

Typical flow:

```python
from benchmark.experiments.kernel_learning.configs import build_ucr_suite_config
from benchmark.industrial.experiments.registry import run_registered_suite

config = build_ucr_suite_config()
bundle = run_registered_suite(config)
```

Keep implementation files focused on parsing and execution; catalog-like constants belong in JSON defaults next to the thematic entrypoint.
