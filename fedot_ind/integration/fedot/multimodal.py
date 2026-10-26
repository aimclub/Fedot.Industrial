"""Explicit named-modality transport through the array-level FEDOT contract.

Packing is lossless routing, not a replacement for learned multimodal fusion.
The terminal operation restores every modality before invoking FUTURE.
"""

from copy import deepcopy
from dataclasses import dataclass
import json

import numpy as np
import torch
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot_ind.core.multimodal.packing import ModalityLayout, pack_modalities, parse_layout
from fedot_ind.core.multimodal.rules import ModalitySpec
from fedot_ind.integration.fedot.contracts import (
    DataProfile, DataStage, IntegrationContractError, IntegrationErrorCode, IntegrationTask,
    ModelExecutionPlan, PredictionBatch, RuntimeState,
)
from fedot_ind.integration.fedot.extensions import industrial_extension_scope
from fedot_ind.integration.fedot.tensor import TensorMultimodalBatch, create_multimodal_tensor_data


@dataclass(frozen=True)
class FusionTensorBatch:
    """Prepared sources and their explicit transport schema; no registry state."""

    sources: TensorMultimodalBatch
    data: TensorData
    layout: tuple[ModalityLayout, ...]


def prepare_multimodal_fusion_data(bundle, *, stage, training=None, idx=None):
    """Check fitted schemas before packing; prediction never reads a target."""
    stage = DataStage(stage)
    if (stage is DataStage.PREDICT) != (training is not None):
        raise IntegrationContractError(IntegrationErrorCode.PLAN_MISMATCH,
                                       "Prediction requires a fitted preparation plan; training must not reuse one.")
    if stage is DataStage.PREDICT:
        bundle = bundle.without_target()
    elif bundle.target is None or not torch.isfinite(bundle.target).all():
        raise IntegrationContractError(IntegrationErrorCode.INVALID_DATA, "A finite training target is required.")
    layout = parse_layout([
        {"name": name.value, "shape": list(value.shape[1:])}
        for name, value in sorted(bundle.modalities.items(), key=lambda pair: pair[0].value)
    ])
    if len(layout) < 2:
        raise IntegrationContractError(
            IntegrationErrorCode.MODALITY_MISMATCH, "Fusion requires at least two modalities.")
    if training is not None and layout != training.layout:
        code = (IntegrationErrorCode.SCHEMA_MISMATCH if {item.name for item in layout}
                == {item.name for item in training.layout} else IntegrationErrorCode.MODALITY_MISMATCH)
        raise IntegrationContractError(code, "Named modality shapes differ from the fitted layout.")
    # The pinned TensorData creator supports up to three axes. Preserve an image's
    # full spatial shape in layout, using only a lossless transport view here.
    transport_values = {
        name: value.reshape(value.shape[0], value.shape[1], -1) if value.ndim == 4 else value
        for name, value in bundle.modalities.items()
    }
    transport_specs = dict(bundle.specs)
    for name, value in bundle.modalities.items():
        if value.ndim == 4:
            transport_specs[name] = ModalitySpec((3,), tuple(transport_values[name].shape[1:]))
    transport = bundle.replace(modalities=transport_values, specs=transport_specs)
    sources = create_multimodal_tensor_data(
        transport, task="classification", stage=stage,
        train_plan=None if training is None else training.sources.plan,
        from_data=None if training is None else training.sources.modalities,
        idx=idx,
    )
    shaped = {item.name: sources.modalities[item.name].features.reshape(-1, *item.shape) for item in layout}
    values = pack_modalities(shaped, layout)
    first = next(iter(sources.modalities.values()))
    if len(np.unique(first.idx)) != len(first.idx):
        raise IntegrationContractError(IntegrationErrorCode.INDEX_MISMATCH, "Fusion requires unique sample indices.")
    # No second obligatory preprocessing: it would erase the per-modality schema.
    data = TensorData(
        task=first.task, data_type=DataTypesEnum.tabular,
        features=values, idx=np.array(first.idx, copy=True),
        target=None if bundle.target is None else bundle.target.detach().clone(),
        numerical_idx=list(range(values.shape[1])),
    )
    return FusionTensorBatch(sources, data, layout)


def build_multimodal_fusion_pipeline(layout, *, parameters=None):
    """Build separate routing branches and bind the root to FEDOT's merge order."""
    layout = parse_layout([
        {"name": item.name, "shape": list(item.shape)} for item in layout
    ])
    if len(layout) < 2:
        raise ValueError("Fusion requires at least two modalities")
    options = deepcopy(dict(parameters or {}))
    if "modality_layout" in options:
        raise ValueError("modality_layout is derived from the prepared sources")
    json.dumps(options, allow_nan=False)
    width = sum(item.width for item in layout)
    branches = []
    start = 0
    for item in layout:
        node = PipelineNode(f"industrial_modality_slice/{item.name}")
        node.parameters = {"input_width": width, "start": start, "stop": start + item.width}
        branches.append((node, item))
        start += item.width
    # FEDOT sorts parent descriptive IDs before merging; never infer order from a dict.
    ordered = sorted(branches, key=lambda pair: pair[0].descriptive_id)
    root = PipelineNode("future_fusion_clf", nodes_from=[node for node, _ in ordered])
    root.parameters = dict(options, modality_layout=[
        {"name": item.name, "shape": list(item.shape)} for _, item in ordered
    ])
    return Pipeline(root)


class TensorMultimodalClassifier:
    """Owned FUTURE composite graph, with preflight checks before model calls."""

    def __init__(self, *, parameters=None):
        self.plan = ModelExecutionPlan(DataProfile.TENSOR, IntegrationTask.CLASSIFICATION,
                                       "future_fusion_clf", parameters or {})
        json.dumps(self.plan.runtime_parameters(), allow_nan=False)
        self._state = RuntimeState.CREATED
        self._pipeline = None
        self._training = None
        self._classes = None

    @property
    def state(self):
        return self._state

    def _require(self, state):
        if self.state is not state:
            raise IntegrationContractError(IntegrationErrorCode.INVALID_STATE,
                                           f"Expected {state.value}, found {self.state.value}.")

    def fit(self, bundle, *, idx=None):
        self._require(RuntimeState.CREATED)
        training = prepare_multimodal_fusion_data(bundle, stage=DataStage.TRAIN, idx=idx)
        pipeline = None
        try:
            with industrial_extension_scope():
                pipeline = build_multimodal_fusion_pipeline(training.layout, parameters=self.plan.runtime_parameters())
                pipeline.fit(training.data, n_jobs=1)
        except Exception:
            if pipeline is not None:
                pipeline.unfit()
            raise
        self._pipeline = pipeline
        self._training = training
        self._classes = np.unique(bundle.target.detach().cpu().numpy())
        self._state = RuntimeState.FITTED
        return self

    def predict(self, bundle, *, probabilities=False, idx=None):
        self._require(RuntimeState.FITTED)
        prediction = prepare_multimodal_fusion_data(
            bundle, stage=DataStage.PREDICT, training=self._training, idx=idx)
        with industrial_extension_scope():
            output = self._pipeline.predict(prediction.data, output_mode="full_probs" if probabilities else "labels")
        values = output.predict
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        return PredictionBatch(values=np.asarray(values), idx=prediction.data.idx, classes=self._classes)

    def close(self):
        try:
            if self._pipeline is not None:
                self._pipeline.unfit()
        finally:
            self._pipeline = self._training = self._classes = None
            self._state = RuntimeState.CLOSED

    def __enter__(self):
        self._require(RuntimeState.CREATED)
        return self

    def __exit__(self, *args):
        self.close()
