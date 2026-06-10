from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from examples.tools_example.contracts import ToolError, ToolRequest, ToolResponse
from examples.tools_example.services.anomaly import detect_anomalies
from examples.tools_example.services.data import list_local_datasets, load_dataset_preview
from examples.tools_example.services.evolution import run_evolution_optimization
from examples.tools_example.services.pdl import run_pdl_training
from examples.tools_example.services.training import train_model

ToolHandler = Callable[[ToolRequest], ToolResponse]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    capability: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "capability": self.capability,
        }


def list_tool_specs() -> tuple[dict[str, Any], ...]:
    return tuple(spec.to_dict() for spec in _SPECS.values())


def invoke_tool(name: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    request = ToolRequest.from_payload(name, payload)
    handler = _HANDLERS.get(name)
    if handler is None:
        known = ", ".join(sorted(_HANDLERS))
        return ToolResponse(
            name=name,
            status="failed",
            dry_run=request.dry_run,
            message=f"Unknown Industrial tool: {name}.",
            error=ToolError(code="unknown_tool", message=f"Known tools: {known}"),
        ).to_dict()
    try:
        return handler(request).to_dict()
    except Exception as exc:
        return ToolResponse(
            name=name,
            status="failed",
            dry_run=request.dry_run,
            message=str(exc),
            error=ToolError(code=exc.__class__.__name__, message=str(exc)),
        ).to_dict()


def _schema(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "execute": {"type": "boolean", "description": "Run the action. Defaults to false/dry-run."},
            "dry_run": {"type": "boolean", "description": "Preview without executing heavy work."},
            **properties,
        },
        "required": list(required),
        "additionalProperties": False,
    }


_SPECS: dict[str, ToolSpec] = {
    "industrial_load_data": ToolSpec(
        name="industrial_load_data",
        description="List local Industrial datasets or preview a specific dataset path.",
        capability="data.load",
        input_schema=_schema(
            {
                "task_type": {"type": "string"},
                "path": {"type": "string"},
            }
        ),
    ),
    "industrial_train_model": ToolSpec(
        name="industrial_train_model",
        description="Train or preview a current Industrial model for classification, regression, or forecasting.",
        capability="model.train",
        input_schema=_schema(
            {
                "task_type": {"type": "string", "enum": ["ts_classification", "ts_regression", "forecasting"]},
                "dataset_name": {"type": "string"},
                "subset": {"type": "string"},
                "sample_size": {"type": ["integer", "null"]},
                "output_dir": {"type": "string"},
                "models": {"type": "array", "items": {"type": "object"}},
                "include_optional_external": {"type": "boolean"},
            }
        ),
    ),
    "industrial_run_evolution": ToolSpec(
        name="industrial_run_evolution",
        description="Run or preview Kernel Learning two-stage evolutionary optimization.",
        capability="optimization.evolution",
        input_schema=_schema(
            {
                "datasets": {"type": "array", "items": {"type": "string"}},
                "stage1_output_dir": {"type": "string"},
                "stage2_output_dir": {"type": "string"},
                "stage1_run_id": {"type": ["string", "null"]},
                "run_stage1": {"type": "boolean"},
                "timeout_minutes": {"type": "integer"},
                "pop_size": {"type": "integer"},
            }
        ),
    ),
    "industrial_run_pdl_training": ToolSpec(
        name="industrial_run_pdl_training",
        description="Run or preview PDL classifier/regressor training through the Industrial benchmark API.",
        capability="model.train.pdl",
        input_schema=_schema(
            {
                "task_type": {"type": "string", "enum": ["ts_classification", "ts_regression"]},
                "dataset_name": {"type": "string"},
                "output_dir": {"type": "string"},
                "pdl_model": {"type": "string"},
            }
        ),
    ),
    "industrial_detect_anomalies": ToolSpec(
        name="industrial_detect_anomalies",
        description="Run or preview a lightweight Industrial anomaly-detection workflow on local SKAB/LIMAN data.",
        capability="anomaly.detect",
        input_schema=_schema(
            {
                "data_root": {"type": "string"},
                "folder": {"type": "string"},
                "dataset": {"type": "string"},
                "threshold_quantile": {"type": "number"},
            }
        ),
    ),
    "industrial_tsc_smoke": ToolSpec(
        name="industrial_tsc_smoke",
        description="Compatibility alias: execute a small time-series classification training run.",
        capability="model.train.compat",
        input_schema=_schema({"output_dir": {"type": "string"}}),
    ),
    "industrial_tser_smoke": ToolSpec(
        name="industrial_tser_smoke",
        description="Compatibility alias: execute a small time-series regression training run.",
        capability="model.train.compat",
        input_schema=_schema({"output_dir": {"type": "string"}}),
    ),
    "industrial_forecasting_config_preview": ToolSpec(
        name="industrial_forecasting_config_preview",
        description="Compatibility alias: preview a forecasting benchmark config.",
        capability="model.train.preview.compat",
        input_schema=_schema({"output_dir": {"type": "string"}}),
    ),
    "industrial_kernel_learning_ucr_preview": ToolSpec(
        name="industrial_kernel_learning_ucr_preview",
        description="Compatibility alias: preview a UCR classification config.",
        capability="model.train.preview.compat",
        input_schema=_schema({"output_dir": {"type": "string"}}),
    ),
    "industrial_anomaly_detection_context": ToolSpec(
        name="industrial_anomaly_detection_context",
        description="Compatibility alias: preview anomaly-detection context.",
        capability="anomaly.context.compat",
        input_schema=_schema({}),
    ),
}

_HANDLERS: dict[str, ToolHandler] = {
    "industrial_load_data": lambda request: (
        load_dataset_preview(request) if request.payload.get("path") else list_local_datasets(request)
    ),
    "industrial_train_model": train_model,
    "industrial_run_evolution": run_evolution_optimization,
    "industrial_run_pdl_training": run_pdl_training,
    "industrial_detect_anomalies": detect_anomalies,
    "industrial_tsc_smoke": lambda request: train_model(
        ToolRequest(name=request.name, payload={**request.payload, "task_type": "ts_classification"}, dry_run=False)
    ),
    "industrial_tser_smoke": lambda request: train_model(
        ToolRequest(name=request.name, payload={**request.payload, "task_type": "ts_regression"}, dry_run=False)
    ),
    "industrial_forecasting_config_preview": lambda request: train_model(
        ToolRequest(name=request.name, payload={**request.payload, "task_type": "forecasting"}, dry_run=True)
    ),
    "industrial_kernel_learning_ucr_preview": lambda request: train_model(
        ToolRequest(name=request.name, payload={**request.payload, "task_type": "ts_classification"}, dry_run=True)
    ),
    "industrial_anomaly_detection_context": lambda request: detect_anomalies(
        ToolRequest(name=request.name, payload=request.payload, dry_run=True)
    ),
}

__all__ = ["ToolSpec", "invoke_tool", "list_tool_specs"]
