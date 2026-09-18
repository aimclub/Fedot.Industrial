from __future__ import annotations

from examples.tools_example.contracts import ToolRequest, ToolResponse
from examples.tools_example.services.training import train_model


def run_pdl_training(request: ToolRequest) -> ToolResponse:
    task_type = str(request.payload.get("task_type", "ts_classification"))
    if task_type == "ts_regression":
        model = {
            "adapter_name": "pdl_regressor",
            "display_name": "PDLRegressor",
            "optional": True,
            "tags": ["industrial", "pdl"],
            "params": {"model": request.payload.get("pdl_model", "treg")},
        }
    else:
        model = {
            "adapter_name": "pdl_classifier",
            "display_name": "PDLClassifier",
            "optional": True,
            "tags": ["industrial", "pdl"],
            "params": {"model": request.payload.get("pdl_model", "rf")},
        }
    payload = {**request.payload, "task_type": task_type, "models": [model]}
    response = train_model(ToolRequest(name=request.name, payload=payload, dry_run=request.dry_run))
    return ToolResponse(
        name=response.name,
        status=response.status,
        dry_run=response.dry_run,
        message="PDL training flow: " + response.message,
        data=response.data,
        artifacts=response.artifacts,
        error=response.error,
    )


__all__ = ["run_pdl_training"]
