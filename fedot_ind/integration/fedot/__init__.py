"""Lazy public index for the FEDOT integration boundary."""

from importlib import import_module


_EXPORTS = {
    "AnomalyInterval": "temporal_contracts",
    "AxisLayout": "contracts",
    "DataPreparationPlan": "contracts",
    "DataProfile": "contracts",
    "DataStage": "contracts",
    "DetectionDataPlan": "temporal_contracts",
    "DetectionExecutionPlan": "temporal_contracts",
    "DetectionMode": "temporal_contracts",
    "DetectionPrediction": "temporal_contracts",
    "DetectionTensorData": "temporal_tensor",
    "FeatureSchema": "contracts",
    "ForecastingDataPlan": "temporal_contracts",
    "ForecastingExecutionPlan": "temporal_contracts",
    "ForecastingPrediction": "temporal_contracts",
    "ForecastingTensorData": "temporal_tensor",
    "IntegrationContractError": "contracts",
    "IntegrationErrorCode": "contracts",
    "IntegrationTask": "contracts",
    "IntervalBoundary": "temporal_contracts",
    "ModelExecutionPlan": "contracts",
    "MultimodalPreparationPlan": "contracts",
    "PredictionBatch": "contracts",
    "PredictionMode": "contracts",
    "PreparedData": "contracts",
    "PreparedDetectionData": "temporal_contracts",
    "PreparedForecastingData": "temporal_contracts",
    "PreparedMultimodalData": "contracts",
    "RegressionRuntime": "runtime",
    "RuntimeSnapshot": "contracts",
    "RuntimeState": "contracts",
    "SupervisedRuntime": "runtime",
    "TemporalModalityPlan": "temporal_contracts",
    "TemporalMultimodalPlan": "temporal_contracts",
    "TemporalOrientation": "temporal_contracts",
    "TemporalRuntimeSnapshot": "temporal_contracts",
    "TemporalSchema": "temporal_contracts",
    "TemporalSignalRole": "temporal_contracts",
    "build_data_plan": "planning",
    "build_multimodal_plan": "planning",
    "build_temporal_multimodal_plan": "temporal_planning",
    "create_detection_runtime": "runtime",
    "create_detection_tensor_data": "temporal_tensor",
    "create_forecasting_runtime": "runtime",
    "create_forecasting_tensor_data": "temporal_tensor",
    "create_multimodal_tensor_data": "runtime",
    "create_regression_runtime": "runtime",
    "create_supervised_runtime": "runtime",
    "detection_prediction_to_output_data": "temporal_tensor",
    "forecasting_prediction_to_output_data": "temporal_tensor",
    "infer_future_time_index": "temporal_planning",
    "intervals_to_labels": "temporal_planning",
    "labels_to_intervals": "temporal_planning",
    "normalize_anomaly_intervals": "temporal_planning",
    "normalize_input_data": "data",
    "normalize_multimodal_input": "data",
    "prepare_detection_data": "temporal_planning",
    "prepare_forecasting_data": "temporal_planning",
    "validate_forecast_coordinates": "temporal_planning",
    "validate_multimodal_prediction_plan": "planning",
    "validate_prediction_plan": "planning",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
