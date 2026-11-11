"""Thin public-framework fitting boundary; receives no TEST targets."""

import numpy as np


def fit_predict(payload):
    config, train_x, train_y, test_x = payload
    if config.scenario == "motions_neural":
        return neural_fit_predict(config, train_x, train_y, test_x)
    if config.scenario == "airline":
        from fedot_ind.integration.fedot import create_forecasting_runtime

        runtime = create_forecasting_runtime(
            profile="tensor", operation_name="lagged_ridge_forecaster",
            horizon=config.parameters.horizon,
            parameters={"window_size": config.parameters.window_size, "alpha": config.parameters.alpha})
        try:
            prediction = runtime.fit(train_x).predict(test_x).values
            return {"prediction": np.asarray(prediction).reshape(-1),
                    "model": "lagged_ridge_forecaster"}
        finally:
            runtime.close()
    from fedot import create_data
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope

    with industrial_extension_scope():
        pipeline = (PipelineBuilder().add_node("quantile_extractor_torch", params={
            "use_cache": False, "use_sliding_window": False, "window_size": 0})
            .add_node("logit", params={"C": config.parameters.c, "max_iter": config.parameters.max_iter,
                                       "random_state": config.seed, "n_jobs": 1}).build())
        train = create_data(train_x, target=train_y,
                            task="classification", data_type="ts", use_cache=False)
        test = create_data(test_x, task="classification",
                           data_type="ts", use_cache=False)
        try:
            pipeline.fit(train)
            values = pipeline.predict(test, output_mode="labels").predict
            return {"prediction": values.detach().cpu().numpy().reshape(-1),
                    "model": "quantile_extractor_torch -> logit"}
        finally:
            pipeline.unfit()


def inference_plan(shape, classes, device):
    from benchmark.industrial.experiments.compute_profile.operation_compute import (
        ComputeDevice, ComputeRejection, EvaluationBackend, MemoryBudget,
        TensorOperationCapabilities, TensorSpec, plan_tensor_operation,
    )
    target = ComputeDevice(device)
    plan = plan_tensor_operation(
        TensorSpec(tuple(shape), "float32", 4, ComputeDevice.CPU),
        TensorSpec((shape[0], classes), "float32", 4, target),
        TensorOperationCapabilities("tiny_cnn_inference", devices=(ComputeDevice.CPU, ComputeDevice.CUDA),
                                    estimated_temporary_bytes=4 * 1024**2),
        device=target, backend=EvaluationBackend.SEQUENTIAL,
        budget=MemoryBudget(cpu_bytes=64 * 1024**2, cuda_bytes=64 * 1024**2))
    if isinstance(plan, ComputeRejection):
        raise ValueError(f"{plan.code.value}: {plan.message}")
    return plan


def neural_fit_predict(config, train_x, train_y, test_x):
    import torch
    from torch import nn
    from fedot_ind.neural.encoders import build_encoder, raw_encoder_config
    from benchmark.industrial.experiments.compute_profile.operation_compute import execute_tensor_operation

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA explicitly requested but unavailable; no CPU fallback")
    params = config.parameters
    classes = len(np.unique(train_y))
    if not np.array_equal(np.unique(train_y), np.arange(classes)):
        raise ValueError("TRAIN labels must be consecutive class ids")
    plan = inference_plan(test_x.shape, classes, config.device)
    devices = [torch.cuda.current_device()] if config.device == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        # Scope the seed to the caller-owned generator state; leave Torch settings alone.
        torch.random.default_generator.manual_seed(config.seed)
        if devices:
            torch.cuda.manual_seed(config.seed)
        model = nn.Sequential(build_encoder(raw_encoder_config(
            in_channels=train_x.shape[1], d_model=params.width,
            hidden_channels=(params.width,), kernel_size=3, dropout=0.0)),
            nn.Linear(params.width, classes)).to(config.device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=params.learning_rate)
        rng = np.random.default_rng(config.seed)
        losses = []
        model.train()
        for _ in range(params.steps):
            indices = rng.choice(len(train_y), size=min(
                params.batch_size, len(train_y)), replace=False)
            x = torch.as_tensor(train_x[indices], device=config.device)
            y = torch.as_tensor(
                train_y[indices], dtype=torch.long, device=config.device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            output = execute_tensor_operation(plan, torch.from_numpy(
                np.ascontiguousarray(test_x)), model).tensor
        return {"prediction": output.argmax(dim=1).cpu().numpy(), "model": "neural.CNNEncoder -> Linear",
                "training_losses": losses, "training_steps": params.steps,
                "inference_memory_estimate": vars(plan.memory)}
