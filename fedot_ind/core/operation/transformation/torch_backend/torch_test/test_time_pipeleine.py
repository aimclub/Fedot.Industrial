from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot_ind.tools.loader import DataLoader
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.core.operation.dummy.dummy_operation import init_input_data_tensor, init_input_data
from fedot_ind.core.architecture.preprocessing.data_convertor import TensorConverter, FedotConverter

from sklearn.metrics import f1_score
import torch 
import os
import shutil
import pandas as pd
import numpy as np
from tabulate import tabulate
import time
from tqdm import tqdm


def remove_folder_completely(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


@torch.no_grad()
def warm_up_cuda_computations(n_iters=5, size=2048, device=None):
    """ Function for CUDA warming. It is used before time measuring.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)
    for _ in tqdm(range(n_iters)):
        C = A @ B
        C = torch.sin(C) * torch.exp(C)
        _ = C.sum()
    if device == "cuda":
        torch.cuda.synchronize()
    print("Warm-up done")


def time_pipeline_test(DATASET_NAME="Beef"):
    cache_path = "/workspaces/Fedot.Industrial/cache"
    remove_folder_completely(cache_path)
    train_data, test_data = DataLoader(dataset_name=DATASET_NAME).load_data()

    #numpy
    with IndustrialModels():
        pipeline = (
            PipelineBuilder()
            .add_node('quantile_extractor', params={'window_size': 20, 'window_mode': True})
            .add_node('rf')
            .build()
        )
        input_data = init_input_data(train_data[0], train_data[1])
        print(input_data.features.shape)
        start_np = time.perf_counter()
        pipeline.fit(input_data)
        t_np = time.perf_counter() - start_np
        preds = pipeline.predict(input_data)
        preds_np = preds.predict
        target_np = preds.target
    # preds_np_classes = np.argmax(preds_np, axis=1)
    # f1_np = f1_score(target_np, preds_np_classes, average='macro')
    # f1_np = f1_score(target_np, preds_np)
    print("np shape:", np.array(preds_np).shape)
    print("time np:", t_np, "\n")
    # print(f1_np)

    # torch cpu
    converter = TensorConverter(data=train_data[0])
    converter_test = TensorConverter(data=test_data[0])
    with IndustrialModels():
        pipeline = (
            PipelineBuilder()
            .add_node('quantile_extractor_torch', params={'window_size': 20, 'window_mode': True})
            .add_node('rf')
            .build()
        )
        input_data = init_input_data_tensor(converter.tensor_data, train_data[1])
        input_data_test = init_input_data_tensor(converter_test.tensor_data, test_data[1])
        start = time.perf_counter()
        pipeline.fit(input_data)
        t_torch = time.perf_counter() - start
        preds = pipeline.predict(input_data)
        preds_torch = preds.predict
        target_torch = preds.target
    # f1_torch = f1_score(target_torch, preds_torch)
    print("torch shape:", np.array(preds_torch).shape)
    print("time torch:", t_torch, "\n")
    # print(f1_torch)

    remove_folder_completely(cache_path)

    #torch gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    warm_up_cuda_computations(device=device)
    with IndustrialModels():
        pipeline = (
            PipelineBuilder()
            .add_node('quantile_extractor_torch', params={'window_size': 20, 'window_mode': True})
            .add_node('rf')
            .build()
        )
        input_data = init_input_data_tensor(converter.tensor_data.to(device), train_data[1])
        # input_data_test = init_input_data_tensor(converter_test.tensor_data, test_data[1])
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        pipeline.fit(input_data)
        end_event.record()
        torch.cuda.synchronize()
        t_torch_gpu = start_event.elapsed_time(end_event) / 1000
        preds = pipeline.predict(input_data)
        preds_torch = preds.predict
        target_torch = preds.target
    # f1_torch_gpu = f1_score(target_torch, preds_torch)
    print("torch shape:", np.array(preds_torch).shape)
    print("time torch (GPU):", t_torch_gpu, "\n")
    return {
        "dataset name": DATASET_NAME,
        "shape of data": input_data.features.shape,
        "numpy CPU time (sec)": t_np,
        "torch CPU time (sec)": t_torch,
        "speedup": round(t_np / t_torch, 2),
        "torch GPU time (sec)": t_torch_gpu,
        "speedup GPU": round(t_np / t_torch_gpu, 2),
    }


def main():
    datasets = ["WormsTwoClass",  "EthanolLevel", "UWaveGestureLibrary", "EMOPain"]
    results = []
    for dn in datasets:
        res = time_pipeline_test(dn)
        results.append(res)
    df = pd.DataFrame(results)
    path = ""
    df.to_csv(path+'stat_pipeline.csv', index=False)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=True))


if __name__ == "__main__":
    main()