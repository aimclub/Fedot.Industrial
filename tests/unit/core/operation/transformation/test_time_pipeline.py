import logging
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.core.architecture.preprocessing.data_convertor import TensorConverter
from fedot_ind.core.operation.dummy.dummy_operation import init_input_data, init_input_data_tensor
from fedot_ind.integration.fedot.extensions import industrial_extension_scope
from fedot_ind.tools.loader import DataLoader
from tests.unit.api.fixtures import warm_up_cuda_computations


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def remove_folder_completely(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


def time_pipeline_test(dataset_name='Beef'):
    cache_path = '/workspaces/Fedot.Industrial/cache'
    remove_folder_completely(cache_path)
    train_data, _ = DataLoader(dataset_name=dataset_name).load_data()

    with industrial_extension_scope():
        pipeline_np = (
            PipelineBuilder()
            .add_node('quantile_extractor', params={'window_size': 20, 'window_mode': True})
            .add_node('rf')
            .build()
        )
        input_data_np = init_input_data(train_data[0], train_data[1])
        start_np = time.perf_counter()
        pipeline_np.fit(input_data_np)
        t_np = time.perf_counter() - start_np

    converter = TensorConverter(data=train_data[0])
    with industrial_extension_scope():
        pipeline_torch = (
            PipelineBuilder()
            .add_node('quantile_extractor_torch', params={'window_size': 20, 'window_mode': True})
            .add_node('rf')
            .build()
        )
        input_data_torch = init_input_data_tensor(converter.tensor_data, train_data[1])
        start_torch = time.perf_counter()
        pipeline_torch.fit(input_data_torch)
        t_torch = time.perf_counter() - start_torch

    remove_folder_completely(cache_path)
    t_torch_gpu = np.nan
    if torch.cuda.is_available():
        warm_up_cuda_computations(device='cuda')
        with industrial_extension_scope():
            pipeline_torch_gpu = (
                PipelineBuilder()
                .add_node('quantile_extractor_torch', params={'window_size': 20, 'window_mode': True})
                .add_node('rf')
                .build()
            )
            input_data_gpu = init_input_data_tensor(
                converter.tensor_data.to('cuda'), train_data[1])
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            pipeline_torch_gpu.fit(input_data_gpu)
            end_event.record()
            torch.cuda.synchronize()
            t_torch_gpu = start_event.elapsed_time(end_event) / 1000

    remove_folder_completely(cache_path)
    assert t_torch < t_np, 'Torch CPU is not faster than NumPy CPU.'
    if torch.cuda.is_available():
        assert t_torch_gpu < t_np, 'Torch GPU is not faster than NumPy CPU.'
        assert t_torch_gpu < t_torch, 'Torch GPU is not faster than Torch CPU.'

    return {
        'dataset name': dataset_name,
        'shape of data': input_data_np.features.shape,
        'numpy CPU time (sec)': t_np,
        'torch CPU time (sec)': t_torch,
        'speedup': round(t_np / t_torch, 2),
        'torch GPU time (sec)': t_torch_gpu,
        'speedup GPU': round(t_np / t_torch_gpu, 2) if torch.cuda.is_available() else np.nan,
    }


def run_pipeline_tests() -> pd.DataFrame:
    logger.info('Start test of pipeline.')
    results = [time_pipeline_test('WormsTwoClass')]
    logger.info('Successful test.')
    return pd.DataFrame(results)


if __name__ == '__main__':
    run_pipeline_tests()
