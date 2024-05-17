from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from benchmark.benchmark_TSC import BenchmarkTSC

experiment_setup = {
    'problem': 'classification',
    'metric': 'accuracy',
    'timeout': 15,
    'num_of_generations': 15,
    'pop_size': 50,
    'logging_level': 10,
    'RAF_workers': 4,
    'output_folder': './riemann',
    'n_jobs': 2,
    'initial_assumption': PipelineBuilder().add_node('riemann_extractor').add_node('logit'),
    'available_operations': [
        'riemann_extractor',
        'quantile_extractor',
        'scaling',
        'normalization',
        'kernel_pca',
        'xgboost',
        'rf',
        'mlp',
        'logit',
        'fourier_basis',
        'wavelet_basis',
        'eigen_basis'],
    'industrial_preprocessing': False,
    'max_pipeline_fit_time': 25,
    'with_tuning': True,
    'early_stopping_iterations': 5,
    'early_stopping_timeout': 100}

if __name__ == "__main__":
    benchmark = BenchmarkTSC(experiment_setup=experiment_setup,
                             custom_datasets=[
                                 # 'EthanolConcentration',
                                 'Handwriting',
                                 'StandWalkJump',
                                 'EigenWorms',
                                 'AtrialFibrillation',
                                 'UWaveGestureLibrary',
                                 'PhonemeSpectra',
                                 'SelfRegulationSCP2',
                                 'ERing',
                                 'Libras'
                             ],
                             use_small_datasets=False)
    benchmark.run()
    _ = 1
