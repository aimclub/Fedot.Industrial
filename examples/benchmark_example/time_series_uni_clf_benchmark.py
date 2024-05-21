from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from benchmark.benchmark_TSC import BenchmarkTSC

experiment_setup = {
    'problem': 'classification',
    'metric': 'accuracy',
    'timeout': 90,
    'num_of_generations': 15,
    'pop_size': 10,
    'logging_level': 10,
    'n_jobs': 2,
    'early_stopping_iterations': 5,
    'initial_assumption': PipelineBuilder().add_node('quantile_extractor').add_node('logit'),
    'early_stopping_timeout': 75}

if __name__ == "__main__":
    benchmark = BenchmarkTSC(experiment_setup=experiment_setup,
                             custom_datasets=[
                                 'Lightning7',
                                 'ToeSegmentation1',
                                 'CricketZ',
                                 'PigArtPressure',
                                 'FacesUCR'
                             ],
                             use_small_datasets=True)
    benchmark.run()
    _ = 1
