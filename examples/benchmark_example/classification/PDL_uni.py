from benchmark.benchmark_TSC import BenchmarkTSC

init_assumption_pdl = ['quantile_extractor', 'pdl_clf']
init_assumption_rf = ['quantile_extractor', 'rf']
comparasion_dict = dict(pairwise_approach=init_assumption_pdl,
                        baseline=init_assumption_rf)
experiment_setup = {
    'problem': 'classification',
    'metric': 'accuracy',
    'timeout': 2,
    'num_of_generations': 15,
    'pop_size': 10,
    'metric_names': ('f1', 'accuracy'),
    'logging_level': 10,
    'n_jobs': -1,
    'initial_assumption': comparasion_dict,
    'finetune': True}

if __name__ == "__main__":
    benchmark = BenchmarkTSC(experiment_setup=experiment_setup,
                             use_small_datasets=True)
    benchmark.run()
    _ = 1
