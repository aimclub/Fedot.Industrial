from core.api.API import Industrial


if __name__ == '__main__':
    config = dict(feature_generator=['topological', 'wavelet'],
                  datasets_list=['UMD', 'Lightning7'],
                  use_cache=True,
                  error_correction=False,
                  launches=3,
                  timeout=10)

    ExperimentHelper = Industrial()
    ExperimentHelper.run_experiment(config=config)
