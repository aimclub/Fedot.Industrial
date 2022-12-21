from core.api.API import Industrial


if __name__ == "__main__":
    config = {
              # 'feature_generator': ['quantile', 'wavelet'],
              'feature_generator': ['window_spectral', 'quantile', 'wavelet'],
              'datasets_list': ['UMD'],
              'use_cache': True,
              'error_correction': False,
              'launches': 1,
              'timeout': 1,
              'n_jobs': 2
               }

    ExperimentHelper = Industrial()
    ExperimentHelper.run_experiment(config=config)
