.. _tsc_advanced:

Advanced TSC experiment
=======================

First, as it was described in :ref:`Basic TSC example<basic-tsc-example>` we import the necessary modules.
Base class :ref:`Industrial<industrial-class-label>` provides all the capabilities
for time series classification.

.. code-block:: python

    from core.api.API import Industrial

Then, instead of config dict with experiment parameters you can use ``yaml`` config
file with extended number of options. Its main fields are the same as in config dict.


.. code-block:: yaml

    feature_generator: ['quantile',
                        'spectral',
                        'ensemble: window_quantile spectral_window']

    error_correction: False
    n_ecm_cycles: 3
    use_cache: True
    ensemble_algorithm: False

    datasets_list: ['Lightning7', 'Earthquakes']

    launches: 3

    feature_generator_params: {
      'window_quantile':
        {
          'window_mode': True,
        },
      'spectral':
        {
          'window_sizes': {
            'ItalyPowerDemand': [3, 6, 9],
            'Earthquakes': [48],
            'Beef': [100]
          },
        },
      'wavelet':
        {
          'wavelet_types': ['sym10']
        }
    }

    fedot_params:
      {
        'problem': 'classification',
        'seed': 42,
        'timeout': 1,
        'max_depth': 10,
        'max_arity': 4,
        'cv_folds': 2,
        'logging_level': 20,
        'n_jobs': 2
      }
    task: 'ts_classification'


It is of a great importance to define corresponding parameters of experiment. The following parameters are required:

- ``feature_generators`` - list of feature generators to use in the experiment
- ``use_cache`` - whether to use cache or not
- ``datasets_list`` - list of datasets to use in the experiment
- ``launches`` - number of launches for each dataset
- ``error_correction`` - flag for application of error correction model in the experiment
- ``n_ecm_cycles`` - number of cycles for error correction model
- ``feature_generator_params`` - hyperparameters for feature generators, for example, ``window_sizes`` for ``spectral`` feature generator
- ``fedot_params`` - hyperparameters for FEDOT. For example, you can adjust here ``n_jobs`` – a number of parallel processes to use in the experiment, or ``timeout`` – a time limit for the pipeline optimization process.
- ``task`` - type of task to solve. For time series classification it is ``ts_classification``
- ``ensemble_algorithm`` - flag for application of model ensemble algorithm in the experiment. It is described in detail in :ref:`Ensemble algorithm<tsc-ensembling>` section.


Finally, we create an instance of the class :ref:`Industrial<industrial-class-label>` and run the experiment.

.. code-block:: python

    path_to_config = 'advanced_tsc_config.yaml'
    ExperimentHelper = Industrial()
    ExperimentHelper.run_experiment(config=path_to_config,
                                    direct_path=True)