Quickstart
==========

First of all, you need to :ref:`install <installation>` FEDOT.Industrial.

To conduct time series classification you need to first
set experiment configuration using file ``cases/config/Config_Classification.yaml``
and then run ``cases/classification_experiment.py`` script, or create your own
with the following code:

.. code-block:: python

    from core.api.tsc_API import Industrial


    if __name__ == '__main__':
        config = {'feature_generator': ['spectral', 'wavelet'],
                  'datasets_list': ['UMD', 'Lightning7'],
                  'use_cache': True,
                  'error_correction': False,
                  'launches': 3,
                  'timeout': 15,
                  'n_jobs': 2}

        ExperimentHelper = Industrial()
        ExperimentHelper.run_experiment(config)