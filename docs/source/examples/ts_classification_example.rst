Time series classification example
==================================
This example shows how to use the framework to perform time series classification.

Basic TSC experiment
--------------------

First, we import the necessary modules. Base class :ref:`Industrial<industrial-class-label>` provides all the capabilities
for time series classification.

.. code-block:: python

    from core.api.tsc_API import Industrial

Then, we set the path to the config file. The config file contains the parameters for the model and the data.

.. code-block:: python

    config_name = 'cases/ts_classification_example/configs_for_examples/BasicConfigCLF.yaml'

It is of a great importance to define corresponding parameters of experiment.
Config file contains the following parameters:

- ``feature_generators`` - list of feature generators to use in the experiment
- ``use_cache`` - whether to use cache or not
- ``datasets_list`` - list of datasets to use in the experiment
- ``launches`` - number of launches for each dataset
- ``feature_generator_params`` - specification for feature generators
- ``fedot_params`` - specification for FEDOT algorithmic kernel
- ``error_correction`` - flag for application of error correction model in the experiment
- ``n_ecm_cycles`` - number of cycles for error correction model

Finally, we create an instance of the class :ref:`Industrial<industrial-class-label>` and run the experiment.

.. code-block:: python

    ExperimentHelper = Industrial()
    ExperimentHelper.run_experiment(config_name)

To accelerate repetitive experiments, the feature caching mechanism is implemented. It allows to dump generated features
to the disk and load them later. To enable this feature, set ``use_cache`` parameter to ``True`` in the config file.

Ensemble experiment
-------------------

The ensemble experiment is a special case of the basic experiment. It allows to use the ensemble of models or the ensemble
of feature generators. To enable feature generators ensemble, set the following option among the feature generators
in the config file:

.. code-block:: yaml

    feature_generators: ['ensemble: topological wavelet window_quantile quantile spectral spectral_window']

This way the ensemble of feature space of topological, wavelet, window_quantile, quantile, spectral and spectral_window
feature generators will be used as a single feature space.

To use an ensemble of models, a method of the class :ref:`Industrial<industrial-class-label>` ``apply_ensemble``
should be called. Its usage described in the :ref:`Advanced approaches<tsc-ensembling>` section.