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
        config_name = 'Config_Classification.yaml'
        ExperimentHelper = Industrial()
        ExperimentHelper.run_experiment(config_name)