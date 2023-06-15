Time Series Data Loader
=======================

This is a tool for loading time series data for classification experiment
either from local directory or from `UCR archive`_. If desired dataset is not found in local
``data`` folder, it will be downloaded from the archive and saved in corresponding directory
in ``.tsv`` format.

.. autoclass:: fedot_ind.core.architecture.preprocessing.DatasetLoader.DataLoader
    :no-undoc-members:

.. _UCR archive: http://www.timeseriesclassification.com