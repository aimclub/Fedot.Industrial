Change point detection example
==============================
This example shows how to use the change point detection algorithm to detect
changes in the mean of a time series.

For this example you need to download a real well-log dataset for an oil well.

Workflow for change point detection task
-----------------------------------

**1. First we need to import necessary libraries.**

Setting up sys path for Fedot.Industrial

.. code-block:: python

    import sys
    sys.path.append(r'...\Fedot.Industrial')

Setting up libraries to work with the dataset.

.. code-block:: python

    import requests
    from io import StringIO
    import pandas as pd
    import numpy as np 

Setting up library to work with vizualisation.

.. code-block:: python

    import matplotlib.pyplot as plt

Finally, setting up main modules for change point detection task.

.. code-block:: python

    from cases.anomaly_detection.WSSAlgorithms import WindowSizeSelection
    from cases.anomaly_detection.SSTdetector import SingularSpectrumTransformation

**2. Data Loading.**

Download the dataset for this expiriment.

.. code-block:: python

    def dataframe_expirement():
        url = "https://storage.yandexcloud.net/cloud-files-public/dataframe.csv" 
        dataframe = pd.read_csv(StringIO(requests.get(url).content.decode('utf-8')), sep='|')
        return dataframe

    dataframe = dataframe_expirement()

**3. Some data engineering.**

First of all you need to define oil well which we are working with.

.. code-block:: python

    dataframe_columns = dataframe.columns 
    first_label_list = dataframe[dataframe_columns[0]].unique() 
    dataframe_aa564g = dataframe[dataframe[dataframe_columns[0]] == first_label_list[0]]

Define important columns to work with. For this expirement you need to use just 2 columns.
They are **uR/h** - rock radioactive rate and **unitless** - class of rock (from 0 to 5).

.. code-block:: python

    dataframe_aa564g_first = dataframe_aa564g.drop(axis=1, labels=(dataframe_aa564g.columns[0])) \
        .drop(axis=1, labels=(dataframe_aa564g.columns[1]))[['uR/h', 'unitless']].reset_index(drop=True)

Before using the well-log you need to skip empty rows.

.. code-block:: python

    dataframe_edited_ = dataframe_aa564g_first.loc[dataframe_aa564g_first['unitless'] >= 0]\
    .loc[dataframe_aa564g_first['uR/h'] >= 0].reset_index(drop=True)
    dataframe = dataframe_edited_

You need to define real change points in the dataframe. Let`s assume that changing rock type in real-time considered as change points here.

.. code-block:: python

    cp_1 = []
    for i in range(len(dataframe)-1):
        if dataframe['unitless'][i] !=  dataframe['unitless'][i+1]
            cp_1.append(1)
        else:
            cp_1.append(0)
    cp_1[0] = 0
    dataframe['change_points'] = cp_1

**4. Look at ground true change point labels.**

Just to be sure that it was done in a right way.

.. code-block:: python

    dataframe.change_points.plot(figsize=(12,3))
    plt.legend()
    plt.show()


.. image:: change_point_detection_example_images/ground_true_labels.png
   :alt: Custom ground true labels for the time series
   :width: 500px
   :align: center

**5. Method applying.**

Define your time series and set hypeparameters via WindowSizeSelection class. In the end use SingularSpectrumTransformation to detect change points in the time series.

.. code-block:: python

    ts = list(dataframe['uR/h'])

Highly recommended to use WindowSizeSelection class to choose appropriate SST hypeparameters.

Also, **highly recommended to use 'summary_statistics_subsequences'** as the fastest algorithm for window size selection.

.. code-block:: python

    ts_window_length = WindowSizeSelection(time_series = ts, wss_algorithm = 'summary_statistics_subsequence').runner_wss()[0]
    trajectory_window_length = WindowSizeSelection(time_series = ts[:ts_window_length], window_max = ts_window_length,  wss_algorithm = 'summary_statistics_subsequence').runner_wss()[0]


Set up SST algorithm. Choose lag parameter.

**Highly recommended to use dynamic_mode True.**

.. code-block:: python

    scorer = SingularSpectrumTransformation(time_series = np.array(ts),
                                            ts_window_length = ts_window_length,
                                            lag = 20,
                                            trajectory_window_length = trajectory_window_length)
    score = scorer.score_offline(dynamic_mode=True)


Save results to the dataframe.

.. code-block:: python

    dataframe['results'] = score + [0]
    
There is a small bug you need to add [0] - 29 november 2022

**6. Results**

.. code-block:: python

    plt.figure(figsize=(12,3), dpi=80)
    dataframe['results'].plot(label='predictions', marker='o', markersize=5)
    dataframe['change_points'].plot(label='true change points', marker='o', markersize=2)
    plt.legend();

.. image:: change_point_detection_example_images/predicted_labels.png
   :alt: Predicted Change Points over true labels
   :width: 500px
   :align: center


