Anomaly search in set of time series
==============

Preparing data
--------------

First of all it's good to set the scene. This module could work with set of time series, example of which shown here:

------------------------------------------------------


.. figure:: Images/AD_time_series.png
       :scale: 100 %
       :align: center
       :alt: Typical time series configuration

As could be notice, data here looks very chaotic and "dirty" with noices, but main trouble hide deeper.
Also there is given a legend for visualization, it's actual for every follow images here:

------------------------------------------------------


.. figure:: Images/Colors.png
       :scale: 100 %
       :align: center
       :alt: Colors of lines

First of all module cleanse data and made a little bit of standartization on time series by smoothing, moving and normalizing them:

------------------------------------------------------


.. figure:: Images/TS.png
       :scale: 100 %
       :align: center
       :alt: Typical time series configuration


       This data is such it could be possible to work with.

Feature aggregation
-------------------

Time series in sets could differents a lot from set to set - in real world it's usual that data strongly depends on a lot of factors.

Because of this we can't reliy upon values-based data of one or several of time series in the set - by using only one or two of time series we would lost a lot of data. But still we have more than one time series, so it could be possible to extract some usefull data based on relative configuration of all time series using feature aggregation.

There is several ways to do so, but it was three that looks most promisable: max-min distance, max average absolute deviation and sum of average absolute deviation. Exaples of this three additional time series show as follow:

------------------------------------------------------

.. figure:: Images/Max_sum_dist.png
       :scale: 100 %
       :align: center
       :alt: Typical time series configuration

       Three graphs between -1 and 0 shows how aggregated time series reflecting changing of configurations of main time series. Green one is max-min distance, dark blues one is max of average absolute deviation and red is sum of average absolute deviation.

------------------------------------------------------

As could be seen max-min distance could show zones when several of main time series goes in countephases. The central anomalie is a good example. But it works worce than average-based time series.
Both average deviations-based graphs shows good results in reflecting big changes of main time series, but maximum-based time series works better. It's so because maximum-based aggregation show if even only one of main time series shows big deviation from it's average.


Anomalies search
----------------

In this example splitting aggregated data to zones are essential for furcher anomaly detection. Cutting additional time series by threshold is suitable for this purpose.

Adaptive threshold cutting
~~~~~~~~~~~~~~~~~~~~~~~~~~

But how to choose the value of threshold? Adaptive threshold provide an answer for this question. User could choose percentage of points on the TS that has to be cut. After that adaptive method search for suitable value of threshold, cut and normilize ts.

------------------------------------------------------

.. figure:: Images/Split_aggregated_data.png
       :scale: 100 %
       :align: center
       :alt: Typical time series configuration

       This data is such it could be possible to work with. TS split by zero-zones.

------------------------------------------------------

.. figure:: Images/Cutted.png
       :scale: 100 %
       :align: center
       :alt: Typical time series configuration

       All zones.

Anomaly zones recognition
-------------------------

For this example was created dataset of labled zones, but future versions will include methods of recognition anomalies without dataset. 

Now we have number of zones of time series that could be worked with without problems. First step of analysis of this zones is features extraction.

Features vector extracting
~~~~~~~~~~~~~~~~~~~~~~~~~~

Question is - what features extract from each zones?

After research and a lot of attempts to combine different features from Time series classification module I find following combination of features best sutable for the task:

max

sum

mean

median

mean_median_distance

compressing zone to 10 points

sum of distance of compress zone to 40 points

------------------------------------------------------

.. figure:: Images/Features_vectors.png
       :scale: 70 %
       :align: center
       :alt: Typical time series configuration

       Examples of vectors of anomaly zones of four types: from noice(green) to critical(red) with light(dark blue) and heavy(yellow). As could be easily noticed - anomalies different dramatically.

Next two stages are connected. First - in using the same reducer that reduce long vectors of features to 2D coordinates. Second - in ensambling of predicts from two methods.

Database method
~~~~~~~~~~~~~~~

Next stage of detection is database comparison. Faster way of this could be compare distances between vectors in 2D coordinate system. Each frame of dataset already has reduced coordinates and saved reducer from database could easy reduce vectors of anomaly zones to such coordinates. 

So for each anomaly zones method looks for point that are closest to the point of reduced features vector of the zone. And parametrs of a zone set by parametrs of respected database frame - type, heaviness and comment.

Clusterization method
~~~~~~~~~~~~~~~~~~~~~

Next stage of analysis is find how critical anomaly is. By research was found that anomalies of groups by several clusters:

------------------------------------------------------

.. figure:: Images/Clusters.png
       :scale: 50 %
       :align: center
       :alt: Typical time series configuration

       Clusters of anomaly zones from dataset. Red(Critical) anomalies groups in several clusters, sumtimes together with brown(heavy) anomalies, in places where border between this types arent't clean.

------------------------------------------------------

.. figure:: Images/Zones_clust.png
       :scale: 80 %
       :align: center
       :alt: Typical time series configuration

       Three zones of grouping of anomalies.


Knowing coordinates of zones where anomalies groups - it's easy to check if reduced 2D coordinates of anomalies lie in each zone.

Final predict
-------------

Final predict creates by ensambling data from CLusterization and Dataset methods. Each anomaly zone got prediction from Dataset method. And then each zone the lie inside grouping zones of Clusterization mathod got additional prediction and, in case this new prediction heavier than old one(in type or heaviness), it got new updated values.

------------------------------------------------------

.. figure:: Images/Results.png
       :scale: 60 %
       :align: center
       :alt: Typical time series configuration

       Results of ensambling.

