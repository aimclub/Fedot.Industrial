Architecture of FEDOT.Industrial
================================

The FEDOT.Industrial framework is a versatile tool for the time-series
engineer. The software is designed to minimize user involvement in the
programming area while maximizing the quality of the results obtained
for tasks such as: time series classification and forecasting, anomaly detection, and
computer vision tasks including object detection and recognition.


We propose the data-driven automated time series classification
approach, which main idea is to combine selection of features in one
feature space and an automated model design using a
graph-based pipeline representation and evolutionary optimization.

.. image:: img_introduction/architecture.png
   :width: 700px
   :align: center
   :alt: Architecture of FEDOT

The input data (Input Data block) in the form of
time series goes to the ``Feature Design`` block. The selected features are the
input data for the ``Model Design`` block, where the evolutionary algorithm
(``Evolutionary Optimizer`` block) selects the optimal Pipeline for solving the problem.
The final pipelines are then applied to obtain predictions, which
are interpreted with application in the corresponding block. To evaluate
the quality of the intermediate results and the final predictions, the
``Quality Analyzer`` block is created, implementing various evaluation metrics.

For the machine learning pipeline creation, represented as an acyclic
graph, we used an approach based on an evolutionary algorithm [1]_.
This pipeline combines multiple methods of feature extraction and
machine learning models. The aim of the evolutionary optimiser is to
obtain the effective but computationally lightweight pipeline avoiding
over-complicated solutions.

It could be used by specialists without machine learning or data
science experience for application tasks in their professional domain,
where the primary data type is time series.



.. [1] Nikitin, Nikolay O., et al. "Automated evolutionary approach
        for the design of composite machine learning pipelines."
        Future Generation Computer Systems 127 (2022): 109-125.