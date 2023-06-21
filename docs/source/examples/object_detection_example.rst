.. _object_detection_example:

Object Detection Example
========================

This example shows how to use the `Fedot Industrial` framework to solve the object detection problem.

Basic object detection example
----------------------------------

.. code-block:: python

    from fedot_ind.api.main import FedotIndustrial

    model = FedotIndustrial(task='object_detection', num_classes=10)
    trained_model = fed.fit(dataset_path='your dataset yaml file in YOLO format')

At the first step we need to import the necessary libraries and packages.
Then, we need to instantiate the class FedotIndustrial with appropriate task type.
Also, as the important parameter either the number of classes or torch model should be passed.
The next step is model training with conventional method fit.
Here we pass the path to the yaml file, which describes the dataset in the
`YOLO format <https://docs.ultralytics.com/datasets/detect/>`_.

To obtain predict one must use the following code:

.. code-block:: python

    predict = fed.predict(data_path='your path to test images')
    predict_proba = fed.predict_proba(data_path='your path to test images')

Here we pass the path to the folder with test images.



To visualize the predictions, you can use the following method:

.. code-block:: python

    from fedot_ind.core.architecture.datasets.visualization import draw_sample_with_bboxes

    for image, prediction in predict_proba:
        fig = draw_sample_with_bboxes(image=image, prediction=prediction)
