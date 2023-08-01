.. _image_classification_example:

Image Classification Example
============================

This example shows how to use the `Fedot Industrial` framework to solve the image classification problem.

Basic image classification example
----------------------------------

.. code-block:: python

    from fedot_ind.api.main import FedotIndustrial

    fed = FedotIndustrial(task='image_classification', num_classes=10)
    trained_model = fed.fit(dataset_path='your dataset path')

At the first step we need to import the necessary libraries and packages.
Then, we need to instantiate the class FedotIndustrial with appropriate task type.
Also, as the important parameter either the number of classes or torch model should be passed.
The next step is model training with conventional method fit.

.. note::

    Here we pass the path to the images folder,  where the images are arranged in this way:

    .. code-block:: python

        dataset_path/class0/xxx.png
        dataset_path/class0/xxy.png
        dataset_path/class0/[...]/xxz.png

        dataset_path/class1/123.png
        dataset_path/class1/nsdf3.png
        dataset_path/class1/[...]/asd932.png

To obtain predict one must use the following code:

.. code-block:: python

    predict = fed.predict(data_path='your path to test images')
    predict_proba = fed.predict_proba(data_path='your path to test images')

Here we pass the path to the folder with test images.

Advanced image classification options
-------------------------------------

When initializing `FedotIndustrial` class, you can pass the following additional parameters:

* **output_folder** - path to record all experiment results.
* **optimization** - model compression method. Can be "svd" or "sfp", where svd is the singular value decomposition method, and sfp is the soft pruning filter method.
* **optimization_params** - parameters passed to optimizer initialization (see :ref:`svd_optimization_api_label` or :ref:`sfp_optimization_api_label`).
* **parameters passed to** :ref:`classification_experimenter_api`.

Also, additional arguments can be passed to the fit method:

* **dataset_name** - name of dataset.
* **num_epochs** - number of training epochs.
* **optimizer** - type of model optimizer, e.g. ``torch.optim.Adam``.
* **lr_scheduler** - Type of learning rate scheduler, e.g ``torch.optim.lr_scheduler.StepLR``.
* **class_metrics** - if ``True``, calculates validation metrics for each class.
* **description** - additional line describing the experiment.
* **finetuning_params** - if you use one of the optimization methods, you can set up the model fine-tuning parameters by passing a dictionary with keys equivalent to the main training: ``dataset_name``, ``num_epochs``, ``optimizer``, ``lr_scheduler``, ``class_metrics``, ``description``.

.. hint::
    If you want to customize part of the optimizer or scheduler options, you can use partial like this:

    .. code-block:: python

        from functools import partial

        fed.fit(
            dataset_path='your dataset path'
            optimizer=partial(
                torch.optim.Adam,
                lr=0.0005,
            )
        )
