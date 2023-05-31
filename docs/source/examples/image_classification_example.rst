.. _image_classification_example:

Image Classification Example
============================

Basic Image classification example
----------------------------------

At the first step we need to import the necessary libraries and packages, and also configure a path to train and
validation datasets

.. code-block:: python

    import os
    import platform

    from torchvision.transforms import ToTensor, Resize, Compose

    from fedot_ind.api.main import FedotIndustrial


    DATASETS_PATH = os.path.abspath('../data/cv/datasets')


Then, we need to instantiate the class FedotIndustrial with appropriate task type. Also, as the important parameter
either the number of classes or torch model should be passed, as well as the device (cuda or cpu, depending on hardware
used)

.. code-block:: python

    task = 'classification'
    num_classes = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = FedotIndustrial(task=task,
                            num_classes=num_classes,
                            device=device)


The next step is model training with conventional method fit. Here we pass dataset_path, transform option and desirable
number of epochs:

.. code-block:: python

    trained_model = fed.fit(dataset_path=os.path.join(DATASETS_PATH, 'Agricultural/train'),
                            transform=Compose([ToTensor(), Resize((256, 256))]),
                            num_epochs=2)


To obtain predict one must use the following code:

.. code-block:: python

    predict = fed.predict(data_path=os.path.join(DATASETS_PATH, 'Agricultural/val'),
                          transform=Compose([ToTensor(), Resize((256, 256))]))

    predict_proba = fed.predict_proba(data_path=os.path.join(DATASETS_PATH, 'Agricultural/val'),
                                  transform=Compose([ToTensor(), Resize((256, 256))]))


Advanced Image classification example
-------------------------------------

To conduct an advanced experiment one should instantiate FedotIndustrial class with optimization method argument and optimization parameters:

.. code-block:: python

    fed = FedotIndustrial(task='image_classification',
                          num_classes=3,
                          optimization='svd',
                          optimization_params={'energy_thresholds': [0.9]},
                          # Taking into account hardware specifics
                          device='cpu' if platform.system() == 'Darwin' else 'cuda')


Method fit also must be provided with additional argument – finetuning_params:

.. code-block:: python

    fitted_model = fed.fit(dataset_path=os.path.join(DATASETS_PATH, 'Agricultural/train'),
                           transform=Compose([ToTensor(), Resize((256, 256))]),
                           num_epochs=2,
                           finetuning_params={'num_epochs': 1})


To obtain predict one must use the following code as before:

.. code-block:: python

    predict = fed.predict(data_path=os.path.join(DATASETS_PATH, 'Agricultural/val'),
                          transform=Compose([ToTensor(), Resize((256, 256))]))

    predict_proba = fed.predict_proba(data_path=os.path.join(DATASETS_PATH, 'Agricultural/val'),
                                  transform=Compose([ToTensor(), Resize((256, 256))]))
