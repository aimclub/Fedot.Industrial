Object detection example
========================
This example shows how to use the object detection API to detect objects in an image.

An example of solving a problem of this class is the problem of detecting objects on images. In this example an experiment with two methods of structure optimization is carried out

The idea of the SVD method is shown: the weights of each layer are decomposed into three matrices, then training takes place with additional constraints to preserve orthogonality of matrices and sparsity of singular values. After training, the least significant singular numbers and corresponding rows and columns in the orthogonal matrices are pruned. Thus, it is possible to achieve a significant reduction in the number of parameters. During the direct pass the product of these matrices is calculated, which corresponds to the weight matrix of the original dimensionality.

The first step of the experiment is to prepare the data in the pytorch dataset format, as shown in the listing below

.. code-block:: python

    import sys
    sys.path.append('../..')

    from torchvision.transforms import Compose, ToTensor, Resize
    from torchvision.datasets import ImageFolder

    dataset_path = 'your_path'

    transform = Compose([ToTensor(), Resize((256, 256))])
    train_dataset = ImageFolder(root=dataset_path + 'train',
                                transform=transform)
    val_dataset = ImageFolder(root=dataset_path + 'validation',
                              transform=transform)

Then it is necessary to set the parameters of the experiments that will be carried out:

- experiment without structure optimization;
- experiment with singular value decomposition of convolutional layers;
- experiments with soft filter pruning


.. code-block:: python

    import torch

    energy_thresholds = [
        0.1, 0.3, 0.5, 0.7, 0.9,
        0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
        0.992, 0.994, 0.996, 0.998, 0.999,
        0.9999, 1
    ]

    exp_params = {
        'dataset_name': 'Land-Use_Scene_Classification',
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'num_classes': 21,
        'dataloader_params': {'batch_size': 32, 'num_workers': 4},
        'model': 'ResNet18',
        'model_params': {},
        'models_saving_path': 'models',
        'optimizer': torch.optim.Adam,
        'optimizer_params': {},
        'target_loss': torch.nn.CrossEntropyLoss,
        'loss_params': {},
        'metric': 'f1',
        'summary_path': 'runs',
        'summary_per_class': True,
        'gpu': True
    }

    optimizations = {
        'none': [{}],
        'SVD': [
            {
                'decomposing_mode': 'spatial',
                'orthogonal_loss_factor': 10,
                'hoer_loss_factor': 0.001,
                'energy_thresholds': energy_thresholds,
                'finetuning_epochs': 1
            },
            {
                'decomposing_mode': 'channel',
                'orthogonal_loss_factor': 10,
                'hoer_loss_factor': 0.001,
                'energy_thresholds': energy_thresholds,
                'finetuning_epochs': 1
            },
            {
                'decomposing_mode': 'spatial',
                'orthogonal_loss_factor': 100,
                'hoer_loss_factor': 0.001,
                'energy_thresholds': energy_thresholds,
                'finetuning_epochs': 1
            },
            {
                'decomposing_mode': 'channel',
                'orthogonal_loss_factor': 100,
                'hoer_loss_factor': 0.001,
                'energy_thresholds': energy_thresholds,
                'finetuning_epochs': 1
            },
        ],
        'SFP': [
            {
                'pruning_ratio': 0.5,
                'finetuning_epochs': 1
            },
            {
                'pruning_ratio': 0.7,
                'finetuning_epochs': 1
            },
            {
                'pruning_ratio': 0.8,
                'finetuning_epochs': 1
            },
            {
                'pruning_ratio': 0.9,
                'finetuning_epochs': 1
            },
        ]
    }


After that, using the basic ClassificationExperimenter class, you can perform experiments for 100 epochs

.. code-block:: python

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    import warnings
    warnings.filterwarnings("ignore")
    from fedot_ind.core.architecture.experiment.CVModule import ClassificationExperimenter

    for optimization, params_list in optimizations.items():
        for params in params_list:
            experimenter = ClassificationExperimenter(
                structure_optimization=optimization,
                structure_optimization_params=params,
                **exp_params
            )
            experimenter.fit(100)


The code below is used to run the optimization of the pre-built model

.. code-block:: python

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    import warnings
    warnings.filterwarnings("ignore")
    from fedot_ind.core.architecture.experiment.CVModule import ClassificationExperimenter

    experimenter = ClassificationExperimenter(
        structure_optimization='SFP',
        structure_optimization_params={
            'pruning_ratio': 0.5,
            'finetuning_epochs': 10
        },
        weights='models/Land-Use_Scene_Classification/ResNet18/trained_model.sd.pt',
        prefix='without_training_',
        **exp_params
    )
    experimenter.structure_optimization.optimize_during_training()
    experimenter.structure_optimization.final_optimize()

    experimenter = ClassificationExperimenter(
        structure_optimization='SVD',
        structure_optimization_params={
            'decomposing_mode': 'channel',
            'orthogonal_loss_factor': 10,
            'hoer_loss_factor': 0,
            'energy_thresholds': energy_thresholds,
            'finetuning_epochs': 3
        },
        weights='models/Land-Use_Scene_Classification/ResNet18/trained_model.sd.pt',
        prefix='without_training_',
        **exp_params
    )
    experimenter.structure_optimization.final_optimize()


Collected in one table the results of the simulation are presented in the table

+------------------+-----------+----------------------+------------------+
| Model            | F1        | Validation accuracy  | Train accuracy   |
+==================+===========+======================+==================+
| Baseline         | 0.937292  | 0.938095             | 1.000000         |
+------------------+-----------+----------------------+------------------+
| SFP 50%          | 0.932722  | 0.933333             | 0.999728         |
+------------------+-----------+----------------------+------------------+
| SVD spatial 100  | 0.923959  | 0.924286             | 0.996060         |
+------------------+-----------+----------------------+------------------+
| SVD channel 100  | 0.903336  | 0.904286             | 0.993750         |
+------------------+-----------+----------------------+------------------+
| SFP 70%          | 0.901154  | 0.901429             | 0.996060         |
+------------------+-----------+----------------------+------------------+
| SVD channel 10   | 0.899932  | 0.900476             | 0.985734         |
+------------------+-----------+----------------------+------------------+
| SVD spatial 10   | 0.897342  | 0.898095             | 0.988043         |
+------------------+-----------+----------------------+------------------+
| SFP 80%          | 0.874307  | 0.874286             | 0.987031         |
+------------------+-----------+----------------------+------------------+
| SFP 90%          | 0.359079  | 0.390476             | 0.793021         |
+------------------+-----------+----------------------+------------------+

Models trained by soft filter pruning optimization can be compressed in a single way, while models trained by singular value decomposition optimization can be compressed in various ways. In this way, an acceptable compromise can be found between the loss of object recognition quality and model size.

.. image:: ./obj_detection_img/obj_det.png
    :width: 600
    :align: center