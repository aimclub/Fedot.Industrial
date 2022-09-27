import torch

from core.operation.utils.cnn_experimenter import Experimenter


if __name__ == "__main__":
    svd_parameters = {
        "decomposing_mode": "spatial",
        "orthogonal_loss": 1,
        "hoer_loss": 0.001,
        "e": [0.1, 0.3, 0.5, 0.7, 0.9,
              0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
              0.992, 0.994, 0.996, 0.998, 0.999,
              0.9999, 1]
    }
    sfp_parameters = {
        "pruning_ratio": 0.4
    }
    exp = Experimenter(
        dataset="MNIST",
        dataset_params={"ds_path": "/home/ann/datasets/MNIST", "batch_size": 100},
        model="SimpleConvNet2",
        model_params={},
        models_saving_path="/home/ann/models",
        optimizer=torch.optim.Adam,
        optimizer_params={},
        loss_fn=torch.nn.CrossEntropyLoss,
        loss_params={},
        compression_mode="SVD",
        compression_params=svd_parameters,
    )
    exp.run(num_epochs=15)
