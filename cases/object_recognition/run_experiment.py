import torch

from core.operation.utils.cv_optimization_experimenter import run_experiment

if __name__ == "__main__":
    root = "/media/n31v/data/"
    experiment_parameters = {
        "dataset": "CIFAR10",
        "dataset_params": {"datasets_folder": root + "datasets", "batch_size": 100},
        "model": "ResNet18",
        "model_params": {},
        "models_saving_path": root + "models",
        "optimizer": torch.optim.Adam,
        "optimizer_params": {},
        "target_loss": torch.nn.CrossEntropyLoss,
        "loss_params": {},
        "target_metric": "f1",
        "summary_per_class": True,
    }
    svd_parameters = {
        "decomposing_mode": "channel",
        "orthogonal_loss_factor": 1,
        "hoer_loss_factor": 0.001,
        "energy_thresholds": [
            0.1, 0.3, 0.5, 0.7, 0.9,
            0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
            0.992, 0.994, 0.996, 0.998, 0.999,
            0.9999, 1
        ]
    }
    sfp_parameters = {
        "pruning_ratio": 0.7
    }

    run_experiment("SVD", experiment_parameters, svd_parameters, 30)

