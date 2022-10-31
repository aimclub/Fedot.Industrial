import torch

from core.operation.utils.cv_experimenters import FasterRCNNExperimenter

if __name__ == "__main__":
    root = "/media/n31v/data/"
    experiment_parameters = {
        "dataset": "ALET",
        "dataset_params": {
            "datasets_folder": root + "datasets",
            "batch_size": 2,
            "num_workers": 4
        },
        "model_params": {"weights": "DEFAULT"},
        "models_saving_path": root + "models",
        "optimizer": torch.optim.SGD,
        "optimizer_params": {"lr": 0.005, "momentum": 0.9, "weight_decay": 0.0005},
        "scheduler_params": {"step_size": 5, "gamma": 0.5},
        "target_metric": "map_50",
        "summary_path": "/home/n31v/workspace/runs",
        "summary_per_class": False,
    }
    svd_parameters = {
        "decomposing_mode": "channel",
        "orthogonal_loss_factor": 10,
        "hoer_loss_factor": 0.001,
        "energy_thresholds": [
            0.1, 0.3, 0.5, 0.7, 0.9,
            0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
            0.992, 0.994, 0.996, 0.998, 0.999,
            0.9999, 1
        ],
        "finetuning_epochs": 5
    }
    sfp_parameters = {
        "pruning_ratio": 0.5
    }
    modes = {"none": {}, "SVD": svd_parameters, "SFP": sfp_parameters}

    mode = "none"

    experiment_parameters["structure_optimization"] = mode
    experiment_parameters["structure_optimization_params"] = modes[mode]
    experimenter = FasterRCNNExperimenter(**experiment_parameters)
    experimenter.run(50)
