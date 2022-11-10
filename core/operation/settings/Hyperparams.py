import numpy as np
from torchvision.transforms import Compose, Normalize, ToTensor, Resize


def quantile(column, q: str):
    return np.quantile(a=column, q=q)


stat_methods_default = {
    'mean_': np.mean,
    'median_': np.median,
    'std_': np.std,
    'var_': np.var,
    'q5_': quantile,
    'q25_': quantile,
    'q75_': quantile,
    'q95_': quantile,
}

stat_methods_full = {
    'mean_': np.mean,
    'median_': np.median,
    'lambda_less_zero': lambda x: x < 0.01,
    'std_': np.std,
    'var_': np.var,
    'max': np.max,
    'min': np.min,
    'q5_': quantile,
    'q25_': quantile,
    'q75_': quantile,
    'q95_': quantile,
    'sum_': np.sum,
    'dif_': np.diff
}

hyper_param_dict = {'statistical_methods': stat_methods_default,
                    'statistical_methods_extra': stat_methods_full}


def select_hyper_param(param_name):
    return hyper_param_dict[param_name]


DATASETS_PARAMETERS = {
    'CIFAR100': {
        'train_ds_params': {
            'train': True,
            'download': True,
            'transform': Compose(
                [
                    ToTensor(),
                    Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
                ]
            ),
        },
        'val_ds_params': {
            'train': False,
            'download': False,
            'transform': Compose(
                [
                    ToTensor(),
                    Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
                ]
            ),
        },
    },
    'CIFAR10': {
        'train_ds_params': {
            'train': True,
            'download': True,
            'transform': Compose(
                [
                    ToTensor(),
                    Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
                ]
            ),
        },
        'val_ds_params': {
            'train': False,
            'download': False,
            'transform': Compose(
                [
                    ToTensor(),
                    Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
                ]
            ),
        },
    },
    'MNIST': {
        'train_ds_params': {
            'train': True,
            'download': True,
            'transform': Compose(
                [
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,)),
                ]
            ),
        },
        'val_ds_params': {
            'train': False,
            'download': False,
            'transform': Compose(
                [
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,)),
                ]
            ),
        },
    },
    'ImageNet': {
        'train_ds_params': {
            'split': 'train',
            'transform': Compose(
                [
                    Resize((256, 265)),
                    ToTensor(),
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            ),
        },
        'val_ds_params': {
            'split': 'val',
            'transform': Compose(
                [
                    Resize((256, 265)),
                    ToTensor(),
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            ),
        },
    },
}
