import os

import pandas as pd

from core.operation.utils.utils import project_path


def read_tsv(file_name: str):
    df_train = pd.read_csv(
        os.path.join(project_path(), 'data', file_name, f'{file_name}_TRAIN.tsv'),
        sep='\t',
        header=None)

    x_train = df_train.iloc[:, 1:]
    y_train = df_train[0].values

    df_test = pd.read_csv(
        os.path.join(project_path(), 'data', file_name, f'{file_name}_TEST.tsv'),
        sep='\t',
        header=None)

    x_test = df_test.iloc[:, 1:]
    y_test = df_test[0].values

    return (x_train, x_test), (y_train, y_test)
