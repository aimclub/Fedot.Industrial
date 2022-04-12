import pandas as pd
import logging
import os
from core.operation.utils.utils import project_path


def read_tsv(file_name: str):
    df_train = pd.read_csv(
        os.path.join(project_path(), 'data', file_name, f'{file_name}_TRAIN.tsv'),
        sep='\t',
        header=None)

    X_train = df_train.iloc[:, 1:]
    y_train = df_train[0].values

    df_test = pd.read_csv(
        os.path.join(project_path(), 'data', file_name, f'{file_name}_TEST.tsv'),
        sep='\t',
        header=None)

    X_test = df_test.iloc[:, 1:]
    y_test = df_test[0].values

    return (X_train, X_test), (y_train, y_test)


def get_logger():
    logger = logging.getLogger('Experiment logger')
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    return logger
