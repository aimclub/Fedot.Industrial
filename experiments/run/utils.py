import pandas as pd
import logging


def read_tsv(file_name: str):
    df_train = pd.read_csv(
        r'C:\Users\User\Desktop\work-folder\industrial_ts\IndustrialTS\data\{}\{}_TRAIN.tsv'.format(file_name, file_name).format(file_name, file_name),
        sep='\t',
        header=None)
    X_train = df_train.iloc[:, 1:]
    y_train = df_train[0].values
    df_test = pd.read_csv(
        r'C:\Users\User\Desktop\work-folder\industrial_ts\IndustrialTS\data\{}\{}_TEST.tsv'.format(file_name, file_name).format(file_name, file_name),
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
