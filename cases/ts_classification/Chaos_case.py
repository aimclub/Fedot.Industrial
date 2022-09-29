import numpy as np
import pandas as pd
import pylab as plt

from core.operation.transformation.TS import TSTransformer
from fedot.api.main import Fedot
from sklearn.metrics import roc_auc_score as roc_auc


class FedotModel:
    def __init__(self, x_data, y_data, time):
        self.fedot_model = Fedot(problem='classification',
                                 timeout=time,
                                 seed=42, logging_level=20,
                                 safe_mode=False,
                                 n_jobs=4)
        self.x_data = x_data
        self.y_data = y_data

    def run_model(self):
        # During fit, the pipeline composition algorithm is started
        self.fedot_model.fit(features=self.x_data,
                             target=self.y_data)

    def evalutate(self, x_test):
        return self.fedot_model.predict_proba(features=x_test)

    def get_metrics(self):
        prediction = self.fedot_model.predict_proba(features=self.x_data)
        print(self.fedot_model)
        print(f'ROC AUC score on training sample: {roc_auc(self.y_data, prediction):.3f}')


def plot_rec_plot_by_class(df):
    df.columns = ["Label" if len(x) == 1 else x for x in df.columns]
    for val in df['Label'].unique():
        tmp = df[df['Label'] == val]
        ru = tmp.iloc[1:2, 1:].values.reshape(-1)
        transformer = TSTransformer(time_series=ru)
        plt.title("Normal")
        plt.subplot(221)
        plt.plot(ru)
        plt.title("Unitary")
        plt.subplot(223)
        plt.imshow(transformer.ts_to_reccurancy_matrix(eps=eps, steps=steps))
        plt.show()


def create_chaos_features(df):
    converted_df = []
    for index, row in df.iterrows():
        transformer = TSTransformer(time_series=row.values[1:])
        metrics = transformer.get_reccurancy_metrics()
        metrics['Label'] = row.values[0]
        converted_df.append(pd.Series(metrics))
    converted_df = pd.concat(converted_df, axis=1).T
    return converted_df


def prepare_data_for_case(dataset):
    df = pd.read_csv(
        f'.\data\{dataset}\{dataset}_TRAIN.tsv',
        sep="\t")
    plot_rec_plot_by_class(df)
    df_encoded_train = create_chaos_features(df)
    df_encoded_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_encoded_train = df_encoded_train.fillna(0)
    train_label = df_encoded_train['Label'].values
    #stat_train = df_encoded_train.groupby(by=['Label']).describe()

    del df_encoded_train['Label']

    df = pd.read_csv(
        f'.\data\{dataset}\{dataset}_TEST.tsv',
        sep="\t")
    df_encoded_test = create_chaos_features(df)
    df_encoded_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_encoded_test = df_encoded_test.fillna(0)
    test_label = df_encoded_test['Label'].values
    #stat_test = df_encoded_test.groupby(by=['Label']).describe()
    del df_encoded_test['Label']
    return df_encoded_train, df_encoded_test, train_label, test_label


def run_case(df_encoded_train, df_encoded_test, train_label, test_label):
    clf_model = FedotModel(x_data=df_encoded_train.values, y_data=train_label, time=30)
    clf_model.run_model()
    predictions = clf_model.evalutate(df_encoded_test.values)
    final_result = roc_auc(test_label, predictions, multi_class='ovo')
    return clf_model, final_result


if __name__ == "__main__":
    eps = 0.1
    steps = 20
    dataset = 'ElectricDevices'
    df_encoded_train, df_encoded_test, train_label, test_label = prepare_data_for_case(dataset)
    model, roc_auc = run_case(df_encoded_train, df_encoded_test, train_label, test_label)
    print(roc_auc)
