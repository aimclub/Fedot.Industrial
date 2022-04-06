import os
from typing import Union
import numpy as np
import pandas as pd


def save_results(predictions: Union[np.ndarray, pd.DataFrame],
                 prediction_proba: Union[np.ndarray, pd.DataFrame],
                 target: Union[np.ndarray, pd.Series],
                 path_to_save: str,
                 inference: float,
                 fit_time: float,
                 window: int,
                 metrics: dict = None):
    path_results = os.path.join(path_to_save, 'test_results')
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    if type(prediction_proba) is not pd.DataFrame:
        df_preds = pd.DataFrame(prediction_proba)
        df_preds['Target'] = target
        df_preds['Preds'] = predictions
    else:
        df_preds = prediction_proba
        df_preds['Target'] = target.values

    if type(metrics) is str:
        df_metrics = pd.DataFrame()
    else:
        df_metrics = pd.DataFrame.from_records(data=[x for x in metrics.items()]).reset_index()
    df_metrics['Inference'] = inference
    df_metrics['Fit_time'] = fit_time
    df_metrics['window'] = window
    for p, d in zip(['probs_preds_target.csv', 'metrics.csv'],
                    [df_preds, df_metrics]):
        full_path = os.path.join(path_results, p)
        d.to_csv(full_path)

    return


def project_path() -> str:
    name_project = 'IndustrialTS'
    abs_path = os.path.abspath(os.path.curdir)
    while os.path.basename(abs_path) != name_project:
        abs_path = os.path.dirname(abs_path)
    return abs_path


def path_to_save_results() -> str:
    path = project_path()
    save_path = os.path.join(path, 'results_of_experiments')
    return save_path




from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap, MDS, LocallyLinearEmbedding, SpectralEmbedding, TSNE

if __name__ == '__main__':
    # eigen_list = []
    #
    # data = InputData.from_csv(
    #     file_path=r'C:\Users\user\Desktop\Репозитории\Huawei_AutoML\data\classification\dionis.csv',
    #     target_columns='class')
    # train_data, test_data = train_test_data_setup(data)
    #
    # # for kernel in ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']:
    # #     transformer = KernelPCA(n_components=10, kernel=kernel, remove_zero_eig=True)
    # #     X_transformed = transformer.fit_transform(train_data.features[:5000])
    # #     eigen_val = transformer.lambdas_
    # #     eigen_val_norm = [x * 100 / sum(eigen_val) for x in eigen_val]
    # #     eigen_list.append(eigen_val_norm)
    # #     print(eigen_val_norm)
    # # _ = 1
    # # target = pd.read_csv(r'C:\Users\user\Desktop\Репозитории\Huawei_AutoML\cases\classification\target.csv',header=None,names=['real'])
    # # predictions = pd.read_csv(
    # #     r'C:\Users\user\Desktop\Репозитории\Huawei_AutoML\cases\classification\predictions.csv',header=None,names=['pred'])
    # # proba_redictions = pd.read_csv(
    # #     r'C:\Users\user\Desktop\Репозитории\Huawei_AutoML\cases\classification\proba_predictions.csv', header=None,sep=',')
    # # train_data.target = predictions
    # # predictions['compare'] = predictions['pred'] == target['real']
    # # print(predictions['compare'].value_counts())
    # # pred = OutputData
    # # pred.predict = target
    # # acc = Accuracy.metric(reference=test_data, predicted=pred)
    # # pred.predict = proba_redictions
    # # roc = ROCAUC.metric(reference=test_data, predicted=pred)
    # # pred.predict = target
    # # f1 = F1.metric(reference=test_data, predicted=pred)
    #
    # transformer = KernelPCA(n_components=4, kernel='poly',degree=3, remove_zero_eig=True)
    # X_transformed = transformer.fit_transform(train_data.features[:5000])
    # train_data.features = transformer.transform(train_data.features)
    # test_data.features = transformer.transform(test_data.features)
    # task_type = data.task.task_type.name
    # fedot = Fedot(problem=task_type, timeout=20,verbose_level=4)
    # fedot.fit(features=train_data)
    # predictions = fedot.predict(test_data)
    # proba_predictions = fedot.predict_proba(test_data)
    # target = test_data.target
    # _, metrics_name = problem_and_metric_for_dataset(task_type=task_type)
    # metric_list = calculate_metrics(metrics_name, target=target, predicted_labels=predictions, convert_flag=False)
    #
    #
    # lle = LocallyLinearEmbedding(n_components=2, n_neighbors=30)
    # embedding = MDS(n_components=3)
    # model = Isomap(n_components=2, n_neighbors=6)
    #
    # X_reduced = lle.fit_transform(X[:1000])
    #
    my_file = open(r'C:\Users\user\Desktop\Репозитории\Huawei_AutoML\results_of_experiments\FEDOT\delta_ailerons.csv\launch_0\predictions.txt')
    text = my_file.read()
    d = json.loads(text)
    dd = [x[0] for x in d]
    my_file = open(r'C:\Users\user\Desktop\Репозитории\Huawei_AutoML\results_of_experiments\FEDOT\delta_ailerons.csv\launch_0\target.txt')
    text = my_file.read()
    d = json.loads(text)
    r2 = r2_score(d,dd)
    rmse = mean_squared_error(d,dd,squared=False)
    # pred = pd.read_csv(r'C:\Users\user\Desktop\Репозитории\Huawei_AutoML\results_of_experiments\AutoGluon\pol.csv\launch_0\utils\predictions.csv',sep=';')
    # y = pd.read_pickle(r'C:\Users\user\Desktop\Репозитории\Huawei_AutoML\results_of_experiments\AutoGluon\pol.csv\launch_0\utils\data\y.pkl')
    # r2 = r2_score(y.values,pred.values)
    # rmse = mean_squared_error(y.values,pred.values,squared=False)
    libraries_to_compare = ['H20',
                            'AutoGluon',
                            'FEDOT']
    df_report, generation_dict, metric_dict = create_report_df(libraries_to_compare,
                                                               project_path,
                                                               profile_flag=False,
                                                               save_flag=True)
    df_report.to_csv('./061221.csv')
    # structure_analysis = get_structure_analysis(generation_dict, metric_dict)
    _ = 1
