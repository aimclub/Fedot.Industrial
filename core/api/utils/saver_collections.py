import os

import pandas as pd
from fedot.core.log import default_log as logger


class ResultSaver:
    def __init__(self):
        self.path = None
        self.logger = logger(self.__class__.__name__)
        self.save_method_dict = {'ECM': self.save_boosting_results,
                                 'Ensemble': self.save_ensemble_results,
                                 'Original': self.save_basic_results}

    def save_basic_results(self,
                           prediction: dict, path, dataset_name: str):
        for model in prediction:
            if prediction[model]:
                path_to_save = os.path.join(path, model, dataset_name)
                for launch in prediction[model]:
                    if prediction[model][launch]:
                        path_test_results = os.path.join(path_to_save, str(launch), 'test_results')
                        path_features = os.path.join(path_to_save, str(launch))
                        path_pipeline = os.path.join(path_to_save, str(launch), 'tsc_pipeline')
                        os.makedirs(path_test_results, exist_ok=True)

                        result_at_launch = prediction[model][launch]
                        feature_target_dict = {'train_features.csv': result_at_launch['train_features'],
                                               'train_target.csv': result_at_launch['train_target'],
                                               'test_features.csv': result_at_launch['test_features'],
                                               'test_target.csv': result_at_launch['test_target']}

                        try:
                            result_at_launch['fitted_predictor'].current_pipeline.save(path_pipeline,
                                                                                       datetime_in_path=False)
                        except Exception as ex:
                            self.logger.error(f'Can not save pipeline: {ex}')

                        for name, features in feature_target_dict.items():
                            pd.DataFrame(features).to_csv(os.path.join(path_features, name))

                        if not isinstance(result_at_launch['class_probability'], pd.DataFrame):
                            df_preds = pd.DataFrame(result_at_launch['class_probability'].round(3), dtype=float)
                            df_preds.columns = [f'Class_{x + 1}' for x in df_preds.columns]
                            df_preds['Target'] = result_at_launch['test_target']
                            df_preds['Predicted_labels'] = result_at_launch['label']
                        else:
                            df_preds = result_at_launch['class_probability']
                            df_preds['Target'] = result_at_launch['test_target'].values

                        if isinstance(result_at_launch['metrics'], str):
                            df_metrics = pd.DataFrame()
                        else:
                            df_metrics = pd.DataFrame.from_records(data=[x for x in result_at_launch['metrics'].items()],
                                                                   columns=['Metric_name', 'Metric_value']).reset_index()
                            del df_metrics['index']
                            df_metrics = df_metrics.T
                            df_metrics = pd.DataFrame(df_metrics.values[1:], columns=df_metrics.iloc[0])

                        for p, d in zip(['probs_preds_target.csv', 'metrics.csv'],
                                        [df_preds, df_metrics]):
                            full_path = os.path.join(path_test_results, p)
                            d.to_csv(full_path)

    def save_boosting_results(self, prediction):
        for launch in prediction:
            path_to_save = prediction[launch]['path_to_save']
            location = os.path.join(path_to_save, 'boosting')
            if not os.path.exists(location):
                os.makedirs(location)
            prediction[launch]['solution_table'].to_csv(os.path.join(location, 'solution_table.csv'))
            prediction[launch]['metrics_table'].to_csv(os.path.join(location, 'metrics_table.csv'))

            models_path = os.path.join(location, 'boosting_pipelines')
            if not os.path.exists(models_path):
                os.makedirs(models_path)
            for index, model in enumerate(prediction['model_list']):
                try:
                    model.current_pipeline.save(path=os.path.join(models_path, f'boost_{index}'),
                                                datetime_in_path=False)
                except Exception:
                    model.save(path=os.path.join(models_path, f'boost_{index}'),
                               datetime_in_path=False)
            if prediction['ensemble_model'] is not None:
                try:
                    prediction[launch]['ensemble_model'].current_pipeline.save(
                        path=os.path.join(models_path, 'boost_ensemble'),
                        datetime_in_path=False)
                except Exception:
                    prediction[launch]['ensemble_model'].save(path=os.path.join(models_path, 'boost_ensemble'),
                                                              datetime_in_path=False)
            else:
                self.logger.info('Ensemble model cannot be saved due to applied SUM method ')

    def save_ensemble_results(self, prediction: dict, path: str, dataset_name: str):
        prediction.update({'dataset': dataset_name})
        path_to_save = os.path.join(path, 'ensemble')
        os.makedirs(path_to_save, exist_ok=True)
        ensemble_results = pd.DataFrame.from_records(data=[x for x in prediction.items()]).T
        ensemble_results.to_csv(os.path.join(path_to_save, f'{dataset_name}_ensemble_results.csv'),
                                header=False)
        self.logger.info(f'Ensemble results saved')
