.. _tsc-ensembling:

TS classification with model ensemble
=====================================

This is a simple example of how to use the model ensemble feature to obtain more accurate results
in time series classification experiment. It is required to conduct several experiments with different
feature generators to obtain a bunch of models and predictions.

It is a good practice to import the following modules:

.. code-block:: python

    import pandas as pd
    from core.operation.utils.result_parser import ResultsParser
    from core.ensemble.static.RankEnsembler import RankEnsemble


Then, it is important to define several functions to read the results from the experiment folders, report
creation and model ensemble creation. The following code shows how to do it:

.. code-block:: python

    def create_report(experiment_results: dict):
        experiment_df = pd.DataFrame.from_dict(experiment_results, orient='index')
        experiment_df = experiment_df.fillna(0)

        if 'Best_ensemble_metric' not in experiment_df.columns:
            experiment_df['Best_ensemble_metric'] = experiment_df['Base_metric']
        experiment_df['Ensemble_gain'] = (experiment_df['Best_ensemble_metric'] - experiment_df['Base_metric']) * 100
        experiment_df['Ensemble_gain'] = experiment_df['Ensemble_gain'].apply(lambda x: x if x > 0 else 0)

        return experiment_df

    def load_results(folder_path: str, launch_type, model_list: list):
        parser = ResultsParser()
        proba_dict, metric_dict = parser.read_proba(path=folder_path, launch=launch_type, exp_folders=model_list)
        return proba_dict, metric_dict

    def apply_rank_ensemble(proba_dict:dict,
                            metric_dict:dict):
        experiment_results = {}
        for dataset in proba_dict:
            print(f'ENSEMBLE FOR DATASET - {dataset}'.center(50, '-'))
            modelling_results = proba_dict[dataset]
            modelling_metrics = metric_dict[dataset]
            rank_ensemble = RankEnsemble(prediction_proba_dict=modelling_results,
                                         metric_dict=modelling_metrics)

            experiment_results.update({dataset: rank_ensemble.ensemble()})
        return experiment_results

Then we are ready to apply the ensemble to the results of the experiment:

.. code-block:: python

    if __name__ == '__main__':
        exp_folders = ['quantile',
                       'window_spectral',
                       'wavelet',
                       'recurrence',
                       'spectral',
                       'window_quantile']
        path = 'subfolder' # path to the sub-folder in the experiment folder
        launch_type = 'max'

        proba_dict, metric_dict = load_results(folder_path=path,
                                               launch_type=launch_type,
                                               model_list=exp_folders)

        experiment_results = apply_rank_ensemble(proba_dict,
                                                 metric_dict)

        report_df = create_report(experiment_results)

The table below shows an example of how the results of an experiment is going to look like:

+-------------------------+------------------+--------------+-----------------------------+-----------------------+----------------+
| Dataset                 | Base_model       | Base_metric  | Ensemble_models             | Best_ensemble_metric  | Ensemble_gain  |
+=========================+==================+==============+=============================+=======================+================+
| CricketZ                | window_quantile  | 0.688        | Models and Ensemble_method  | 0.729                 | 4.1            |
+-------------------------+------------------+--------------+-----------------------------+-----------------------+----------------+
| Car                     | window_quantile  | 0.885        | Models and Ensemble_method  | 0.933                 | 4.8            |
+-------------------------+------------------+--------------+-----------------------------+-----------------------+----------------+
| LargeKitchenAppliances  | quantile         | 0.803        | Models and Ensemble_method  | 0.816                 | 1.29           |
+-------------------------+------------------+--------------+-----------------------------+-----------------------+----------------+

The column ``Ensemble_models`` contains the list of models and method that were used to create the ensemble.
In terms of documentation space it is not possible to show the whole table. For instance, the columns
could contain the following information:

- Models: ['quantile', 'recurrence', 'window_quantile'].
- Ensemble_method: ProductEnsemble


