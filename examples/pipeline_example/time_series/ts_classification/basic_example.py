import matplotlib
import pandas as pd
from fedot import Fedot
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.verification import common_rules
from sklearn.metrics import accuracy_score

from examples.example_utils import evaluate_metric
from examples.example_utils import init_input_data
from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from fedot_ind.tools.loader import DataLoader
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from tsml_eval._wip.results.results_by_classifier import *

matplotlib.use('TkAgg')
model_dict = {'basic_quantile': PipelineBuilder().add_node('quantile_extractor',
                                                           params={'window_size': 10,
                                                                   'stride': 5}).add_node('rf'),
              'basic_topological': PipelineBuilder().add_node('topological_extractor',
                                                              params={'window_size': 10,
                                                                      'stride': 5}).add_node('rf'),
              'basic_recurrence': PipelineBuilder().add_node('recurrence_extractor',
                                                             params={'window_size': 10,
                                                                     'stride': 5}).add_node('rf'),
              'advanced_quantile': PipelineBuilder().add_node('fourier_basis').add_node('quantile_extractor',
                                                                                        params={
                                                                                            'window_size': 10}).add_node(
                  'rf'),
              'advanced_topological': PipelineBuilder().add_node('eigen_basis').add_node('topological_extractor',
                                                                                         params={'stride': 20,
                                                                                                 'window_size': 10}).add_node(
                  'rf'),
              'advanced_reccurence': PipelineBuilder().add_node('wavelet_basis').add_node(
                  'recurrence_extractor',
                  params={'window_size': 0}).add_node(
                  'rf')
              }
metric_dict = {}
ts_clf_operations = [
    'eigen_basis',
    'dimension_reduction',
    'inception_model',
    'logit',
    'minirocket_extractor',
    'normalization',
    'omniscale_model',
    'pca',
    'mlp',
    'quantile_extractor',
    'resample',
    'scaling',
    'signal_extractor',
    'topological_features'
]

# 14 Regression equal length no missing problems [1]
monash_regression = [
    "AppliancesEnergy",
    "AustraliaRainfall",
    "BIDMC32HR",
    "BIDMC32RR",
    "BIDMC32SpO2",
    "Covid3Month",
    "FloodModeling1",
    "FloodModeling2",
    "FloodModeling3",
    "IEEEPPG",
    "LiveFuelMoistureContent",
    "NewsHeadlineSentiment",
    "NewsTitleSentiment",
    "PPGDalia",
]

# 19 Regression problems from [1] with interpolated missing and truncated unequal
monash_regression_nm_eq = [
    "AppliancesEnergy",
    "AustraliaRainfall",
    "BeijingPM10Quality-no-missing",
    "BeijingPM25Quality-no-missing",
    "BenzeneConcentration-no-missing",
    "BIDMC32HR",
    "BIDMC32RR",
    "BIDMC32SpO2",
    "Covid3Month",
    "FloodModeling1",
    "FloodModeling2",
    "FloodModeling3",
    "HouseholdPowerConsumption1-no-missing",
    "HouseholdPowerConsumption2-no-missing",
    "IEEEPPG",
    "LiveFuelMoistureContent",
    "NewsHeadlineSentiment",
    "NewsTitleSentiment",
    "PPGDalia-equal-length",
]

if __name__ == "__main__":
    OperationTypesRepository = IndustrialModels().setup_repository()
    results = get_averaged_results_from_web(datasets=multivariate_equal_length, classifiers=valid_multi_classifiers)
    results = pd.DataFrame(results)
    results.columns = valid_multi_classifiers
    results.index = multivariate_equal_length
    results['Fedot_Ind'] = 0
    multivariate_equal_length = [
        #  'DuckDuckGeese',
        # 'MotorImagery',
        # 'Heartbeat',
        # 'Handwriting',
        # 'EigenWorms',
        # 'Epilepsy',
        # 'EthanolConcentration',
        # 'FaceDetection',
        'RacketSports',
        'LSST',
        'SelfRegulationSCP1',
        'SelfRegulationSCP2',
        'StandWalkJump',
    ]
    # error_model = PipelineBuilder().add_node('resample').add_node('resample', branch_idx=1) \
    #     .add_node('minirocket_extractor', branch_idx=1).add_node('quantile_extractor', branch_idx=1).join_branches(
    #     'logit').build()
    #error_model = PipelineBuilder().add_node('logit').add_node('logit').build()
    # error_model = PipelineBuilder().add_node('pca').add_node('resample', branch_idx=1).add_node('quantile_extractor', branch_idx=1).join_branches(
    #     'logit').build()
    #error_model = PipelineBuilder().add_node('pca').add_node('logit').build()
    for dataset in multivariate_equal_length:
        train_data, test_data = DataLoader(dataset_name=dataset).load_data()
        input_data = init_input_data(train_data[0], train_data[1])
        val_data = init_input_data(test_data[0], test_data[1])
        model = Fedot(problem='classification',
                      logging_level=10,
                      n_jobs=2,
                      metric='accuracy',
                      pop_size=20,
                      num_of_generations=20,
                      optimizer=IndustrialEvoOptimizer,
                      available_operations=ts_clf_operations,
                      timeout=30,
                      with_tuning=False
                      )
        #model = error_model
        model.fit(input_data)
        features = model.predict(val_data)
        metric = evaluate_metric(target=val_data.target, prediction=features)
        try:
            acc = accuracy_score(y_true=val_data.target, y_pred=features.predict)
        except Exception:
            acc = accuracy_score(y_true=val_data.target, y_pred=np.argmax(features, axis=1))
        metric_dict.update({model: metric})
        model.history.save(f"{dataset}classification_history.json")
        model.current_pipeline.show(save_path=f'./{dataset}_best_model.png')
        # model.history.show.fitness_box(best_fraction=0.5, dpi=100)
        # model.history.show.operations_kde(dpi=100)
        model.history.show.operations_animated_bar(save_path=f'./{dataset}_history_animated_bars.gif',
                                                   show_fitness=True, dpi=100)

        results.loc[dataset, 'Fedot_Ind'] = acc
        results.to_csv('./multi_ts_clf_run4.csv')
    _ = 1
