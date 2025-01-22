import numpy as np

from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, \
    DEFAULT_AUTOML_LEARNING_CONFIG

INDUSTRIAL_PARAMS = {'data_type': 'tensor',
                     'learning_strategy': 'ts2tabular'
                     }

# DEFINE ALL CONFIG FOR API
AUTOML_LEARNING_STRATEGY = DEFAULT_AUTOML_LEARNING_CONFIG
COMPUTE_CONFIG = DEFAULT_COMPUTE_CONFIG
AUTOML_CONFIG = {'task': 'classification'}
LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                   'optimisation_loss': {'quality_loss': 'f1'}}
INDUSTRIAL_CONFIG = {'problem': 'classification',
                     'strategy': 'tabular',
                     'strategy_params': INDUSTRIAL_PARAMS
                     }
API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
              'automl_config': AUTOML_CONFIG,
              'learning_config': LEARNING_CONFIG,
              'compute_config': COMPUTE_CONFIG}

if __name__ == "__main__":
    dataset_name = 'Libras'
    finetune = False
    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=0.1,
                      n_jobs=2,
                      logging_level=20)
    api_client = ApiTemplate(api_config=API_CONFIG,
                             metric_list=('f1', 'accuracy'))
    result_dict = api_client.eval(dataset=dataset_name, finetune=finetune)
    uncalibrated_labels, uncalibrated_probs = result_dict['industrial_model'].predicted_labels, \
        result_dict['industrial_model'].predicted_probs
    calibrated_probs = result_dict['industrial_model'].predict_proba(predict_data=api_client.test_data,
                                                                     calibrate_probs=True)
    calibrated_labels = np.argmax(calibrated_probs, axis=1) + np.min(uncalibrated_labels)
