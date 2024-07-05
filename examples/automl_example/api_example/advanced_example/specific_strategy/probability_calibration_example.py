import numpy as np

from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate

if __name__ == "__main__":
    dataset_name = 'Libras'
    finetune = False
    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=0.1,
                      n_jobs=2,
                      logging_level=20)
    api_client = ApiTemplate(api_config=api_config,
                             metric_list=('f1', 'accuracy'))
    result_dict = api_client.eval(dataset=dataset_name, finetune=finetune)
    uncalibrated_labels, uncalibrated_probs = result_dict['industrial_model'].predicted_labels, \
        result_dict['industrial_model'].predicted_probs
    calibrated_probs = result_dict['industrial_model'].predict_proba(predict_data=api_client.test_data,
                                                                     calibrate_probs=True)
    calibrated_labels = np.argmax(calibrated_probs, axis=1) + np.min(uncalibrated_labels)
    _ = 1
