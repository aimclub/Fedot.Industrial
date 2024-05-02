from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.tools.example_utils import industrial_common_modelling_loop

if __name__ == "__main__":
    return_history = True
    opt_hist = PROJECT_PATH + '/examples/data/forecasting/D1679_opt_history/'
    dataset_name = 'Lightning7'
    finetune = False
    metric_names = ('f1', 'accuracy', 'precision', 'roc_auc')
    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=5,
                      pop_size=10,
                      with_tuning=False,
                      n_jobs=2,
                      logging_level=10)

    industrial, labels, metrics = industrial_common_modelling_loop(api_config=api_config,
                                                                   dataset_name=dataset_name,
                                                                   finetune=finetune)
    if return_history:
        opt_hist = industrial.save_optimization_history(return_history=True)
    else:
        # tutorial sample of opt history
        opt_hist = PROJECT_PATH + '/examples/data/forecasting/D1679_opt_history/'
    opt_hist = industrial.vis_optimisation_history(
        opt_history_path=opt_hist, return_history=True)
