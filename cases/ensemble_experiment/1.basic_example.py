from core.api.API import Industrial

if __name__ == '__main__':
    config_name = 'configs_for_examples/supplementary_data/BasicConfigEnsemble.yaml'
    ExperimentHelper = Industrial()
    ExperimentHelper.run_experiment(config_name)
