from core.api.API import Industrial

if __name__ == '__main__':
    config_name = 'cases/config/configs_for_examples/BasicConfigEnsemble.yaml'
    ExperimentHelper = Industrial()
    ExperimentHelper.run_experiment(config_name, direct_path=False)
