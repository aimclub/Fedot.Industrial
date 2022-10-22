from core.api.tsc_API import Industrial


if __name__ == '__main__':
    config_name = 'Config_Classification.yaml'
    ExperimentHelper = Industrial()
    ExperimentHelper.run_experiment(config_name)