from core.api.tsc_API import Industrial


if __name__ == '__main__':
    config_name = 'cases/ts_classification_example/configs_for_examples/BasicConfigCLF.yaml'
    ExperimentHelper = Industrial()
    ExperimentHelper.run_experiment(config_name)
