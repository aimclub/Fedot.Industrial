from core.api.API import Industrial


if __name__ == '__main__':
    config_path = 'cases/ts_classification_example/supplementary_data/configs_for_examples/AdvancedConfigCLF.yaml'
    ExperimentHelper = Industrial()
    ExperimentHelper.run_experiment(config_path)
