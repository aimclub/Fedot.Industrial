from core.api.API import Industrial


if __name__ == '__main__':
    config_name = 'ImageCLF/BasicConfigIMAGECLF.yaml'
    ExperimentHelper = Industrial()
    ExperimentHelper.run_experiment(config_name)
