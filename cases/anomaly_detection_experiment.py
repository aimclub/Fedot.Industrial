from cases.API import Industrial

if __name__ == '__main__':
    config_name = 'Config_Anomaly.yaml'
    ExperimentHelper = Industrial()
    ExperimentHelper.run_experiment(config_name)
