#-*- coding: utf-8 -*-
from core.api.tsc_API import Industrial

if __name__ == '__main__':
    config_name = 'configs_for_examples/BasicConfigEnsemble.yaml'
    ExperimentHelper = Industrial()
    ExperimentHelper.run_experiment(config_name)
