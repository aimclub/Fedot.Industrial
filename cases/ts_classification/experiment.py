from launch.run.experiment import Experimenter
from experiments.experiment_setup import ExperimentHelper
import sys
import matplotlib
matplotlib.use('Agg')
if __name__ == '__main__':
    config_name = 'ConfigTPOT.yaml'
    if len(sys.argv) < 2:
        print('Config name was not specified. ConfigAll was choosen')
    else:
        config_name = sys.argv[1]
    ExperimentHelper = ExperimentHelper()
    yaml_config = ExperimentHelper.read_yaml_config(config_name)
    dataset_info = ExperimentHelper.create_data_paths(dataset_names=yaml_config['dataset_list'])
    exp = Experimenter(working_dir=None,
                       datasets_info=dataset_info,
                       helper=ExperimentHelper,
                       additional_data=yaml_config)
    exp.tabular_experiment(**yaml_config)
