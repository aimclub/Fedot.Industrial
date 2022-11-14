import sys
import os
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)
sys.path.append(os.path.join(import_path, "../"))

from TSAnomalyDetector.abstract_classes.DataObject import DataObject
from TSAnomalyDetector.utils.settings_args import SettingsArgs

class MainPipeline:
    """
    There is main class of the pipline. 

    classes of predicts 

        0 - false anomaly  
        1 - no-critical anomaly
        2 - heavy anomaly
        3 - critical anomaly

    Returns:
        : _description_
    """
    elements_of_pipeline: list = None
    visualize: bool = False #True
    print_logs: bool = True
    settings_args: SettingsArgs

    def __init__(self) -> None:
        self.settings_args = SettingsArgs(visualize=self.visualize,
                                          print_logs=self.print_logs)
        self.elements_of_pipeline: list = []
        self.init_pipeline: list = []
        self.detector_pipeline: list = []
        self.data_object: DataObject = None

    def init_run(self):
        """
        In case of you need to use several detectors combinations on the same data
        it is much faster to load data(this method) and then make attempts to use
        detectors combintaions(detector_run). Data stored in self.current_data.
        self.current_data is a dict, se be careful with rewrition or clean data. For cleaning there is 
        file clean_pipe.py in utils - it's just return pipline class to state just after read data.

        Example:
            pipe.init_pipeline = \
                [
                    reader
                ]
            pipe.detector_pipeline = \
            [
                transformer_1,
                elector_1,
                dy_det_1,
                dy_det_2,
                dy_det_3,
                metrics,
                visualisator
            ]
        First pipline just read data and save it inide the class.
        Second just works with this data. 

        """
        self.init_pipeline[0].set_settings(self.settings_args)
        self.init_pipeline[0].input_data()
        self.init_pipeline[0].run()
        self.data_object = self.init_pipeline[0].output_data()
        if len(self.init_pipeline) > 1:
            for i in range(1, len(self.init_pipeline)):
                self.init_pipeline[i].set_settings(self.settings_args)
                self.init_pipeline[i].input_data(self.data_object)
                self.init_pipeline[i].run()
                self.data_object = self.init_pipeline[i].output_data()
            
    def detector_run(self):
        """
        Warch previous method's description.

        Raises:
            Exception: if there is too few elements if pipeline

        Returns:
            float: Metric of detection. If no metric - returns 0
        """
        if len(self.detector_pipeline) > 1:
            for i in range(0, len(self.detector_pipeline)):
                self.detector_pipeline[i].set_settings(self.settings_args)
                self.detector_pipeline[i].input_data(self.data_object)
                self.detector_pipeline[i].run()
                self.data_object = self.detector_pipeline[i].output_data()
        else:
            raise Exception('Pipeline has not enough nodes')
        return 0

    def get_data_object(self) -> DataObject:
        return self.data_object