"""
Data format:

Data between elements goes in following format: 
{
    data_type: <type>,
    data_body: <main data>,
    data_flags: <{dict of flags}>
}

element of pipeline have three methods:

    init
    input_data
    run
    output_data
    set_settings

fields:

    element_name
    element_type??


Data_reader 

Out format:
    "raw_labels": self.refined_data,
    "raw_data": self.refined_labels,
    "raw_columns": self.all_labels
    "labels_for_show": labels for visualisation

All fields in data:
    "data" - main data with all columns
    "labels" - labels for trainings
    "columns" - columns labels

    // Transform data //
    "transformed_data" - data in shape of list of dataset, 
        when dataset is list of ts. Each ts is a list of numbers.

    // We HAVE to choose som ts for work - just fo filtering //
    "labels_of_elected_ts" - labels of columns for choosing ts
    "elected_data" - data with only ts has been chosen by "labels of needed ts"
    
    // Window cutter //
    "window_step" - step of window
    "window_len" - size of window
    "windows_list" - list of windows. Shape len will be increased by one. Make check!!!

    // Standard analizator of type 1. Get raw data. //

    // Standard analizator of type 2. Get data from previous analizator. //

pipeline
"""

import os

from cases.anomaly_detection.clear_architecture.detectors.areas_detector import AreasDetector
from cases.anomaly_detection.clear_architecture.detectors.areas_detector_by_zero import AreasDetectorByZero
from cases.anomaly_detection.clear_architecture.detectors.min_max_detector import MinMaxDetector
from cases.anomaly_detection.clear_architecture.detectors.vector_detector import VectorDetector
from cases.anomaly_detection.clear_architecture.operations.read_data import DataReader
from cases.anomaly_detection.clear_architecture.operations.transform_data import DataTransformer
from cases.anomaly_detection.clear_architecture.detectors.ts_elector import TsElector
from cases.anomaly_detection.clear_architecture.operations.visualisation_old import DataVisualizer
from cases.anomaly_detection.clear_architecture.detectors.window_cutting import WindowCutter
from cases.anomaly_detection.clear_architecture.utils.settings_args import SettingsArgs
from core.operation.utils.utils import PROJECT_PATH


class MainPipeline:
    elements_of_pipeline: list = None
    visualize: bool = True
    print_logs: bool = True
    settings_args: SettingsArgs

    def __init__(self, elements_list: list) -> None:
        self.elements_of_pipeline = elements_list
        self.settings_args = SettingsArgs(visualize=self.visualize,
                                          print_logs=self.print_logs)

    def run(self):
        self.elements_of_pipeline[0].set_settings(self.settings_args)
        self.elements_of_pipeline[0].input_data()
        self.elements_of_pipeline[0].run()
        out_dict = self.elements_of_pipeline[0].output_data()
        if len(self.elements_of_pipeline) > 1:
            for pipeline_node in self.elements_of_pipeline:
                pipeline_node.set_settings(self.settings_args)
                pipeline_node.input_data(out_dict)
                pipeline_node.run()
                out_dict = pipeline_node.output_data()
        else:
            raise Exception('Pipeline has not enough nodes')


if __name__ == '__main__':

    path = os.path.join(PROJECT_PATH, "data/anomaly_detection/monitoring/CSV2")
    labels_path = os.path.join(PROJECT_PATH, "data/anomaly_detection/anomalies_new_nocount_2.csv")

    reader = DataReader(path, labels_path)
    transformer = DataTransformer()
    labels_1 = ["N", "DIST", "Xu", "Yu", "Zu", "Xd", "Yd", "Zd", "Vu", "Vd", "LAT", "LNG", "Depth"]
    labels_2 = ["Vd", "Vu"]
    labels_3 = ["Zd", "Zu"]
    labels_4 = ["Xu", "Xd"]
    labels_5 = ["Zd", "Vu"]
    # labels_5 = ["Vd", "Zu"] # 1500, 100, 0.98, 50
    # labels_5 = ["Xu", "Yd"]
    # labels_3 = ["N"]
    # Xu 
    # "Yu", "Xu" - good 
    elector_1 = TsElector(labels_5)
    elector_2 = TsElector(labels_5)
    elector_3 = TsElector(labels_2)
    elector_4 = TsElector(labels_2)
    cutter_1 = WindowCutter(100, 10)
    cutter_2 = WindowCutter(3000, 50)
    cutter_3 = WindowCutter(1200, 100)
    cutter_4 = WindowCutter(900, 40)

    area_detector = AreasDetector(0.95, 3, False)
    area_detector_1 = AreasDetectorByZero(0.92, 2, False)

    min_max_detector = MinMaxDetector(0.90, 2, False)

    vector_detector_1 = VectorDetector(0.95, 50, False)
    vector_detector_2 = VectorDetector(0.98, 500, False)
    vector_detector_3 = VectorDetector(0.98, 3, False)
    vector_detector_4 = VectorDetector(0.98, 4, False)
    # vector_detector_FaL_1 = VectorDetectorFaL(0.99)
    # vector_detector_FaL_2 = VectorDetectorFaL(0.99)
    visualizer = DataVisualizer()
    pipe = MainPipeline([
        reader,
        transformer,
        elector_1,
        cutter_1,
        min_max_detector,
        # elector_2,
        # cutter_2,
        # area_detector,
        # elector_3,
        # cutter_3,
        # vector_detector_3,
        # elector_4,
        # cutter_4,
        # vector_detector_4,
        visualizer
    ])
    pipe.run()
