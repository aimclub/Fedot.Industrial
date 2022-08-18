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
    "raw_labels": self.refied_data,
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
    "elected_data" - data with only ts choosed by "labels of needed ts"
    
    // Window cutter //
    "window_step" - step of window
    "window_len" - size of window
    "windows_list" - list of windows. Shape len will be increased by one. Make check!!!

    // Standard analizator of type 1. Get raw data. //

    // Standard analizator of type 2. Get data from previous analizator. //

pipeline
"""

import os

from areas_detector import AreasDetector
from areas_detector_by_zero import AreasDetectorByZero
from core.operation.utils.utils import project_path
from min_max_detector import MinMaxsDetector
from read_data import DataReader
from settings_args import SettingsArgs
from transform_data import DataTransform
from ts_elector import TsElector
from vector_detector import VectorDetector
from visualisation_old import DataVisualisator
from window_cutting import WindowCut


#
# import_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(import_path)
# sys.path.append(os.path.join(import_path, "../../"))
#
# from anomaly_detection.clear_architecture.settings_args \
#     import SettingsArgs
# from anomaly_detection.clear_architecture.read_data \
#     import DataReader
# from anomaly_detection.clear_architecture.ts_elector \
#     import TsElector
# from anomaly_detection.clear_architecture.transform_data \
#     import DataTransform
# from anomaly_detection.clear_architecture.window_cutting \
#     import WindowCut
# from anomaly_detection.clear_architecture.vector_detector \
#     import VectorDetector
# from anomaly_detection.clear_architecture.vector_detector_first_and_last \
#     import VectorDetectorFaL
# from anomaly_detection.clear_architecture.visualisation_old \
#     import DataVisualisator
# from anomaly_detection.clear_architecture.areas_detector \
#     import AreasDetector
# from anomaly_detection.clear_architecture.areas_detector_by_zero \
#     import AreasDetectorByZero
# from anomaly_detection.clear_architecture.min_max_detector \
#     import MinMaxsDetector


class MainPipeline:
    elements_of_pipeline: list = None
    visualize: bool = True
    print_logs: bool = True
    settings_args: SettingsArgs

    def __init__(self, elements_list: list) -> None:
        self.elements_of_pipeline = elements_list
        self.settings_args = SettingsArgs(
            visualize=self.visualize,
            print_logs=self.print_logs
        )

    def run(self):
        self.elements_of_pipeline[0].set_settings(self.settings_args)
        self.elements_of_pipeline[0].input_data()
        self.elements_of_pipeline[0].run()
        out_dict = self.elements_of_pipeline[0].output_data()
        if len(self.elements_of_pipeline) > 1:
            for i in range(1, len(self.elements_of_pipeline)):
                self.elements_of_pipeline[i].set_settings(self.settings_args)
                self.elements_of_pipeline[i].input_data(out_dict)
                self.elements_of_pipeline[i].run()
                out_dict = self.elements_of_pipeline[i].output_data()
        # print(out_dict["data_body"]["windows_list"])


if __name__ == '__main__':
    proj_path = project_path()

    path = os.path.join(proj_path, "data/anomaly_detection/data/data/CSV2")
    labels_path = os.path.join(proj_path, "data/anomaly_detection/anomalies_new_nocount_2.csv")

    # path = "/media/nikita/HDD/Data_part_1/data/¥¼íá/CSV/"
    # labels_path = "/media/nikita/HDD/anomalies_new_nocount_2.csv"

    reader = DataReader(path, labels_path)
    transformer = DataTransform()
    labels_1 = ["N", "DIST", "Xu", "Yu", "Zu", "Xd", "Yd", "Zd", "Vu", "Vd", "LAT", "LNG", "Depth"]
    labels_2 = ["Vd", "Vu"]
    labels_3 = ["Zd", "Zu"]
    labels_4 = ["Xu", "Xd"]
    labels_5 = ["Zd", "Vu"]
    # lables_5 = ["Vd", "Zu"] # 1500, 100, 0.98, 50
    # lables_5 = ["Xu", "Yd"]
    # lables_3 = ["N"]
    # Xu 
    # "Yu", "Xu" - good 
    elector_1 = TsElector(labels_5)
    elector_2 = TsElector(labels_5)
    elector_3 = TsElector(labels_2)
    elector_4 = TsElector(labels_2)
    cutter_1 = WindowCut(100, 10)
    cutter_2 = WindowCut(3000, 50)
    cutter_3 = WindowCut(1200, 100)
    cutter_4 = WindowCut(900, 40)

    area_detector = AreasDetector(0.95, 3, False)
    area_detector_1 = AreasDetectorByZero(0.92, 2, False)

    min_max_detector = MinMaxsDetector(0.90, 2, False)

    vector_detector_1 = VectorDetector(0.95, 50, False)
    vector_detector_2 = VectorDetector(0.98, 500, False)
    vector_detector_3 = VectorDetector(0.98, 3, False)
    vector_detector_4 = VectorDetector(0.98, 4, False)
    # vector_detector_FaL_1 = VectorDetectorFaL(0.99)
    # vector_detector_FaL_2 = VectorDetectorFaL(0.99)
    visualisator = DataVisualisator()
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
        visualisator
    ])
    pipe.run()
