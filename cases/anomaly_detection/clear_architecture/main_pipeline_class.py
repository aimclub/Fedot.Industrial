"""
Data format:

Data between elements goes in following format: 
{
    data_type: <type>,
    data_body: <main data>,
    data_flags: <{dict of flags}>
}

element of paipline have three methods:

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
    "raw_lables": self.refied_data, 
    "raw_data": self.refined_lables, 
    "raw_columns": self.all_lables
    "lables_for_show": lables for visualisation

All fields in data:
    "data" - main data with all columns
    "lables" - lables for trainings
    "columns" - columns lables

    // Transform data //
    "transformed_data" - data in shape of list of dataset, 
        when dataset is list of ts. Each ts is a list of numbers.

    // We HAVE to choose som ts for work - just fo filtering //
    "lables_of_elected_ts" - lables of columns for choosing ts
    "elected_data" - data with only ts choosed by "lables of needed ts"
    
    // Window cutter //
    "window_step" - step of window
    "window_len" - size of window
    "windows_list" - list of windows. Shape len will be increased by one. Make check!!!

    // Standart analisator of type 1. Get raw data. //

    // Standart analisator of type 2. Get data from previous analysator. //

pipeline
"""

import sys
import os
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)
sys.path.append(os.path.join(import_path, "../../"))

from anomaly_detection.clear_architecture.settings_args \
    import SettingsArgs
from anomaly_detection.clear_architecture.read_data \
    import DataReader
from anomaly_detection.clear_architecture.ts_elector \
    import TsElector
from anomaly_detection.clear_architecture.transform_data \
    import DataTransform
from anomaly_detection.clear_architecture.window_cutting \
    import WindowCut   
from anomaly_detection.clear_architecture.vector_detector \
    import VectorDetector  
from anomaly_detection.clear_architecture.vector_detector_first_and_last \
    import VectorDetectorFaL  
from anomaly_detection.clear_architecture.visualisation_old \
    import DataVisualisator  
from anomaly_detection.clear_architecture.areas_detector \
    import AreasDetector 
from anomaly_detection.clear_architecture.areas_detector_by_zero \
    import AreasDetectorByZero 

class MainPipeline:
    elements_of_pipeline: list = None
    visualisate: bool = False
    print_logs: bool = True
    settings_args: SettingsArgs

    def __init__(self, elements_list: list) -> None:
        self.elements_of_pipeline = elements_list
        self.settings_args = SettingsArgs(
            visualisate = self.visualisate, 
            print_logs = self.print_logs
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
        #print(out_dict["data_body"]["windows_list"])


if __name__ == '__main__':
    path = "/media/nikita/HDD/Data_part_1/data/¥¼íá/CSV/"
    lables_path = "/media/nikita/HDD/anomalies_new_nocount_2.csv"

    reader = DataReader(path, lables_path)
    transformer = DataTransform()
    lables_1 = ["N", "DIST", "Xu", "Yu", "Zu", "Xd", "Yd", "Zd", "Vu", "Vd", "LAT", "LNG", "Depth"]
    lables_2 = ["Vd", "Vu"]
    lables_3 = ["Zd", "Zu"]
    lables_4 = ["Xu", "Xd"]
    lables_5 = ["Vd", "Vu"]
    #lables_5 = ["Vd", "Zu"] # 1500, 100, 0.98, 50
    #lables_5 = ["Xu", "Yd"]
    #lables_3 = ["N"]
    # Xu 
    # "Yu", "Xu" - good 
    elector_1 = TsElector(lables_5)
    elector_2 = TsElector(lables_5)
    elector_3 = TsElector(lables_2)
    elector_4 = TsElector(lables_2)
    cutter_1 = WindowCut(1500, 100)
    cutter_2 = WindowCut(3000, 50)
    cutter_3 = WindowCut(1200, 100)
    cutter_4 = WindowCut(900, 40)

    area_detector = AreasDetector(0.95, 2, False)
    area_detector_1 = AreasDetectorByZero(0.95, 2, False)

    vector_detector_1 = VectorDetector(0.93, 50, False)
    vector_detector_2 = VectorDetector(0.98, 500, False)
    vector_detector_3 = VectorDetector(0.98, 3, False)
    vector_detector_4 = VectorDetector(0.98, 4, False)
    #vector_detector_FaL_1 = VectorDetectorFaL(0.99)
    #vector_detector_FaL_2 = VectorDetectorFaL(0.99)
    visualisator = DataVisualisator()
    pipe = MainPipeline([
        reader, 
        transformer,
        elector_1,
        cutter_1,
        area_detector_1,
        #elector_2,
        #cutter_2,
        #vector_detector_2,
        #elector_3,
        #cutter_3,
        #vector_detector_3,
        #elector_4,
        #cutter_4,
        #vector_detector_4,
        visualisator
    ])
    pipe.run()