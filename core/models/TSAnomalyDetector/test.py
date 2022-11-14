import sys
import os
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)
sys.path.append(os.path.join(import_path, "../"))
from TSAnomalyDetector.TSAnomalyDetectorRunner import TSAnomalyDetectorRunner
main_labels_path = "/media/nikita/HDD/Data_part_1/Good_lables.csv"
main_data_path = "/media/nikita/HDD/Data_part_1/all_csv_2"

ts_det = TSAnomalyDetectorRunner(main_data_path, main_labels_path)
ts_det.get_features()