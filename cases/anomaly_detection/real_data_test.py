# libraries importing
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import datetime
# additional modules
import sys
import os
from sqlalchemy import false
from tqdm import tqdm
from datetime import datetime
from tsad.evaluating.evaluating import evaluating
import csv
sys.path.append('../utils')
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)
sys.path.append(os.path.join(import_path, "../../"))
from SSTdetector import SingularSpectrumTransformation
from itertools import chain, combinations

from bayes_opt import BayesianOptimization

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class DataContainer:
    def __init__(self, path: str, lables_path: str) -> None:
        self.all_lables = ["N", "DIST", "Xu", "Yu", "Zu", "Xd", "Yd", "Zd", "Vu", "Vd", "LAT", "LNG", "Time", "Depth"]
        self.lables = self.read_lables_csv_from_file(lables_path)
        self.refied_data = self.read_data_csv_in_folder(path)
        print(f"Time series count: {len(self.refied_data[0])}, lables count: {len(self.all_lables)}")
        if len(self.refied_data[0]) != len(self.all_lables):
            raise ValueError("Lens of lables and data have to be the same!")
        self.get_time_series_from_data()
        all_lables = ["Vd", "Vu"] #["Xu", "Yu", "Zu", "Xd", "Yd", "Zd", "Vd", "Vu"]

        # Vd Vu
        not_all_lables = ["Vd", "Vu"]
        self.best_score = 0
        self.best_lables = []
        self.best_window = 0
        self.best_step = 0
        self.best_filt_window = 0
        self.best_filt_thresh = 0 
        attempt = 0
        self.get_several_time_series(not_all_lables)
        self.labeling()
        self.good_lbl = np.asarray(self.labeles)

        visualisate = False

        
        if visualisate:
            print(self.processing(visualisate, 180, 24, 70, 3, 15)) #180, 24, 70, 3, 15)
            plt.show()

        x0 = [50, 5, 36, 4]
        optimizer = BayesianOptimization(
            f=self.funk_for_opt,
            pbounds={'window': (110, 200), 
            'step': (20, 40), 
            'f_window': (60, 90), 
            'f_thresh': (3, 5),
            "dist": (5, 15)},
            verbose=2,
            random_state=1,
        )
        optimizer.maximize(alpha=1e-3, n_iter=100)
        print("++++++++++++++++++++++++++++++++++++")
        print("Best score:")
        print(f"Best time series: {self.best_lables}")
        print(f"Best Window len : {self.best_window}")
        print(f"Best Step : {self.best_step}")
        print(f"Best F1 metric : {self.best_score}")
        print(f"Best filtering window len : {self.best_filt_window}")
        print(f"Best filtering threshold : {self.best_filt_thresh}")
        print("++++++++++++++++++++++++++++++++++++")
        with open('/home/nikita/IndustrialTS/cases/anomaly_detection/log.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Lables", "win_len", 
            "Step","metric","filtering window len","filtering threshold"])
            for window in range(40, 300, 10):
                for step in range(5, 100, 2):
                    for filter_window in range(10, 50, 2):
                        for filter_thresh in range(2, 4, 1):
                            #for i, x in enumerate(self.powerset(all_lables)):
                            print(f"Attempt number {attempt}")
                            print(f"Window len : {window}")
                            print(f"Step : {step}")
                            print(f"Filter Window len : {filter_window}")
                            print(f"Filter thresh : {filter_thresh}")
                            attempt += 1
                            print(f"Current time series: {not_all_lables}")
                            print("++++++++++++++++++++++++++++++++++++")
                            print("Best score:")
                            print(f"Best time series: {self.best_lables}")
                            print(f"Best Window len : {self.best_window}")
                            print(f"Best Step : {self.best_step}")
                            print(f"Best metric : {self.best_score}")
                            print(f"Best filtering window len : {self.best_filt_window}")
                            print(f"Best filtering threshold : {self.best_filt_thresh}")
                            print("++++++++++++++++++++++++++++++++++++")

                            if len(not_all_lables) > 1:
                                self.score = 0
                                #for lables in not_all_lables:
                                self.get_several_time_series(not_all_lables)
                                #self.labeling()
                                self.good_lbl = np.asarray(self.labeles)
                                self.processing(False, window, step, filter_window, filter_thresh)
                                print(f"F1 metric : {self.score}")
                                writer.writerow([not_all_lables, window, step, self.score, filter_window, filter_thresh])
                                if self.score > self.best_score: 
                                    self.best_score = self.score
                                    self.best_lables = not_all_lables
                                    self.best_window = window
                                    self.best_step = step
                                    self.best_filt_window = filter_window
                                    self.best_filt_thresh = filter_thresh
                                    print("---------------------------------")
                                    print("Best score found!")
                                    print(f"Current time series: {not_all_lables}")
                                    print(f"Window len : {window}")
                                    print(f"Step : {step}")
                                    print(f"F1 metric : {self.score}")
                                    print("---------------------------------")
        print("++++++++++++++++++++++++++++++++++++")
        print("Best score:")
        print(f"Best time series: {self.best_lables}")
        print(f"Best Window len : {self.best_window}")
        print(f"Best Step : {self.best_step}")
        print(f"Best F1 metric : {self.best_score}")
        print(f"Best filtering window len : {self.best_filt_window}")
        print(f"Best filtering threshold : {self.best_filt_thresh}")
        print("++++++++++++++++++++++++++++++++++++")
        #plt.show()
        #self.window_analisys(350, 30) # 350, 800

    def funk_for_opt(self,  window, step, f_window, f_thresh, dist):
        window = int(window)
        step = int(step) 
        filter_window = int(f_window) 
        filter_thresh = int(f_thresh) 
        dist = int(dist)
        lables = ["Vd", "Vu"]
        if len(lables) > 1:
            self.score = 0
            #for lables in not_all_lables:
            self.get_several_time_series(lables)
            self.good_lbl = np.asarray(self.labeles)
            result = self.processing(False, window, step, 
            filter_window, filter_thresh, dist)
            #print(f"F1 metric : {self.score}")
            if self.score > self.best_score: 
                self.best_score = self.score
                self.best_lables = lables
                self.best_window = window
                self.best_step = step
                self.best_filt_window = filter_window
                self.best_filt_thresh = filter_thresh
            return result
        else: return 0


    def powerset(self, list_name):
        s = list(list_name)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def labeling(self):
        self.labeles = [0] * (len(self.time_series_for_work))
        approx = 0.00028
        for i in range(0, len(self.time_series_for_work)):
            for label in self.lables:
                if label[0] - approx <= self.coordinates[0][i] <= label[0] + approx and \
                    label[1] - approx <= self.coordinates[1][i] <= label[1] + approx:
                    self.labeles[i] = 1

    def read_lables_csv_from_file(self, filename: str) -> list:
        with open(filename, 'r', encoding="iso-8859-1") as file:
            lines = file.readlines()
        list_to_save = []
        for i in tqdm(range(0, len(lines), 1)): #len(lines)-2
            v = bytes(lines[i], encoding="iso-8859-1")
            good_str = v.decode("iso-8859-1").replace('\x00','').replace(' ','')
            arr = good_str.split(";")
            temp_arr = []
            for j in range(len(arr)):
                temp_arr.append(float(arr[j]))
            list_to_save.append(temp_arr)
        return list_to_save

    def read_data_csv_from_file(self, filename: str) -> list:
        with open(filename, 'r', encoding="iso-8859-1") as file:
            lines = file.readlines()
        list_to_save = []
        for i in tqdm(range(500, len(lines), 1)): #len(lines)-2
            v = bytes(lines[i], encoding="iso-8859-1")
            good_str = v.decode("iso-8859-1").replace('\x00','').replace(' ','')
            arr = good_str.split(";")
            temp_arr = []
            for j in range(0, len(self.all_lables)):
                if j == 12:
                    temp_arr.append(datetime.strptime(arr[j], "%H:%M:%S").time())
                elif j == 0:
                    temp_arr.append(int(arr[j]))
                else:   
                    temp_arr.append(float(arr[j]))
            list_to_save.append(temp_arr)
        return list_to_save

    def read_data_csv_in_folder(self, path_to_folder: str):
        files = []
        for file in os.listdir(path_to_folder):
            if file.endswith(".CSV"):
                print(os.path.join(path, file))
                files.append(os.path.join(path, file))
        formatted_data = []
        #files = ["/media/nikita/HDD/Data_part_1/data/¥¼íá/CSV/190918_1310.CSV"]
        for file in files:
            formatted_data.extend(self.read_data_csv_from_file(file))

        return formatted_data
 
    def get_time_series_from_data(self):
        self.time_series = []
        for _ in range(len(self.all_lables)):
            self.time_series.append([])

        for line in self.refied_data:
           for i in range(len(self.all_lables)) :
               self.time_series[i].append(line[i])
        self.df_list = pd.DataFrame(self.refied_data, columns = self.all_lables)
        # random dataset visualizing
        print(self.df_list)
        
        self.df_list = [pd.DataFrame(self.refied_data, columns = self.all_lables)]

    def get_several_time_series(self, columns_names: list) -> None:
        self.time_series_for_work = []
        lat_long = ["LAT", "LNG"]
        self.coordinates = [[], []]
        # this is for new analisys
        self.alterative_time_series_for_work = []
        for _ in columns_names:
            self.alterative_time_series_for_work.append([])
        from datetime import date
        self.time_list = []
        for line in self.refied_data:
            temp_list = []
            for name in columns_names:
                temp_list.append(line[self.all_lables.index(name)])
            for i, name in enumerate(columns_names):
                self.alterative_time_series_for_work[i].append(line[self.all_lables.index(name)])
            for i, name in enumerate(lat_long):
                self.coordinates[i].append(line[self.all_lables.index(name)])
            self.time_list.append(datetime.combine(date(2020, 1, 1),
                line[self.all_lables.index('Time')]))
            self.time_series_for_work.append(temp_list)
        self.new_lables = columns_names
        self.time_list = pd.to_datetime(pd.Series(self.time_list))
        self.idx = pd.date_range("2018-01-01", periods=len(self.time_series_for_work), freq="S")
        #print(self.new_lables)
    
    def plot_data_and_score(self, raw_data, score, true_lables):
        f, ax = plt.subplots(3, 1, figsize=(20, 10))
        ax[0].plot(raw_data)
        ax[0].set_title(f"raw data {self.new_lables }")
        ax[1].plot(score, "r")
        ax[1].set_title("score")
        ax[2].plot(true_lables, "r")
        ax[2].set_title("lables")

    def _window_filter(self, predict, win_len: int, threshold: int):
        for i in range(0, len(predict) - win_len//2):
            counter = 0
            if predict[i] == 1:
                for j in range(i-win_len//2, i + win_len//2):
                    if j>=0:
                        if predict[j] == 1:
                            counter += 1
                if counter < threshold:
                    predict[i] = 0
                    print(f"Anomaly at {i} is a mistake...")
        return predict

    def _win_unite(self, predict, distance: int) -> list:
        def _fill_gap(predict, start, end):
            for i in range(start, end):
                predict[i] = 1
            return predict
        last_point = 0
        i = 0
        while i < len(predict) - distance:
            i += 1
            if predict[i] == 1:
                for j in range(i+1, i + distance):
                    if predict[j] == 1:
                        last_point = j
                        predict = _fill_gap(predict, i, last_point)
                        i = last_point-1
        for i in range(len(predict)-30, len(predict)):
            predict[i] = 0
        return predict

    def processing(self, visualisate: bool, 
    window: int, step: int, filter_window, 
    filter_threshold, unite_distance):
        array = np.asarray(self.alterative_time_series_for_work)
        scorer = SingularSpectrumTransformation(time_series=array,
                                                ts_window_length=window,
                                                lag=50,
                                                trajectory_window_length=step)
        score = scorer.score_offline_2d(dynamic_mode=False)
        from sklearn import preprocessing
        normalized_arr = preprocessing.normalize([score])
        true_lables = []
        index = []
        for i in range(0, len(score)):
            index.append(pd.to_timedelta(i))
            value = 0
            real_value = i * (len(self.labeles)//len(score))
            for n in range(real_value, real_value+window):
                if self.labeles[n] == 1:
                    value = 1
                    break
            true_lables.append(value)
        blockPrint()
        score_1 = self._window_filter(score, filter_window, filter_threshold)
        #score_1 = self._window_filter(score_1, filter_window, filter_threshold)
        score_1 = self._win_unite(score_1, unite_distance)
        predicted_cp = pd.Series(score_1)
        if visualisate:
            self.plot_data_and_score(np.transpose(array), score_1, true_lables)
        idx = pd.date_range("2018-01-01", periods=len(score), freq="S")
        predicted_cp.index = idx 
        true_cp = pd.Series(true_lables)
        true_cp.index = idx 
        #add = evaluating(true_cp, 
        #    predicted_cp, 
        #    metric='nab', 
        #    numenta_time='6 sec',
        #    verbose=True)
        #print(add)
        eval = False
        if eval:
            add = evaluating(true_cp, 
                predicted_cp, 
                metric='nab', 
                numenta_time='10 sec',
                verbose=True,
                plot_figure=False)
            self.score = add['Standart'] #
            print(add)
        enablePrint()
        self.score = f1_score(true_lables, score_1, average='macro')
        return self.score

if __name__ == '__main__':
    path = "/media/nikita/HDD/Data_part_1/data/¥¼íá/CSV/"
    lables_path = "/media/nikita/HDD/Data_part_1/data/¥¼íá/lables1.csv"

    data_container = DataContainer(path, lables_path)