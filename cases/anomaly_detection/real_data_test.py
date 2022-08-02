# libraries importing
import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import datetime
# additional modules
import sys
import os
sys.path.append('../utils')
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)
sys.path.append(os.path.join(import_path, "../../"))
from sqlalchemy import false
from tqdm import tqdm
from datetime import datetime
from tsad.evaluating.evaluating import evaluating
import csv

from SSTdetector import SingularSpectrumTransformation
from itertools import chain, combinations

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from data.lable_reader import read_lables_csv_from_file
from sklearn.ensemble import IsolationForest
# block output

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
import os

class DataContainer:
    def __init__(self, path: str, lables_path: str) -> None:
        self.all_lables = ["N", "DIST", "Xu", "Yu", "Zu", "Xd", "Yd", "Zd", "Vu", "Vd", "LAT", "LNG", "Time", "Depth"]
        self.lables = read_lables_csv_from_file(lables_path)
        self.count_from_the_start = 3
        self.refied_data, self.refined_lables = self.read_data_csv_in_folder(path)
        print(f"Time series count: {len(self.refied_data[0])}, lables count: {len(self.all_lables)}")
        if len(self.refied_data[0]) != len(self.refined_lables[0]):
            raise ValueError("Lens of lables and data have to be the same!")
        self.get_time_series_from_data()
        all_lables = ["Xu", "Yu", "Zu", "Xd", "Yd", "Zd", "Vd", "Vu"]
        
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
        #self.data_test_visualisation(self.horizontal_time_series_for_work, self.refined_lables)
        self.good_lbl = np.asarray(self.refined_lables)
        
        visualisate = True
        self.filtering = True
        self.array_of_arrays_of_ts = []
        for L in range(0, len(all_lables)+1):
            for subset in itertools.combinations(all_lables, L):
                if len(subset) <= 2:
                    self.array_of_arrays_of_ts.append(subset)

        print(len(self.array_of_arrays_of_ts))
        self.array_of_arrays_of_ts = [["Vd", "Vu"]]
        if visualisate:
            print(self.processing(visualisate, 
                100, 
                15, 
                60, 
                3, 
                10
            )) #180, 24, 70, 3, 15)
            plt.show()

        x0 = [50, 5, 36, 4]
        optimizer = BayesianOptimization(
            f=self.funk_for_opt,
            pbounds={'window': (20, 280), 
            'step': (5, 80), 
            'f_window': (15, 140), 
            'f_thresh': (3, 6),
            "dist": (2, 20)},
            #"combination_number":(0, len(self.array_of_arrays_of_ts)-1)},
            verbose=2,
            random_state=1,
        )
        print("Start optimizing...")
        logger = JSONLogger(path="/home/nikita/IndustrialTS/cases/anomaly_detection/logs.json")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.maximize(alpha=1e-3, n_iter=1000)
        #os.system("sudo shutdown now -h")
    

    def funk_for_opt(self,  window, step, f_window, f_thresh, dist):
        try:
            window = int(window)
            step = int(step) 
            filter_window = int(f_window) 
            filter_thresh = int(f_thresh) 
            dist = int(dist)
            lables = self.array_of_arrays_of_ts[0]
            if len(lables) > 1:
                self.score = 0
                #for lables in not_all_lables:
                self.get_several_time_series(lables)
                #self.good_lbl = np.asarray(self.refined_lables)
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
        except:
            return 0

    def data_test_visualisation(self, raw_data, true_lables):
        f, ax = plt.subplots(len(raw_data) * 2, 1, figsize=(20, 10))
        counter = 0
        for i in range(0, len(raw_data)):
            ax[counter].plot(raw_data[i])
            #ax[counter].set_title(f"raw data {self.new_lables }")
            counter += 1
            ax[counter].plot(true_lables[i], "r")
            #ax[counter].set_title("lables")
            counter += 1
        plt.show()
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
        for i in tqdm(range(self.count_from_the_start, len(lines), 1)): #len(lines)-2
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
                files.append(os.path.join(path, file))
        formatted_data = []
        fromatted_lables = []
        #files = ["/media/nikita/HDD/Data_part_1/data/¥¼íá/CSV/190918_1310.CSV"]
        for file in files:
            data = self.read_data_csv_from_file(file)
            #get file's lables
            lable = []
            filename = os.path.splitext(os.path.basename(file))[0]
            for i in range(len(self.lables)):
                if self.lables[i][0] == filename: 
                    lable = self.lables[i][1]
            temp_lable_arr = [0] * len(data)
            approx_count = 30
            for i, line in enumerate(data):
                for lable_line in lable:
                    if int(lable_line[0]) - approx_count <= line[0]+self.count_from_the_start <= int(lable_line[1])+approx_count:
                        temp_lable_arr[i] = 1
            formatted_data.append(data)
            fromatted_lables.append(temp_lable_arr)

        return formatted_data, fromatted_lables
 
    def get_time_series_from_data(self):
        self.time_series = []
        #for _ in range(len(self.all_lables)):
        #    self.time_series.append([])



        for j, data in enumerate(self.refied_data):
            self.time_series.append([])
            for _ in range(len(self.all_lables)):
                self.time_series[j].append([])
            for line in data:
                for i in range(len(self.all_lables)) :
                    self.time_series[j][i].append(line[i])
        #self.df_list = []
        #for item in self.refied_data:
        #    self.df_list.append(pd.DataFrame(item, columns = self.all_lables))
        # random dataset visualizing
        #print(self.df_list)
        
        #self.df_list = [pd.DataFrame(self.refied_data, columns = self.all_lables)]

    def get_several_time_series(self, columns_names: list) -> None:
        self.time_series_for_work = []
        self.vertical_time_series_for_work = []
        self.horizontal_time_series_for_work = []
        lat_long = ["LAT", "LNG"]
        self.coordinates = [[], []]
        # this is for new analisys
        self.alterative_time_series_for_work = []

        from datetime import date
        self.temp_time_list = []
        for j, data in enumerate(self.refied_data):
            self.vertical_time_series_for_work.append([])
            self.horizontal_time_series_for_work.append([]) 
            self.temp_time_list.append([])
            for _ in columns_names:
                #self.alterative_time_series_for_work.append([])
                self.vertical_time_series_for_work[j].append([])
                #self.horizontal_time_series_for_work[j].append([]) 
            for line in data:
                temp_list = []
                for name in columns_names:
                    temp_list.append(line[self.all_lables.index(name)])
                for i, name in enumerate(columns_names):
                    self.vertical_time_series_for_work[j][i].append(line[self.all_lables.index(name)])
                for i, name in enumerate(lat_long):
                    self.coordinates[i].append(line[self.all_lables.index(name)])
                self.temp_time_list[j].append(datetime.combine(date(2020, 1, 1),
                    line[self.all_lables.index('Time')]))
                self.horizontal_time_series_for_work[j].append(temp_list)
        self.new_lables = columns_names
        self.time_list = []
        self.idx = []
        for i in range(len(self.temp_time_list)):
            self.time_list.append(pd.to_datetime(pd.Series(self.temp_time_list[i])))
            self.idx.append(pd.date_range("2018-01-01", periods=len(self.horizontal_time_series_for_work[i]), freq="S"))
        #print(self.new_lables)
    
    def plot_data_and_score(self, raw_data, score, true_lables):
        f, ax = plt.subplots(2, 1, figsize=(20, 10))
        temp_list = [score, true_lables]
        #out_list = np.asarray([[row[i] for row in temp_list] for i in range(len(temp_list[0]))])        
        ax[0].plot(raw_data)
        ax[0].set_title(f"raw data {self.new_lables }")
        ax[1].plot(score, "b")
        ax[1].plot(true_lables, "r")
        ax[1].set_title("score")
        #ax[2].plot(true_lables, "r")
        #ax[2].set_title("lables")

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
        predicts_list = []
        true_lables_list = []
        predicts = []
        for number, ts in enumerate(self.vertical_time_series_for_work):
            array = np.asarray(ts)
            scorer = SingularSpectrumTransformation(time_series=array,
                                                    ts_window_length=window,
                                                    lag=50,
                                                    trajectory_window_length=step,
                                                    is_scaled=False)
            score = scorer.score_offline_2d_average(dynamic_mode=True)
            rng = np.random.RandomState(42)
            clf = IsolationForest(n_estimators=100, max_samples='auto', 
                        contamination=float(.12), 
                        max_features=1.0, bootstrap=False, 
                        n_jobs=-1, random_state=42, verbose=0)
            metrics_df=pd.DataFrame(array.T)
            #clf.fit(metrics_df)
            #score = clf.predict(metrics_df)
            from sklearn import preprocessing
            normalized_arr = preprocessing.normalize([score])
            true_lables = []
            index = []
            for i in range(0, len(score)):
                index.append(pd.to_timedelta(i))
                value = 0
                real_value = i * (len(self.refined_lables[number])//len(score))
                for n in range(real_value, real_value+window):
                    if n < len(self.refined_lables[number]):
                        if self.refined_lables[number][n] == 1:
                            value = 1
                            break
                true_lables.append(value)
            blockPrint()
            #for i in range(0, len(score)):
            #    if score[i] < 0: score[i] = 0
            if self.filtering:
                score_1 = self._window_filter(score, filter_window, filter_threshold)
                #score_1 = self._window_filter(score_1, filter_window, filter_threshold)
                score_1 = self._win_unite(score_1, unite_distance)
            else:
                score_1 = score
            predicted_cp = pd.Series(score_1)
            predicts_list.append(score_1)
            true_lables_list.append(true_lables)
            if False:
                self.plot_data_and_score(np.transpose(array), score_1, true_lables)
            idx = pd.date_range("2018-01-01", periods=len(score), freq="S")
            predicted_cp.index = idx 
            true_cp = pd.Series(true_lables)
            true_cp.index = idx 
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
            print(self.score)
            predicts.append(self.score)
        print("-------------------------------------")
        self.score = sum(predicts) / len(predicts)
        print("Average predict:")
        print(sum(predicts) / len(predicts))
        print("-------------------------------------")
        if True:
            self.data_predicts_visualisation(
                self.horizontal_time_series_for_work, 
                true_lables_list, 
                predicts_list)
        return self.score


    def data_predicts_visualisation(self, raw_data, true_lables, predicts):
        f, ax = plt.subplots(len(raw_data) * 2, 1, figsize=(20, 10))
        counter = 0
        for i in range(0, len(raw_data)):
            ax[counter].plot(raw_data[i])
            #ax[counter].set_title(f"raw data {self.new_lables }")
            counter += 1
            ax[counter].plot(true_lables[i], "r")
            ax[counter].plot(predicts[i], "b")
            #ax[counter].set_title("lables")
            counter += 1
        plt.show()
    
if __name__ == '__main__':
    path = "/media/nikita/HDD/Data_part_1/data/¥¼íá/CSV2/"
    lables_path = "/media/nikita/HDD/anomalies_new_nocount_2.csv"

    data_container = DataContainer(path, lables_path)