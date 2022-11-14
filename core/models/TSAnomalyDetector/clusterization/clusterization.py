from TSAnomalyDetector.utils.settings_args \
    import SettingsArgs
from TSAnomalyDetector.utils.get_time \
    import get_current_time
from scipy import spatial
from sklearn.metrics import f1_score
from tslearn.clustering import KShape, TimeSeriesKMeans
from sklearn.cluster import DBSCAN, Birch, OPTICS, SpectralClustering, MeanShift, \
        MiniBatchKMeans, FeatureAgglomeration
from sklearn import mixture
import hdbscan
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import plotly.express as px
import os
from TSAnomalyDetector.abstract_classes.AbstractDataOperation import AbstractDataOperation
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from TSAnomalyDetector.abstract_classes.DataObject import DataObject
from scipy import spatial

from TSAnomalyDetector.utils.get_features import generate_features_from_one_ts
import umap
import pickle
import csv
from typing import List
from tqdm import tqdm
import plotly.graph_objects as go

class Clusterization(AbstractDataOperation):
    args: SettingsArgs
    raw_data: list
    transformed_data: list = []
    def __init__(self, slider_1: float, slider_2: float) -> None:
        self.clasters_number = 10
        self.clusterisators = {
            "TimeSeriesKMeans": TimeSeriesKMeans(
                n_clusters=self.clasters_number, 
                tol=slider_1,
                max_iter=20, 
                n_init=5, 
                metric="euclidean", 
                verbose=1),
            "KShape": KShape(
                n_clusters=self.clasters_number, 
                max_iter=100, 
                n_init=100, 
                verbose=1), # good
            "DBSCAN": DBSCAN(
                eps=slider_1, p=slider_2,
                min_samples=self.clasters_number),
            "OPTICS": OPTICS(
                min_samples=self.clasters_number,
                xi=slider_1),
            "GaussianMixture": mixture.GaussianMixture(
                n_components=self.clasters_number, 
                tol=slider_1,
                covariance_type="full"),
            "SpectralClustering": SpectralClustering(
                n_clusters=self.clasters_number),
            "MeanShift": MeanShift(
                bandwidth=slider_1, 
                min_bin_freq=slider_2),
            "MiniBatchKMeans": MiniBatchKMeans(
                n_clusters=self.clasters_number, 
                batch_size = 8), # ++
            "HDBSCAN": hdbscan.HDBSCAN(
                algorithm="best", alpha=slider_1, 
                cluster_selection_epsilon=slider_2,
                approx_min_span_tree=True,
                gen_min_span_tree=False, 
                leaf_size=40,
                metric='euclidean', 
                min_cluster_size=5, 
                min_samples=6, p=None, 
                allow_single_cluster=False,
                prediction_data=True), # ++
            #"AffinityPropagation": AffinityPropagation(), # broked!!
            "Birch": Birch(
                n_clusters=self.clasters_number, threshold=slider_1), #middle
            "FeaturdeAgglomeration": FeatureAgglomeration(
                n_clusters=self.clasters_number)
        }
        self.name = "DBSCAN"
        self.clusterisator = self.clusterisators[self.name]
        
    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{get_current_time()} Clusterization: settings was set.")
        self._print_logs(f"{get_current_time()} Clusterization: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Clusterization: Print logs = {self.args.print_logs}")

    def input_data(self, data_object: DataObject) -> None:
        self._print_logs(f"{get_current_time()} Clusterization: Data read!")
        self.data_object = data_object

    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Clusterization: Loading metrics...")
        self._clusterization()
        self._print_logs(f"{get_current_time()} Clusterization: Ready!")

    def output_data(self) -> DataObject:
        return self.data_object

    def get_dataset_predict_1(self, vector: list) -> int:
        if self.data_object.database is None:
            raise ValueError("Database of anomalies was not loaded!")
        current_type: int = 0
        current_min_dist: float = None

        temp_distances_list = []
        for data_object in self.data_object.database:
            temp_distances_list.append(spatial.distance.cosine(vector, data_object.features))
        temp_distances_list_copy = temp_distances_list.copy()
        temp_distances_list.sort()
        first_three_dist = temp_distances_list[-2:]
        for dist in first_three_dist:
            index = self.data_object.database[temp_distances_list_copy.index(dist)].anomaly_class
            if index > current_type: current_type = index
        return current_type

    def get_dataset_predict(self, vector: list) -> int:
        if self.data_object.database is None:
            return ValueError("Database of anomalies was not loaded!")
        current_type: int = None
        current_min_dist: float = None
        current_comment: str = None
        current_heaviness: int = 0

        for data_object in self.data_object.database:
            if current_type is None:
                current_type = data_object.anomaly_class
                current_comment = data_object.comment
                current_heaviness = data_object.heaviness
                current_min_dist = spatial.distance.cosine(vector, data_object.features)
            else:
                current_dist = spatial.distance.cosine(vector, data_object.features)
                if current_dist<current_min_dist:
                    current_min_dist = current_dist
                    current_type = data_object.anomaly_class
                    current_comment = data_object.comment
                    current_heaviness = data_object.heaviness
        #print(current_comment)
        return current_type, current_heaviness, current_comment

    def _clusterization(self) -> None:
        # demention number
        n_components = 2

        # Creating temporary lists for dataframes and clusterization
        vectors: List[List[float]] = []
        lables: List[str] = []
        lables_of_ts: List[str] = []
        dataset_preds: List[int] = []

        counter = 0
        inner_counter = 0
        count_of_zones = 0
        dataset_predict_comment = []
        dataset_predict_heaviness = []
        for filename in self.data_object.get_list_of_files():
            if len(self.data_object.ensambled_prediction[filename]) == 0:
                self._print_logs(f"{get_current_time()} Clusterization: For file {filename} no anomalies found!")
            else:
                inner_counter = 0
                for i in tqdm(range(len(self.data_object.ensambled_prediction[filename]))):
                    self.data_object.ensambled_prediction[filename][i].dataset_type, heaviness, comment = self.get_dataset_predict(self.data_object.ensambled_prediction[filename][i].features_for_datasets)
                    self.data_object.ensambled_prediction[filename][i].heaviness = heaviness
                    self.data_object.ensambled_prediction[filename][i].comment = comment
                    dataset_predict_heaviness.append(heaviness)
                    dataset_predict_comment.append(comment)
                    dataset_preds.append(self.data_object.ensambled_prediction[filename][i].dataset_type)
                    vectors.append(self.data_object.ensambled_prediction[filename][i].features)
                    lables_of_ts.append(str(counter))
                    lables.append(f"{counter} - {inner_counter}")
                    inner_counter += 1
                    count_of_zones += 1
                counter += 1
        # transform array to np.array



        dataset_vectors = []
        dataset_len = len(self.data_object.database)
        for object in self.data_object.database:
            dataset_vectors.append(object.features)
            #vectors.append(object.features)
        vectors = np.array([np.array(xi) for xi in vectors])
        train = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(vectors)
        #resape
        nsamples, nx, ny = train.shape
        train = train.reshape((nsamples,nx*ny))
        #train = np.transpose(train)
        train = self.__check_Nan(train)
        if False:
            path = "/home/nikita/Desktop/Fedot.Industrial/cases/anomaly_detection/clusterization/0.9752567693744166"
            path_clust = os.path.join(path, "Clusterizator.pickle")
            path_reducer = os.path.join(path, "Reducer.pickle")
            with open(path_clust, 'rb') as f:
                self.clusterisator = pickle.load(f)
            with open(path_reducer, 'rb') as f:
                self.reducer = pickle.load(f)
            self.tsne_result_2D = self.reducer.transform(vectors)
        else:
            #self.reducer = umap.UMAP()
            self.reducer = umap.UMAP(random_state=42).fit(vectors)


            path = "/media/nikita/HDD/created_database/reducer.pickle"
            with open(path, 'rb') as f:
                self.reducer = pickle.load(f) # fit_transform
            self.tsne_result_2D = self.reducer.transform(vectors)
            train = self.__check_Nan(train)
            self.tsne_result_2D_all = self.reducer.transform(vectors)
            self.clusterisator.fit(self.tsne_result_2D)
            #self.clusterisator.fit([self.reducer.embedding_[:, 0], self.reducer.embedding_[:, -1]])

        self.all_clusters = []
        self.tsne_2D  = TSNE(n_components)
        #self.tsne_result_2D = self.tsne_2D .fit_transform(vectors)
        if self.name == "HDBSCAN":
            soft_preds = hdbscan.all_points_membership_vectors(self.clusterisator)
            predictions = [np.argmax(x) for x in soft_preds]
            #self.tsne_result_2D = soft_preds
        else:
            #predictions = self.clusterisator.predict(self.tsne_result_2D_all)

            predictions = self.clusterisator.labels_
            #predictions = self.custom_clusterizator(temp_vec)


            #print(f"Success! Predict is ready.")
            #print(f"Fail! Ignored. Attempt tp use labels_ field.")
            #predictions = self.clusterisator.labels_
                #predictions = hdbscan.all_points_membership_vectors(self.clusterisator.labels_)
                #redictions = self.clusterisator.all_points_membership_vectorsabels_(train)
            #print(f"Success! Predict is ready.")
        
        
        self._print_logs(f"{get_current_time()} Clusterization: Predicts: {predictions}")

        predictions = list(predictions)
        all_predict = predictions.copy
        dataset_len
        #predictions = predictions[:len(predictions) - dataset_len]
        # predictions could be with mising clusters, we have to eliminate them
        clusters = set(predictions)
        clusters = [*clusters, ]
        clusters.sort()
        if False:
            is_good_format = True
            if clusters[0] == 0 or clusters[0] == -1:
                for number in range(0, len(clusters)-1):
                    if not clusters[number+1] == clusters[number]:
                        is_good_format = False
                        break
            else:
                is_good_format = False
            if not is_good_format:
                # filter predicts to avoid holes in numbers of clusters
                new_clusters = []
                if clusters[0] == -1:
                    for number, cluster in enumerate(clusters):
                        new_clusters.append(number-1)
                else:
                    for number, _ in enumerate(clusters):
                        new_clusters.append(number)
                for number in range(len(predictions)):
                    predictions[number] = new_clusters[clusters.index(predictions[number])]
        timp_dict = {}
        temp_list = []
        for clust in clusters:
            timp_dict[int(clust)] = predictions.count(int(clust))
            temp_list.append(predictions.count(int(clust)))
        self._print_logs(f"{get_current_time()} Clusterization: Clusters: {timp_dict}")

        #print(predictions)
        if len(predictions) != count_of_zones: raise ValueError("Something went wrong! Count of predictions doesn't match count of zones!")
        counter = 0
        for filename in self.data_object.get_list_of_files():
            if not len(self.data_object.ensambled_prediction[filename]) == 0:
                for i in range(len(self.data_object.ensambled_prediction[filename])):
                    self.data_object.ensambled_prediction[filename][i].cluster_type = predictions[counter]
                    counter += 1
        
        # looking for suspicious clusters
        # clustr 0 - none-critical
        # the rest start with 1, 2, 3... suspitious

        counter = 0
        for cluster in clusters:
            self.data_object.clusters.append(counter)
            self._print_logs(f"{get_current_time()} Clusterization: Check cluster {cluster}...")
            temp_cluster_len: int = 0
            temp_cluster_items: dict = {} # <dataset predict>: count of elements in cluster
            for filename in self.data_object.get_list_of_files():
                if not len(self.data_object.ensambled_prediction[filename]) == 0:
                    for i in range(len(self.data_object.ensambled_prediction[filename])):
                        if self.data_object.ensambled_prediction[filename][i].cluster_type == cluster:
                            temp_cluster_len += 1
                            if self.data_object.ensambled_prediction[filename][i].dataset_type in temp_cluster_items:
                                temp_cluster_items[self.data_object.ensambled_prediction[filename][i].dataset_type] += 1
                            else:
                                temp_cluster_items[self.data_object.ensambled_prediction[filename][i].dataset_type] = 1
            print(temp_cluster_items)
            if 0 in temp_cluster_items:
                if temp_cluster_items[0] > temp_cluster_len * 0.99:
                    new_cluster = 0
                else:
                    new_cluster = counter
                    counter += 1
            else:
                new_cluster = counter
                counter += 1
            for filename in self.data_object.get_list_of_files():
                if not len(self.data_object.ensambled_prediction[filename]) == 0:
                    for i in range(len(self.data_object.ensambled_prediction[filename])):
                        if self.data_object.ensambled_prediction[filename][i].cluster_type == cluster:
                            self.data_object.ensambled_prediction[filename][i].cluster_type = new_cluster
        

        predictions = []
        for filename in self.data_object.get_list_of_files():
            if not len(self.data_object.ensambled_prediction[filename]) == 0:
                for i in range(len(self.data_object.ensambled_prediction[filename])):
                    predictions.append(self.data_object.ensambled_prediction[filename][i].cluster_type)
        
        print(f"Cluster count: {counter}")
        if False:
            for filename in self.data_object.get_list_of_files():
                if not len(self.data_object.ensambled_prediction[filename]) == 0:
                    for i in range(len(self.data_object.ensambled_prediction[filename])):
                        if 0 == self.data_object.ensambled_prediction[filename][i].cluster_type:
                            self.data_object.ensambled_prediction[filename][i].predicted_type = 0



        for current_cluster in range(0, counter):
            print(f"Creating cluster {current_cluster}...")
            temp_adress_list = [] # list of tuples where <filename>, <number>
            for filename in self.data_object.get_list_of_files():
                if not len(self.data_object.ensambled_prediction[filename]) == 0:
                    for i in range(len(self.data_object.ensambled_prediction[filename])):
                        if current_cluster == self.data_object.ensambled_prediction[filename][i].cluster_type:
                            temp_adress_list.append([filename, i])
            self.data_object.clusters_adress.append(temp_adress_list)
        #for cluster_number in clusters:
        temp_counter = 0
        for filename in self.data_object.get_list_of_files():
            if not len(self.data_object.ensambled_prediction[filename]) == 0:
                for i in range(len(self.data_object.ensambled_prediction[filename])):
                    self.data_object.ensambled_prediction[filename][i].x = self.tsne_result_2D[:,0][temp_counter]
                    self.data_object.ensambled_prediction[filename][i].y = self.tsne_result_2D[:,1][temp_counter]
                    temp_counter+=1
        
        
        self.data_object.make_last_predict()

        zones = [
            [-5, -2.5, -3.5, 2], # 2
            [-5.5, -7, -4, -4], #2
            [-3.5, -8, -1,-5], # 2-3
            [-3.8, -10, 2, -6], # 3
            [-1, 2.5, 1, 10] # 2
        ]
        for filename in self.data_object.get_list_of_files():
            if not len(self.data_object.ensambled_prediction[filename]) == 0:
                for i in range(len(self.data_object.ensambled_prediction[filename])):
                    x = self.data_object.ensambled_prediction[filename][i].x
                    y = self.data_object.ensambled_prediction[filename][i].y
                    if self.check_point_for_zone(zones[0], x, y):
                        self.data_object.ensambled_prediction[filename][i].predicted_type = 2
                        print(f"{x} - {y} - {0}")
                    elif self.check_point_for_zone(zones[1], x, y):
                        self.data_object.ensambled_prediction[filename][i].predicted_type = 2
                        print(f"{x} - {y} - {1}")
                    elif self.check_point_for_zone(zones[2], x, y):
                        self.data_object.ensambled_prediction[filename][i].predicted_type = 2
                        print(f"{x} - {y} - {2}")
                    elif self.check_point_for_zone(zones[3], x, y):
                        self.data_object.ensambled_prediction[filename][i].predicted_type = 3
                        print(f"{x} - {y} - {3}")
                    elif self.check_point_for_zone(zones[4], x, y):
                        self.data_object.ensambled_prediction[filename][i].predicted_type = 3
                        print(f"{x} - {y} - {4}")
    
        if True:
            predictions = []
            dataset_preds = []
            for filename in self.data_object.get_list_of_files():
                if not len(self.data_object.ensambled_prediction[filename]) == 0:
                    for i in range(len(self.data_object.ensambled_prediction[filename])):
                        predictions.append(self.data_object.ensambled_prediction[filename][i].cluster_type)
                        dataset_preds.append(self.data_object.ensambled_prediction[filename][i].dataset_type)
            int_pred = []
            symbol_pred = []
            for i in dataset_preds:
                int_pred.append(str(i))
            for i in predictions:
                symbol_pred.append(str(i))
            color_list = [
                '#636EFA', 
                '#EF553B', 
                '#00CC96', 
                '#AB63FA', 
                '#FFA15A', 
                '#19D3F3', 
                '#FF6692', 
                '#B6E880', 
                '#FF97FF', 
                '#FECB52',
                '#636EFA', 
                '#EF553B', 
                '#00CC96', 
                '#AB63FA', 
                '#FFA15A', 
                '#19D3F3', 
                '#FF6692', 
                '#B6E880', 
                '#FF97FF', 
                '#FECB52',]
            cols = [
                "#1AF201",
                "#0169F2",
                "#F2F001",
                "#F20101"
            ]
            colors = []
            for predict in predictions:
                colors.append(color_list[predict])
            type_color = []
            for predict in dataset_preds:
                type_color.append(color_list[predict])
            """
            tsne_result_df = pd.DataFrame(
                {
                'x': self.tsne_result_2D[:,0], 
                'y': self.tsne_result_2D[:,1], 
                #'z': self.tsne_result_2D[:,2],
                'clusters': int_pred,
                "lables": lables,
                "ts_lables": lables_of_ts,
                "symbol": symbol_pred
                }
            )

            self.fig = px.scatter(tsne_result_df, x="x", y="y", 
                hover_name='lables', color='clusters', symbol="symbol")
            self.fig.update_layout(height=600)   
            """
            self.fig = go.Figure()
            self.fig.add_trace(
                go.Scatter(
                    mode='markers',
                    x=self.tsne_result_2D[:,0],
                    y=self.tsne_result_2D[:,1],
                    marker=dict(
                        color=colors,
                        size=25,
                        opacity=0.1,
                    ),
                    name='Clusters'
                )
            )
            for j in range(len(self.tsne_result_2D[:,0])):
                if dataset_predict_comment[j] != "0":
                    self.fig.add_trace(go.Scatter(
                                x=[self.tsne_result_2D[:,0][j]],
                                y=[self.tsne_result_2D[:,1][j]+0.2],
                                text=[f"{dataset_predict_heaviness[j]} {dataset_predict_comment[j]}"],
                                mode="text",
                                ))


            cols = [
                "#1AF201",
                "#0169F2",
                "#F2F001",
                "#F20101"
            ]
            lables = [
                "No anomalies",
                "Light anomalies",
                "Heavy anomalies",
                "Critical anomalies!"
            ]
            for i in range(0, 4):
                temp_x_arr = []
                temp_y_arr = []
                for j in range(len(dataset_preds)):
                    if dataset_preds[j] == i:
                       temp_x_arr.append(self.tsne_result_2D[:,0][j])
                       temp_y_arr.append(self.tsne_result_2D[:,1][j])  
                self.fig.add_trace(
                    go.Scatter(
                        mode='markers',
                        x=temp_x_arr,
                        y=temp_y_arr,
                        marker=dict(
                            color=cols[i],
                            size=8,
                            #line=dict(
                            #    color='DarkSlateGrey',
                            #    width=2
                            #)
                        ),
                        name=lables[i]
                    )
                )
            self.fig.update_layout(
                        font=dict(
                            family="Courier New, monospace",
                            size=6,  # Set the font size here
                            color="RebeccaPurple"
                        )                 
                        )
            self.fig.show()

        print('Done')
    
    
    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)
    
    
    def __check_Nan(self, ts):
        #if np.isnan(ts):
        ts = np.nan_to_num(ts, nan=0)
        return ts


    def check_point_for_zone(self, zone: list, x: float, y: float) -> bool:
        """

        Args:
            zone (list): [x1, y1, x2, y2] - float
            x (float): _description_
            y (float): _description_

        Returns:
            bool: true if in zone, folse otherwise
        """
        if float(zone[0]) <= float(x) <= float(zone[2]) and \
            float(zone[1]) <= float(y) <= float(zone[3]): return True
        return False