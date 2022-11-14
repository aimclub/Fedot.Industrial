import copy
from cases.anomaly_detection.utils.settings_args \
    import SettingsArgs
from cases.anomaly_detection.utils.get_time \
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
from cases.anomaly_detection.abstract_classes.AbstractDataOperation import AbstractDataOperation
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from cases.anomaly_detection.abstract_classes.DataObject import DataObject
from scipy import spatial

from cases.anomaly_detection.utils.get_features import generate_features_from_one_ts
import umap
import pickle
import csv
from typing import List
from tqdm import tqdm
import plotly.graph_objects as go

class FurcherClusterization(AbstractDataOperation):
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
        self._print_logs(f"{get_current_time()} Next clusterization: settings was set.")
        self._print_logs(f"{get_current_time()} Next clusterization: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Next clusterization: Print logs = {self.args.print_logs}")

    def input_data(self, data_object: DataObject) -> None:
        self._print_logs(f"{get_current_time()} Next clusterization: Data read!")
        self.data_object = data_object

    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Next clusterization: Loading metrics...")
        self._clusterization()
        self._print_logs(f"{get_current_time()} Next clusterization: Ready!")

    def output_data(self) -> DataObject:
        return self.data_object

    def get_dataset_predict(self, vector: list) -> int:
        if self.data_object.database is None:
            return -100
        current_type: int = None
        current_min_dist: float = None
        for data_object in self.data_object.database:
            if current_type is None:
                current_type = data_object.anomaly_class
                current_min_dist = spatial.distance.cosine(vector, data_object.features)
            else:
                current_dist = spatial.distance.cosine(vector, data_object.features)
                if current_dist<current_min_dist:
                    current_min_dist = current_dist
                    current_type = data_object.anomaly_class
        return current_type

    def _clusterization(self) -> None:
        # demention number
        n_components = 2
        for cluster_number, list_of_element_in_cluster in enumerate(self.data_object.clusters_adress):
            self._print_logs(f"{get_current_time()} Next clusterization: Clusterization of cluster {cluster_number}")
            self._print_logs(f"{get_current_time()} Next clusterization: Cluster len: {len(list_of_element_in_cluster)}")
            # Creating temporary lists for dataframes and clusterization
            vectors: List[List[float]] = []
            lables: List[str] = []
            lables_of_ts: List[str] = []
            dataset_preds: List[int] = []
            if len(list_of_element_in_cluster):
                temp_list_of_anomalies = []
                for item in list_of_element_in_cluster:
                    temp_list_of_anomalies.append(copy.copy(self.data_object.ensambled_prediction[item[0]][item[1]]))

                # get dataset predict anew or not? not yet I guess
                for item in temp_list_of_anomalies:
                    vectors.append(item.features)
                    dataset_preds.append(item.dataset_type)

                # transform array to np.array
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
                    self.reducer = umap.UMAP()
                    self.tsne_result_2D = self.reducer.fit_transform(vectors)
                    train = self.__check_Nan(train)
                    self.tsne_result_2D_all = self.reducer.transform(vectors)
                    self.clusterisator.fit(self.tsne_result_2D)

                self.all_clusters = []
                self.tsne_2D  = TSNE(n_components)
                print(self.tsne_result_2D.shape)
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
                # predictions could be with mising clusters, we have to eliminate them
                clusters = set(predictions)
                clusters = [*clusters, ]
                clusters.sort()
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
                    print(predictions.count(int(clust)))
                    timp_dict[predictions.count(int(clust))] = int(clust)
                    temp_list.append(predictions.count(int(clust)))
                temp_list.sort()
                temp_list = temp_list[:len(temp_list)-1]
                choosen_clusters = []
                for item in temp_list:
                    choosen_clusters.append(timp_dict[item])
                print(temp_list)
                print(clusters)
                #print(predictions)
                if len(predictions) != len(list_of_element_in_cluster): raise ValueError("Something went wrong! Count of predictions doesn't match count of zones!")
                
                # looking for suspicious clusters
                new_predict = [0] * len(predictions)
                for cluster in clusters:
                    self._print_logs(f"{get_current_time()} Next clusterization: Work with cluster {cluster}")
                    temp_list = []
                    temp_dict_of_items = {}
                    # create list of items of current cluster
                    for number, item in enumerate(list_of_element_in_cluster):
                        if cluster == predictions[number]:
                            temp_list.append(item)
                            if dataset_preds[number] not in temp_dict_of_items:
                                temp_dict_of_items[dataset_preds[number]] = 1
                            else:
                                temp_dict_of_items[dataset_preds[number]] += 1
                    max_count = 0
                    max_type = 0
                    for key in temp_dict_of_items.keys():
                        if max_count < temp_dict_of_items[key]:
                            max_count = temp_dict_of_items[key]
                            max_type = key
                    print(f"----------------------------------------------")  
                    print(f"{temp_dict_of_items}")
                    if 3 in temp_dict_of_items:
                        print(f"{temp_dict_of_items[3]}")
                    print(f"----------------------------------------------")  
                    if 0 in temp_dict_of_items:
                        max_type = 0
                        rest_count = len(temp_list) - temp_dict_of_items[0]
                        if rest_count < temp_dict_of_items[0]:
                            max_type = 0
                        else:
                            if 3 in temp_dict_of_items:
                                second_rest_count = rest_count - temp_dict_of_items[3]
                                if second_rest_count < temp_dict_of_items[3] * 5:
                                    max_type = 3
                                else:
                                    if 2 in temp_dict_of_items:
                                        third_rest_count = rest_count - temp_dict_of_items[2] * 5 - temp_dict_of_items[3] * 5
                                        if third_rest_count < temp_dict_of_items[2]:
                                            max_type = 2
                                        else:
                                            max_type = 1
                                    else:
                                        max_type = 1
                            else:
                                if 2 in temp_dict_of_items:
                                    third_rest_count = rest_count - temp_dict_of_items[2]
                                    if third_rest_count < temp_dict_of_items[2]:
                                        max_type = 2
                                    else:
                                        max_type = 1
                                else:
                                    max_type = 1

                    for number, item in enumerate(list_of_element_in_cluster):
                        if cluster == predictions[number]:
                            new_predict[number] = max_type

                for number, item in enumerate(list_of_element_in_cluster):
                    self.data_object.ensambled_prediction[item[0]][item[1]].predicted_type = new_predict[number]
                self.data_object.make_last_predict()
                if False:
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
                        '#636AAA', 
                        '#EF513B', 
                        '#00CA96', 
                        '#AB01FA', 
                        '#FFA15A', 
                        '#19D1F3', 
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
                    for predict in new_predict:
                        colors.append(cols[predict])
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
                    self.fig.update_layout(title={'text':f'Visualization of cluster {cluster_number}', 'x':.45},
                        template='plotly_white', hovermode='x',
                        showlegend=True,
                        xaxis_title="Length",
                        yaxis_title="Amplitude",
                        legend_tracegroupgap=90,
                        height=600,
                        font=dict(
                            family="Courier New, monospace",
                            size=8,  # Set the font size here
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


        