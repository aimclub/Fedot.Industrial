from matplotlib import pyplot as plt

from TSAnomalyDetector.abstract_classes.AbstractDataOperation import AbstractDataOperation
from TSAnomalyDetector.utils.settings_args \
    import SettingsArgs
from TSAnomalyDetector.utils.get_time \
    import get_current_time
from plotly.graph_objs.scatter import Line
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

from TSAnomalyDetector.utils.get_features import generate_features_from_one_ts

from TSAnomalyDetector.abstract_classes.DataObject import DataObject


class DataVisualizatorNew(AbstractDataOperation):
    args: SettingsArgs
    raw_data: list
    transformed_data: list = []

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{get_current_time()} Visualisator: settings was set.")
        self._print_logs(f"{get_current_time()} Visualisator: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Visualisator: Print logs = {self.args.print_logs}")

    def input_data(self, data_object: DataObject) -> None:
        self._print_logs(f"{get_current_time()} Visualisator: Data read!")
        self.ts_by_clust_figs = []
        self.data_object = data_object
        """
        self.data = self.input_dict[DATA_BODY][ELECTED_DATA]
        self.columns = self.input_dict[DATA_BODY][ELECTED_LABLES]
        self.lables = self.input_dict[DATA_BODY][LABLES_FOR_VISUALIZATION]
        self.lables_for_metrics = self.input_dict[DATA_BODY][RAW_LABLES]
        self.files = self.input_dict[DATA_BODY][FILES_LIST]

        self.ensamble_predict = False
        self.clustered_predict = False
        if DETECTIONS in self.input_dict[DATA_BODY]:
            self.visualisate_pred = True
            self.predicts = self.input_dict[DATA_BODY][DETECTIONS][RAW_PREDICTIONS]
            self.predicts_q = self.input_dict[DATA_BODY][DETECTIONS][QUANTILE_PREDICTIONS]
            self.predicts_q_for_show = self.input_dict[DATA_BODY][DETECTIONS][PREDICTIONS_FOR_VISUALIZATION]
            if ENSAMBLED_PREDICTION_FOR__VISUALIZATION in self.input_dict[DATA_BODY][DETECTIONS]:
                self.ensamble_predict = self.input_dict[DATA_BODY][DETECTIONS][ENSAMBLED_PREDICTION_FOR__VISUALIZATION]
            if CLUSTERED_PREDICT in self.input_dict[DATA_BODY][DETECTIONS]:
                self.clustered_predict = self.input_dict[DATA_BODY][DETECTIONS][CLUSTERED_PREDICT]
                self.clustered_parts = self.input_dict[DATA_BODY][DETECTIONS][CLUSTERS_PARTS]

        else: 
            self.visualisate_pred = False
        """
    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Visualisator: Loading visualisation...")
        self._visualisator()
        #self._visualisator_by_cluster()
        self._print_logs(f"{get_current_time()} Visualisator: Ready!")

    def output_data(self) -> DataObject:
        return self.data_object

        
    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)


    def _visualisator(self) -> None:
        files = self.data_object.get_list_of_files()
        numbers = []
        for i in range(len(files)):
            numbers.append(i)
        fig = make_subplots(rows=len(files),
                        cols=1,
                        subplot_titles=files)
        counters_for_creating_dataset = [0] * len(self.data_object.clusters)
        for i, filename in enumerate(files):
            # get data in format of two list - <data> - <lables in format of numbers from 0 to n>
            temp_lables = []
            #for j, data in enumerate(self.data[i]):
            #    temp_data.append([])
            #    for u, item in enumerate(data):
            #        temp_data[j].append(item)
            #        temp_lables.append(u)
            temp_data = {}
            columns = self.data_object.current_elected_columns

            temp_data = self.data_object.transformed_data[filename]
            #temp_data = self.data_object.main_data_dict[filename]
            #temp_data = self.data_object.experimented_data[filename]
            #temp_lables = self.data_object.get_lables_for_visualization(filename)
            temp_dict = {}
            columns = self.data_object.current_elected_columns
            #columns = ["Xu", "Yu", "Zu", "Xd", "Yd", "Zd", "Vu", "Vd"]
            #for col in columns:
            #    temp_dict[f"Fig-{i} - {col}"] = temp_data[col]
            #temp_dict[f"Fig-{i} - min"] = self.data_object.min_ts[filename]
            #temp_dict[f"Fig-{i} - max"] = self.data_object.max_ts[filename]
            #temp_dict[f"Fig-{i} - dist"] = self.data_object.distance_ts[filename]
            temp_dict[f"Fig-{i} - dist_trans"] = self.data_object.distance_ts_for_exp[filename]
            #if len(temp_lables):
            #    temp_dict[f"Fig-{i} - Main lables"] = temp_lables
            #if self.visualisate_pred and False:
            #predict = self.data_object.get_ts_of_predict(filename, 3)
            #if len(predict):
            #    temp_dict[f"Fig-{i} - Predict: Critical anomalies"] = predict
            #predict = self.data_object.get_ts_of_predict(filename, 2)
            #if len(predict):
            #    temp_dict[f"Fig-{i} - Predict: Heavy anomalies"] = predict
            #predict = self.data_object.get_ts_of_predict(filename, 1)
            #if len(predict):
            #    temp_dict[f"Fig-{i} - Predict: Middle anomalies"] = predict
            #    for number, predict in enumerate(self.predicts[i]):
            #        temp_dict[f"Fig-{i} - Predict-{number}"] = predict
            #if self.clustered_predict and True:
            #    for number, predict in enumerate(self.clustered_predict[i]):
            #        temp_dict[f"Fig-{i} - Predict of cluster-{number}"] = predict
            #if self.ensamble_predict and False:
            #    temp_dict[f"Fig-{i} - Ensambled"] = self.ensamble_predict[i]
            #target_regions = df.query('Region == @r').drop('Region', axis=1).set_index('Serie').T "columns": temp_columns
            
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
            color_dict = {}
            for numb, columns_in_data in enumerate(temp_dict.keys()):
                color_dict[columns_in_data] = color_list[numb]
            
            target_regions = pd.DataFrame(data=temp_dict)
            #for c in target_regions.columns[:2]:
                #for item, item1 in zip(temp_data, temp_lables):
                    #print(f"{item} - {item1}")
            
            for columns_in_data in temp_dict.keys():
                #for item, item1 in zip(target_regions[c], target_regions.index):
               #    print(f"{item} - {item1}")
                fig = fig.add_trace(
                    go.Scatter(x=target_regions.index,
                            y=target_regions[columns_in_data],
                            name=columns_in_data,
                            mode='lines', 
                            #legendgroup=str(i),
                            showlegend=True,
                            line = Line({'color':color_dict[columns_in_data], 'width': 2})
                            ), row=i+1, col=1)
            cols = [
                "#1AF201",
                "#0169F2",
                "#F2F001",
                "#F20101"
            ]
            # <anomaly start>, <anomaly end>, <index>, <Description>
            lables = self.data_object.get_lables_list(filename)
            if lables:
                for line in lables:
                    if line[1] < self.data_object.get_len_of_dataset(filename):
                        fig = fig.add_trace(go.Scatter(
                            x=[line[0],line[0],line[1], line[1]], y=[0,-1,-1,0], 
                            name=f"{line[3]} {line[2]}", fill="toself", line=dict(
                                color="#141616",
                                width=1,
                            ),
                            fillcolor="#141616",
                            opacity=0.2), row=i+1, col=1)
                    
                        fig = fig.add_trace(go.Scatter(
                            x=[int((line[1]-line[0])/2+line[0])],
                            #y=[-0.6],
                            y=[-1.5],
                            text=[f"{line[3]} {line[2]}"],
                            mode="text",
                            ), row=i+1, col=1)
            import json
            text_position = 1.5
            for type_number in [2, 3]:
                predictions = self.data_object.get_predicts_list(filename, type_number)
                if predictions:
                    for zone in predictions:
                        if zone.get_end() < self.data_object.get_len_of_dataset(filename):
                            if zone.comment != "111":
                                fig = fig.add_trace(go.Scatter(
                                    x=[zone.get_start(),zone.get_start(),zone.get_end(), zone.get_end()], 
                                    y=[0,1,1,0], 
                                    name=f"{zone.comment} {type_number}", fill="toself", line=dict(
                                        color=cols[zone.predicted_type],
                                        width=1,
                                    ),
                                    fillcolor=cols[zone.predicted_type],
                                    line_color="RoyalBlue",
                                    opacity=0.6), row=i+1, col=1)
                            
                                if text_position == 1.5: text_position = 1.3
                                elif text_position == 1.3: text_position = 1.1
                                else: text_position = 1.5

                                fig = fig.add_trace(go.Scatter(
                                    x=[int((zone.get_end()-zone.get_start())/2+zone.get_start())],
                                    y=[text_position],
                                    text=[f"{zone.comment} {zone.heaviness}"],
                                    #text=[f"{counters_for_creating_dataset[type_number]}"],
                                    mode="text",
                                    ), row=i+1, col=1)

                            temp_dict = {}
                            temp_dict["type"] =  zone.predicted_type
                            temp_dict["heaviness"] = zone.heaviness
                            temp_dict["comment"] = zone.comment
                            for key in zone.data.keys():
                                if key != "Time":
                                    temp_dict[key] = zone.data[key]
                            temp_dict["min"] = zone.min_data.tolist()
                            temp_dict["max"] = zone.max_data.tolist()
                            temp_dict["dist"] = zone.distance_data
                            temp_dict["dist_transform"] = zone.distance_data_trans
                            if False:
                                with open(f'/home/nikita/Desktop/Fedot.Industrial/cases/anomaly_detection/database/temp_4/vector_{0.4}_{counters_for_creating_dataset[type_number]}.json', 'w') as filehandle:
                                    json.dump(temp_dict, filehandle)
                            
                            
                            counters_for_creating_dataset[type_number] +=1

            
        fig.update_layout(title={'text':f'Visualization of all data', 'x':.45},
                        template='plotly_white', hovermode='x',
                        showlegend=True,
                        xaxis_title="Length",
                        yaxis_title="Amplitude",
                        legend_tracegroupgap=90,
                        height=300*len(files),
                        font=dict(
                            family="Courier New, monospace",
                            size=8,  # Set the font size here
                            color="RebeccaPurple"
                        )                 
                        )
        #fig.update_yaxes(range = [-0.8,0.8])
        fig.update_xaxes(range = [0,50000])
        fig.update_yaxes(range = [-1.6,1.6])
        """
        
        fig.add_annotation(x=0.75, y=0.00,
                        text = 'Fuente: Datos obtenidos desde el Ministerio de Ciencia:', 
                        showarrow = False, 
                        xref='paper',
                        yref='paper', 
                        xanchor='right',
                        yanchor='bottom',
                        xshift=0,
                        yshift=-30
                        )   
        """
        self.all_ts = fig
        fig.show()

    