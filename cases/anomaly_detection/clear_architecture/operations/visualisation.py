import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from cases.anomaly_detection.clear_architecture.operations.AbstractDataOperation import AbstractDataOperation
from cases.anomaly_detection.clear_architecture.utils.get_time import time_now


class DataVisualizer(AbstractDataOperation):
    """
    Visualizes data obtained from anomaly detection experiment
    """

    def __init__(self):
        super().__init__(name='Data Visualizer', operation='visualisation')
        self.labels = None
        self.predicts = None
        self.data = None
        self.input_dict = None
        self.transformed_data = list()

    def input_data(self, dictionary: dict) -> None:
        self._print_logs(f"{time_now()} {self.name}: Data read!")
        self.input_dict = dictionary
        self.data = self.input_dict["data_body"]["elected_data"]
        self.labels = self.input_dict["data_body"]["labels_for_show"]
        self.predicts = self.input_dict["data_body"]["detection"]

    def output_data(self) -> dict:
        self.input_dict["data_body"]["transformed_data"] = self.transformed_data
        return self.input_dict

    def _do_analysis(self) -> None:
        """
        Visualize data
        :return: None
        """
        x_labels = []
        y_data = []
        for i in range(len(self.data[0][0])):
            x_labels.append(i)
            y_data.append([self.data[0][0][i], self.data[0][1][i]])

        visualize_df = pd.DataFrame(dict(x=x_labels,
                                         y1=self.data[0][0],
                                         y2=self.data[0][1]))

        plots = make_subplots(rows=2, cols=1)
        fig = go.Figure([
            go.Scatter(name='y1',
                       x=visualize_df['x'],
                       y=visualize_df['y1'],
                       mode='lines',
                       marker=dict(color='red', size=2),
                       showlegend=True),

            go.Scatter(name='y2',
                       x=visualize_df['x'],
                       y=visualize_df['y2'],
                       mode='lines',
                       marker=dict(color="#444"),
                       line=dict(width=1),
                       showlegend=False)
        ])
        fig.update_layout(autosize=False, width=1400, height=400)
        plots.append_trace(fig, row=1, col=1)
        plots.append_trace(fig, row=2, col=1)
        plots.update_layout(height=900, width=1400, title_text="Side By Side Subplots")
        plots.show()
