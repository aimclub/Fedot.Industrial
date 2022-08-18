import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from cases.anomaly_detection.clear_architecture.utils.get_time \
    import get_current_time
from cases.anomaly_detection.clear_architecture.utils.settings_args \
    import SettingsArgs

"""
some description   
"""


class DataVisualizer:
    args: SettingsArgs
    raw_data: list
    transformed_data: list = []

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{get_current_time()} Visualizer: settings was set.")
        self._print_logs(f"{get_current_time()} Visualizer: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Visualizer: Print logs = {self.args.print_logs}")

    def input_data(self, dictionary: dict) -> None:
        self._print_logs(f"{get_current_time()} Visualizer: Data read!")
        self.input_dict = dictionary
        self.data = self.input_dict["data_body"]["elected_data"]
        self.lables = self.input_dict["data_body"]["labels_for_show"]
        self.predicts = self.input_dict["data_body"]["detection"]

    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Visualizer: Loading visualisation...")
        self._visualizer()
        self._print_logs(f"{get_current_time()} Visualizer: Ready!")

    def output_data(self) -> dict:
        self.input_dict["data_body"]["transformed_data"] = self.transformed_data
        return self.input_dict

    def _visualizer(self) -> None:
        x_labels = []
        y_data = []
        for i in range(len(self.data[0][0])):
            x_labels.append(i)
            y_data.append([self.data[0][0][i], self.data[0][1][i]])
        visualisate_df = pd.DataFrame(
            {
                'x': x_labels,
                'y1': self.data[0][0],
                'y2': self.data[0][1],
            }
            )
        
        #fig = px.scatter(visualisate_df,
        #    x = 'x', y=['y1', 'y2'], # replace with your own data source
        #    title="Test fig", height=325
        #)
        plots = make_subplots(rows=2, cols=1)
        fig = go.Figure([
            go.Scatter(
                name='y1',
                x=visualisate_df['x'],
                y=visualisate_df['y1'],
                mode='lines',
                marker=dict(color='red', size=2),
                showlegend=True
            ),
            go.Scatter(
                name='y2',
                x=visualisate_df['x'],
                y=visualisate_df['y2'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=1),
                showlegend=False
            )
        ])
        fig.update_layout(
            autosize=False,
            width=1400,
            height=400)
        plots.append_trace(fig, row=1, col=1)
        plots.append_trace(fig, row=2, col=1)
        plots.update_layout(height=900, width=1400, title_text="Side By Side Subplots")
        plots.show()
        """
        app = Dash(__name__)

        app.layout = html.Div([
            html.H4('Displaying figure structure as JSON'),
            dcc.Graph(id="graph", figure=fig),
            dcc.Clipboard(target_id="structure"),
            html.Pre(
                id='structure',
                style={
                    'border': 'thin lightgrey solid', 
                    'overflowY': 'scroll',
                    'height': '275px'
                }
            ),
        ])
        """
    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)
