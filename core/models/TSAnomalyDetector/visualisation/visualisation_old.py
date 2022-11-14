from matplotlib import pyplot as plt
from cases.anomaly_detection.abstract_classes.AbstractDataOperation import AbstractDataOperation
from cases.anomaly_detection.utils.settings_args \
    import SettingsArgs
from cases.anomaly_detection.utils.get_time \
    import get_current_time
from cases.anomaly_detection.constants.current_data_const import \
    DATA_BODY, PREDICTIONS_FOR_VISUALIZATION, QUANTILE_PREDICTIONS, RAW_LABLES, RAW_PREDICTIONS,\
    ELECTED_DATA, DETECTIONS, LABLES_FOR_VISUALIZATION, \
        ENSAMBLED_PREDICTION, ENSAMBLED_PREDICTION_FOR__VISUALIZATION
"""


    
"""
class DataVisualisator(AbstractDataOperation):
    args: SettingsArgs
    raw_data: list
    transformed_data: list = []

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{get_current_time()} Visualisator: settings was set.")
        self._print_logs(f"{get_current_time()} Visualisator: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Visualisator: Print logs = {self.args.print_logs}")

    def input_data(self, dictionary: dict) -> None:
        self._print_logs(f"{get_current_time()} Visualisator: Data read!")
        self.input_dict = dictionary
        self.data = self.input_dict[DATA_BODY][ELECTED_DATA]
        self.lables = self.input_dict[DATA_BODY][LABLES_FOR_VISUALIZATION]
        self.lables_for_metrics = self.input_dict[DATA_BODY][RAW_LABLES]
        """
            predictions
            quantile_predictions
            predictions_for_show
        """
        self.ensamble_predict = False
        if DETECTIONS in self.input_dict[DATA_BODY]:
            self.visualisate_pred = True
            self.predicts = self.input_dict[DATA_BODY][DETECTIONS][RAW_PREDICTIONS]
            self.predicts_q = self.input_dict[DATA_BODY][DETECTIONS][QUANTILE_PREDICTIONS]
            self.predicts_q_for_show = self.input_dict[DATA_BODY][DETECTIONS][PREDICTIONS_FOR_VISUALIZATION]
            if ENSAMBLED_PREDICTION_FOR__VISUALIZATION in self.input_dict[DATA_BODY][DETECTIONS]:
                self.ensamble_predict = self.input_dict[DATA_BODY][DETECTIONS][ENSAMBLED_PREDICTION_FOR__VISUALIZATION]
        else: 
            self.visualisate_pred = False

    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Visualisator: Loading visualisation...")
        self._visualisator()
        self._print_logs(f"{get_current_time()} Visualisator: Ready!")

    def output_data(self) -> dict:
        return self.input_dict

    def _visualisator(self) -> None:
        x_lables = []
        y_data = []
        
        
        f, ax = plt.subplots(len(self.data) * 1, 1, figsize=(20, 10))
        counter = 0
        score = []
        for i in range(0, len(self.data)):
            y_data = []
            for j in range(len(self.data[i][0])):
                temp_list = []
                for t in range(len(self.data[i])):
                    temp_list.append(self.data[i][t][j])
                y_data.append(temp_list)
            """
            for j in range(0, len(self.predicts[i])):
                index.append(pd.to_timedelta(j))
                value = 0
                real_value = j * (len(self.lables[i])//len(self.predicts[i]))
                for n in range(real_value, real_value+self.window):
                    if n < len(self.lables[i]):
                        if self.lables[i][n] == 1:
                            value = 1
                            break
                true_lables.append(value)
            """
            zero_line = [0] * len(y_data)
            ax[counter].plot(y_data)
            ax[counter].plot(zero_line)
            ax[counter].plot(self.lables[i], "g")
            #ax[counter].set_title(f"raw data {self.new_lables }")
            #counter += 1
            #ax[counter].plot(self.lables[i], "r")
            if self.visualisate_pred:
                for predict in self.predicts[i]:
                    ax[counter].plot(predict)
                if self.ensamble_predict:
                    ax[counter].plot(self.ensamble_predict[i], "r")
                #for predict in self.predicts_q_for_show[i]:
                #    ax[counter].plot(predict, "r")  
                ax[counter].plot(self.predicts[i][0], "b")
            #ax[counter].plot(odd_new_predicts_1, "r")
            #ax[counter].set_title("lables")
            counter += 1
        print("--------------------------")
        #print(mean(score))
        plt.show()

        #ax[2].plot(true_lables, "r")
        #ax[2].set_title("lables")
        
    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)
