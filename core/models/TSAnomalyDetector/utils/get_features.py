from core.models.statistical.stat_features_extractor import StatFeaturesExtractor
import pandas as pd
import numpy as np



def get_features_vector_from_prediction(prediction: list) -> list:
        features: list = []
        temp_list_max = [0, 0]
        temp_list_min = [0, 0]
        for i, ts in enumerate(prediction):
            temp_features = generate_features_from_one_ts(ts)
            features.extend(temp_features)
        #features.extend(abs(temp_list_min[0]) - abs(temp_list_max[1]))
        return features

def generate_features_from_one_ts(time_series, features: list) -> list:
    features_name = [
        "mean",
        "median",
        "lambda_less_zero", 
        "std", 
        "var", 
        "max", 
        "min", 
        "q5", 
        "q25", 
        "q75", 
        "q95", 
        "sum"]
    time_series = np.asarray(time_series)
    time_series = np.reshape(time_series, (1, time_series.shape[0]))
    time_series = pd.DataFrame(time_series, dtype=float)
    aggregator = StatFeaturesExtractor()
    feat = aggregator.create_baseline_features(time_series)
    """
    mean_ 0
    median_ 1
    lambda_less_zero 2 ??
    std_ 3 -/+
    var_ 4 -/+
    max 5
    min 6
    q5_ 7
    q25_ 8
    q75_ 9
    q95_ 10
    sum_ 11
    len_ 12 &&&
    delta_ 13
    dist 14 - dist between start and end
    v20 15 - dist between start value and value at 20%
    v40 16
    v60 17
    v80 18
    v10 19 - 10, 20, 30, .., 90
    v1
    """

    keys = feat.columns
    values = feat._values
    out_values = []
    delta = values[0][5] - values[0][6]
    for feature in features:
        if feature == "len":
            out_values.append(len(time_series.values[0]))
        elif feature == "delta":
            out_values.append(delta)
        elif feature == "dist":
            res = abs(abs(time_series.values[0][0]) - abs(time_series.values[0][-1]))
            out_values.append(res)
        elif feature == "v20":
            res = abs(
                abs(time_series.values[0][0]) 
                - abs(time_series.values[0][int(len(time_series.values[0])*0.2)]))
            out_values.append(res)
        elif feature == "v40":
            res = abs(
                abs(time_series.values[0][0]) 
                - abs(time_series.values[0][int(len(time_series.values[0])*0.4)]))
            out_values.append(res)
        elif feature == "v60":
            res = abs(
                abs(time_series.values[0][0]) 
                - abs(time_series.values[0][int(len(time_series.values[0])*0.6)]))
            out_values.append(res)
        elif feature == "v80":
            res = abs(
                abs(time_series.values[0][0]) 
                - abs(time_series.values[0][int(len(time_series.values[0])*0.8)]))
            out_values.append(res)
        elif feature == "v10":
            length_of_ts = len(time_series.values[0])
            count = 10
            multiplier = 1/count
            for i in range(1, count):
                start = abs(time_series.values[0][0]) 
                part = time_series.values[0][int((i-1)*multiplier*length_of_ts):int((i)*multiplier*length_of_ts)]

                res = abs(start - max(part))
                out_values.append(res)
        elif feature == "v1":
            for i in range(3, 96):
                res = abs(
                    abs(time_series.values[0][0]) 
                    - abs(time_series.values[0][int(len(time_series.values[0])*(0.01*i))]))
                out_values.append(res)
        elif feature == "sum":
            res = time_series.values[0][0] * 2
            for value in time_series.values[0]:
                res += value - time_series.values[0][0]
            out_values.append(float(res))
        else:
            out_values.append(values[0][features_name.index(feature)])
    #out_values.append(values[0][0]) # mean_
    #out_values.append(values[0][1]) # median_
    #out_values.append(values[0][2]) # lambda_less_zero
    #out_values.append(values[0][3]) # std_
    #out_values.append(values[0][4]) # var_
    
    #out_values.append(values[0][5]) # max
    #out_values.append(values[0][6]) # min
    #out_values.append(values[0][7]) # q5_
    #out_values.append(values[0][8]) # q25_
    #out_values.append(values[0][9]) # q75_
    #out_values.append(values[0][10]) # q95_
    #out_values.append(values[0][11]) # sum_
    #out_values = np.array(out_values) 
    #reshaped_values = [out_values] 
    return out_values