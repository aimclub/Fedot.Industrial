import numpy as np
from statistics import mean
from scipy.signal import savgol_filter
from fedot_ind.core.models.detection.utils.get_time import get_current_time


def NormalizeData(ts: list, return_numpy: bool = False):
    """
    Normalization method for time seroes. 
    Output time series' values are between 0.0 and 1.0

    Args:
        ts (list): time series, ould be list or numpy.array
        return_numpy (bool, optional): If set to True, method result will be in 
        Numply.array format. Defaults to False.

    Raises:
        ValueError: If ts isn't list or Numpy.array

    Returns:
        list or Numpy.array: Normilized time series
    """

    #norm_1 = (ts - np.min(ts)) / (np.max(ts) - np.min(ts))
    if not isinstance(ts, list):
        #print(f"{get_current_time()} Math utils - Normalization: In normalization method not a list was sent. Possible np.array. Attempting to convert...")
        try:
            ts = ts.tolist()
            #print(f"{get_current_time()} Math utils - Normalization: Succesfull converted!")
        except:
            raise ValueError(F"Math utils - Normalization: An unexpected type of time series has been meet!")

    norm_list: list = list()
    min_value = min(ts)
    distance = max(ts) - min_value
    if distance == 0: 
        print(f"---[{get_current_time()}] Math utils - Normalization: Distance between min and max is zero! Ts without changes!")
        norm_list = [0] * len(ts)
    else:
        for value in ts:
            norm_list.append((value - min_value) / distance)
        if return_numpy:
            norm_list = np.array(norm_list)

    return norm_list

def NormalizeData_1(data):
    from sklearn.preprocessing import minmax_scale
    data = minmax_scale(data, feature_range=(0,1))
    return data

def NormalizeDataForDetectors(data):
    data = np.append(data, [0])
    output = (data - np.min(data)) / (np.max(data) - np.min(data))
    output = output[:-1]
    #output = [float(i)/sum(data) for i in data]
    return output

def Cut_data(data: list, threshold: float):
    """
    This method cut data - everython under threshold will be put to zero

    Args:
        data (list): list
        threshold (float): float

    Returns:
        out_data (list): cutted data
    """
    #values = []
    #values_dict = {}
    #for value in data:
    #    if not value in values_dict: 
    #        values.append(value)
    #        values_dict[value] = 0
    #values.sort()
    #length = len(values)
    if not 0 <= threshold <= 1:
        threshold = 0.99
    #q = int(length * threshold)
    #if q >= len(values): q -= 1
    #quantile = values[int(length * threshold)]
    quantile = threshold
    if data[0] < 0:  
        out_data = []
        for element in data:
            if element >= quantile: out_data.append(0)
            else: out_data.append(element)
    else:
        out_data = []
        for element in data:
            if element <= quantile: out_data.append(0)
            else: out_data.append(element)
    
    return out_data

def clean_begin_of_ts(ts: list, start_part_len: int, desirable_value: float = 0) -> list:
    """
    Because of very big nouces at the start and the end of time series 
    we have to clean it - made it 0 or another value

    Args:
        ts (list): some time serie
        start_part_len (int): len of start part. Method will change all
            elements of ts from 0 to <start_part_len>
        desirable_value (float): desirable value of elements in 
            this part, 0 by default
    Returns:
        list: changed time serie
    """
    mult = 5
    if len(ts) > start_part_len:
        for j in range(start_part_len):
            ts[j] = desirable_value
        if len(ts) < start_part_len * mult:
            print(f"---[{get_current_time()}] Math utils - Begin cleaner warning: Time series is short. <{start_part_len} * {mult}! Anomalies could be detected with errors or doesn't detected at all!")
            #warnings.warn(f"Transformation warning: Time series is short. <{start_part_len} * {mult}! Anomalies could be detected with errors or doesn't detected at all!")
    else:
        raise ValueError(f"Transformation error: Time series is too short. <{start_part_len}!")
    return ts

def clean_end_of_ts(ts: list, end_part_len: int, desirable_value: float = 0) -> list:
    """
    Because of very big nouces at the start and the end of time series 
    we have to clean it - made it 0 or another value

    Args:
        ts (list): some time serie
        end_part_len (int): len of start part. Method will change all
            elements of ts from <start_part_len> to end of ts
        desirable_value (float): desirable value of elements in 
            this part, 0 by default
    Returns:
        list: changed time serie
    """
    mult = 5
    if len(ts) > end_part_len:
        for j in range(len(ts)-end_part_len, len(ts)):
            ts[j] = desirable_value
        if len(ts) < end_part_len * mult:
            print(f"---[{get_current_time()}] Math utils - End cleaner warning: Time series is short. <{end_part_len} * {mult}! Anomalies could be detected with errors or doesn't detected at all!")
            #warnings.warn(f"Transformation warning: Time series is short. <{end_part_len} * {mult}! Anomalies could be detected with errors or doesn't detected at all!")
    else:
        raise ValueError(f"Transformation error: Time series is too short. <{end_part_len}!")
    return ts

def move_ts_to_desirable_average_level(ts: list, desirable_average: float = 0) -> list:
    """
    This method moves time series up or down along Y-axis
    to the level when average of this time series became equial to desirable

    Args:
        ts (list): list of float numbers(possible any number values)
        desirable_average (float, optional): Desirable level of average. Defaults to 0.

    Returns:
        list: changed ts
    """
    if mean(ts) < desirable_average:
        mean_distance = desirable_average - mean(ts)
        for j in range(len(ts)):
            ts[j] = ts[j] + abs(mean_distance)
    else:
        mean_distance = desirable_average - mean(ts)
        for j in range(len(ts)):
            ts[j] = ts[j] - abs(mean_distance)
    return ts

def smooth_ts_for_transforming(
    ts: list, 
    cycles_count: int, 
    window_len_of_first_filer: int = 87,
    window_len_of_second_filer: int = 31) -> list:
    """
    Method smooth time series.

    Args:
        ts (list): time serie
        cycles_count (int): count of cycles of smoothing
        window_len_of_first_filer (int, optional): An odd int value. Magic len of window for first of two smoother methods. Defaults to 87.
        window_len_of_second_filer (int, optional): An odd int value. Magic len of window for second of two smoother methods. Defaults to 31.

    Returns:
        list: smoothed ts
    """
    
    for _ in range(cycles_count):
        ts = savgol_filter(ts, window_len_of_first_filer, 1) 
        ts = savgol_filter(ts, window_len_of_second_filer, 1)
    return ts