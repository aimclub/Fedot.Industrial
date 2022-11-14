import numpy as np

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

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
    values = []
    values_dict = {}
    for value in data:
        if not value in values_dict: 
            values.append(value)
            values_dict[value] = 0
    values.sort()
    length = len(values)
    if not 0 <= threshold <= 1:
        threshold = 0.99
    q = int(length * threshold)
    if q >= len(values): q -= 1
    quantile = values[int(length * threshold)]
    if data[0] < 0:  
        #quantile = quantile * -1
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