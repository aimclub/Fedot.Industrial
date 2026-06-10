#!/bin/bash

mkdir -p ./data/repositories

# Yahoo dataset is for non-commerical uses only.
# Can manually request at: https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70


# Numenta Anomaly detection Benchmark (NAB) - AGPL-3.0 license, commercial use allowed
# https://github.com/numenta/NAB
mkdir -p ./data/anomaly_detection/NAB
if cd ./data/repositories/NAB; then git pull && cd ../../../; else git clone https://github.com/numenta/NAB.git ./data/repositories/NAB; fi

cp -r ./data/repositories/NAB/data/* ./data/anomaly_detection/NAB
cp -r ./data/repositories/NAB/labels/combined_labels.json ./data/anomaly_detection/NAB/combined_labels.json
cp -r ./data/repositories/NAB/labels/combined_windows.json ./data/anomaly_detection/NAB/combined_windows.json


# SKAB - Skoltech Anomaly Benchmark - AGPL-3.0 license, commercial use allowed
# https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab

# Can download data as one file: https://www.kaggle.com/datasets/caesarlupum/benckmark-anomaly-timeseries-skab?resource=download

# Alternatively, can clone original repository, but data is divided into many files:
# if cd ./data/repositories/SKAB; then git pull && cd ../../../; else git clone https://github.com/waico/SKAB.git ./data/repositories/SKAB; fi
# mkdir -p ./data/anomaly_detection/SKAB
# cp -r ./data/repositories/SKAB/data/* ./data/anomaly_detection/SKAB


# HAI Security Dataset - CC BY-SA 4.0, commerical use, must credit
# https://www.kaggle.com/datasets/icsdataset/hai-security-dataset
# https://github.com/icsdataset/hai/blob/master/hai_dataset_technical_details_v3.0.pdf
if cd ./data/anomaly_detection/HAI; then git pull && cd ../../../; else git clone https://github.com/icsdataset/hai.git ./data/anomaly_detection/HAI; fi


# https://www.kaggle.com/datasets/drscarlat/time-series?select=TimeSeries.csv
# License?


# 3W dataset - CC BY 4.0, commerical, credit
# https://www.kaggle.com/datasets/afrniomelo/3w-dataset
# Each of the 8 directories is a seperate dataset
mkdir -p ./data/anomaly_detection/3W
mkdir -p ./data/anomaly_detection/3W/dataset
if cd ./data/repositories/3W; then git pull && cd ../../../; else git clone https://github.com/petrobras/3W.git ./data/repositories/3W; fi
cp -r ./data/repositories/3W/dataset/* ./data/anomaly_detection/3W


# Anomaly Detection Falling People - "refactored" from a UCI repository - open use?
# Download manually from: https://www.kaggle.com/datasets/jorekai/anomaly-detection-falling-people-events
# Original: https://archive.ics.uci.edu/ml/datasets/Localization+Data+for+Person+Activity


# BETH Dataset - CC0: Public Domain
# Download manually from: https://www.kaggle.com/datasets/katehighnam/beth-dataset
# See also: https://github.com/jinxmirror13/BETH_Dataset_Analysis


# More at: https://archive.ics.uci.edu/ml/datasets.php?format=&task=cla&att=&area=&numAtt=&numIns=&type=ts&sort=nameUp&view=table

echo Script finished.
