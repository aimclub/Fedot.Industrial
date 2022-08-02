
import sys
import os

sys.path.append('../utils')
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)
sys.path.append(os.path.join(import_path, "../../"))
from tqdm import tqdm
# format output is list of files and list of anomalies with type and descriptions


def read_lables_csv_from_file(filename: str) -> list:
    temp_list = [[], [], [], [], []]
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    for i in tqdm(range(0, len(lines), 1)): #len(lines)-2
        temp_line = lines[i].strip().split(";")
        for j in range(len(temp_list)):
            temp_list[j].append(temp_line[j])
    out_list = []
    current_filename = ""
    temp_element = []
    temp_anomalies_list = []
    for i in range(len(temp_list[0])):
        if current_filename == "":
            current_filename = temp_list[0][i].strip()
            temp_anomalies_list.append([temp_list[1][i], temp_list[2][i], temp_list[3][i], temp_list[4][i]])
            continue
        if current_filename!=temp_list[0][i].strip():
            temp_element = [current_filename, temp_anomalies_list]
            out_list.append(temp_element)
            current_filename = temp_list[0][i].strip()
            temp_anomalies_list = []
            temp_anomalies_list.append([temp_list[1][i], temp_list[2][i], temp_list[3][i], temp_list[4][i]])
        else:
            temp_anomalies_list.append([temp_list[1][i], temp_list[2][i], temp_list[3][i], temp_list[4][i]])
    temp_element = [current_filename, temp_anomalies_list]
    out_list.append(temp_element)
    return out_list


if __name__ == '__main__':
    path = "/media/nikita/HDD/anomalies_new_nocount_2.csv"
    path_1 = "/media/nikita/HDD/anomalies_new_nocount.csv"

    data_container = read_lables_csv_from_file(path)
    data_container = read_lables_csv_from_file(path_1)