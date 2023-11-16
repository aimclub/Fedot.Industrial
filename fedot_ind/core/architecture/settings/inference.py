import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fedot_ind.core.models.topological.topological_extractor import TopologicalExtractor

# for rank in [10, 20, 30, 40, 50]:
#     time_computational = []
#     time_computational_low_rank = []
#     matrix_sizes = []
#     features = []
#     for matrix_size in range(100, 10000, 100):
#         random_matrix = np.random.randn(matrix_size, matrix_size)
#         random_vector = np.random.randn(matrix_size, 1).flatten()
#         low_rank = int(matrix_size / rank)
#         start = time.time()
#         dot_product = random_vector.T @ random_matrix @ random_matrix
#         end = time.time()
#         time_computational.append(end - start)
#         start = time.time()
#         dot_product_low_rank = random_vector.T @ random_matrix[:, :low_rank] @ random_matrix[:low_rank, :]
#         end = time.time()
#         time_computational_low_rank.append(end - start)
#         matrix_sizes.append(matrix_size)
#     df = pd.DataFrame()
#     time_computational[0] = time_computational[1]
#     df['matrix_size'] = matrix_sizes
#     df['time_for_matrix_computation'] = time_computational
#     df['time_for_matrix_computation_low_rank'] = time_computational_low_rank
#     matplotlib.use('TkAgg')
#     df.plot(x='matrix_size', y=['time_for_matrix_computation_low_rank', 'time_for_matrix_computation'])
#     plt.show()

for w_size in [10, 20, 30]:
    time_computational = []
    matrix_sizes = []
    features = []
    for matrix_size in range(100, 2000, 100):
        random_matrix = np.random.randn(matrix_size, 1).flatten()
        start = time.time()
        topo_model = TopologicalExtractor(params={'window_size': w_size,
                                                  'stride': 1})
        topo_features = topo_model.generate_features_from_ts(random_matrix)
        features.append(topo_features)
        end = time.time()
        time_computational.append(end - start)
        matrix_sizes.append(matrix_size)
    df = pd.DataFrame()
    time_computational[0] = time_computational[1]
    df['ts_length'] = matrix_sizes
    df['time_for_feature_generation'] = time_computational
    matplotlib.use('TkAgg')
    df.plot(x='ts_length', y='time_for_feature_generation')
    plt.show()
_ = 1
