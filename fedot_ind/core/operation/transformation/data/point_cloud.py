from attr import dataclass
import pandas as pd
from gtda.time_series import SingleTakensEmbedding
from gtda.homology import VietorisRipsPersistence, WeakAlphaPersistence
from gtda.diagrams import Scaler
from ripser import Rips, ripser
import ripserplusplus as rpp
from scipy import sparse
import torch
from typing import Literal, List, Tuple, Union, Union
import warnings
import numpy as np

@dataclass
class TopologicalEmbeddingConfig:
    window_size: int
    stride: int = 1
    delay: int = 1
    multivariate_strategy: Literal['independent', 'joint'] = 'independent'

    def validate(self, ts_length: int):
        effective_window = (self.window_size - 1) * self.delay + 1
        
        if ts_length < effective_window:
            raise ValueError(
                f"Time series length ({ts_length}) is too short. "
                f"With window_size={self.window_size} and delay={self.delay}, "
                f"effective phase space window requires at least {effective_window} points."
            )
            
        if self.stride < 1:
            raise ValueError(f"Stride must be >= 1, got {self.stride}.")
            
        if self.delay < 1:
            raise ValueError(f"Delay must be >= 1, got {self.delay}.")


class PointCloudBuilder:
    """
    Class for constructing a point cloud from time series data using Takens' embedding theorem.
    This implementation allows for flexible configuration of the embedding parameters, including window size,
    """
    def __init__(self, config: TopologicalEmbeddingConfig):
        self.config = config

    def build(self, ts_data: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
        ts_tensor = self._prepare_input(ts_data)
        
        b, c, n = ts_tensor.shape
        self.config.validate(n)

        w = self.config.window_size
        tau = self.config.delay
        stride = self.config.stride

        effective_window = (w - 1) * tau + 1
        num_windows = (n - effective_window) // stride + 1

        batch_stride, channel_stride, time_stride = ts_tensor.stride()

        embedding = ts_tensor.as_strided(
            size=(b, c, num_windows, w),
            stride=(batch_stride, channel_stride, stride * time_stride, tau * time_stride)
        )

        if self.config.multivariate_strategy == 'joint':

            embedding = embedding.transpose(1, 2).reshape(b, num_windows, c * w)

        return embedding

    def _prepare_input(self, ts_data: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
        """
        Prepares the input time series data for embedding. Converts input to a 3D tensor of shape (B, C, N),
        where B is the batch size, C is the number of channels (1 for univariate), and N is the length of the time series.
        """
        if not isinstance(ts_data, torch.Tensor):
            ts_data = torch.tensor(ts_data, dtype=torch.float32)

        if ts_data.ndim == 1:
            ts_data = ts_data.view(1, 1, -1)
        elif ts_data.ndim == 2:
            ts_data = ts_data.unsqueeze(1)
        elif ts_data.ndim > 3:
            raise ValueError(f"Expected <= 3 dimensions (B, C, N), got {ts_data.ndim}")

        if not ts_data.is_contiguous():
            ts_data = ts_data.contiguous()

        return ts_data

@dataclass
class PersistenceConfig:
    homology_dimensions: Tuple[int, ...] = (0, 1, 2)
    backend: Literal['gtda', 'ripser++'] = 'gtda'
    filtration_type: Literal['vietoris-rips', 'alpha'] = 'vietoris-rips'
    normalize: bool = True
    distance_metric: Literal['euclidean', 'manhattan', 'cosine'] = 'euclidean'
    distance_device: Literal['cpu', 'cuda'] = 'cuda'


class PersistenceDiagramsExtractor:
    """
    Extracts persistence diagrams from point clouds using the specified backend and filtration type.
    """
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.max_dimension = max(config.homology_dimensions)
        self._device = torch.device(
            self.config.distance_device if torch.cuda.is_available() else 'cpu'
        )

    def transform(self, point_clouds: torch.Tensor) -> np.ndarray:
        if not isinstance(point_clouds, torch.Tensor):
            point_clouds = torch.tensor(point_clouds, dtype=torch.float32)

        if point_clouds.ndim == 2:
            point_clouds = point_clouds.unsqueeze(0)
        elif point_clouds.ndim != 3:
            raise ValueError(f"Expected tensor of shape (B, M, d) or (M, d), got {point_clouds.ndim} dimensions.")
            
        if self.config.filtration_type == 'alpha':
            return self._compute_alpha(point_clouds)
        elif self.config.filtration_type == 'vietoris-rips':
            return self._compute_vietoris_rips(point_clouds)
        else:
            raise ValueError(f"Unknown filtration_type: {self.config.filtration_type}")

    def _compute_distance_matrix(self, point_clouds: torch.Tensor) -> np.ndarray:
        pc_tensor = point_clouds.to(self._device)
        with torch.no_grad():
            dist_matrix = torch.cdist(pc_tensor, pc_tensor, p=self.config.distance_metric)
        return dist_matrix.cpu().numpy()

    def _compute_alpha(self, point_clouds: torch.Tensor) -> np.ndarray:
        if self.config.backend == 'ripser++':
            warnings.warn(
                "ripser++ does not support alpha filtration. "
                "Calculation has been forcibly switched to gtda."
            )
            
        pc_array = point_clouds.detach().cpu().numpy()
        alpha = WeakAlphaPersistence(
            homology_dimensions=self.config.homology_dimensions,
            n_jobs=-1
        )
        diagrams = alpha.fit_transform(pc_array)
        return self._normalize(diagrams)

    def _compute_vietoris_rips(self, point_clouds: torch.Tensor) -> np.ndarray:
        dist_matrices = self._compute_distance_matrix(point_clouds)

        if self.config.backend == 'gtda':
            vr = VietorisRipsPersistence(
                metric='precomputed',
                homology_dimensions=self.config.homology_dimensions,
                n_jobs=-1
            )
            diagrams = vr.fit_transform(dist_matrices)
            return self._normalize(diagrams)
            
        elif self.config.backend == 'ripser++':
            batch_size = dist_matrices.shape[0]
            raw_diagrams: List[np.ndarray] = []
            
            for i in range(batch_size):
                rpp_dict = rpp.run(
                    f"--format distance --dim {self.max_dimension}", 
                    dist_matrices[i]
                )
                
                formatted_holes = []
                for dim, holes in rpp_dict.items():
                    if dim not in self.config.homology_dimensions:
                        continue
                    for birth, death in holes:
                        if np.isinf(death):
                            continue
                        formatted_holes.append([birth, death, dim])
                
                raw_diagrams.append(
                    np.array(formatted_holes, dtype=float) if formatted_holes else np.empty((0, 3))
                )

            # Pad diagrams to have the same number of points for batch processing
            max_k = max((len(dgm) for dgm in raw_diagrams), default=0)
            if max_k == 0:
                return np.zeros((batch_size, 1, 3))

            padded_diagrams = np.zeros((batch_size, max_k, 3), dtype=float)
            for i, dgm in enumerate(raw_diagrams):
                if (k := len(dgm)) > 0:
                    padded_diagrams[i, :k, :] = dgm
                    
            return self._normalize(padded_diagrams)

    def _normalize(self, diagrams: np.ndarray) -> np.ndarray:
        if not self.config.normalize or diagrams.size == 0:
            return diagrams
            
        if self.config.backend == 'gtda' or self.config.filtration_type == 'alpha':
            scaler = Scaler(function=lambda x: float(np.max(x)), n_jobs=-1)
            return scaler.fit_transform(diagrams)
        else:
            for i in range(diagrams.shape[0]):
                max_val = np.max(diagrams[i, :, :2])
                if max_val > 0:
                    diagrams[i, :, :2] /= max_val
            return diagrams



class TopologicalTransformation:
    """Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
    recorded at equal intervals.

    Args:
        time_series: Time series to be decomposed.
        max_simplex_dim: Maximum dimension of the simplices to be used in the Rips filtration.
        epsilon: Maximum distance between two points to be considered connected by an edge in the Rips filtration.
        persistence_params: ...
        window_length: Length of the window to be used in the rolling window function.

    Attributes:
        epsilon_range (np.ndarray): Range of epsilon values to be used in the Rips filtration.

    """

    def __init__(self,
                 time_series: np.ndarray = None,
                 max_simplex_dim: int = None,
                 epsilon: int = 10,
                 persistence_params: dict = None,
                 window_length: int = None,
                 stride: int = 1):
        self.time_series = time_series
        self.stride = stride
        self.max_simplex_dim = max_simplex_dim
        self.epsilon_range = self.__create_epsilon_range(epsilon)
        self.persistence_params = persistence_params

        if self.persistence_params is None:
            self.persistence_params = {
                'coeff': 2,
                'do_cocycles': False,
                'verbose': False}

        self.__window_length = window_length

    @staticmethod
    def __create_epsilon_range(epsilon):
        return np.array([y * float(1 / epsilon) for y in range(epsilon)])

    @staticmethod
    def __compute_persistence_landscapes(ts):

        N = len(ts)
        I = np.arange(N - 1)
        J = np.arange(1, N)
        V = np.maximum(ts[0:-1], ts[1::])

        # Add vertex birth times along the diagonal of the distance matrix
        I = np.concatenate((I, np.arange(N)))
        J = np.concatenate((J, np.arange(N)))
        V = np.concatenate((V, ts))

        # Create the sparse distance matrix
        D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
        dgm0 = ripser(D, maxdim=0, distance_matrix=True)['dgms'][0]
        dgm0 = dgm0[dgm0[:, 1] - dgm0[:, 0] > 1e-3, :]

        allgrid = np.unique(dgm0.flatten())
        allgrid = allgrid[allgrid < np.inf]

        xs = np.unique(dgm0[:, 0])
        ys = np.unique(dgm0[:, 1])
        ys = ys[ys < np.inf]

    def time_series_to_point_cloud(self,
                                   input_data: np.array = None,
                                   dimension_embed=3,
                                   use_gtda=False) -> np.array:
        """Convert a time series into a point cloud in the dimension specified by dimension_embed.

        Args:
            input_data: Time series to be converted.
            dimension_embed: dimension of Euclidean space in which to embed the time series into by taking
            windows of dimension_embed length, e.g. if the time series is ``[t_1,...,t_n]`` and dimension_embed
            is ``2``, then the point cloud would be ``[(t_0, t_1), (t_1, t_2),...,(t_(n-1), t_n)]``

        Returns:
            Collection of points embedded into Euclidean space of dimension = dimension_embed, constructed
            in the manner explained above.

        """

        if self.__window_length is None:
            self.__window_length = dimension_embed
        if use_gtda:
            pcd = self.gtda_time_series_to_pcd(input_data, dimension_embed)
        else:
            trajectory_transformer = HankelMatrix(time_series=input_data,
                                                  window_size=self.__window_length,
                                                  strides=self.stride)
            pcd = trajectory_transformer.trajectory_matrix
        return pcd

    def gtda_time_series_to_pcd(self,
                                input_data: np.array = None,
                                dimension_embed=3) -> np.array:
        embedder_periodic = SingleTakensEmbedding(
            parameters_type="fixed",
            n_jobs=2,
            time_delay=self.__window_length,
            dimension=dimension_embed,
            stride=self.stride,
        )
        embedding = embedder_periodic.fit_transform(input_data)
        return embedding

    def point_cloud_to_persistent_cohomology_ripser(
            self, point_cloud: np.array = None, max_simplex_dim: int = 1):

        # ensure epsilon_range is a numpy array
        epsilon_range = self.epsilon_range

        # build filtration
        self.persistence_params['maxdim'] = max_simplex_dim
        filtration = Rips(**self.persistence_params)

        if point_cloud is None:
            point_cloud = self.time_series_to_point_cloud()

        # initialize persistence diagrams
        diagrams = filtration.fit_transform(point_cloud)
        # Instantiate persistence landscape transformer
        # plot_diagrams(diagrams)

        # normalize epsilon distance in diagrams so max is 1
        diagrams = [np.array([dg for dg in diag if np.isfinite(dg).all()])
                    for diag in diagrams]
        # diagrams = diagrams / max([np.array([dg for dg in diag if np.isfinite(
        # dg).all()]).max() for diag in diagrams if diag.shape[0] > 0])

        diagrams = [d / max([np.array([dg for dg in diag if np.isfinite(
            dg).all()]).max() for diag in diagrams if diag.shape[0] > 0]) for d in diagrams]

        ep_ran_len = len(epsilon_range)

        homology = {dimension: np.zeros(ep_ran_len).tolist(
        ) for dimension in range(max_simplex_dim + 1)}

        for dimension, diagram in enumerate(diagrams):
            if dimension <= max_simplex_dim and len(diagram) > 0:
                homology[dimension] = np.array(
                    [np.array(((epsilon_range >= point[0]) & (epsilon_range <= point[1])).astype(int))
                     for point in diagram
                     ]).sum(axis=0).tolist()

        return homology

    def time_series_to_persistent_cohomology_ripser(
            self, time_series: np.array, max_simplex_dim: int) -> dict:
        """Wrapper function that takes in a time series and outputs the persistent homology object, along with other
        auxiliary objects.

        Args:
            time_series: Time series to be converted.
            max_simplex_dim: Maximum dimension of the simplicial complex to be constructed.

        Returns:
            Persistent homology object. Dictionary with keys in ``range(max_simplex_dim)`` and, the value ``hom[i]``
            is an array of length equal to ``len(epsilon_range)`` containing the betti numbers of the ``i-th`` homology
            groups for the Rips filtration.

        """

        homology = self.point_cloud_to_persistent_cohomology_ripser(
            point_cloud=time_series, max_simplex_dim=max_simplex_dim)
        return homology

    def time_series_rolling_betti_ripser(self, ts):

        point_cloud = self.rolling_window(
            array=ts, window=self.__window_length)
        homology = self.time_series_to_persistent_cohomology_ripser(
            point_cloud, max_simplex_dim=self.max_simplex_dim)
        df_features = pd.DataFrame(data=homology)
        cols = ["Betti_{}".format(i) for i in range(df_features.shape[1])]
        df_features.columns = cols
        df_features['Betti_sum'] = df_features.sum(axis=1)
        return df_features

    def rolling_window(self, array, window):
        if window <= 0:
            raise ValueError("Window size must be a positive integer.")
        if window > len(array):
            raise ValueError(
                "Window size cannot exceed the length of the array.")
        return np.array([array[i:i + window]
                         for i in range(len(array) - window + 1)])
