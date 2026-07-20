from attr import dataclass
import pandas as pd
from gtda.time_series import SingleTakensEmbedding
from gtda.homology import VietorisRipsPersistence, WeakAlphaPersistence
from ripser import Rips
import torch
from typing import Literal, List, Tuple, Union
import warnings
import numpy as np

from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix


def _load_ripserplusplus():
    try:
        import ripserplusplus as rpp
    except ModuleNotFoundError as ex:
        raise ImportError(
            "ripserplusplus is required only for PersistenceConfig(backend='ripser++'). "
            "Install it or use backend='gtda'."
        ) from ex
    return rpp


class TopologicalTransformation:
    """Backward-compatible facade for legacy topological point-cloud helpers."""

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
        self.epsilon_range = self._create_epsilon_range(epsilon)
        self.persistence_params = persistence_params or {
            'coeff': 2,
            'do_cocycles': False,
            'verbose': False}
        self._window_length = window_length

    @staticmethod
    def _create_epsilon_range(epsilon):
        return np.array([value / float(epsilon) for value in range(epsilon)])

    def time_series_to_point_cloud(self,
                                   input_data: np.array = None,
                                   dimension_embed=3,
                                   use_gtda=False) -> np.array:
        if input_data is None:
            input_data = self.time_series
        if self._window_length is None:
            self._window_length = dimension_embed
        if use_gtda:
            return self.gtda_time_series_to_pcd(input_data, dimension_embed)

        trajectory_transformer = HankelMatrix(time_series=input_data,
                                              window_size=self._window_length,
                                              strides=self.stride)
        return trajectory_transformer.trajectory_matrix

    def gtda_time_series_to_pcd(self,
                                input_data: np.array = None,
                                dimension_embed=3) -> np.array:
        if input_data is None:
            input_data = self.time_series
        embedder_periodic = SingleTakensEmbedding(
            parameters_type="fixed",
            n_jobs=2,
            time_delay=self._window_length,
            dimension=dimension_embed,
            stride=self.stride,
        )
        return embedder_periodic.fit_transform(input_data)

    def point_cloud_to_persistent_cohomology_ripser(
            self, point_cloud: np.array = None, max_simplex_dim: int = 1):
        if point_cloud is None:
            point_cloud = self.time_series_to_point_cloud()

        self.persistence_params['maxdim'] = max_simplex_dim
        filtration = Rips(**self.persistence_params)
        diagrams = filtration.fit_transform(point_cloud)
        finite_diagrams = [np.array([point for point in diagram
                                     if np.isfinite(point).all()])
                           for diagram in diagrams]
        scale = max((diagram.max() for diagram in finite_diagrams
                     if diagram.shape[0] > 0),
                    default=1.0)
        if scale > 0:
            finite_diagrams = [diagram / scale for diagram in finite_diagrams]

        homology = {dimension: np.zeros(len(self.epsilon_range)).tolist()
                    for dimension in range(max_simplex_dim + 1)}
        for dimension, diagram in enumerate(finite_diagrams):
            if dimension > max_simplex_dim or len(diagram) == 0:
                continue
            homology[dimension] = np.array([
                ((self.epsilon_range >= point[0]) &
                 (self.epsilon_range <= point[1])).astype(int)
                for point in diagram
            ]).sum(axis=0).tolist()
        return homology

    def time_series_to_persistent_cohomology_ripser(
            self, time_series: np.array, max_simplex_dim: int) -> dict:
        return self.point_cloud_to_persistent_cohomology_ripser(
            point_cloud=time_series, max_simplex_dim=max_simplex_dim)

    def time_series_rolling_betti_ripser(self, ts):
        point_cloud = self.rolling_window(array=ts, window=self._window_length)
        homology = self.time_series_to_persistent_cohomology_ripser(
            point_cloud, max_simplex_dim=self.max_simplex_dim)
        df_features = pd.DataFrame(data=homology)
        df_features.columns = [
            f"Betti_{idx}" for idx in range(df_features.shape[1])]
        df_features['Betti_sum'] = df_features.sum(axis=1)
        return df_features

    @staticmethod
    def rolling_window(array, window):
        if window <= 0:
            raise ValueError("Window size must be a positive integer.")
        if window > len(array):
            raise ValueError(
                "Window size cannot exceed the length of the array.")
        return np.array([array[i:i + window]
                         for i in range(len(array) - window + 1)])


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
        """Building a point cloud from the given time series data using Takens' embedding theorem.
        Args:
            ts_data: Time series data as a tensor, numpy array, or list.
        Returns:
            A tensor representing the constructed point cloud.
        """
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
            stride=(batch_stride, channel_stride, stride *
                    time_stride, tau * time_stride)
        )

        if self.config.multivariate_strategy == 'joint':

            embedding = embedding.transpose(
                1, 2).reshape(b, num_windows, c * w)

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
            raise ValueError(
                f"Expected <= 3 dimensions (B, C, N), got {ts_data.ndim}")

        if not ts_data.is_contiguous():
            ts_data = ts_data.contiguous()

        return ts_data

    def build_trajectory_matrix(self, ts_data: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
        """
        Constructs the trajectory matrix from the time series data based on the embedding configuration.
        The trajectory matrix is a 3D tensor of shape (B, C, num_windows, W) for 'independent' strategy
        or (B, num_windows, C * W) for 'joint' strategy.
        """
        # (B, num_windows, C * W)
        point_cloud = self.build(ts_data)

        if self.config.multivariate_strategy == 'joint':
            # (B, num_windows, C * W) -> (B, C * W, num_windows)
            trajectory_matrix = point_cloud.transpose(1, 2)
        else:
            # (B, num_windows, C * W) -> (B, C, W, num_windows)
            trajectory_matrix = point_cloud.transpose(2, 3)

        return trajectory_matrix


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
            raise ValueError(
                f"Expected tensor of shape (B, M, d) or (M, d), got {point_clouds.ndim} dimensions.")

        if self.config.filtration_type == 'alpha':
            return self._compute_alpha(point_clouds)
        elif self.config.filtration_type == 'vietoris-rips':
            return self._compute_vietoris_rips(point_clouds)
        else:
            raise ValueError(
                f"Unknown filtration_type: {self.config.filtration_type}")

    def _compute_distance_matrix(self, point_clouds: torch.Tensor) -> np.ndarray:
        pc_tensor = point_clouds.to(self._device)
        with torch.no_grad():
            if self.config.distance_metric == 'euclidean':
                dist_matrix = torch.cdist(pc_tensor, pc_tensor, p=2.0)
            elif self.config.distance_metric == 'manhattan':
                dist_matrix = torch.cdist(pc_tensor, pc_tensor, p=1.0)
            elif self.config.distance_metric == 'cosine':
                pc_norm = torch.nn.functional.normalize(pc_tensor, p=2, dim=-1)
                dist_matrix = 1.0 - torch.bmm(pc_norm, pc_norm.transpose(1, 2))
                dist_matrix = torch.clamp(dist_matrix, min=0.0)
            else:
                raise ValueError(
                    f"Unsupported metric: {self.config.distance_metric}")
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
            n_jobs=1
        )
        diagrams = alpha.fit_transform(pc_array)
        return self._normalize(diagrams)

    def _compute_vietoris_rips(self, point_clouds: torch.Tensor) -> np.ndarray:
        dist_matrices = self._compute_distance_matrix(point_clouds)

        if self.config.backend == 'gtda':
            vr = VietorisRipsPersistence(
                metric='precomputed',
                homology_dimensions=self.config.homology_dimensions,
                n_jobs=1
            )
            diagrams = vr.fit_transform(dist_matrices)
            return self._normalize(diagrams)

        elif self.config.backend == 'ripser++':
            rpp = _load_ripserplusplus()
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
                    np.array(formatted_holes, dtype=float) if formatted_holes else np.empty(
                        (0, 3))
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

        coords = diagrams[:, :, :2]
        finite_mask = np.isfinite(coords)
        if not np.any(finite_mask):
            return diagrams

        max_val = float(np.max(coords[finite_mask]))

        if max_val > 0:
            coords[finite_mask] /= max_val
            diagrams[:, :, :2] = coords

        return diagrams
