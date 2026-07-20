from abc import ABC, abstractmethod

import torch


class TopologicalFeaturesExtractor:
    def __init__(self, persistence_diagram_features: dict):
        self.persistence_diagram_features_ = persistence_diagram_features

    def transform(self, diagrams_batch: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        """
        Extracts features from a batch of persistence diagrams.
        Parameters:
            diagrams_batch: torch.Tensor
                A batch of persistence diagrams with shape (B, K, 3), where B is the batch size, K is the number of points in each diagram, and 3 corresponds to (birth, death, dimension).
        Returns:
            tuple[torch.Tensor, list[str]]
                A tuple containing the tensor of extracted features and the list of column names for each persistence diagram in the batch.
        """
        batch_features = []
        column_list = []
        B = diagrams_batch.shape[0]

        for feature_name, feature_model in self.persistence_diagram_features_.items():
            try:
                x_features = feature_model.fit_transform(diagrams_batch)
                batch_features.append(x_features)

                for dim in range(x_features.shape[1]):
                    column_list.append(f'{feature_name}_{dim}')

            except Exception:
                expected_len = getattr(
                    feature_model, 'max_homology_dim', 2) + 1
                batch_features.append(torch.zeros(
                    (B, expected_len), device=diagrams_batch.device))
                for dim in range(expected_len):
                    column_list.append(f'{feature_name}_{dim}')

        x_transformed = torch.cat(batch_features, dim=1)
        return x_transformed, column_list


class PersistenceDiagramFeatureExtractor(ABC):
    """Abstract class for persistence diagrams features extractor."""

    def __init__(self, max_homology_dim: int = 2):
        self.max_homology_dim = max_homology_dim

    def _get_diagram_components(self, persistence_diagram: torch.Tensor):
        B = persistence_diagram.shape[0]
        b = persistence_diagram[:, :, 0]
        d = persistence_diagram[:, :, 1]

        finite_d = torch.where(torch.isinf(
            d), torch.full_like(d, float('-inf')), d)
        max_finite_d = finite_d.max(dim=1, keepdim=True).values
        max_finite_d = torch.where(torch.isinf(
            max_finite_d), torch.zeros_like(max_finite_d), max_finite_d)

        d_clipped = torch.where(torch.isinf(d), max_finite_d, d)

        lifetimes = d_clipped - b
        mask = lifetimes > 0
        dims = persistence_diagram[:, :, 2].long()

        return B, lifetimes, mask, dims

    def _get_gtda_grid(self, b: torch.Tensor, d: torch.Tensor, mask: torch.Tensor, dims: torch.Tensor, n_bins: int):
        """Build a grid for each homology dimension based on the birth and death times of the persistence diagram."""
        b.shape[0]
        grids = []
        for dim in range(self.max_homology_dim + 1):
            dim_mask = mask & (dims == dim)

            # min birth (ignoring padding)
            b_dim = torch.where(dim_mask, b, torch.full_like(b, float('inf')))
            min_b = b_dim.min(dim=1, keepdim=True).values
            min_b = torch.where(torch.isinf(
                min_b), torch.zeros_like(min_b), min_b)

            # max death (ignoring padding AND infinity)
            valid_d = dim_mask & ~torch.isinf(d)
            d_dim = torch.where(valid_d, d, torch.full_like(d, float('-inf')))
            max_d = d_dim.max(dim=1, keepdim=True).values
            max_d = torch.where(torch.isinf(max_d), min_b, max_d)  # fallback

            # Grid for the current dimension (B, n_bins)
            steps = torch.linspace(
                0, 1, n_bins, device=b.device).view(1, n_bins)
            dim_grid = min_b + (max_d - min_b) * steps
            grids.append(dim_grid.unsqueeze(1))

        return torch.cat(grids, dim=1)  # (B, max_homology_dim + 1, n_bins)

    @abstractmethod
    def extract_feature_(self, persistence_diagram: torch.Tensor) -> torch.Tensor:
        pass

    def fit_transform(self, x_pd: torch.Tensor) -> torch.Tensor:
        return self.extract_feature_(x_pd)


class HolesNumberFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self, max_homology_dim: int = 2):
        super().__init__(max_homology_dim)

    def extract_feature_(self, persistence_diagram: torch.Tensor) -> torch.Tensor:
        B, _, mask, dims = self._get_diagram_components(persistence_diagram)

        feature = torch.zeros(B, self.max_homology_dim + 1,
                              device=persistence_diagram.device)
        for dim in range(self.max_homology_dim + 1):
            feature[:, dim] = (mask & (dims == dim)).sum(dim=1)

        return feature


class MaxHoleLifeTimeFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self, max_homology_dim: int = 2):
        super().__init__(max_homology_dim)

    def extract_feature_(self, persistence_diagram: torch.Tensor) -> torch.Tensor:
        B, lifetimes, mask, dims = self._get_diagram_components(
            persistence_diagram)

        feature = torch.zeros(B, self.max_homology_dim + 1,
                              device=persistence_diagram.device)
        for dim in range(self.max_homology_dim + 1):
            dim_mask = mask & (dims == dim)
            dim_lifetimes = torch.where(
                dim_mask, lifetimes, torch.zeros_like(lifetimes))
            feature[:, dim] = dim_lifetimes.max(dim=1).values

        return feature


class RelevantHolesNumber(PersistenceDiagramFeatureExtractor):
    def __init__(self, max_homology_dim: int = 2, ratio: float = 0.7):
        super().__init__(max_homology_dim)
        self.ratio_ = ratio

    def extract_feature_(self, persistence_diagram: torch.Tensor) -> torch.Tensor:
        B, lifetimes, mask, dims = self._get_diagram_components(
            persistence_diagram)

        max_lifetimes = torch.zeros(
            B, self.max_homology_dim + 1, device=persistence_diagram.device)
        for dim in range(self.max_homology_dim + 1):
            dim_mask = mask & (dims == dim)
            dim_lifetimes = torch.where(
                dim_mask, lifetimes, torch.zeros_like(lifetimes))
            max_lifetimes[:, dim] = dim_lifetimes.max(dim=1).values

        feature = torch.zeros(B, self.max_homology_dim + 1,
                              device=persistence_diagram.device)
        for dim in range(self.max_homology_dim + 1):
            dim_mask = mask & (dims == dim)
            target_lifetimes = self.ratio_ * max_lifetimes[:, dim].unsqueeze(1)

            relevant_mask = dim_mask & (lifetimes >= target_lifetimes)
            feature[:, dim] = relevant_mask.sum(dim=1)

        return feature


class AverageHoleLifetimeFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self, max_homology_dim: int = 2):
        super().__init__(max_homology_dim)

    def extract_feature_(self, persistence_diagram: torch.Tensor) -> torch.Tensor:
        B, lifetimes, mask, dims = self._get_diagram_components(
            persistence_diagram)

        feature = torch.zeros(B, self.max_homology_dim + 1,
                              device=persistence_diagram.device)
        for dim in range(self.max_homology_dim + 1):
            dim_mask = mask & (dims == dim)
            dim_lifetimes = torch.where(
                dim_mask, lifetimes, torch.zeros_like(lifetimes))

            sums = dim_lifetimes.sum(dim=1)
            counts = dim_mask.sum(dim=1).float()

            feature[:, dim] = torch.where(
                counts > 0, sums / counts, torch.zeros_like(sums))

        return feature


class SumHoleLifetimeFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self, max_homology_dim: int = 2):
        super().__init__(max_homology_dim)

    def extract_feature_(self, persistence_diagram: torch.Tensor) -> torch.Tensor:
        B, lifetimes, mask, dims = self._get_diagram_components(
            persistence_diagram)

        feature = torch.zeros(B, self.max_homology_dim + 1,
                              device=persistence_diagram.device)
        for dim in range(self.max_homology_dim + 1):
            dim_mask = mask & (dims == dim)
            dim_lifetimes = torch.where(
                dim_mask, lifetimes, torch.zeros_like(lifetimes))
            feature[:, dim] = dim_lifetimes.sum(dim=1)

        return feature


class PersistenceEntropyFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self, max_homology_dim: int = 2):
        super().__init__(max_homology_dim)

    def extract_feature_(self, persistence_diagram: torch.Tensor) -> torch.Tensor:
        B, lifetimes, mask, dims = self._get_diagram_components(
            persistence_diagram)

        feature = torch.zeros(B, self.max_homology_dim + 1,
                              device=persistence_diagram.device)
        for dim in range(self.max_homology_dim + 1):
            dim_mask = mask & (dims == dim)
            dim_lifetimes = torch.where(
                dim_mask, lifetimes, torch.zeros_like(lifetimes))

            lifetimes_sum = dim_lifetimes.sum(dim=1, keepdim=True)

            p = torch.where(lifetimes_sum > 0, dim_lifetimes /
                            lifetimes_sum, torch.zeros_like(dim_lifetimes))

            safe_log_p = torch.where(p > 0, torch.log(p), torch.zeros_like(p))
            entropy = -(p * safe_log_p).sum(dim=1)

            feature[:, dim] = entropy

        return feature


class SimultaneousAliveHolesFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self, max_homology_dim: int = 2):
        super().__init__(max_homology_dim)

    def extract_feature_(self, persistence_diagram: torch.Tensor) -> torch.Tensor:
        B, lifetimes, mask, dims = self._get_diagram_components(
            persistence_diagram)

        b = persistence_diagram[:, :, 0]
        d = persistence_diagram[:, :, 1]

        max_b = torch.max(b.unsqueeze(2), b.unsqueeze(1))
        min_d = torch.min(d.unsqueeze(2), d.unsqueeze(1))
        overlap_matrix = max_b <= min_d

        feature = torch.zeros(B, self.max_homology_dim + 1,
                              device=persistence_diagram.device)
        for dim in range(self.max_homology_dim + 1):
            dim_mask = mask & (dims == dim)  # (B, K)

            valid_overlaps = overlap_matrix & dim_mask.unsqueeze(
                2) & dim_mask.unsqueeze(1)
            total_overlap = valid_overlaps.sum(dim=(1, 2)).float()
            N = dim_mask.sum(dim=1).float()

            feature[:, dim] = torch.where(
                N > 0,
                (total_overlap + N) / (2 * N),
                torch.zeros_like(N)
            )

        return feature


class AveragePersistenceLandscapeFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self, max_homology_dim: int = 2, n_bins: int = 100):
        super().__init__(max_homology_dim)
        self.n_bins = n_bins

    def extract_feature_(self, persistence_diagram: torch.Tensor) -> torch.Tensor:
        B, _, mask, dims = self._get_diagram_components(persistence_diagram)
        b = persistence_diagram[:, :, 0]
        d = persistence_diagram[:, :, 1]
        grid = self._get_gtda_grid(
            b, d, mask, dims, self.n_bins)  # (B, D, n_bins)

        feature = torch.zeros(B, self.max_homology_dim + 1,
                              device=persistence_diagram.device)
        for dim in range(self.max_homology_dim + 1):
            dim_mask = mask & (dims == dim)
            dim_grid = grid[:, dim, :].unsqueeze(1)  # (B, 1, n_bins)

            L = torch.clamp(torch.min(dim_grid - b.unsqueeze(2),
                            d.unsqueeze(2) - dim_grid), min=0.0)
            L_dim = L * dim_mask.unsqueeze(2).float()

            lambda_1 = L_dim.max(dim=1).values
            feature[:, dim] = lambda_1.mean(dim=1)

        return feature


class BettiNumbersSumFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self, max_homology_dim: int = 2, n_bins: int = 100):
        super().__init__(max_homology_dim)
        self.n_bins = n_bins

    def extract_feature_(self, persistence_diagram: torch.Tensor) -> torch.Tensor:
        B, _, mask, dims = self._get_diagram_components(persistence_diagram)
        b = persistence_diagram[:, :, 0]
        d = persistence_diagram[:, :, 1]
        grid = self._get_gtda_grid(b, d, mask, dims, self.n_bins)

        feature = torch.zeros(B, self.max_homology_dim + 1,
                              device=persistence_diagram.device)
        for dim in range(self.max_homology_dim + 1):
            dim_mask = mask & (dims == dim)
            dim_grid = grid[:, dim, :].unsqueeze(1)

            alive_dim = (b.unsqueeze(2) <= dim_grid) & (
                dim_grid < d.unsqueeze(2)) & dim_mask.unsqueeze(2)

            betti_curve = alive_dim.sum(dim=1).float()
            feature[:, dim] = betti_curve.sum(dim=1)

        return feature


class RadiusAtMaxBNFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self, max_homology_dim: int = 2, n_bins: int = 100):
        super().__init__(max_homology_dim)
        self.n_bins = n_bins

    def extract_feature_(self, persistence_diagram: torch.Tensor) -> torch.Tensor:
        B, _, mask, dims = self._get_diagram_components(persistence_diagram)
        b = persistence_diagram[:, :, 0]
        d = persistence_diagram[:, :, 1]
        grid = self._get_gtda_grid(b, d, mask, dims, self.n_bins)

        feature = torch.zeros(B, self.max_homology_dim + 1,
                              device=persistence_diagram.device)
        for dim in range(self.max_homology_dim + 1):
            dim_mask = mask & (dims == dim)
            dim_grid = grid[:, dim, :].unsqueeze(1)

            alive_dim = (b.unsqueeze(2) <= dim_grid) & (
                dim_grid < d.unsqueeze(2)) & dim_mask.unsqueeze(2)

            betti_curve = alive_dim.sum(dim=1).float()
            max_idx = betti_curve.argmax(dim=1).float()

            feature[:, dim] = max_idx / \
                (self.n_bins * (self.max_homology_dim + 1))

        return feature
