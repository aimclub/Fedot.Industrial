from __future__ import division, print_function

import torch


class RecurrenceFeatureExtractorTorch:
    def __init__(self, recurrence_matrix: torch.Tensor = None):
        self.recurrence_matrix = recurrence_matrix.float()

    def quantification_analysis(self, MDL: int = 3, MVL: int = 3, MWVL: int = 2):
        n_vectors = self.recurrence_matrix.shape[0]
        recurrence_rate = torch.sum(self.recurrence_matrix) / (n_vectors ** 2)

        diagonal_frequency_dist = self.calculate_diagonal_frequency(n_vectors)
        vertical_frequency_dist = self.calculate_vertical_frequency(n_vectors, not_white=1)
        white_vertical_frequency_dist = self.calculate_vertical_frequency(n_vectors, not_white=0)

        determinism = self.laminarity_or_determinism(MDL, n_vectors, diagonal_frequency_dist, lam=False)
        laminarity = self.laminarity_or_determinism(MVL, n_vectors, vertical_frequency_dist, lam=True)

        average_diagonal_line_length = self.average_line_length(MDL, n_vectors, diagonal_frequency_dist)
        average_vertical_line_length = self.average_line_length(MVL, n_vectors, vertical_frequency_dist)
        average_white_vertical_line_length = self.average_line_length(MWVL, n_vectors, white_vertical_frequency_dist)

        longest_diagonal_line_length = self.longest_line_length(diagonal_frequency_dist, n_vectors)
        longest_vertical_line_length = self.longest_line_length(vertical_frequency_dist, n_vectors)
        longest_white_vertical_line_length = self.longest_line_length(white_vertical_frequency_dist, n_vectors)

        entropy_diagonal_lines = self.entropy_lines(MDL, n_vectors, diagonal_frequency_dist)
        entropy_vertical_lines = self.entropy_lines(MVL, n_vectors, vertical_frequency_dist)
        entropy_white_vertical_lines = self.entropy_lines(MWVL, n_vectors, white_vertical_frequency_dist)

        return {
            'RR': recurrence_rate,
            'DET': determinism,
            'ADLL': average_diagonal_line_length,
            'LDLL': longest_diagonal_line_length,
            'DIV': (1. / longest_diagonal_line_length) if longest_diagonal_line_length > 0 else 0.,
            'EDL': entropy_diagonal_lines,
            'LAM': laminarity,
            'AVLL': average_vertical_line_length,
            'LVLL': longest_vertical_line_length,
            'EVL': entropy_vertical_lines,
            'AWLL': average_white_vertical_line_length,
            'LWLL': longest_white_vertical_line_length,
            'EWLL': entropy_white_vertical_lines,
            'RDRR': (determinism / recurrence_rate) if recurrence_rate > 0 else 0.,
            'RLD': (laminarity / determinism) if determinism > 0 else 0.,
        }
    
    def calculate_vertical_frequency(self, number_of_vectors, not_white: int):
        vertical_frequency_distribution = torch.zeros(number_of_vectors + 1, 
                                                      device=self.recurrence_matrix.device)
        m = (self.recurrence_matrix == not_white).float()
        for i in range(number_of_vectors):
            col = m[i]
            diff = torch.cat([torch.tensor([0.], device=m.device), col, torch.tensor([0.], device=m.device)])
            edges = diff[1:] - diff[:-1]
            lengths = torch.where(edges == -1)[0] - torch.where(edges == 1)[0]
            for l in lengths:
                vertical_frequency_distribution[int(l)] += 1
        return vertical_frequency_distribution

    def calculate_diagonal_frequency(self, number_of_vectors):
        diag_freq = torch.zeros(number_of_vectors + 1, device=self.recurrence_matrix.device)
        m = self.recurrence_matrix.float()
        for i in range(number_of_vectors):
            diag = torch.diagonal(m, offset=-i)
            if len(diag) < 2:
                continue
            diff = torch.cat([torch.tensor([0.], device=m.device), diag, torch.tensor([0.], device=m.device)])
            edges = diff[1:] - diff[:-1]
            lengths = torch.where(edges == -1)[0] - torch.where(edges == 1)[0]
            for l in lengths:
                diag_freq[int(l)] += 1
        for i in range(1, number_of_vectors):
            diag = torch.diagonal(m, offset=i)
            if len(diag) < 2:
                continue
            diff = torch.cat([torch.tensor([0.], device=m.device), diag, torch.tensor([0.], device=m.device)])
            edges = diff[1:] - diff[:-1]
            lengths = torch.where(edges == -1)[0] - torch.where(edges == 1)[0]
            for l in lengths:
                diag_freq[int(l)] += 1
        return diag_freq
    
    def entropy_lines(self, factor, number_of_vectors, distribution:torch.Tensor):
        sum_frequency_distribution = torch.sum(distribution[factor:])
        if sum_frequency_distribution == 0:
            return 0.
        probs = distribution[factor:number_of_vectors] / sum_frequency_distribution
        mask = probs > 0
        return -torch.sum(probs[mask] * torch.log(probs[mask]))
    
    def laminarity_or_determinism(self, factor, number_of_vectors, distribution, lam: bool):
        if lam:
            number_of_vectors = number_of_vectors + 1
        idx = torch.arange(1, number_of_vectors, device=distribution.device, dtype=distribution.dtype)
        numerator = torch.sum(idx[factor - 1:] * distribution[factor:number_of_vectors])
        denominator = torch.sum(idx * distribution[1:number_of_vectors])
        return numerator / denominator

    def longest_line_length(self, frequency_distribution: torch.Tensor, number_of_vectors):
        lines = torch.where(frequency_distribution > 0)[0]
        if len(lines) != 0:
            return lines[-1]
        return 1.

    def average_line_length(self, factor, number_of_vectors, distribution: torch.Tensor):
        i = torch.arange(number_of_vectors + 1, device=distribution.device)
        num = torch.sum(i[factor:] * distribution[factor:])
        den = torch.sum(distribution[factor:])
        return num / den if den > 0 else 0.
