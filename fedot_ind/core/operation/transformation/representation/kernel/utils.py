import numpy as np
from scipy.special import gamma


def mittag_leffler(z, q, beta=1.0, n_terms=100):
    """Вычисление функции Миттаг-Леффлера E_q(z)"""
    result = 0.0

    for k in range(n_terms):
        try:
            term = z ** k / gamma(q * k + beta)
            if np.any(np.isnan(term)) or np.any(np.isinf(term)):
                break
            result += term
        except BaseException:
            break

    return result


def compute_memory_weights(n_points, q):
    """Вычисление весов памяти для n точек"""
    weights = np.array([(n_points - i) ** (-q) / gamma(1 - q)
                        for i in range(n_points)])

    # Обработка особых случаев
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    if weights.sum() > 0:
        weights = weights / weights.sum()

    return weights


def validate_fractional_order(q):
    """Валидация дробного порядка"""
    if not (0 < q <= 1):
        raise ValueError(f"Дробный порядок q должен быть в (0, 1], получено {q}")
    return q
