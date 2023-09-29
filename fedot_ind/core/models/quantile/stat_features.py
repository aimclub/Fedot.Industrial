import warnings

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import entropy, linregress
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


def lambda_less_zero(array: np.array) -> int:
    mask = np.array(list(map(lambda x: x < 0.01, array)), dtype=int)
    return np.sum(mask)


def q5(array: np.array) -> float:
    return np.quantile(array, 0.05)


def q25(array: np.array) -> float:
    return np.quantile(array, 0.25)


def q75(array: np.array) -> float:
    return np.quantile(array, 0.75)


def q95(array: np.array) -> float:
    return np.quantile(array, 0.95)


def diff(array: np.array) -> float:
    return np.diff(array, n=len(array) - 1)[0]


# Extra methods for quantile features extraction
def skewness(array: np.array) -> float:
    if not isinstance(array, pd.Series):
        array = pd.Series(array)

    return pd.Series.skew(array)


def kurtosis(array: np.array) -> float:
    if not isinstance(array, pd.Series):
        array = pd.Series(array)
    return pd.Series.kurtosis(array)


def n_peaks(array: np.array) -> int:
    return len(find_peaks(array)[0])


def mean_ptp_distance(array: np.array):
    peaks, _ = find_peaks(array)
    return np.mean(np.diff(peaks))


def slope(array: np.array) -> float:
    return linregress(range(len(array)), array).slope


def ben_corr(x):
    """Useful for anomaly detection applications [1][2]. Returns the correlation from first digit distribution when
     compared to the Newcomb-Benford's Law distribution [3][4].

     Args:
            x (np.array): time series to calculate the feature of

     Returns:
            float: the value of this feature


     .. math::

         P(d)=\\log_{10}\\left(1+\\frac{1}{d}\\right)

     where :math:`P(d)` is the Newcomb-Benford distribution for :math:`d` that is the leading digit of the number
     {1, 2, 3, 4, 5, 6, 7, 8, 9}.

     .. rubric:: References

     |  [1] A Statistical Derivation of the Significant-Digit Law, Theodore P. Hill, Statistical Science, 1995
     |  [2] The significant-digit phenomenon, Theodore P. Hill, The American Mathematical Monthly, 1995
     |  [3] The law of anomalous numbers, Frank Benford, Proceedings of the American philosophical society, 1938
     |  [4] Note on the frequency of use of the different digits in natural numbers, Simon Newcomb, American Journal of
     |  mathematics, 1881

    """
    x = np.asarray(x)

    # retrieve first digit from data
    x = np.array(
        [int(str(np.format_float_scientific(i))[:1]) for i in np.abs(np.nan_to_num(x))]
    )

    # benford distribution
    benford_distribution = np.array([np.log10(1 + 1 / n) for n in range(1, 10)])

    data_distribution = np.array([(x == n).mean() for n in range(1, 10)])

    # np.corrcoef outputs the normalized covariance (correlation) between benford_distribution and data_distribution.
    # In this case returns a 2x2 matrix, the  [0, 1] and [1, 1] are the values between the two arrays
    return np.corrcoef(benford_distribution, data_distribution)[0, 1]


def interquartile_range(array: np.array) -> float:
    return q75(array) - q25(array)


def energy(array: np.array) -> float:
    return np.sum(np.power(array, 2)) / len(array)


def autocorrelation(array: np.array) -> float:
    """Autocorrelation of the time series with its lagged version
    """
    lagged_ts = np.roll(array, 1)
    return np.corrcoef(array, lagged_ts)[0, 1]


def zero_crossing_rate(array: np.array) -> float:
    """Returns the rate of sign-changes of the time series for a scaled version of it.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_array = scaler.fit_transform(array.reshape(-1, 1)).flatten()
    signs = np.sign(scaled_array)
    signs[signs == 0] = -1
    return np.sum((signs[1:] - signs[:-1]) != 0) / len(scaled_array)


def shannon_entropy(array: np.array) -> float:
    """Returns the Shannon Entropy of the time series.
    """
    p = np.unique(array, return_counts=True)[1] / len(array)
    return -np.sum(p * np.log2(p))


def base_entropy(array: np.array) -> float:
    """Returns the Shannon Entropy of the time series.
    """
    # Normalize the time series to sum up to 1
    normalized_series = array / np.sum(array)
    return entropy(normalized_series)

def ptp_amp(array: np.array) -> float:
    """Returns the peak-to-peak amplitude of the time series.
    """
    return np.ptp(array)


def crest_factor(array: np.array) -> float:
    """Returns the crest factor of the time series.
    """
    return np.max(np.abs(array)) / np.sqrt(np.mean(np.square(array)))


def mean_ema(array: np.array) -> float:
    """Returns the exponential moving average of the time series.
    """
    span = int(len(array) / 10)
    if span in [0, 1]:
        span = 2
    return pd.Series(array).ewm(span=span).mean().iloc[-1]


def mean_moving_median(array: np.array) -> float:
    span = int(len(array) / 10)
    if span in [0, 1]:
        span = 2
    return pd.Series(array).rolling(window=span, center=False).median().mean()


def hjorth_mobility(array):
    # Compute the first-order differential sequence
    diff_sequence = np.diff(array)
    # Calculate the mean power of the first-order differential sequence
    M2 = np.sum(np.power(diff_sequence, 2)) / len(diff_sequence)
    # Calculate the total power of the time series
    TP = np.sum(np.power(array, 2)) / len(array)
    # Calculate Hjorth mobility
    mobility = np.sqrt(M2 / TP)
    return mobility


def hjorth_complexity(array):
    # Compute the first-order differential sequence
    diff_sequence = np.diff(array)
    # Calculate the mean power of the first-order differential sequence
    M2 = np.sum(np.power(diff_sequence, 2)) / len(diff_sequence)
    # Calculate the total power of the time series
    TP = np.sum(np.power(array, 2)) / len(array)
    # Calculate the fourth central moment of the first-order differential sequence
    M4 = sum([(diff_sequence[i] - diff_sequence[i - 1]) ** 2 for i in range(1, len(diff_sequence))]) / len(
        diff_sequence)
    # Calculate Hjorth complexity
    complexity = np.sqrt((M4 * TP) / (M2 * M2))
    # complexity = (M4 * TP) / (M2 * M2)
    return complexity


def hurst_exponent(array):
    """ Compute the Hurst Exponent of the time series. The Hurst exponent is used as a measure of long-term memory of
    time series. It relates to the autocorrelations of the time series, and the rate at which these decrease as the
    lag between pairs of values increases. For a stationary time series, the Hurst exponent is equivalent to the
    autocorrelation exponent.

    Args:
        array: Time series

    Returns:
        hurst_exponent: Hurst exponent of the time series

    Notes:
        Author of this function is Xin Liu

    """
    X = np.array(array)
    N = X.size
    T = np.arange(1, N + 1)
    Y = np.cumsum(X)
    Ave_T = Y / T

    S_T = np.zeros(N)
    R_T = np.zeros(N)

    for i in range(N):
        S_T[i] = np.std(X[:i + 1])
        X_T = Y - T * Ave_T[i]
        R_T[i] = np.ptp(X_T[:i + 1])

    R_S = R_T / S_T
    R_S = np.log(R_S)[1:]
    n = np.log(T)[1:]
    A = np.column_stack((n, np.ones(n.size)))
    [m, c] = np.linalg.lstsq(A, R_S,rcond=None)[0]
    H = m
    return H


def pfd(X, D=None):
    """The Petrosian fractal dimension (PFD) is a chaotic algorithm used to calculate EEG signal complexity
    Compute Petrosian Fractal Dimension of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, the first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed using Numpy's difference function.

    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.
    """
    if D is None:
        D = np.diff(X)
        D = D.tolist()
    N_delta = 0  # number of sign changes in derivative of the signal
    for i in range(1, len(D)):
        if D[i] * D[i - 1] < 0:
            N_delta += 1
    n = len(X)
    return np.log10(n) / (
        np.log10(n) + np.log10(n / n + 0.4 * N_delta)
    )
