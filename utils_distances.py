import numpy as np
from aeon.distances import dtw_distance

# from tslearn.metrics import dtw, lcss, soft_dtw
from tslearn.metrics import dtw as dtw_d_tslearn
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def baseline_distance(multivariate_signal_1, multivariate_signal_2):
    """Baseline: distance between the difference in lengths (which is the number of samples)."""
    return np.abs(len(multivariate_signal_2) - len(multivariate_signal_1))


def dtw_i_tslearn(
    X1: np.ndarray,
    X2: np.ndarray,
):
    """
    Multivariate distance with independent strategy, built on the univariate
        (dependent) distance.
    Simply sums over the results of applying the distance separately to each
        dimension.
    The inputs must be multivariate.

    X1 and X2 are of shape (n_timepoints, n_channels), as in tslearn
    """

    if len(X1.shape) <= 1 or len(X2.shape) <= 1:
        raise TypeError("The inputs must be multivariate.")
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("The inputs must have the same dimension.")

    total_d = X1.shape[1]  # dimension
    dtw_cost = 0
    for d in range(total_d):
        dtw_cost += dtw_d_tslearn(
            np.array(X1[:, d]),
            np.array(X2[:, d]),
        )
    return dtw_cost


def multivariate_euclidean(X1, X2):
    return np.linalg.norm(X1 - X2)


def derivate_ts(X):
    """
    For DDTW.

    TODO: look into sktime.transformations.series.difference
    """
    X_deriv = list()
    n = len(X)
    for i in range(1, n - 1):
        elem = X[i] - X[i - 1] + (X[i + 1] - X[i - 1]) / 2
        X_deriv.append(elem)
    return np.array(X_deriv)


def set_weight(i, j, g, L):
    """
    For WDTW: modified logistic weight function (MLWF).
    """
    weight_max = 1
    return weight_max / (1 + np.exp(-g * (abs(i - j) - L) / 2))


def build_weight_matrix(i, j, g, L):
    """
    For WDTW.
    """
    M = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            M[i, j] = set_weight(i, j, g, L)
    return M


def multivariate_dtw_dependent(
    X1: np.ndarray,
    X2: np.ndarray,
    is_weighted: bool = False,
    g: float = None,
    is_derivate: bool = False,
):
    """
    Multivariate DTW with dependent strategy. Included variants:
        - DTW
        - WDTW
        - DDTW
        - WDDTW (first derivated, then weighted)

    Can also be considered as univariate DTW if the inputs are univariate.

    The inputs can be univariate or multivariate.

    TODO:
    - include a window size
    - for WDTW, choose `g` using cross-validation (requires supervised setting)

    Parameters
    ----------
    X1 : array-like of shape (n_samples_1, d)
        First multivariate signal

    X2 : array-like of shape (n_samples_2, d)
        Second multivariate signal

    is_weighted : bool, default=False
        True when the WDTW (Weighted DTW) is called.

    g : float or None, default=None
        For WDTW.

    is_derivate : bool, default=False
        True when the DDTW (Derivate DTW) is called.
    """

    if type(X1) != type(X2):
        raise TypeError("The inputs must have the same type.")
    if is_weighted:
        if g is None:
            raise ValueError("When using WDTW, set `g`.")
        if len(X1) != len(X2):
            raise ValueError("When using WDTW, the inputs must have the same length.")

    if is_derivate:
        X1 = derivate_ts(X1)
        X2 = derivate_ts(X2)

    n_samples_1 = len(X1)
    n_samples_2 = len(X2)
    D = np.zeros((n_samples_1, n_samples_2))
    C = np.zeros((n_samples_1, n_samples_2))

    if is_weighted:
        weight_matrix = np.zeros((n_samples_1, n_samples_1))
        for i in range(n_samples_1):
            for j in range(n_samples_1):
                weight_matrix[i, j] = set_weight(i, j, g, n_samples_1)

    for i in range(n_samples_1):
        for j in range(n_samples_2):
            D[i, j] = np.sum((X1[i] - X2[j]) ** 2)

    C[0, :] = D[0, :]
    C[:, 0] = D[:, 0]

    if not (is_weighted):
        for i in range(1, n_samples_1):
            for j in range(1, n_samples_2):
                C[i, j] = D[i, j] + min(C[i - 1, j - 1], C[i - 1, j], C[i, j - 1])
    else:
        for i in range(1, n_samples_1):
            for j in range(1, n_samples_2):
                C[i, j] = weight_matrix[i, j] * D[i, j] + min(
                    C[i - 1, j - 1], C[i - 1, j], C[i, j - 1]
                )

    dtw_cost = np.sqrt(C[n_samples_1 - 1, n_samples_2 - 1])
    return dtw_cost


def dtw_d(X1, X2):
    """
    Multivariate DTW with dependent strategy.
    """
    dtw_cost = multivariate_dtw_dependent(
        X1=X1, X2=X2, is_weighted=False, g=None, is_derivate=False
    )
    return dtw_cost


def dtwsc_d(X1, X2):
    """
    Multivariate DTW with Sakoe-Chiba constraints with dependent strategy.
    """
    dtw_cost = dtw(
        X1,
        X2,
        global_constraint="sakoe_chiba",
        sakoe_chiba_radius=3,
    )
    return dtw_cost


def ddtw_d(X1, X2):
    """
    Multivariate DDTW with dependent strategy.
    """
    dtw_cost = multivariate_dtw_dependent(
        X1=X1, X2=X2, is_weighted=False, g=None, is_derivate=True
    )
    return dtw_cost


def wdtw_d(X1, X2):
    """
    Multivariate WDTW with dependent strategy.
    """
    dtw_cost = multivariate_dtw_dependent(
        X1=X1, X2=X2, is_weighted=True, g=0.1, is_derivate=False
    )
    return dtw_cost


def wddtw_d(X1, X2):
    """
    Multivariate WDDTW with dependent strategy.
    """
    dtw_cost = multivariate_dtw_dependent(
        X1=X1, X2=X2, is_weighted=True, g=0.1, is_derivate=True
    )
    return dtw_cost


def soft_dtw_d(X1, X2):
    """
    Multivariate Soft-DTW with dependent strategy.
    """
    dtw_cost = soft_dtw(
        X1,
        X2,
    )
    return dtw_cost


def lcss_d(X1, X2):
    """
    LCSS with dependent strategy.
    """
    lcss_cost = lcss(
        X1,
        X2,
    )
    return lcss_cost


def multivariate_distance_independent(
    X1: np.ndarray,
    X2: np.ndarray,
    distance_str: str,
):
    """
    Multivariate distance with independent strategy, built on the univariate
        (dependent) distance.
    Simply sums over the results of applying the distance separately to each
        dimension.
    The inputs must be multivariate.

    Included multivariate dependent distances:
        - DTW
        - DTW with Sakoe-Chiba constraints
        - WDTW
        - DDTW
        - WDDTW (first derivated, then weighted)
        - Soft-DTW
        - LCSS

    Parameters
    ----------
    X1 : array-like of shape (n_samples_1, d)
        First multivariate signal

    X2 : array-like of shape (n_samples_2, d)
        Second multivariate signal

    distance_str : {'DTW', 'DTWSC', 'DDTW', 'WDTW', 'WDDTW', 'Soft-DTW', 'LCSS'}, default='DTW'
        Multivariate DTW variant.
    """

    d_func = {
        "DTW": dtw_d,
        "DTWSC": dtwsc_d,
        "DDTW": ddtw_d,
        "WDTW": wdtw_d,
        "WDDTW": wddtw_d,
        "Soft-DTW": soft_dtw_d,
        "LCSS": lcss_d,
    }

    if len(X1.shape) <= 1 or len(X2.shape) <= 1:
        raise TypeError("The inputs must be multivariate.")
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("The inputs must have the same dimension.")

    total_d = X1.shape[1]  # dimension
    dtw_cost = 0
    for d in range(total_d):
        dtw_cost += d_func[distance_str](
            X1=X1[:, d],
            X2=X2[:, d],
        )
    return dtw_cost


def dtw_i(X1, X2):
    """
    Multivariate DTW with independent strategy.
    """
    dtw_cost = multivariate_distance_independent(
        X1=X1,
        X2=X2,
        distance_str="DTW",
    )
    return dtw_cost


def dtwsc_i(X1, X2):
    """
    Multivariate DTW with Sakoe-Chiba constraints with independent strategy.
    """
    dtw_cost = multivariate_distance_independent(
        X1=X1,
        X2=X2,
        distance_str="DTWSC",
    )
    return dtw_cost


def ddtw_i(X1, X2):
    """
    Multivariate DDTW with independent strategy.
    """
    dtw_cost = multivariate_distance_independent(
        X1=X1,
        X2=X2,
        distance_str="DDTW",
    )
    return dtw_cost


def wdtw_i(X1, X2):
    """
    Multivariate WDTW with independent strategy.
    """
    dtw_cost = multivariate_distance_independent(
        X1=X1,
        X2=X2,
        distance_str="WDTW",
    )
    return dtw_cost


def wddtw_i(X1, X2):
    """
    Multivariate WDDTW with independent strategy.
    """
    dtw_cost = multivariate_distance_independent(
        X1=X1,
        X2=X2,
        distance_str="WDDTW",
    )
    return dtw_cost


def soft_dtw_i(X1, X2):
    dtw_cost = multivariate_distance_independent(
        X1=X2,
        X2=X2,
        distance_str="Soft-DTW",
    )
    return dtw_cost


def lcss_i(X1, X2):
    dtw_cost = multivariate_distance_independent(
        X1=X2,
        X2=X2,
        distance_str="LCSS",
    )
    return dtw_cost


# NEW


def numpy3d_to_listoflists(X):
    """
    Transfrom a 3D numpay array of shape (n_ts, dimension, length) into a list of mulvariate signals.

    A data set of `N` multivariate signals of dimension `d` is:
        a list of `N` multivariate signals
        each multivariate signal being a list of `d` univariate signals
    They are not stored as numpy arrays because the signals are not required to be equal-sized.
    """
    if X.shape[1] > X.shape[2]:
        raise ValueError("The shape of `X` does not seem right!")

    list_of_multivariate_signals = list()
    for i in range(X.shape[0]):
        multivariate_signal = X[i]
        list_of_univariate_signals = list()
        for d in range(X.shape[1]):
            univariate_signal = multivariate_signal[d]
            list_of_univariate_signals.append(univariate_signal)
        list_of_multivariate_signals.append(list_of_univariate_signals)
    return list_of_multivariate_signals


def scale_dataset_of_multivariate_signals(X):
    """
    Scale each dimension of each multivariate signal.
    """

    list_of_scaled_univariate_signals = list()
    for d in range(X.shape[1]):
        # data set of univariate signals (for one dimension)
        univariate_signals = X[:, d, :]
        scaled_univariate_signals = TimeSeriesScalerMeanVariance().fit_transform(
            univariate_signals
        )
        list_of_scaled_univariate_signals.append(scaled_univariate_signals)
    # list of numpy arrays for each dimension
    return list_of_scaled_univariate_signals
