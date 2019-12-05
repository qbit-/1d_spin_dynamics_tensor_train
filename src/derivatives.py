"""
This module calculates various derivative quantities from
dynamics
"""
import functools
import numpy as np


def mean_sd(x):
    """
    Get mean and SD of an array along
    the first dimension of x
    Parameters:
    -----------
    x: np.array

    Returns:
    --------
    (mean, std): (float, float)
    """
    return np.mean(x, axis=0), np.std(x, axis=0) / np.sqrt(x.shape[0])


def autocorr(x, t):
    """
    Calculates a discrete autocorrelation on [0, inf)
    x is assumed 1D array-like, t are time steps
    """
    # discrete correlation on (-inf, inf)
    y = np.correlate(x, x, 'full')
    # take only positive part [0, +inf)
    part = int((y.shape[0]+1)/2-1)
    y = y[part:]
    sigma = np.sum(x ** 2) / t[-1]
    y = y / sigma / (t[-1] - t + 1e-10)

    return y


def autocorr_windowed(x, n=1):
    """
    Calculates a discrete autocorrelation on [0, (len(x)-1)/n]
    x is assumed 1D array-like, n is the shrinking factor

    Parameters:
    -----------
    x: 1D np.array
       array to calculate the autocorrelation
    n: int
       Number of chunks to divide the signal
    """
    len_total = len(x)
    len_active = int((len_total - 1) / n)

    raw = np.correlate(x, x, mode='full')
    start = len_total - 1
    stop = len_total + len_active

    return raw[start:stop] / range(len_total, len_total-len_active-1, -1)


def corr_windowed(xy, n=1):
    """
    Calculates a discrete correlation on [0, (len(x)-1)/n]
    xy is assumed 2D array-like and correlation is calculated
    along columns
    n is the shrinking factor

    Parameters:
    -----------
    xy: 1D np.array
       array to calculate the correlation. Should contain x and y
       as columns, e.g. each element of xy is a tuple (x_k, y_k)
    n: int
       Number of chunks to divide the signal
    """
    xy = xy.reshape([-1, 2])
    len_total = xy.shape[0]
    len_active = int((len_total - 1) // n)

    raw = 0.5 * (np.correlate(xy[:, 0], xy[:, 1], mode='full')
                 + np.correlate(xy[:, 1], xy[:, 0], mode='full'))
    start = len_total - 1
    stop = len_total + len_active

    return raw[start:stop] / range(len_total, len_total-len_active-1, -1)


def autocorr_windowed_over_axis(X, n=1, axis_time=1):
    """
    Calculates a discrete autocorrelation on [0, (len(x)-1)/n]
    X is assumed 2D array-like, the autocorrelation is calculated
    for each row n is the shrinking factor

    Parameters:
    -----------
    x: 2D np.array
       array to calculate the autocorrelation
    n: int, default 1
       Number of chunks to divide the signal
    axis: int, default 1
       Axis along which to calculate
    """
    return np.apply_along_axis(
        functools.partial(autocorr_windowed, n=n), axis_time, X)


def corr_windowed_over_axis(X, n=1, axis_time=1, axes_which_data=[0, 1]):
    """
    Calculates a dicrete autocorrelation between two indices in the
    third dimension of the 3D array along a specified axis 1.
    """
    new_shape = X.shape[:axis_time] + (-1,)
    X = X[..., axes_which_data].reshape(new_shape)
    return np.apply_along_axis(
        functools.partial(corr_windowed, n=n), axis_time, X)
