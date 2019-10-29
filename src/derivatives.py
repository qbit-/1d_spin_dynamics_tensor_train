"""
This module calculates various derivative quantities from
dynamics
"""
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


def get_autocorrelation(x, t):
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


def get_autocorrelation_symm(x, n=1):
    """
    Calculates a discrete autocorrelation on [0, (len(x)-1)/n]
    x is assumed 1D array-like, n is the shrinking factor

    This is an optimized version of the get_autocorrelation_symm_naive
    """
    len_total = len(x)
    len_active = int((len_total - 1) / n)

    raw = np.correlate(x, x, mode='full')
    start = len_total - 1
    stop = len_total + len_active

    return raw[start:stop] / range(len_total, len_total-len_active-1, -1)
