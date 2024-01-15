"""
Code for the project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi
"""

from math import ceil
from scipy.special import gamma, loggamma
import numpy as np

def log_multi_beta(m):
    """Computes the natural logarithm of the generalized beta function."""
    return np.sum(loggamma(m)) - loggamma(np.sum(m))

def multi_beta(m):
    """Computes the generalized beta function, see
    https://en.wikipedia.org/wiki/Beta_function#Multivariate_beta_function

    .. math::

        B(m) = \frac{\prod_{i=1}^n\Gamma(m_i)}{\Gamma\left(\sum_{i=1}^n m_i\right)}
    """
    # Use log functions to avoid float overflows
    return np.exp(log_multi_beta(m))

def harmonic_partial_sum(x, n):
    """Computes the sum for i from 1 to n of 1 / (x + i - 1):

    .. math::

        \sum_{i=1}^n\frac 1{x + i - 1}
    """
    return np.sum(1 / (x + np.arange(n)))

def all_harmonic_partial_sums(x, n):
    return np.hstack((0, np.cumsum(1 / (x + np.arange(n)))))

def rising_factorial(x, n):
    """Computes the product :math:`x(x+1)...(x+n-1)`."""
    return np.prod(x + np.arange(n))

def all_rising_factorials(x, n):
    res = [1]
    for i in range(n):
        res.append(res[-1] * (x + i))
    return np.array(res)

def falling_factorial(x, n):
    """Computes the product :math:`x(x-1)...(x-n+1)`."""
    return np.prod(x - np.arange(n))

def double_factorial(n):
    """Compute the product :math:`n(n-2)(n-4)...` ending at 1 if n is odd, or
    two if n is even.
    """
    return np.prod(n - 2 * np.arange(ceil(n / 2)))

def stirling_1(n, k):
    """Stirling number of the first kind.
    https://en.wikipedia.org/wiki/Stirling_numbers_of_the_first_kind
    """
    if n == k >= 0:
        return 1
    elif n > 0 and k == 0:
        return 0
    elif k > n:
        return 0
    else:
        return (n-1) * stirling_1(n - 1, k) + stirling_1(n - 1, k - 1)


def stirling_2(n, k):
    """Stirling number of the second kind.
    https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind
    """
    if n == k >= 0:
        return 1
    elif n > 0 and k == 0:
        return 0
    elif k > n:
        return 0
    else:
        return k * stirling_2(n - 1, k) + stirling_2(n - 1, k - 1)
