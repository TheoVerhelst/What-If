"""
Code for the project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi

This file contains various functions related to a bivariate beta
distribution, from the paper
Ingram Olkin and Thomas A Trikalinos. Constructions for a bivariate beta
distribution. Statistics & Probability Letters, 96:54–60, 2015.
"""

import numpy as np
import scipy as sc
from scipy import integrate as sci
from joblib import Parallel, delayed
import pandas as pd
import mpmath as mp
import whatif.cf.math as wim


def moment_n_to_rs(n):
    """Gives the exponents of the moment S_0^r * S_1^s, given the index
    of the moment. For example, the first moments are
    n = 0 -> r = 0, s = 0
    n = 1 -> r = 1, s = 0
    n = 2 -> r = 0, s = 1
    n = 3 -> r = 2, s = 0
    n = 4 -> r = 1, s = 1
    ...
    """
    order = 0
    r = 0
    for t in range(n):
        r -= 1
        if r < 0:
            order += 1
            r = order
    return r, order - r


def moment_rs_to_n(r, s):
    """Inverse of the function moment_n_to_rs."""
    return (r+s) * (r+s+1) // 2 + s


def compute_sample_moments(S_0, S_1, moment_indices, w=None):
    """Compute the sample moments of a bivariate distribution from
    samples (S_0, S_1, w), where S_0 and S_1 are the two components of
    the distribution, and w is the vector of sample weights.
    """
    moments = []
    if w is None:
        w = np.ones(S_0.shape)
    w /= np.sum(w)
    for n in moment_indices:
        r, s = moment_n_to_rs(n)
        moments.append(np.dot(S_0**r * S_1**s, w))
    return np.array(moments)


def compute_exact_moments(dis, moment_indices):
    moments = []
    for n in moment_indices:
        r, s = moment_n_to_rs(n)
        moments.append(dis.exact_moment(r, s))
    return np.array(moments)


def compute_full_jacobian(dis, moment_indices):
    jacobian = []
    for n in moment_indices:
        r, s = moment_n_to_rs(n)
        jacobian.append(dis.jacobian(r, s))
    return np.vstack(jacobian)


def fit_error(dis, moment_indices, sample_moments):
        """Returns the function to optimize when fitting the bivariate
        beta function to data. Note that we use scipy.optimize.least_squares,
        which takes care of squaring the result of this function.
        """
        return compute_exact_moments(dis, moment_indices) - sample_moments


def integrate(fun_sc, fun_mp, a, b, args=()):
    res = sci.quad(lambda x: fun_sc(x, *args), a, b)[0]
    if not np.isfinite(res):
        res = mp.quad(lambda x: fun_mp(x, *args), [a, b])
    return res


def double_integrate(fun_sc, fun_mp, a, b, c, d, args=()):
    res = sci.dblquad(lambda y, x: fun_sc(x, y, *args), a, b, c, d)[0]
    if not np.isfinite(res):
        res = mp.quad(lambda x, y: fun_mp(x, y, *args), [a, b], [c, d], verbose=True)
    return res


def mpmath_to_float(x, tol=1e-20):
    """Converts a numpy array of mpmath numbers to a float array."""
    return np.vectorize(mp.re, otypes=[float])(x).astype("float")


def log_likelihood(S_0, S_1, model, show_progress=False):
    return np.sum(model.log_pdf(S_0, S_1, show_progress))


def estimate_cf(dis, S_0, S_1, show_progress=False, parallel=False):
    # At the boundaries of the domain of (S_0, S_1), the exact value of
    # the counterfactuals is known, no need to integrate over the
    # probability distribution. This function returns the values in
    # these cases, and a mask indicating the remaining values inside the
    # domain.
    mask_00 = S_0 == 0
    mask_01 = S_0 == 1
    mask_10 = S_1 == 0
    mask_11 = S_1 == 1
    mask_inside = ~(mask_00 | mask_01 | mask_10 | mask_11)
    zeros = np.zeros(S_0.shape)
    res = np.empty((S_0.shape[0], 4))
    res[mask_00] = np.array((1 - S_1[mask_00], zeros[mask_00],
                             S_1[mask_00],     zeros[mask_00])).T
    res[mask_01] = np.array((zeros[mask_01],   1 - S_1[mask_01],
                             zeros[mask_01],   S_1[mask_01])).T
    res[mask_10] = np.array((1 - S_0[mask_10], S_0[mask_10],
                             zeros[mask_10],   zeros[mask_10])).T
    res[mask_11] = np.array((zeros[mask_11],   zeros[mask_11],
                             1 - S_0[mask_11], S_0[mask_11])).T

    if len(mask_inside) > 0:
        S_0_inside = S_0[mask_inside]
        if isinstance(S_0_inside, pd.Series):
            S_0_inside = S_0_inside.reset_index(drop=True)

        S_1_inside = S_1[mask_inside]
        if isinstance(S_1_inside, pd.Series):
            S_1_inside = S_1_inside.reset_index(drop=True)

        log_p = dis.log_pdf(S_0_inside, S_1_inside, show_progress)

        models, factors = dis.shifted_mu()
        if parallel:
            log_p_m = np.column_stack(Parallel(n_jobs=4)(
                delayed(models[i].log_pdf)(
                    S_0_inside, S_1_inside, show_progress
                ) for i in range(4)
            ))
        else:
            log_p_m = np.column_stack([
                models[i].log_pdf(
                    S_0_inside, S_1_inside, show_progress
                ) for i in range(4)
            ])
        res[mask_inside] = np.exp(log_p_m - log_p[:,None]) * factors[None,:]
    return res


def unbiased_moments_binomial_noise(S_0, S_1, n_0, n_1, n_moments):
    """Same as unbiased_moments_beta_noise, but assuming the noisy
    scores are distributed as Binomial(n_t, S_t) / n_t.
    """
    moment_indices = list(range(n_moments))
    sample_moments = compute_sample_moments(S_0, S_1, moment_indices)
    moments = np.zeros((n_moments))
    for n in range(n_moments):
        r, s = moment_n_to_rs(n)
        moments[n] = n_0**r * n_1**s * sample_moments[n]
        for p in range(r + 1):
            for q in range (s + 1):
                if p < r or q < s:
                    moments[n] -= wim.stirling_2(r, p) * wim.stirling_2(s, q) \
                            * wim.falling_factorial(n_0, p) \
                            * wim.falling_factorial(n_1, q) \
                            * moments[moment_rs_to_n(p, q)]
        moments[n] /= wim.falling_factorial(n_0, r) * wim.falling_factorial(n_1, s)
    return moments
