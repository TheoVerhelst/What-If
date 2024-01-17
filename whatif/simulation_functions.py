"""
Code for the project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi
"""

import pandas as pd
from scipy.stats import binom, norm, multivariate_normal
import numpy as np


def add_norm_noise(rng, S_0, S_1, var_0, var_1):
    S_0_hat = rng.normal(loc = S_0, scale = np.sqrt(var_p))
    S_1_hat = rng.normal(loc = S_1, scale = np.sqrt(var_p))
    return S_0_hat, S_1_hat


def zero_add_espilon(x):
    return np.clip(x, 1e-6, 1 - 1e-6)


def add_beta_noise(rng, S_0, S_1, l_0, l_1):
    S_0_eps = zero_add_espilon(S_0)
    S_0_hat = rng.beta(l_0 * S_0_eps, l_0 * (1 - S_0_eps))
    S_1_eps = zero_add_espilon(S_1)
    S_1_hat = rng.beta(l_1 * S_1_eps, l_1 * (1 - S_1_eps))
    return S_0_hat, S_1_hat


def add_binom_noise(rng, S_0, S_1, n_0, n_1):
    S_0_hat = rng.binomial(n_0, S_0) / n_0
    S_1_hat = rng.binomial(n_1, S_1) / n_1
    return S_0_hat, S_1_hat


def simulate_uplift_bb(a, size, use_churn_convention=True, random_state=None,
                        noise=None, var_0=None, var_1=None, l_0=None, l_1=None,
                        n_0=None, n_1=None):
    """Create simulated data suitable to evaluate uplift approaches.
    It is based on the bivariate beta distribution described in
    Ingram Olkin and Thomas A Trikalinos. Constructions for a bivariate
    beta distribution. Statistics & Probability Letters, 96:54–60, 2015.

    We return noisy estimates S_0_hat, S_1_hat of S_0, S_1, where the noise is
    sampled depending on the value of `noise`. If `noise == "norm"`, we use

    .. math::
        \widehat S_t \sim \mathcal N(S_t, \mathrm{var}_t),

    if `noise == "beta"`, we use

    .. math::
        \widehat S_t \sim \mathrm{Beta}(S_t l_t, (1 - S_t) l_t),

    and if `noise == "binom"`, we use

    .. math::
        \widehat S_t \sim \\frac 1{n_t} \mathrm{Bin}(S_t, n_t)

    Parameters
    ----------
    a:
        an array of size 4 of the Dirichlet parameters

    size:
        the number of samples to generate

    use_churn_convention:
        if True, the uplift is S_0-S_1, otherwise it
        is S_1-S_0.

    random_state:
        seed for the random device.

    noise:
        type of noise to add to S_0 and S_1, one of None, "norm", "beta", or "binom"

    var_0:
        the variance of the Gaussian noise on S_0, ignored if noise != "norm"

    var_1:
        the variance of the Gaussian noise on S_1, ignored if noise != "norm"

    l_0:
        the scale parameter of the beta noise on S_0, ignored if noise != "beta"

    l_1:
        the scale parameter of the beta noise on S_1, ignored if noise != "beta"

    n_0:
        the number of repetition of the binomial noise on S_0, ignored if noise != "binom"

    n_1:
        the number of repetition of the binomial noise on S_1, ignored if noise != "binom"

    Returns
    -------
    res: dict
        A dict containing all the generated data.
    """
    rng = np.random.default_rng(seed=random_state)
    mu = rng.dirichlet(a, size=size)
    S_0 = mu[:, 1] + mu[:, 3]
    S_1 = mu[:, 2] + mu[:, 3]

    # Estimators
    if noise == "norm":
        assert var_0 is not None and var_1 is not None
        S_0_hat, S_1_hat = add_norm_noise(rng, S_0, S_1, var_0, var_1)
    elif noise == "beta":
        assert l_0 is not None and l_1 is not None
        S_0_hat, S_1_hat = add_beta_noise(rng, S_0, S_1, l_0, l_1)
    elif noise == "binom":
        assert n_0 is not None and n_1 is not None
        S_0_hat, S_1_hat = add_binom_noise(rng, S_0, S_1, n_0, n_1)
    else:
        S_0_hat, S_1_hat = S_0.copy(), S_1.copy()

    if use_churn_convention:
        uplift = S_0 - S_1
        uplift_hat = S_0_hat - S_1_hat
    else:
        uplift = S_1 - S_0
        uplift_hat = S_1_hat - S_0_hat

    return pd.DataFrame.from_dict({
        "alpha": mu[:, 0],
        "beta" : mu[:, 1],
        "gamma": mu[:, 2],
        "delta": mu[:, 3],
        "S_0": S_0,
        "S_1": S_1,
        "S_0_hat": S_0_hat,
        "S_1_hat": S_1_hat,
        "uplift": uplift,
        "uplift_hat": uplift_hat
    })


def simulate_uplift_norm(N, mu, Sigma, lambda_0, lambda_1, eta_0, eta_1, var_0, var_1, random_state=None):
    """Create simulated data suitable to evaluate uplift approaches.
    It uses features sampled from a multivariate normal distribution,
    and the outcomes Y_0 and Y_1 are determined by whether a given
    linear combination of the features is above a certain threshold.
    A normal noise is added to the scores S_0 and S_1.

    Parameters
    ----------

    N:
        the number of samples to generate

    mu:
        the mean vector of the features

    Sigma:
        the covariance matrix of the features

    lambda_0, lambda_1:
        the coefficients of the linear combinations
        used to determine Y_0 and Y_1

    eta_0, eta_1:
        the thresholds used to determine Y_0 and Y_1

    var_0, var_1:
        the variance of the noisy estimators of S_0 and S_1

    Returns
    -------
    res: dict
        A dict containing all the generated data.
    """
    # Generate the data
    rng = np.random.default_rng(seed=random_state)
    X = rng.multivariate_normal(mean=mu, cov=Sigma, size=N)
    noise = rng.norm(size=N)
    Q_0 = np.dot(X, lambda_0) + noise
    Q_1 = np.dot(X, lambda_1) + noise

    Y_0 = Q_0 > eta_0
    Y_1 = Q_1 > eta_1
    S_0 = norm.cdf(np.dot(X, lambda_0) - eta_0)
    S_1 = norm.cdf(np.dot(X, lambda_1) - eta_1)

    alpha = norm.cdf(np.minimum(eta_0 - np.dot(X, lambda_0), eta_1 - np.dot(X, lambda_1)))
    delta = norm.cdf(np.minimum(np.dot(X, lambda_0) - eta_0, np.dot(X, lambda_1) - eta_1))
    beta  = np.maximum(0, norm.cdf(eta_1 - np.dot(X, lambda_1)) - norm.cdf(eta_0 - np.dot(X, lambda_0)))
    gamma = np.maximum(0, norm.cdf(eta_0 - np.dot(X, lambda_0)) - norm.cdf(eta_1 - np.dot(X, lambda_1)))

    S_0_hat = rng.norm(loc = S_0, scale = np.sqrt(var_0))
    S_1_hat = rng.norm(loc = S_1, scale = np.sqrt(var_1))
    S_0_hat[S_0_hat < 0] = 0
    S_0_hat[S_0_hat > 1] = 1
    S_1_hat[S_1_hat < 0] = 0
    S_1_hat[S_1_hat > 1] = 1

    return pd.DataFrame.from_dict({
        "noise": noise,
        "alpha": alpha,
        "beta" : beta,
        "gamma": gamma,
        "delta": delta,
        "Y_0": Y_0,
        "Y_1": Y_1,
        "S_0": S_0,
        "S_1": S_1,
        "Q_0": Q_0,
        "Q_1": Q_1,
        "S_0_hat": S_0_hat,
        "S_1_hat": S_1_hat
    })
