"""
Code for the project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi
"""

import pandas as pd
from scipy.stats import binom, norm, multivariate_normal
import numpy as np

def simulate_uplift_dir_norm(a, size, var_p, var_u,
                             use_churn_convention=True, random_state=None):
    """Create simulated data suitable to evaluate uplift approaches.
    It is based on the bivariate beta distribution described in
    Ingram Olkin and Thomas A Trikalinos. Constructions for a bivariate
    beta distribution. Statistics & Probability Letters, 96:54–60, 2015.
    
    A normal noise is added to the scores S_0 and S_1.
    
    Parameters:
    a: an array of size 4 of the Dirichlet parameters
    size: the number of samples to generate
    var_p: the variance of the predictive approach (S_0)
    var_u: the variance of the uplift approach (S_0 - S_1)
    use_churn_convention: if True, the uplift is S_0-S_1, otherwise it
        is S_1-S_0.
    random_state: seed for the random device.
    output: a dict containing all the generated data.
    """
    rng = np.random.default_rng(seed=random_state)
    mu = rng.dirichlet(a, size=size)
    S_0 = mu[:, 1] + mu[:, 3]
    S_1 = mu[:, 2] + mu[:, 3]
    uplift = S_0 - S_1
    
    # Estimators
    S_0_hat = rng.normal(loc = S_0, scale = np.sqrt(var_p))
    S_1_hat = rng.normal(loc = S_1, scale = np.sqrt(var_p))
    uplift_hat = rng.normal(loc = uplift, scale = np.sqrt(var_u))
    
    if not use_churn_convention:
        uplift = -uplift
        uplift_hat = -uplift_hat
        
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


def simulate_uplift_norm(N, mu, Sigma, lambda_0, lambda_1, eta_0, eta_1, var_0, var_1):
    """Create simulated data suitable to evaluate uplift approaches.
    It uses features sampled from a multivariate normal distribution,
    and the outcomes Y_0 and Y_1 are determined by whether a given
    linear combination of the features is above a certain threshold.
    A normal noise is added to the scores S_0 and S_1.
    
    Parameters:
    N: the number of samples to generate
    mu: the mean vector of the features
    Sigma: the covariance matrix of the features
    lambda_0, lambda_1: the coefficients of the linear combinations
        used to determine Y_0 and Y_1
    eta_0, eta_1: the thresholds used to determine Y_0 and Y_1
    var_0, var_1: the variance of the noisy estimators of S_0 and S_1
    output: a dict containing all the generated data.
    """
    # Generate the data
    X = multivariate_normal.rvs(mean=mu, cov=Sigma, size=N)
    noise = norm.rvs(size=N)
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

    S_0_hat = norm.rvs(loc = S_0, scale = np.sqrt(var_0))
    S_1_hat = norm.rvs(loc = S_1, scale = np.sqrt(var_1))
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


def simulate_uplift_binom(a, size, n_p_0, n_u_0, n_u_1,
                          use_churn_convention=True, random_state=None):
    """Create simulated data suitable to evaluate uplift approaches.
    It is based on the bivariate beta distribution described in
    Ingram Olkin and Thomas A Trikalinos. Constructions for a bivariate
    beta distribution. Statistics & Probability Letters, 96:54–60, 2015.
    
    The noisy estimators of S_0 and S_1 are computed from binomial
    distributions with probability S_0 and S_1, then divided by its
    n parameter.
    
    Parameters:
    a: an array of size 4 of the Dirichlet parameters
    size: the number of samples to generate
    n_p_0: The noise parameter for S_0_hat for the predictive approach
    n_u_0: The noise parameter for S_0_hat for the uplift approach
    n_u_1: The noise parameter for S_1_hat for the uplift approach
    use_churn_convention: if True, the uplift is S_0-S_1, otherwise it
        is S_1-S_0.
    random_state: seed for the random device.
    output: a dict containing all the generated data.
    """
    rng = np.random.default_rng(seed=random_state)
    mu = rng.dirichlet(a, size=size)
    S_0 = mu[:, 1] + mu[:, 3]
    S_1 = mu[:, 2] + mu[:, 3]
    uplift = S_0 - S_1
    
    # Estimators
    S_0_hat = rng.binomial(n_p_0, S_0) / n_p_0
    S_1_hat = rng.binomial(n_u_1, S_1) / n_u_1
    S_0_u = rng.binomial(n_u_0, S_0) / n_u_0
    S_1_u = rng.binomial(n_u_1, S_1) / n_u_1
    uplift_hat = S_0_u - S_1_u
    
    if not use_churn_convention:
        uplift = -uplift
        uplift_hat = -uplift_hat
        
    return pd.DataFrame.from_dict({
        "alpha": mu[:, 0],
        "beta" : mu[:, 1],
        "gamma": mu[:, 2],
        "delta": mu[:, 3],
        "S_0": S_0,
        "S_1": S_1,
        "S_0_hat": S_0_hat,
        "S_1_hat": S_1_hat,
        "S_0_u": S_0_u,
        "S_1_u": S_1_u,
        "uplift": uplift,
        "uplift_hat": uplift_hat
    })

def zero_add_espilon(x):
    return np.clip(x, 1e-6, 1 - 1e-6)


def simulate_uplift_beta(a, size, l_0, l_1, use_churn_convention=True, random_state=None):
    """Create simulated data suitable to evaluate uplift approaches.
    It is based on the bivariate beta distribution described in
    Ingram Olkin and Thomas A Trikalinos. Constructions for a bivariate
    beta distribution. Statistics & Probability Letters, 96:54–60, 2015.
    
    The noisy estimators of S_0 and S_1 are computed from beta
    distributions with means S_0 and S_1, and scale parameters l_0 and
    l_1.
    
    Parameters:
    a: an array of size 4 of the Dirichlet parameters
    size: the number of samples to generate
    l_0: The noise parameter for S_0_hat, or "inf" for no noise
    l_1: The noise parameter for S_1_hat, or "inf" for no noise
    use_churn_convention: if True, the uplift is S_0 - S_1, otherwise it
        is S_1 - S_0.
    random_state: seed for the random device.
    output: a dataframe containing all the generated data.
    """
    rng = np.random.default_rng(seed=random_state)
    mu = rng.dirichlet(a, size=size)
    S_0 = mu[:, 1] + mu[:, 3]
    S_1 = mu[:, 2] + mu[:, 3]
    
    # Estimators
    if l_0 == "inf":
        S_0_hat = S_0.copy()
    else:
        S_0_eps = zero_add_espilon(S_0)
        S_0_hat = rng.beta(l_0 * S_0_eps, l_0 * (1 - S_0_eps))
        
    if l_1 == "inf":
        S_1_hat = S_1.copy()
    else:
        S_1_eps = zero_add_espilon(S_1)
        S_1_hat = rng.beta(l_1 * S_1_eps, l_1 * (1 - S_1_eps))
        
    uplift = S_1 - S_0
    uplift_hat = S_1_hat - S_0_hat
    if use_churn_convention:
        uplift = -uplift
        uplift_hat = -uplift_hat
        
    return pd.DataFrame.from_dict({
        "alpha": mu[:, 0],
        "beta" : mu[:, 1],
        "gamma": mu[:, 2],
        "delta": mu[:, 3],
        "S_0": S_0,
        "S_1": S_1,
        "uplift": uplift,
        "S_0_hat": S_0_hat,
        "S_1_hat": S_1_hat,
        "uplift_hat": uplift_hat
    })