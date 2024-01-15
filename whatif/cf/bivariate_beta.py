"""
Code for the project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi
"""

import numpy as np
import scipy as sc
from scipy.special import comb
from tqdm.autonotebook import trange
from joblib import Parallel, delayed
from whatif.cf.math import all_rising_factorials, all_harmonic_partial_sums, log_multi_beta
from whatif.cf.optimization_helpers import fit_error, compute_full_jacobian, mpmath_to_float, integrate, estimate_cf, compute_sample_moments

class BivariateBeta:
    n_max = 6
    pascal = sc.linalg.pascal(n_max, kind="lower", exact=False)

    def __init__(self, m=None):
        assert (m is None) or (len(m) == 4 and np.all(np.greater(m, 0)))
        if m is not None:
            self.m = np.array(m)
            self.m = np.maximum(1e-6, self.m) # To avoid NaNs
            a, b, c, d = self.m
            A = np.sum(self.m)
            self.b_rising = all_rising_factorials(b, BivariateBeta.n_max)
            self.c_rising = all_rising_factorials(c, BivariateBeta.n_max)
            self.d_rising = all_rising_factorials(d, BivariateBeta.n_max)
            self.A_rising = all_rising_factorials(A, BivariateBeta.n_max)
            self.b_partial = all_harmonic_partial_sums(b, BivariateBeta.n_max)
            self.c_partial = all_harmonic_partial_sums(c, BivariateBeta.n_max)
            self.d_partial = all_harmonic_partial_sums(d, BivariateBeta.n_max)
            self.A_partial = all_harmonic_partial_sums(A, BivariateBeta.n_max)
            self.log_integration_constant = -log_multi_beta(self.m)

    def params(self):
        return self.m

    def exact_moment(self, r, s):
        moment = 0
        for p in range(r + 1):
            for q in range(s + 1):
                C = BivariateBeta.pascal[r, p] * BivariateBeta.pascal[s, q]
                moment +=  C * self.b_rising[r - p] * self.c_rising[s - q] * self.d_rising[p + q]
        return moment / self.A_rising[r + s]

    def jacobian(self, r, s):
        """Computes the Jacobian matrix of the function that returns the
        moment given the parameter vector m. This is used in the
        optimization procedure.
        """
        J = np.zeros(self.m.size)
        for p in range(r + 1):
            for q in range(s + 1):
                partial = BivariateBeta.pascal[r, p] * BivariateBeta.pascal[s, q] \
                            * self.b_rising[r - p] * self.c_rising[s - q] \
                            * self.d_rising[p + q] / self.A_rising[r + s]
                J[0] += partial * (0                     - self.A_partial[r + s])
                J[1] += partial * (self.b_partial[r - p] - self.A_partial[r + s])
                J[2] += partial * (self.c_partial[s - q] - self.A_partial[r + s])
                J[3] += partial * (self.d_partial[p + q] - self.A_partial[r + s])
        return J

    @staticmethod
    def initial_guess(moment_indices, sample_moments):
        """Makes an educated guess for the value of the distribution
        parameters, given the 5 first sample moments.
        """
        assert all(m in moment_indices for m in [1, 2, 3, 5])
        R_10, R_01, R_20, R_02 = (sample_moments[moment_indices.index(m)] for m in [1, 2, 3, 5])
        A_0 = (R_10 - R_20) / (R_20 - R_10**2)
        A_1 = (R_01 - R_02) / (R_02 - R_01**2)
        if A_0 > 0 and A_1 > 0:
            A = (A_0 + A_1) / 2
        elif A_0 > 0:
            A = A_0
        elif A_1 > 0:
            A = A_1
        else:
            A = 1
        a = (1 - R_10) * (1 - R_01)
        b = R_10       * (1 - R_01)
        c = (1 - R_10) * R_01
        d = R_10       * R_01
        return np.array([a, b, c, d]) * A

    def fit(self, S_0, S_1, moment_indices = None, **kwargs):
        if moment_indices is None:
            moment_indices = [1, 2, 3, 5]

        sample_moments = compute_sample_moments(S_0, S_1, moment_indices)
        self.fit_from_moments(moment_indices, sample_moments, **kwargs)

    def fit_from_moments(self, moment_indices, sample_moments, params_init=None, **kwargs):
        if params_init is None:
            params_init = BivariateBeta.initial_guess(moment_indices, sample_moments)

        res = sc.optimize.least_squares(
            lambda m: fit_error(BivariateBeta(m), moment_indices, sample_moments),
            x0=params_init,
            jac=lambda m: compute_full_jacobian(BivariateBeta(m), moment_indices),
            bounds=(0, np.inf),
            **kwargs
        )
        self.__init__(res.x)
        self.m_init = params_init
        return res.cost

    def pdf_integrand(self, delta, S_0, S_1, mpmath=False):
        """Function integrated in the CDF of the bivariate beta distribution.
        The function is integrated over delta.
        """
        exp = mp.exp if mpmath else np.exp
        log = mp.log if mpmath else np.log
        return exp(
               (self.m[0] - 1) * log(1 - S_0 - S_1 + delta) \
             + (self.m[1] - 1) * log(S_0 - delta) \
             + (self.m[2] - 1) * log(S_1 - delta) \
             + (self.m[3] - 1) * log(delta)
        )

    def log_pdf(self, S_0, S_1, show_progress=False):
        if show_progress:
            r = trange(S_0.size)
        else:
            r = range(S_0.size)
        # We integrate over delta, first we compute its bounds of integration
        LB = np.maximum(0, S_0 + S_1 - 1)
        UB = np.minimum(S_0, S_1)
        return np.log(mpmath_to_float(np.array([integrate(
                lambda delta: self.pdf_integrand(delta, S_0[i], S_1[i], False),
                lambda delta: self.pdf_integrand(delta, S_0[i], S_1[i], True),
                LB[i], UB[i], ()
            ) for i in r
        ]))) + self.log_integration_constant

    def pdf(self, S_0, S_1, show_progress=False):
        return np.exp(self.log_pdf(S_0, S_1, show_progress))

    def shifted_mu(self):
        return (
            [BivariateBeta(self.m + np.eye(4)[i]) for i in range(4)],
            self.m / np.sum(self.m)
        )

    def rvs(self, size):
        mu = sc.stats.dirichlet.rvs(self.m, size=size)
        return np.column_stack((
            mu[:, 1] + mu[:, 3],
            mu[:, 2] + mu[:, 3]
        ))

    def individual_cf(self, S_0, S_1, show_progress=False, parallel=False):
        return estimate_cf(self, S_0, S_1, show_progress, parallel)

    def population_cf(self):
        return self.m / np.sum(self.m)

    def expectation_maximization(S_0, S_1, mu, n_iter=10, exact_mu=None, n_moments=6):
        """Applies the expectation-maximization (EM) algorithm by using
        the bivariate beta distribution, and using the counterfactual
        category of the customer as a latent variable.
        """
        # Probability for each sample to belong to each class
        mu = mu.copy()
        # Vectors of parameters for the four conditional distributions
        M = np.empty((4, 4))
        init_step = True
        # Prior probabilities of each class
        tau = np.mean(mu, axis=0)
        for t in trange(n_iter):
            if exact_mu is not None:
                se = ((mu - exact_mu)**2).flatten()
                #print("Error: {:.3%} +- {:.3%}".format(np.nanmean(se), np.nanstd(se)))

            # M-step: fit the distributions
            for i in range(4):
                sample_moments = bb_sample_moments(S_0, S_1, n_moments, mu[:,i])
                if init_step:
                    M[:,i] = bb_initial_guess(sample_moments)
                M[:,i] = bb_fit(sample_moments, M[:,i])
            init_step = False

            # E-step: compute the new weights of the data samples
            new_mu = np.array(Parallel(n_jobs=4)(
                delayed(bb_pdf)(
                    S_0, S_1,
                    BivariateBeta(M[:,i]).pdf_integrand
                ) for i in range(4)
            )).T
            B = np.array([multi_beta(M[:,i]) for i in range(4)])
            new_mu = new_mu / B[None, :]
            # Multiply by the prior probabilities
            new_mu *= tau[None, :]
            # Normalize to sum up to one
            new_mu /= np.sum(new_mu, axis=1)[:, None]
            # Use past values where nan occurs
            new_mu[np.isnan(new_mu)] = mu[np.isnan(new_mu)]
            mu = new_mu

        return M, tau, mu
