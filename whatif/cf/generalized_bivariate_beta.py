"""
Code for the project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi
"""

import numpy as np
import scipy as sc
from scipy.special import comb
from tqdm.autonotebook import trange
from whatif.cf.math import all_rising_factorials, all_harmonic_partial_sums, log_multi_beta
from whatif.cf.optimization_helpers import fit_error, compute_full_jacobian, mpmath_to_float, integrate, estimate_cf, compute_sample_moments, compute_exact_moments
from whatif.cf import BivariateBeta, UnivariateBeta

class GeneralizedBivariateBeta:
    n_max = 6
    pascal = sc.linalg.pascal(n_max, kind="lower", exact=False)

    def __init__(self, params=None):
        assert (params is None) or (len(params) == 6 and np.all(np.greater(params,  0)))
        if params is not None:
            self.a = np.array(params[0:3])
            self.b = np.array(params[3:6])
            self.z = [UnivariateBeta(self.a[i], self.b[i]) for i in range(3)]
            self.log_integration_constant = -np.sum(sc.special.betaln(self.a, self.b))

    def params(self):
        return np.hstack((self.a, self.b))

    def exact_moment(self, r, s):
        moment = 0
        for p in range(r + 1):
            for q in range(s + 1):
                moment += GeneralizedBivariateBeta.pascal[r, p] \
                        * GeneralizedBivariateBeta.pascal[s, q] \
                        * self.z[0].exact_moment(0, r + s) \
                        * self.z[1].exact_moment(r - p, s + p) \
                        * self.z[2].exact_moment(s - q, p + q)
        return moment

    def jacobian(self, r, s):
        J = np.zeros(self.a.size + self.b.size)
        for p in range(r + 1):
            for q in range (s + 1):
                C = GeneralizedBivariateBeta.pascal[r, p] * GeneralizedBivariateBeta.pascal[s, q]
                moment_0 = self.z[0].exact_moment(0, r + s)
                moment_1 = self.z[1].exact_moment(r - p, s + p)
                moment_2 = self.z[2].exact_moment(s - q, p + q)
                J_0 = C * moment_1 * moment_2 * self.z[0].jacobian(0, r + s)
                J_1 = C * moment_0 * moment_2 * self.z[1].jacobian(r - p, s + p)
                J_2 = C * moment_0 * moment_1 * self.z[2].jacobian(s - q, p + q)
                J[0] += J_0[0]
                J[1] += J_1[0]
                J[2] += J_2[0]
                J[3] += J_0[1]
                J[4] += J_1[1]
                J[5] += J_2[1]
        return J

    @staticmethod
    def initial_guess(moment_indices, sample_moments):
        tol = 1e-15
        params = {
            "verbose": 0,
            "method": "trf",
            "max_nfev": 1000,
            "gtol": tol,
            "xtol": tol,
            "ftol": tol,
            "x_scale": "jac",
            "tr_solver": "exact"
        }
        bb = BivariateBeta()
        bb.fit_from_moments(moment_indices, sample_moments, **params)
        a = bb.m[0:3]
        b = np.empty(3)
        b[2] = bb.m[3]
        b[1] = b[2] + a[2]
        b[0] = b[1] + a[1]
        return np.hstack((a, b))

    def fit(self, S_0, S_1, moment_indices = None, **kwargs):
        if moment_indices is None:
            moment_indices = [1, 2, 3, 5]

        sample_moments = compute_sample_moments(S_0, S_1, moment_indices)
        return self.fit_from_moments(moment_indices, sample_moments, **kwargs)

    def fit_from_moments(self, moment_indices, sample_moments, params_init=None, C=1e-4, **kwargs):
        if params_init is None:
            params_init = GeneralizedBivariateBeta.initial_guess(moment_indices, sample_moments)

        res = sc.optimize.least_squares(
            lambda params: np.hstack((
                compute_exact_moments(
                    GeneralizedBivariateBeta(params), moment_indices
                ) - sample_moments,
                C * (params - params_init) # Normalization term
            )),
            x0=params_init,
            jac=lambda params: np.vstack((
                compute_full_jacobian(
                    GeneralizedBivariateBeta(params),
                    moment_indices
                ),
                np.diagflat(2 * C * (params - params_init))
            )),
            bounds=(0, 2 * np.max(params_init)),
            **kwargs
        )
        self.a_init = params_init[0:3]
        self.b_init = params_init[3:6]
        self.__init__(res.x)
        return res.cost

    def pdf_integrand(self, delta, S_0, S_1, mpmath=False):
        beta = S_0 - delta
        gamma = S_1 - delta
        alpha = 1 - beta - gamma - delta
        exp = mp.exp if mpmath else np.exp
        log = mp.log if mpmath else np.log
        return exp(
              (self.a[0]-1) * log(alpha) \
            + (self.a[1]-1) * log(beta)  \
            + (self.a[2]-1) * log(gamma) \
            + (self.b[2]-1) * log(delta) \
            + (self.b[0] - self.b[1] - self.a[1]) * log(1-alpha) \
            + (self.b[1] - self.b[2] - self.a[2]) * log(S_1)
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
        # Values to add to the four shifted models
        parameter_offsets = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1]
        ])
        # Factors to use to keep the pdf normalized to one
        factors = []
        for i in range(4):
            o = parameter_offsets[i]
            factors.append(
                np.prod([self.z[j].exact_moment(o[j], o[j + 3]) for j in range(3)])
            )

        return [
            GeneralizedBivariateBeta(np.hstack((self.a, self.b)) + parameter_offsets[i]) \
            for i in range(4)
        ], np.array(factors)

    def rvs(self, size):
        z_1 = sc.stats.beta.rvs(self.a[0], self.b[0], size=size)
        z_2 = sc.stats.beta.rvs(self.a[1], self.b[1], size=size)
        z_3 = sc.stats.beta.rvs(self.a[2], self.b[2], size=size)
        beta = z_2 * (1 - z_1)
        gamma = z_3 * (1 - z_2) * (1 - z_1)
        delta = (1 - z_3) * (1 - z_2) * (1 - z_1)
        return np.column_stack((
            beta + delta, gamma + delta
        ))

    def individual_cf(self, S_0, S_1, show_progress=False, parallel=False):
        return estimate_cf(self, S_0, S_1, show_progress, parallel)

    def population_cf(self):
        a = self.a
        b = self.b
        l = a + b
        return np.array([
            a[0]               / l[0],
            b[0] * a[1]        / l[0] / l[1],
            b[0] * b[1] * a[2] / l[0] / l[1] / l[2],
            b[0] * b[1] * b[2] / l[0] / l[1] / l[2]
        ])
