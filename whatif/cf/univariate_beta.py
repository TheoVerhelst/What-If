"""
Code for the project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi
"""

import numpy as np
import scipy as sc
from whatif.cf.math import all_rising_factorials, all_harmonic_partial_sums

class UnivariateBeta:
    n_max = 6

    def __init__(self, a=None, b=None):
        assert (a is None and b is None) or (a > 0 and b > 0)
        self.a = a
        self.b = b
        if a is not None and b is not None:
            self.l = a + b
            self.S = a / self.l

            self.a_rising = all_rising_factorials(self.a, UnivariateBeta.n_max)
            self.b_rising = all_rising_factorials(self.b, UnivariateBeta.n_max)
            self.ab_rising = all_rising_factorials(self.a + self.b, UnivariateBeta.n_max)
            self.a_partial = all_harmonic_partial_sums(self.a, UnivariateBeta.n_max)
            self.b_partial = all_harmonic_partial_sums(self.b, UnivariateBeta.n_max)
            self.ab_partial = all_harmonic_partial_sums(self.a + self.b, UnivariateBeta.n_max)
            self.log_integration_constant = -sc.special.betaln(self.a, self.b)

    def exact_moment(self, r, s):
        return self.a_rising[r] * self.b_rising[s] / self.ab_rising[r + s]

    def jacobian(self, r, s):
        moment = self.exact_moment(r, s)
        return np.array([
            moment * (self.a_partial[r] - self.ab_partial[r + s]),
            moment * (self.b_partial[s] - self.ab_partial[r + s])
        ])

    def fit_moments(self, R_1, R_2):
        self.S = R_1
        self.l = (R_1 - R_2) / (R_2 - R_1**2)
        self.__init__(self.l * self.S, self.l * (1 - self.S))

    def fit(self, samples):
        self.fit_moments(np.mean(samples), np.mean(samples**2))

    def pdf(self, x):
        return np.exp(self.log_pdf(x))

    def log_pdf(self, x):
        return (self.a - 1) * np.log(x) + (self.b - 1) * np.log(1-x) \
            + self.log_integration_constant
