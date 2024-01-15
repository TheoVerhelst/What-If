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
from whatif.cf.math import all_harmonic_partial_sums, all_rising_factorials, stirling_1
from whatif.cf.optimization_helpers import compute_sample_moments, compute_full_jacobian, fit_error, integrate
from whatif.cf import BivariateBeta

class NoisyBeta:
    n_max = 6

    def __init__(self, l_0, l_1, inner_class, params=None):
        assert l_0 > 0 and l_1 > 0 and isinstance(inner_class, type)
        self.l_0 = l_0
        self.l_1 = l_1
        self.inner_class = inner_class
        self.inner_dis = inner_class(params)
        self.integration_cache = {}
        self.l_0_rising = all_rising_factorials(self.l_0, NoisyBeta.n_max)
        self.l_1_rising = all_rising_factorials(self.l_1, NoisyBeta.n_max)
        self.l_0_partial = all_harmonic_partial_sums(self.l_0, NoisyBeta.n_max)
        self.l_1_partial = all_harmonic_partial_sums(self.l_1, NoisyBeta.n_max)

    def exact_moment(self, r, s):
        moment = 0
        for p in range(r + 1):
            for q in range (s + 1):
                moment += stirling_1(r, p) * stirling_1(s, q) \
                        * self.l_0**p * self.l_1**q \
                        * self.inner_dis.exact_moment(p, q)
        return moment / self.l_0_rising[r] / self.l_1_rising[s]

    def unbiased_moments(self, S_0, S_1, n_moments):
        """Computes the moments of the underlying bivariate distribution
        without noise, from the noisy sample moments. Remarquably, this does
        not require to know anything about the underlying bivariate
        distribution, only the two noise parameters. These moments then
        allows to make a good guess at the parameters of the underlying
        bivariate distribution.
        """
        moment_indices = list(range(n_moments))
        sample_moments = compute_sample_moments(S_0, S_1, moment_indices)
        moments = np.zeros(n_moments)
        for n in range(n_moments):
            r, s = moment_n_to_rs(n)
            moments[n] = self.l_0_rising[r] \
                    * self.l_1_rising[s] \
                    * sample_moments[n]
            for p in range(r + 1):
                for q in range (s + 1):
                    if p + q < r + s:
                        moments[n] -= stirling_1(r, p) * stirling_1(s, q) \
                                * self.l_0**p * self.l_1**q \
                                * moments[moment_rs_to_n(p, q)]
            moments[n] /= self.l_0**r * self.l_1**s
        return moments

    def jacobian(self, r, s):
        n_params = len(self.inner_dis.params()) + 2 # + 2 for l_1 and l_2
        J = np.zeros(n_params)
        for p in range(r + 1):
            for q in range (s + 1):
                s_rp = stirling_1(r, p)
                s_sq = stirling_1(s, q)
                # Derivative with respect to l_0 and l_1
                R_pq = self.inner_dis.exact_moment(p, q)
                J[0] += s_rp * s_sq * R_pq * self.l_0**(p-1) * self.l_1**q * \
                    (p - self.l_0 * self.l_0_partial[r])
                J[1] += s_rp * s_sq * R_pq * self.l_0**p     * self.l_1**(q-1) * \
                    (q - self.l_1 * self.l_1_partial[s])
                # Derivative with respect to m
                J_pq_bb = self.inner_dis.jacobian(p, q)
                for k in range(4):
                    J[2 + k] += s_rp * s_sq * self.l_0**p * self.l_1**q * J_pq_bb[k]

        return J / self.l_0_rising[r] / self.l_1_rising[s]

    def initial_guess(self, S_0, S_1):
        n_moments = 6
        unbiased_moments = self.unbiased_moments(S_0, S_1, n_moments)
        return self.inner_class.initial_guess(range(n_moments), unbiased_moments)

    def fit(self, moment_indices, sample_moments, params_init, fix_l=True, **kwargs):
        if fix_l:
            return self.fit_fixed_l(moment_indices, sample_moments, params_init, **kwargs)
        else:
            return self.fit_variable_l(moment_indices, sample_moments, params_init, **kwargs)

    def fit_variable_l(self, moment_indices, sample_moments, params_init, **kwargs):
        res = sc.optimize.least_squares(
            lambda params: fit_error(
                NoisyBeta(params[0], params[1], self.inner_class, params[2:]),
                moment_indices,
                sample_moments
            ),
            x0=np.hstack((self.l_0, self.l_1, params_init)),
            jac=lambda params: compute_full_jacobian(
                NoisyBeta(params[0], params[1], self.inner_class, params[2:]),
                moment_indices
            ),
            bounds=(0, np.inf),
            **kwargs
        )
        self.l_0_init = self.l_0
        self.l_1_init = self.l_1
        self.params_init = params_init
        self.__init__(res.x[0], res.x[1], self.inner_class, res.x[2:])
        return res.cost

    def fit_fixed_l(self, moment_indices, sample_moments, params_init, **kwargs):
        res = sc.optimize.least_squares(
            lambda params: fit_error(
                NoisyBeta(self.l_0, self.l_1, self.inner_class, params),
                moment_indices,
                sample_moments
            ),
            x0=params_init,
            jac=lambda params: compute_full_jacobian(
                NoisyBeta(self.l_0, self.l_1, self.inner_class, params),
                moment_indices
            )[:, 2:],
            bounds=(0, np.inf),
            **kwargs
        )
        self.l_0_init = self.l_0
        self.l_1_init = self.l_1
        self.params_init = params_init
        self.__init__(self.l_0, self.l_1, self.inner_class, res.x)
        return res.cost

    def pdf_log_inner_integrand(self, S_0, S_1, mpmath=False, inner_dis=None):
        """Computes
            log(f_m(S_0, S_1) / B(S_0*l_0, (1-S_0)*l_0)
                              / B(S_1*l_1, (1-S_1)*l_1)
            )
        and caches the result.
        """
        if inner_dis is None:
            inner_dis = self.inner_dis
        log = mp.log if mpmath else np.log
        logbeta = sc.special.betaln
        if mpmath:
            logbeta = lambda x, y: mp.log(mp.beta(x, y))

        # This function is called repeatitively with the same values,
        # we obtain a 25x speedup by using a cache.
        value = self.integration_cache.get((S_0, S_1))
        if value is None:
            value = float(mp.re(integrate(
                lambda delta: inner_dis.pdf_integrand(delta, S_0, S_1, False),
                lambda delta: inner_dis.pdf_integrand(delta, S_0, S_1, True),
                max(0, S_0 + S_1 - 1), min(S_0, S_1), ()
            )))

            value = log(value) \
                    - logbeta(S_0 * self.l_0, (1 - S_0) * self.l_0) \
                    - logbeta(S_1 * self.l_1, (1 - S_1) * self.l_1)
            self.integration_cache[(S_0, S_1)] = value
        return value

    def pdf_integrand(self, S_0, S_1, S_0_hat, S_1_hat, mpmath=False, inner_dis=None):
        exp = mp.exp if mpmath else np.exp
        log = mp.log if mpmath else np.log
        a_0 = self.l_0 * S_0
        b_0 = self.l_0 * (1 - S_0)
        a_1 = self.l_1 * S_1
        b_1 = self.l_1 * (1 - S_1)
        return exp(
              (a_0 - 1) * log(S_0_hat) + (b_0 - 1) * log(1 - S_0_hat)
            + (a_1 - 1) * log(S_1_hat) + (b_1 - 1) * log(1 - S_1_hat)
            + self.pdf_log_inner_integrand(S_0, S_1, mpmath, inner_dis)
        )

    def integration_bounds(self, S_0_hat, S_1_hat):
        """Computes the bounds of the integration over S_0, S_1, such
        that we avoid integrating over the region where the density is
        smaller than machine epsilon. This allows to have more
        integration points over the relevant region.
        """
        eps = np.finfo(float).eps
        c_0 = np.exp((np.log(eps) + sc.special.betaln(self.l_0/2, self.l_0/2)) * 2 / (self.l_0 - 2))
        c_1 = np.exp((np.log(eps) + sc.special.betaln(self.l_1/2, self.l_1/2)) * 2 / (self.l_1 - 2))
        diff_0 = (1 - 4*c_0)**(1/2) / 2
        diff_1 = (1 - 4*c_1)**(1/2) / 2

        lb_0 = np.maximum(0, np.minimum(1 - diff_0, S_0_hat) - diff_0)
        ub_0 = np.minimum(1, np.maximum(diff_0, S_0_hat) + diff_0)
        lb_1 = np.maximum(0, np.minimum(1 - diff_1, S_1_hat) - diff_1)
        ub_1 = np.minimum(1, np.maximum(diff_1, S_1_hat) + diff_1)

        return (lb_0, ub_0), (lb_1, ub_1)

    def log_pdf(self, S_0_hat, S_1_hat, show_progress=False, inner_dis=None):
        if show_progress:
            r = trange(S_0_hat.size)
        else:
            r = range(S_0_hat.size)
        self.integration_cache = {}

        (lb_0, ub_0), (lb_1, ub_1) = self.integration_bounds(S_0_hat, S_1_hat)

        res = np.array([
            double_integrate(
                lambda S_0, S_1: self.pdf_integrand(S_0, S_1, S_0_hat[i], S_1_hat[i], False, inner_dis),
                lambda S_0, S_1: self.pdf_integrand(S_0, S_1, S_0_hat[i], S_1_hat[i], True, inner_dis),
                lb_0[i], ub_0[i], lb_1[i], ub_1[i], ()
            ) for i in r
        ])
        cst = self.inner_dis.log_integration_constant if inner_dis is None else inner_dis.log_integration_constant
        return np.log(mpmath_to_float(res)) + cst

    def pdf(self, S_0_hat, S_1_hat, show_progress=False):
        return np.exp(self.log_pdf(S_0_hat, S_1_hat, show_progress))

    def rvs(self, size):
        S_01 = self.inner_dis.rvs(size)
        S_0 = S_01[:, 0]
        S_1 = S_01[:, 1]
        return np.column_stack((
            sc.stats.beta.rvs(S_0 * self.l_0, (1 - S_0) * self.l_0),
            sc.stats.beta.rvs(S_1 * self.l_1, (1 - S_1) * self.l_1)
        ))

    def individual_cf(self, S_0_hat, S_1_hat, parallel=False, show_progress=False):
        log_p = self.log_pdf(S_0_hat, S_1_hat, show_progress)
        shifted_inner_mu, factors = self.inner_dis.shifted_mu()

        if parallel:
            log_p_m = np.column_stack(Parallel(n_jobs=4)(
                delayed(self.log_pdf)(
                    S_0_hat, S_1_hat, show_progress, shifted_inner_mu[i]
                ) for i in range(4)
            ))
        else:
            log_p_m = np.column_stack([
                self.log_pdf(
                    S_0_hat, S_1_hat, show_progress, shifted_inner_mu[i]
                ) for i in range(4)
            ])
        return np.exp(log_p_m - log_p[:,None]) * factors

    def population_cf(self):
        return self.inner_dis.population_cf()
