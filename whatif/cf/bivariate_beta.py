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
    """Bivariate beta distribution from [1]_. This is a bivariante random
    distribution whose domain is the square :math:`[0,1]^2`, and the marginal
    distributions are beta distributions. See [2]_, Annex D for the details of
    the computations.

    In this page, we note the random variables with bivariate beta distribution
    as :math:`\mathbf S_0, \mathbf S_1`.

    Parameters
    ----------
        m: array-like of shape (4,) with dtype `float`, default=None
            Parameters of the distribution. Can be `None`, since :func:`fit`
            will set its value as well. Must have only nonnegative values.


    References
    ----------
    .. [1] Olkin, Ingram and Thomas A Trikalinos (2015). "Constructions for a
           bivariate beta distribution". In: Statistics & Probability Letters
           96. Publisher: Elsevier, pp. 54–60.

    .. [2] Théo Verhelst (2024). "Causal and predictive modeling of customer
           churn: lessons learned from empirical and theoretical research". PhD
           thesis, Université Libre de Bruxelles.
    """
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
        """Returns the value of the distribution parameters."""
        return self.m

    def exact_moment(self, r, s):
        """Compute the distribution moments of order :math:`r,s`. This
        corresponds to :math:`\mathbb E[\mathbf S_0^r\mathbf S_1^s]`.

        Parameters
        ----------
        r, s: int
            Order of the moments. Must be integers.

        Returns
        -------
        M: float
            Exact value of the moment.
        """
        moment = 0
        for p in range(r + 1):
            for q in range(s + 1):
                C = BivariateBeta.pascal[r, p] * BivariateBeta.pascal[s, q]
                moment +=  C * self.b_rising[r - p] * self.c_rising[s - q] * self.d_rising[p + q]
        return moment / self.A_rising[r + s]

    def jacobian(self, r, s):
        """Computes the Jacobian matrix of the function that returns the
        distribution moments from the parameter vector `m`, that is,

        .. math::
            \\frac{\partial\mathbb E[\mathbf S_0^r,\mathbf S_1^s]}{\partial m_i}.

        This is used in the optimization procedure. In fact, this is only one
        row of the Jacobian matrix, for a given value of `r` and `s`.

        Parameters
        ----------
        r, s: int
            Order of the moments. Must be integers.

        Returns
        -------
        d: ndarray of shape (4,) and dtype `float`
            Partial derivatives.
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
        parameters `m`, given the sample moment of indices 1, 2, 3 and 5, that
        is, :math:`\mathbb E[\mathbf S_0],\mathbb E[\mathbf S_1],
        \mathbb E[\mathbf S_0^2]` and :math:`\mathbb E[\mathbf S_1^2]`
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

    def fit(self, S_0, S_1, moment_indices=None, **kwargs):
        """Fits the distribution parameters to a series of samples from
        :math:`\mathbf S_0,\mathbf S_1`. It uses the function
        `scipy.optimize.least_squares` internally for the optimization
        procedure. It finds the distribution parameters that most closely match
        the samples moments of the input.

        Parameters
        ----------
        S_0, S_1 : 1D array-like of shape (N,) and dtype `float`
            Samples of :math:`\mathbf S_0,\mathbf S_1`. Must have the same
            length.

        moment_indices : array-like with dtype `int`, default=None
            Indices of the moments to use for the optimization. There should be
            at least four moments, since the distribution has four parameters.
            If `None`, it is set to `[1, 2, 3, 5]`.

        **kwargs :
            parameters forwarded to `scipy.optimize.least_squares`.

        Returns
        -------
        C: float
            The final cost computed by the optimization procedure (lower means
            a better fit to the data).

        """

        if moment_indices is None:
            moment_indices = [1, 2, 3, 5]

        sample_moments = compute_sample_moments(S_0, S_1, moment_indices)
        return self.fit_from_moments(moment_indices, sample_moments, **kwargs)

    def fit_from_moments(self, moment_indices, sample_moments, params_init=None, **kwargs):
        """Same as `fit`, but takes as input already-computed moments, and
        optionally initial values for the distribution parameters.

        Parameters
        ----------
        moment_indices: 1D array-like with dtype `int`
            See :func:`fit`.

        sample_moments: 1D array-like with dtype `float`
            Sample moments corresponding to `moment_indices`.

        params_init: array-like of shape (4,) with dtype `float`
            Initial value for `m` during optimization. Must have only
            nonnegative values.

        **kwargs:
            parameters forwarded to `scipy.optimize.least_squares`.

        Returns
        -------
        C: float
            The final cost computed by the optimization procedure (lower means
            a better fit to the data).
        """

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
        """Function that is integrated when computing the pdf of the
        distribution. The function is integrated over delta.

        Parameters
        ----------
        delta: float
            dependent variable in the function

        S_0: float
            Current value of S_0

        S_1: float
            Current value of S_1

        mpmath: bool, default=False
            If `True`, use `mpmath` functions to compute exponentials and
            logarithms. Otherwise, use the `numpy` version.

        Returns
        -------

        res: float

            .. math::
                (1-S_0-S_1+\delta)^{m_1-1} (S_0-\delta)^{m_2-1}
                (S_1-\delta)^{m_3-1} \delta^{m_4-1}

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
        """Computes the logarithm of the pdf for the given samples. It uses
        mpmath for samples when the integration suffers too much from numerical
        errors, and numpy otherwise.

        Parameters
        ----------
        S_0: 1D ndarray with dtype `float`

        S_1: 1D ndarray with dtype `float`

        show_progress: bool, default=False
            If `True`, uses `tqdm.trange` to show the progress of the
            computation.

        Returns
        -------
        pdf: ndarray with dtype `float`
            The log density for each value in `S_0` and `S_1`.
        """
        assert S_0.ndim == 1 and S_1.ndim == 1, "S_0 and S_1 should be 1D vectors"

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
        """Computes the logarithm of the pdf for the given samples. It uses
        mpmath for samples when the integration suffers too much from numerical
        errors, and numpy otherwise.

        Parameters
        ----------
        S_0: 1D ndarray with dtype `float`

        S_1: 1D ndarray with dtype `float`

        show_progress: bool, default=False
            If `True`, uses `tqdm.trange` to show the progress of the
            computation.

        Returns
        -------
        pdf: ndarray with dtype `float`
            The density for each value in `S_0` and `S_1`.
        """
        return np.exp(self.log_pdf(S_0, S_1, show_progress))

    def shifted_mu(self):
        """Returns four new instances where one element of `m` is increased
        in each, and an estimation of the population-level distribution of
        counterfactuals.

        Returns
        -------
        dis: list of size 4
            BivariateBeta instances with shifted parameters

        mu: ndarray of shape (4,)
            Estimated population counterfactuals
        """
        return (
            [BivariateBeta(self.m + np.eye(4)[i]) for i in range(4)],
            self.m / np.sum(self.m)
        )

    def rvs(self, size):
        """Generates random samples from the distribution.

        Parameters
        ----------
        size: int
            Number of samples to generate

        Returns
        -------
        mu: ndarray of shape(size, 4)
            Generated samples
        """
        mu = sc.stats.dirichlet.rvs(self.m, size=size)
        return np.column_stack((
            mu[:, 1] + mu[:, 3],
            mu[:, 2] + mu[:, 3]
        ))

    def individual_cf(self, S_0, S_1, show_progress=False, parallel=False):
        """Computes the individual-level counterfactuals from samples of
        :math:`\mathbf S_0,\mathbf S_1`.

        Parameters
        ----------
        S_0: 1D ndarray with dtype `float`

        S_1: 1D ndarray with dtype `float`

        show_progress: bool, default=False
            If `True`, uses `tqdm.trange` to show the progress of the
            computation.

        parallel: bool, default=False
            If `True`, the computation for the four counterfactuals are done
            in parallel using multiple threads.

        Returns
        -------
        mu: ndarray of shape (N, 4) with dtype `float`
            The countefactuals for each value in `S_0` and `S_1`.
        """
        return estimate_cf(self, S_0, S_1, show_progress, parallel)

    def population_cf(self):
        """Returns an estimation of the population-level distribution of
        counterfactuals.

        Returns
        -------
        mu: ndarray of shape (4,)
            Estimated population counterfactuals
        """
        return self.m / np.sum(self.m)

    def expectation_maximization(S_0, S_1, mu, n_iter=10, exact_mu=None, n_moments=6):
        """Applies the expectation-maximization (EM) algorithm by using
        the bivariate beta distribution, and using the counterfactual
        category of the customer as a latent variable.

        Warning
        -------
        This function is not up-to-date and needs to be re-written.
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
