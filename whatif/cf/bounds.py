"""
Code for the project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi
"""

import numpy as np


def frechet_bounds(S_0, S_1):
    """Returns the Fréchet bounds on the joint probability distribution of
    (y_0, y_1) given the marginal probabilities S_0 = P(y_0 = 1) and
    S_1 = P(y_1 = 1). The input can be scalars or numpy arrays.

    Reference:
    Fréchet, Maurice (1935). "Généralisation du théoreme des probabilités
    totales". In: Fundamenta mathematicae 1.25, pp. 379–387.
    """
    lb_ind = np.array([
        np.maximum(0, 1 - S_0 - S_1),
        np.maximum(0, S_0 - S_1),
        np.maximum(0, S_1 - S_0),
        np.maximum(0, S_0 + S_1 - 1)
    ]).T
    ub_ind = np.array([
        np.minimum(1 - S_0, 1 - S_1),
        np.minimum(S_0,     1 - S_1),
        np.minimum(1 - S_0, S_1),
        np.minimum(S_0,     S_1)
    ]).T
    return lb_ind, ub_ind


def uplift_bounds(S_0, S_1):
    """Compute uplift bounds on the probability of counterfactuals from the
    predictions of an uplift model. More precisely, we give bounds on joint
    probability distribution of (y_0, y_1), such as P(y_0 = 1, y_1 = 0), given N
    realizations of uplift scores S_0(x) = P(y_0 = 1 | x) and
    S_1(x) = P(y_1 = 1 | x).

    Reference:
    Théo Verhelst, Denis Mercier, et al. (2023). "Partial counterfactual
    identification and uplift modeling: theoretical results and real-world
    assessment". In: Machine Learning. doi: 10.1007/s10994-023-06317-w.
    """
    lb_ind, ub_ind = frechet_bounds(S_0, S_1)
    return np.mean(lb_ind, axis=0), np.mean(ub_ind, axis=0)
