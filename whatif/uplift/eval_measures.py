"""
Code for the project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi
"""

import numpy as np
import pandas as pd

__all__ = ["extract_cb", "uplift_curve", "profit_curve", "cf_profit_curve", "lift_curve", "calibrate_score"]

def extract_cb(CB):
    """Returns a 4-tuple of len 4 from the values from the cost-benefit matrix
    CB, which is either a tuple ((a, b), (c, d)) or a (N, 2, 2) numpy array.
    In the latter case, each element of the result contains the  corresponding
    array column.
    """
    if len(CB.shape) == 2:
        (CB_00, CB_01), (CB_10, CB_11) = CB
    else:
        CB_00 = CB[:, 0, 0]
        CB_01 = CB[:, 0, 1]
        CB_10 = CB[:, 1, 0]
        CB_11 = CB[:, 1, 1]
    return CB_00, CB_01, CB_10, CB_11

def uplift_curve(y, t, score, use_churn_convention=True):
    """Computes the conventional uplift curve."""
    if use_churn_convention:
        return profit_curve(y, t, score, np.array([[1, 1], [0, 0]]))
    else:
        return profit_curve(y, t, score, np.array([[0, 0], [1, 1]]))

def profit_curve(y, t, score, CB=np.array([[1, 1], [0, 0]])):
    """Computes the profit curve, a version of the uplift curve
    weighted by a cost-benefit matrix CB.
    """
    score_ranking = np.argsort(score)[::-1]
    y = y[score_ranking]
    t = t[score_ranking]
    N = y.shape[0]
    score = score[score_ranking]
    CB_00, CB_01, CB_10, CB_11 = extract_cb(CB)

    grid = pd.DataFrame(data={
        "k": np.arange(N),
        "n_1": np.cumsum(t),
        "n_0": np.cumsum(1 - t),
        "score": score,
        "r_00": np.cumsum((1 - y) * (1 - t) * CB_00),
        "r_01": np.cumsum((1 - y) * t * CB_01),
        "r_10": np.cumsum(y * (1 - t) * CB_10),
        "r_11": np.cumsum(y * t * CB_11)
    })
    # Avoid NaNs, they will be replaced by zeros
    grid.loc[grid["n_0"] == 0, ["n_0"]] = 1
    grid.loc[grid["n_1"] == 0, ["n_1"]] = 1

    grid["profit"] = (
        ((grid.r_01 + grid.r_11) / grid.n_1) -
        ((grid.r_00 + grid.r_10) / grid.n_0)
    ) * grid["k"]

    return grid

def cf_profit_curve(score, S_0, S_1, CB=np.array([[1, 1], [0, 0]])):
    """Computes the profit curve, assuming we know the conditional
    probabilities S_0 and S_1. This is more precise than using y and t,
    but this can only be used in simulations (predictions from real models might
    be biased).
    """
    score_ranking = np.argsort(score)[::-1]
    score = score[score_ranking]
    N = score.shape[0]
    S_0 = S_0[score_ranking]
    S_1 = S_1[score_ranking]
    CB_00, CB_01, CB_10, CB_11 = extract_cb(CB)

    grid = pd.DataFrame(data={
        "k": np.arange(N),
        "score": score
    })
    grid["profit"] = np.cumsum(
        CB_01 * (1 - S_1) + CB_11 * S_1
        - CB_00 * (1 - S_0) - CB_10 * (1 - S_0)
    ) / N
    return grid

def lift_curve(y, score):
    """Lift curve, i.e., the churn rate as a function of the treatment rate,
    divided by the overall churn rate.
    """
    score_ranking = np.argsort(score)[::-1]
    y = y[score_ranking]
    score = score[score_ranking]
    N = y.shape[0]
    S = y.mean()
    grid = pd.DataFrame(data={
        "k": np.arange(N),
        "score": score,
        "y": y,
        "r": np.cumsum(y)
    })
    grid["lift"] = (grid["r"] / grid["k"]) / S
    return grid


def calibrate_score(score, p):
    """
    Calibration of posterior probabilities as shown in
    Dal Pozzolo, Andrea, et al. "Calibrating probability with undersampling for
    unbalanced classification." 2015 IEEE Symposium Series on Computational
    Intelligence. IEEE, 2015.

    score: the probability (e.g., output of a model) to calibrate
    p: the prior probability of Y=1.
    """
    p_s = np.mean(score)
    ratio = p * (p_s - 1) / (p_s * (p - 1))
    return ratio * score / (ratio * score - score + 1)
