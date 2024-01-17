"""
Code for the project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi

This file contains wrappers for various uplift meta-models. This allows to
use seamlessly the functions.dataset.Dataset class without having to
specify the X, y, t, or r vectors. This also harmonizes the various
interfaces between sklearn and causalml models.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone


__all__ = ["TLearnerWrapper", "SLearnerWrapper", "TransformedOutcomeWrapper", "SKLearnWrapper", "CausalMLWrapper"]


class TLearnerWrapper:
    def __init__(self, model):
        self.model_0 = clone(model)
        self.model_1 = clone(model)

    def fit(self, data, *args, **kwargs):
        return (
            self.model_0.fit(data.X[~data.t], data.y[~data.t], *args, **kwargs),
            self.model_1.fit(data.X[data.t], data.y[data.t], *args, **kwargs)
        )

    def predict(self, data, *args, **kwargs):
        return pd.DataFrame(data={
            "control": self.model_0.predict_proba(data.X, *args, **kwargs)[:, 1],
            "target": self.model_1.predict_proba(data.X, *args, **kwargs)[:, 1]
        })


class SLearnerWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, data, *args, **kwargs):
        X_t = self.add_t(data.X, data.t)
        return self.model.fit(X_t, data.y, *args, **kwargs)

    def predict(self, data, *args, **kwargs):
        X_0 = self.add_t(data.X, np.full(data.X.shape[0], False))
        X_1 = self.add_t(data.X, np.full(data.X.shape[0], True))
        return pd.DataFrame(data={
            "control": self.model.predict_proba(X_0, *args, **kwargs)[:, 1],
            "target": self.model.predict_proba(X_1, *args, **kwargs)[:, 1]
        })

    def add_t(self, X, t):
        return np.hstack((X, t.reshape(t.shape[0], 1)))


class TransformedOutcomeWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, data, *args, **kwargs):
        z = np.logical_xor(data.t, data.y)
        return self.model.fit(data.X, z, *args, **kwargs)

    def predict(self, data, *args, **kwargs):
        return 2 * self.model.predict_proba(data.X, *args, **kwargs)[:, 1] - 1


class SKLearnWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, data, *args, **kwargs):
        # Train on control group only
        return self.model.fit(data.X[~data.t], data.y[~data.t], *args, **kwargs)

    def predict(self, data, *args, **kwargs):
        return self.model.predict_proba(data.X, *args, **kwargs)[:, 1]


class CausalMLWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, data, *args, **kwargs):
        t = np.array(["target" if t else self.model.control_name for t in data.t])
        return self.model.fit(data.X, t, data.y, *args, **kwargs)

    def predict(self, data, *args, **kwargs):
        return self.model.predict(data.X, *args, **kwargs)
