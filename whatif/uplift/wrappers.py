"""
Code for the project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi

This file contains wrappers for various uplift models. This allows to
use seamlessly the functions.dataset.Dataset class without having to
specify the X, y, t, or r vectors. This also harmonizes the various
interfaces between sklearn and causalml models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#from causalml.inference.tree import UpliftRandomForestClassifier
#from causalml.inference.meta import BaseXClassifier
#from causalml.inference.nn.cevae import CEVAE

class RandomForestWrapper:
    def __init__(self, *args, **kwargs):
        self.model = RandomForestClassifier(*args, **kwargs)
    
    def fit(self, data, *args, **kwargs):
        # Train on control group only
        return self.model.fit(data.X[~data.t], data.y[~data.t], *args, **kwargs)
    
    def predict(self, data, *args, **kwargs):
        return self.model.predict_proba(data.X, *args, **kwargs)[:, 1]

class TLearnerWrapper:
    def __init__(self, *args, **kwargs):
        self.model_0 = RandomForestClassifier(*args, **kwargs)
        self.model_1 = RandomForestClassifier(*args, **kwargs)
    
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
    def __init__(self, *args, **kwargs):
        self.model = RandomForestClassifier(*args, **kwargs)
    
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
    def __init__(self, *args, **kwargs):
        self.model = RandomForestClassifier(*args, **kwargs)
    
    def fit(self, data, *args, **kwargs):
        z = np.logical_xor(data.t, data.y)
        return self.model.fit(data.X, z, *args, **kwargs)
    
    def predict(self, data, *args, **kwargs):
        return 2 * self.model.predict_proba(data.X, *args, **kwargs)[:, 1] - 1
    

class URFCWrapper:
    def __init__(self, *args, **kwargs):
        self.model = UpliftRandomForestClassifier(control_name="control", *args, **kwargs)
    
    def fit(self, data, *args, **kwargs):
        t = np.array(["target" if t else "control" for t in data.t])
        return self.model.fit(data.X, t, data.y, *args, **kwargs)
    
    def predict(self, data, *args, **kwargs):
        return self.model.predict(data.X, *args, **kwargs)
    
    
class XClassifierWrapper:
    def __init__(self, *args, **kwargs):
        self.model = BaseXClassifier(control_name="control", *args, **kwargs)
    
    def fit(self, data, *args, **kwargs):
        t = np.array(["target" if t else "control" for t in data.t])
        self.p = np.full(data.y.shape, data.t.mean())
        return self.model.fit(X=data.X, treatment=t, y=data.y, p=self.p, *args, **kwargs)
    
    def predict(self, data, *args, **kwargs):
        p = np.full((data.X.shape[0],), self.p.mean())
        return self.model.predict(data.X, p=p, *args, **kwargs).flatten()

    
class CEVAEWrapper:
    def __init__(self, *args, **kwargs):
        self.model = CEVAE(*args, **kwargs)
    
    def fit(self, data, *args, **kwargs):
        return self.model.fit(data.X, data.t.astype("float"), data.y.astype("float"), *args, **kwargs)
    
    def predict(self, data, *args, **kwargs):
        return self.model.predict(data.X, *args, **kwargs)