"""
Code for the project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi
"""

import numpy as np
from whatif import Dataset

__all__ = ["RFeature"]

class RFeature:
    """A model that uses the reach indicator to improve predictions.
    A model is trained to predict r (the reach) given X (the features),
    then the predictions of that model are used as a new feature to
    predict y (the outcome).
    """
    def __init__(self, reach_model, uplift_model, verbose=False):
        self.reach_model = reach_model
        self.uplift_model = uplift_model
        self.verbose = verbose

    def fit(self, dataset):
        """Fits the R-feature model.
        dataset: a functions.dataset.Dataset object.
        """
        if self.verbose:
            print("Training model to predict reach")
        self.reach_model.fit(dataset.X, dataset.r)
        X_r = self._add_reach(dataset.X)
        if self.verbose:
            print("Training uplift model")
        self.uplift_model.fit(Dataset(X = X_r, t = dataset.t, y = dataset.y))

    def _add_reach(self, X):
        r_hat = self.reach_model.predict_proba(X)[:,1]
        r_hat = r_hat.reshape((r_hat.shape[0], 1))
        return np.hstack((X, r_hat))

    def predict(self, dataset, *args, **kwargs):
        X_r = self._add_reach(dataset.X)
        return self.uplift_model.predict(Dataset(X = X_r), *args, **kwargs)


def add_r_feature(model, dataset):
    """Fits a reach model and adds its predictions to a dataset.
    Accepts a functions.dataset.Dataset object, and return the augmented
    X component of the Dataset.
    """
    model.fit(dataset.X, dataset.r)
    r_hat = model.predict_proba(dataset.X)[:,1]
    return np.hstack((dataset.X, r_hat.reshape((r_hat.shape[0], 1))))
