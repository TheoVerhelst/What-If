"""
Code for the project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi
"""

import numpy as np

class Independence:
    def fit(self, S_0, S_1):
        self.S_0 = S_0
        self.S_1 = S_1

    def individual_cf(self, S_0, S_1):
        """Returns the estimated distribution of counterfactuals given
        the probabilities S_0(x)=P(Y_0=1 | x) and S_1(x) = P(Y_1=1 | x),
        assuming that the potential outcomes are independent given X.
        """
        return np.array([
            (1 - S_0) * (1 - S_1),
            S_0       * (1 - S_1),
            (1 - S_0) * S_1,
            S_0       * S_1
        ]).T


    def population_cf(self):
        return np.mean(self.individual_cf(self.S_0, self.S_1), axis=0)
