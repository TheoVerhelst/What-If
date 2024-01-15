"""
Code for the project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from openml.datasets import get_dataset
from whatif import Dataset
from whatif.uplift import EasyEnsemble, TLearnerWrapper
from whatif.cf import frechet_bounds, uplift_bounds, Independence, \
                    BivariateBeta, GeneralizedBivariateBeta, NoisyBeta


if False:
    dataset = get_dataset("churn-uplift-orange").get_data()[0]
    dataset = Dataset(
        X = dataset.drop(["y", "t"], axis=1),
        y = (dataset.y == 1).to_numpy(),
        t = (dataset.t == 1).to_numpy()
    )
    dataset.X = pd.get_dummies(dataset.X).to_numpy().astype("float32")


    model = EasyEnsemble(
        TLearnerWrapper(n_estimators=100),
        n_folds=10
    )
    model.fit(dataset)
    pred = model.predict(dataset)

    S_0 = pred["control"]
    S_1 = pred["target"]
else:
    S_0 = np.random.rand(100)
    S_1 = np.random.rand(100)

u_bounds = uplift_bounds(S_0, S_1)
f_bounds = frechet_bounds(np.mean(S_0), np.mean(S_1))

ind_point = Independence()
ind_point.fit(S_0, S_1)

bb_point = BivariateBeta()
bb_point.fit(S_0, S_1)

gbb_point = GeneralizedBivariateBeta()
gbb_point.fit(S_0, S_1)

print("u_bounds:", u_bounds)
print("f_bounds:", f_bounds)
print("bb_point:", bb_point.population_cf())
print("gbb_point:", gbb_point.population_cf())
