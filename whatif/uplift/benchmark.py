"""
Code for the project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi
"""

from math import floor
from copy import deepcopy
import numpy as np
from tqdm.autonotebook import tqdm
from whatif.uplift import EasyEnsemble
from whatif.uplift import uplift_curve


def generate_splits(N, k_folds, n_repeats, rng):
    """Creates data splits suitable for repeated k-fold.
    
    N: size of the dataset.
    k_folds: number of folds.
    n_repeats: number of times to repeat k-folds.
    rng: random device.
    output: a list of k_folds*n_repeats indices arrays
    """
    res = []
    fold_size = N // k_folds
    for i in range(n_repeats):
        I = rng.permutation(N)
        for j in range(k_folds):
            res.append(I[j * fold_size : min((j + 1) * fold_size, N)])
    return res

def get_splits_from_results(past_results):
    """Extracts the train-test splits from the results data structure."""
    return list([split["test_indices"] for split in past_results])

def benchmark(dataset, models, k_folds=5, n_repeats=4, seed=None,
              verbose=False, fit_params={}, predict_params={}, past_results=None):
    """Perform a benchmark on a dataset with a series of models.
    dataset: a whatif.Dataset object
    models: a dictionnary of {name: model} pairs
    k_folds: the number of folds
    n_repeats: the number of times to repeat the k-fold
    seed: a random seed
    verbose: prints progress if True
    fit_params: a dict of params to forward to models when training.
    predict_params: a dict of params to forward to models when predicting.
    past_results: if given, use test splits from these result data.
    output: a data structure containing the test indices, and for each
    model, the trained model object and its predictions on each split.
    """
    rng = np.random.default_rng(seed=seed)
    N = dataset.X.shape[0]
    all_indices = np.arange(N)
    if past_results is None:
        splits = generate_splits(N, k_folds, n_repeats, rng)
    else:
        splits = get_splits_from_results(past_results)
    n_splits = len(splits)
    
    res = []
    
    for i, split in tqdm(enumerate(splits), total=n_splits):
        if verbose:
            print("Split {}/{}".format(i, n_splits))
                  
        data_test = dataset[split]
        # Shuffle also the train indices just to be sure
        train_indices = rng.permutation(np.setdiff1d(all_indices, split))
        data_train = dataset[train_indices]
        
        res_split = {"test_indices": split, "results": {}}
        
        for model_name, model in models.items():
            model = deepcopy(model)
            if verbose:
                print("Fitting", model_name)
            model.fit(data_train, **fit_params.get(model_name, {}))
            if verbose:
                print("Predicting", model_name)
            pred = model.predict(data_test, **predict_params.get(model_name, {}))
            
            res_split["results"][model_name] = {}
            res_split["results"][model_name]["pred"] = pred
            res_split["results"][model_name]["model"] = model
        res.append(res_split)
    return res


def get_preds_array(results, model_name, n_samples, k_folds=3, n_repeats=10):
    all_preds = np.empty((n_samples, n_repeats))
    for i, split in enumerate(results):
        i_repeat = i // k_folds
        preds = split["results"][model_name]["pred"]
        if model_name.startswith("urf") \
            or model_name.startswith("rf_tlearner") \
            or model_name.startswith("rf_slearner"):
            preds = preds["control"] - preds["target"]
        all_preds[split["test_indices"], i_repeat] = preds
    all_preds[np.isnan(all_preds)] = 0
    all_preds[all_preds <= -1] = -1
    all_preds[all_preds >= 1] = 1
    return all_preds

def estimator_variance(results, model_name, n_samples, k_folds=3, n_repeats=10):
    all_preds = get_preds_array(results, model_name, n_samples, k_folds, n_repeats)
    return np.mean(np.var(all_preds, axis=1))


def auucs(results, model_name, dataset, k_folds=3, n_repeats=10, use_churn_convention=False):
    auucs = []
    for i, split in enumerate(results):
        i_repeat = i // k_folds
        preds = split["results"][model_name]["pred"]
        if model_name.startswith("urf") \
            or model_name.startswith("rf_tlearner") \
            or model_name.startswith("rf_slearner"):
            preds = preds["target"] - preds["control"]
        if use_churn_convention:
            preds = -preds
            if model_name in ("urf", "rf"):
                preds = -preds
        preds[np.isnan(preds)] = 0
        preds[preds > 1e10] = 0 # Sometimes some models give huge scores
        indices = split["test_indices"]
        curve = uplift_curve(dataset.y[indices], dataset.t[indices], preds, use_churn_convention)
        auucs.append(curve.profit.mean() / len(indices))
    return np.array(auucs)

def KL_divergence(mu_0, mu_1_array, sigma_0, sigma_1_array):
    """
    KL-divergence between a distribution N(mu_0, sigma_0^2) and a series
    of other distribution N(mu_1[i], sigma_1[i]^2).
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    """
    return np.log(sigma_1_array / sigma_0) - 1/2 \
        + (sigma_0**2 + (mu_0 - mu_1_array)**2) / (2 * sigma_1_array**2)

def ranking_variance(results, model_name, n_samples, k_folds=3, n_repeats=10):
    """
    Computes the variability in the rankings induced by the scores,
    given multiple independent realizations of these scores for each
    individual. It assumes a Gaussian distribution for the scores, and
    computes the average KL-divergence between all pairs of individuals.
    
    preds: a (N, n) array of N samples and n realizations of the scores
    """
    preds = get_preds_array(results, model_name, n_samples, k_folds, n_repeats)
    mu = np.mean(preds, axis=1)
    sigma = np.std(preds, axis=1)
    average_kl = 0
    for i in range(preds.shape[0]):
        average_kl += KL_divergence(mu[i], np.delete(mu, i), sigma[i], np.delete(sigma, i)).mean()
    return 1 / (average_kl / preds.shape[0])