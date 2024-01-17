"""Wrapper for dataset objects.

Code for the project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi
"""

from sklearn.model_selection import train_test_split
import numpy as np

__all__ = ["Dataset"]

class Dataset:
    """Represents a dataset such that the features X, the outcomes y,
    the treatment t, or the reach r can be easily accessed and
    subscripted together. This is useful to avoid duplicating code.
    Any number of attributes (X, y, t, r, w, ...) can be used.

    Example:
    ```
    d = Dataset(X=data[:, 0:10], y=data[:, 10], t=data[:, 11])
    d[3:6] # New Dataset object containing a slice of the data
    d[3:6].t # Slice of the treatment values
    ```
    """
    def __init__(self, **arrays):
        self.__dict__.update(arrays)

    def __getitem__(self, key):
        res = Dataset()
        res.__dict__.update(self.__dict__)
        for name in res.__dict__:
            res.__dict__[name] = res.__dict__[name][key].copy()
        return res

    def train_test_split(self, *args, **kwargs):
        """Calls the function train_test_split from sklearn on the
        data objects stored in this Dataset. All parameters are
        forwarded to sklearn.train_test_split.
        """
        ordered_dict = list(self.__dict__.items())
        res = train_test_split(*list(value for name, value in ordered_dict), *args, **kwargs)
        return (
            Dataset(**{ordered_dict[i][0]: res[i * 2] for i in range(len(ordered_dict))}),
            Dataset(**{ordered_dict[i][0]: res[i * 2 + 1] for i in range(len(ordered_dict))})
        )
