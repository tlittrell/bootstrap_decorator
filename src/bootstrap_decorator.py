import functools
import numpy as np
from sklearn.utils import resample


def bootstrap(*, n=100, summarize=True):
    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            samples = [func(*resample(*args), **kwargs) for _ in range(n)]
            if summarize:
                return calc_summary_statistics(samples)
            else:
                return samples

        return wrapper

    return inner


def calc_summary_statistics(samples):
    vals = np.array(samples)
    mean = vals.mean()
    std = vals.std()
    return mean, std
