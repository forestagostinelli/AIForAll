from typing import Tuple
import numpy as np
from utils.log_reg_utils import get_xor_data
from sklearn.datasets import make_moons
from numpy.typing import NDArray


def get_2d_data(dataset: str) -> Tuple[NDArray, NDArray]:
    if dataset == "lin":
        n: int = 200
        x_pos = np.random.multivariate_normal([-1, 1], np.array([[0.1, 0], [0, 0.1]]), n)
        x_neg = np.random.multivariate_normal([1, -1], np.array([[0.1, 0], [0, 0.1]]), n)
        x = np.concatenate((x_pos, x_neg), axis=0)
        y = np.array([1] * n + [0] * n)
        return x, y
    elif dataset == "xor":
        return get_xor_data()
    elif dataset == "half_moons":
        n: int = 200
        return make_moons(n_samples=n, noise=0.1)
    else:
        raise ValueError("Unknown dataset %s" % dataset)
