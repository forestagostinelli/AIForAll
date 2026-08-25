from typing import Tuple

import numpy as np


def get_xor_data() -> Tuple[np.ndarray, np.ndarray]:
    point_locs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    point_labels = [0, 1, 1, 0]
    x_l = []
    y_l = []
    for point_loc, point_label in zip(point_locs, point_labels):
        num_loc: int = 50
        x_loc = np.random.multivariate_normal(point_loc, np.array([[0.01, 0], [0, 0.01]]), num_loc)
        y_loc = np.array([point_label] * num_loc)
        x_l.append(x_loc)
        y_l.append(y_loc)

    x = np.concatenate(x_l, axis=0)
    y = np.concatenate(y_l, axis=0)

    return x, y
