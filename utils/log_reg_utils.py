from typing import Optional, Tuple
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import torch


def update_decision_boundary(ax, x, y, x1_mesh, x2_mesh, mesh_points, mesh_size, nnet,
                             xlim: Optional[Tuple[float, float]] = None, ylim: Optional[Tuple[float, float]] = None):
    nnet.eval()

    title: str = ax.get_title()
    ax.cla()

    cm_decision = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    y_hat_mesh_flat = nnet(torch.tensor(mesh_points)).detach().numpy()

    y_hat_mesh = y_hat_mesh_flat.reshape((mesh_size, mesh_size))

    ax.contourf(x1_mesh, x2_mesh, y_hat_mesh, cmap=cm_decision)
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright)

    if xlim is None:
        ax.set_xlim([np.min(x[:, 0]), np.max(x[:, 0])])
    else:
        ax.set_xlim([xlim[0], xlim[1]])

    if ylim is None:
        ax.set_ylim([np.min(x[:, 1]), np.max(x[:, 1])])
    else:
        ax.set_ylim([ylim[0], ylim[1]])

    # ax.set_xlabel('x0')
    # ax.set_ylabel('x1')
    ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])


def get_xor_data():
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
