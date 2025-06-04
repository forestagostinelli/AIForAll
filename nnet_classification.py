from typing import Any, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from argparse import ArgumentParser

from sklearn.datasets import make_moons


class Abs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)


class NNet(nn.Module):
    def __init__(self):
        super().__init__()
        # EDIT here
        self.model = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid(),
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x.float())

        return x


def update_decision_boundary(ax, x, y, x1_mesh, x2_mesh, mesh_points, mesh_size, nnet):
    nnet.eval()

    ax.cla()

    cm_decision = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    y_hat_mesh = nnet(torch.tensor(mesh_points)).detach().numpy()

    y_hat_mesh = y_hat_mesh.reshape((mesh_size, mesh_size))

    ax.contourf(x1_mesh, x2_mesh, y_hat_mesh, cmap=cm_decision)

    ax.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright)

    ax.set_xlim([np.min(x[:, 0]), np.max(x[:, 0])])
    ax.set_ylim([np.min(x[:, 1]), np.max(x[:, 1])])
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--dataset', type=str, default="half_moons", help="lin, xor, half_moons")
    parser.add_argument('--steps', type=int, default=500, help="Number of training steps")
    args = parser.parse_args()

    torch.set_num_threads(1)
    np.random.seed(42)
    plt.ion()

    if args.dataset == "lin":
        n: int = 200
        x_pos = np.random.multivariate_normal([-1, 1], np.array([[0.1, 0], [0, 0.1]]), n)
        x_neg = np.random.multivariate_normal([1, -1], np.array([[0.1, 0], [0, 0.1]]), n)
        x = np.concatenate((x_pos, x_neg), axis=0)
        y = np.array([1] * n + [0] * n)
    elif args.dataset == "xor":
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
    elif args.dataset == "half_moons":
        n: int = 200
        x, y = make_moons(n_samples=n, noise=0.1)
    else:
        raise ValueError("Unknown dataset %s" % args.dataset)

    fig, ax = plt.subplots(1, 1)
    ax.set(adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])

    mesh_size: int = 100
    x1_vals_contour = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), mesh_size)
    x2_vals_contour = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), mesh_size)

    x1_mesh, x2_mesh = np.meshgrid(x1_vals_contour, x2_vals_contour)
    mesh_points = np.stack((x1_mesh.reshape(mesh_size * mesh_size), x2_mesh.reshape(mesh_size * mesh_size)), axis=1)

    nnet = NNet()
    nnet.eval()
    update_decision_boundary(ax, x, y, x1_mesh, x2_mesh, mesh_points, mesh_size, nnet)

    plt.pause(0.5)

    criterion = nn.BCELoss()
    optimizer: Optimizer = optim.Adam(nnet.parameters(), lr=args.lr, weight_decay=0.0)
    for i in range(args.steps):
        # plot
        nnet.eval()
        update_decision_boundary(ax, x, y, x1_mesh, x2_mesh, mesh_points, mesh_size, nnet)

        # forward (train)
        nnet.train()
        optimizer.zero_grad()

        y_hat = nnet(torch.tensor(x))

        # loss
        loss = criterion(y_hat[:, 0], torch.tensor(y).float())

        # backwards
        loss.backward()

        # step
        optimizer.step()
        print("Itrs: %i, Train: %.2E" % (i, loss.item()))

        plt.pause(0.01)
        plt.draw()

    plt.show(block=True)


if __name__ == "__main__":
    main()
