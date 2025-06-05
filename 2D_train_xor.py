from typing import List
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from argparse import ArgumentParser

from utils.log_reg_utils import get_xor_data, update_decision_boundary


class NNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid(),
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x.float())

        return x


class NNet1L(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x.float())

        return x


def update_all_decision_boundaries(axs, x, y, x1_mesh, x2_mesh, mesh_points, mesh_size, nnet, nnet_1l):
    x_hid_l: List = []
    for neuron_idx in range(2):
        neuron_idx_p1: int = neuron_idx + 1
        nnet_1l.state_dict()['model.0.bias'].copy_(nnet.state_dict()['model.0.bias'][neuron_idx:neuron_idx_p1])
        nnet_1l.state_dict()['model.0.weight'].copy_(nnet.state_dict()['model.0.weight'][neuron_idx:neuron_idx_p1])
        update_decision_boundary(axs[neuron_idx], x, y, x1_mesh, x2_mesh, mesh_points, mesh_size, nnet_1l)

        y_hat_hid = nnet_1l(torch.tensor(x)).detach().numpy()
        x_hid_l.append(y_hat_hid)

    nnet_1l.state_dict()['model.0.bias'].copy_(nnet.state_dict()['model.2.bias'])
    nnet_1l.state_dict()['model.0.weight'].copy_(nnet.state_dict()['model.2.weight'])
    x_hid = np.concatenate(x_hid_l, axis=1)
    update_decision_boundary(axs[2], x_hid, y, x1_mesh, x2_mesh, mesh_points, mesh_size, nnet_1l,
                             xlim=(np.min(x[:, 0]), np.max(x[:, 0])), ylim=(np.min(x[:, 1]), np.max(x[:, 1])))
    update_decision_boundary(axs[3], x, y, x1_mesh, x2_mesh, mesh_points, mesh_size, nnet)


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--steps', type=int, default=500, help="Number of training steps")
    args = parser.parse_args()

    torch.set_num_threads(1)
    np.random.seed(42)
    plt.ion()

    x, y = get_xor_data()

    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()
    for ax in axs:
        ax.set(adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].set_title("Hidden Layer, Neuron 0")
    axs[1].set_title("Hidden Layer, Neuron 1")
    axs[2].set_title("Output with Hidden Layer Inputs")
    axs[3].set_title("Output")
    # fig.tight_layout()

    mesh_size: int = 100
    x1_vals_contour = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), mesh_size)
    x2_vals_contour = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), mesh_size)

    x1_mesh, x2_mesh = np.meshgrid(x1_vals_contour, x2_vals_contour)
    mesh_points = np.stack((x1_mesh.reshape(mesh_size * mesh_size), x2_mesh.reshape(mesh_size * mesh_size)), axis=1)

    nnet = NNet()
    nnet_1l = NNet1L()
    nnet.eval()
    update_all_decision_boundaries(axs, x, y, x1_mesh, x2_mesh, mesh_points, mesh_size, nnet, nnet_1l)

    plt.pause(0.5)

    criterion = nn.BCELoss()
    optimizer: Optimizer = optim.Adam(nnet.parameters(), lr=args.lr, weight_decay=0.0)
    for i in range(args.steps):
        # plot
        nnet.eval()
        update_all_decision_boundaries(axs, x, y, x1_mesh, x2_mesh, mesh_points, mesh_size, nnet, nnet_1l)

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
