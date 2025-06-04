from typing import Any, List
import torch
import torch.nn as nn
import numpy as np
from torch.optim.optimizer import Optimizer
import torch.optim as optim
from torch import Tensor
import time


def evaluate_nnet(nnet: nn.Module, data_input_np, data_labels_np):
    nnet.eval()
    criterion = nn.CrossEntropyLoss()

    val_input = torch.tensor(data_input_np).float()
    val_labels = torch.tensor(data_labels_np).long()
    nnet_output: Tensor = nnet(val_input)

    loss = criterion(nnet_output, val_labels)

    nnet_label = np.argmax(nnet_output.data.numpy(), axis=1)
    acc: float = 100 * np.mean(nnet_label == val_labels.data.numpy())

    return loss.item(), acc


def get_act_fn(act: str):
    act = act.upper()
    if act == "RELU":
        act_fn = nn.ReLU()
    elif act == "SIGMOID":
        act_fn = nn.Sigmoid()
    elif act == "TANH":
        act_fn = nn.Tanh()
    else:
        raise ValueError("Un-defined activation type %s" % act)

    return act_fn


class FullyConnectedModel(nn.Module):
    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, input_dim: int, dims: List[int], acts: List[str], dropouts: List[float]):
        super().__init__()
        self.layers: nn.ModuleList[nn.ModuleList] = nn.ModuleList()

        # layers
        for dim, act, dropout in zip(dims, acts, dropouts):
            module_list = nn.ModuleList()

            # linear
            linear_layer = nn.Linear(input_dim, dim)
            module_list.append(linear_layer)

            # activation
            if act.upper() != "LINEAR":
                module_list.append(get_act_fn(act))

            # dropout
            if dropout > 0.0:
                module_list.append(nn.Dropout(dropout))

            self.layers.append(module_list)

            input_dim = dim

    def forward(self, x):
        x = x.float()

        module_list: nn.ModuleList
        for module_list in self.layers:
            for module in module_list:
                x = module(x)

        return x


def get_nnet_lin() -> nn.Module:
    class NNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(784, 10)

        def forward(self, x):
            x = self.lin(x)
            return x

    return NNet()


def train_nnet(nnet: nn.Module, train_input_np: np.ndarray, train_labels_np: np.array, val_input_np: np.ndarray,
               val_labels_np: np.array) -> nn.Module:
    """

    :param nnet: neural network to train
    :param train_input_np: training inputs
    :param train_labels_np: training labels
    :param val_input_np: validation inputs
    :param val_labels_np: validation labels
    :return: trained neural network
    """

    # optimization
    train_itr: int = 0
    batch_size: int = 100
    num_itrs: int = 10000

    display_itrs = 100
    criterion = nn.CrossEntropyLoss()
    lr: float = 0.001
    lr_d: float = 0.99996
    optimizer: Optimizer = optim.Adam(nnet.parameters(), lr=lr)
    # optimizer: Optimizer = optim.SGD(nnet.parameters(), lr=lr, momentum=0.9)

    # initialize status tracking
    start_time = time.time()

    nnet.train()
    max_itrs: int = train_itr + num_itrs

    while train_itr < max_itrs:
        # zero the parameter gradients
        optimizer.zero_grad()
        lr_itr: float = lr * (lr_d ** train_itr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_itr

        # get data
        batch_idxs = np.random.randint(0, train_input_np.shape[0], size=batch_size)
        data_input_b = torch.tensor(train_input_np[batch_idxs]).float()
        data_labels_b = torch.tensor(train_labels_np[batch_idxs]).long()

        # forward
        nnet_output_b: Tensor = nnet(data_input_b)

        # cost
        loss = criterion(nnet_output_b, data_labels_b)

        # backwards
        loss.backward()

        # step
        optimizer.step()

        # display progress
        if train_itr % display_itrs == 0:
            nnet_label = np.argmax(nnet_output_b.data.numpy(), axis=1)
            acc: float = 100 * np.mean(nnet_label == data_labels_b.data.numpy())

            nnet.eval()
            loss_val, acc_val = evaluate_nnet(nnet, val_input_np, val_labels_np)
            nnet.train()

            print("Itr: %i, lr: %.2E, loss: %.2E, acc: %.2f, loss_val: %.2E, acc_val: %.2f, Time: %.2f" % (
                      train_itr, lr_itr, loss.item(), acc, loss_val, acc_val, time.time() - start_time))

            start_time = time.time()

        train_itr = train_itr + 1

    return nnet
