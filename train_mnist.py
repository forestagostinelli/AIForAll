import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import pickle
import time
from argparse import ArgumentParser

from examples.train_mnist_utils import train_nnet, get_nnet_lin


def get_nnet() -> nn.Module:
    class NNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=(3, 3)),
                nn.ReLU(),
                torch.nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(1014, 10),
                nn.LogSoftmax(dim=-1),
            )

        def forward(self, x):
            x = self.model(x.float())

            return x

    return NNet()


def evaluate_nnet(nnet: nn.Module, data_input_np, data_labels_np):
    nnet.eval()
    criterion = nn.CrossEntropyLoss()

    val_input = torch.tensor(data_input_np).float()
    val_labels = torch.tensor(data_labels_np).long()
    nnet_output: Tensor = nnet(val_input).detach()

    loss = criterion(nnet_output, val_labels)

    nnet_label = np.argmax(nnet_output.data.numpy(), axis=1)
    acc: float = 100 * np.mean(nnet_label == val_labels.data.numpy())

    return loss.item(), acc


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--save', type=str, default=None, help="")
    parser.add_argument('--lin', action='store_true', default=False, help="")
    args = parser.parse_args()

    # parse data
    train_input_np, train_labels_np = pickle.load(open("data/mnist/mnist_train.pkl", "rb"))
    # train_input_np = np.concatenate((train_input_np, np.rot90(train_input_np, k=2, axes=(1,2))), axis=0)
    train_input_np = train_input_np.reshape(-1, 1, 28, 28)
    # train_input_np = train_input_np.reshape(-1, 28 * 28)

    # train_labels_np = train_labels_np[rand_idxs]
    # train_labels_np = np.concatenate((train_labels_np, train_labels_np), axis=0)

    val_input_np, val_labels_np = pickle.load(open("data/mnist/mnist_val.pkl", "rb"))
    # val_input_np = np.concatenate((val_input_np, np.rot90(val_input_np, k=2, axes=(1,2))), axis=0)
    val_input_np = val_input_np.reshape(-1, 1, 28, 28)
    # val_input_np = val_input_np.reshape(-1, 28 * 28)
    # val_labels_np = np.concatenate((val_labels_np, val_labels_np), axis=0)

    print(f"Training input shape: {train_input_np.shape}, Validation data shape: {val_input_np.shape}")

    # get nnet
    start_time = time.time()
    if args.lin:
        nnet = get_nnet_lin()
    else:
        nnet = get_nnet()
    train_nnet(nnet, train_input_np, train_labels_np, val_input_np, val_labels_np)
    loss, acc = evaluate_nnet(nnet, val_input_np, val_labels_np)
    print(f"Loss: %.5f, Accuracy: %.2f%%, Time: %.2f seconds" % (loss, acc, time.time() - start_time))

    if args.save is not None:
        torch.save(nnet.state_dict(), args.save)


if __name__ == "__main__":
    main()
