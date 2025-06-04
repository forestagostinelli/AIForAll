from typing import Any, List
import torch
import torch.nn as nn
import numpy as np
from torch.optim.optimizer import Optimizer
import torch.optim as optim
from torch import Tensor
import time
from argparse import ArgumentParser
import pickle
import os
import matplotlib.pyplot as plt


def evaluate_nnet(nnet: nn.Module, val_input_np, val_labels_np, fig, axs, cvae: bool):
    nnet.eval()
    criterion = nn.MSELoss()

    val_input = torch.tensor(val_input_np).float()
    val_labels = torch.tensor(val_labels_np).float()
    if cvae:
        ae_input = torch.cat((val_input, torch.unsqueeze(val_labels, 1)), dim=1)
    else:
        ae_input = val_input

    nnet_output: Tensor = nnet(ae_input)

    loss = criterion(nnet_output, val_input)

    plt_idxs = np.linspace(0, val_input.shape[0] - 1, axs.shape[1]).astype(int)
    for plot_num in range(axs.shape[1]):
        ax_in, ax_out = axs[:, plot_num]
        for ax in [ax_in, ax_out]:
            ax.cla()

        plot_idx: int = plt_idxs[plot_num]
        ax_in.imshow(val_input[plot_idx, :].reshape((28, 28)), cmap="gray")
        ax_out.imshow(nnet_output.cpu().data.numpy()[plot_idx, :].reshape((28, 28)), cmap="gray")

    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()
    plt.pause(0.01)

    return loss.item()


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

    def __init__(self, input_dim: int, dims: List[int], acts: List[str]):
        super().__init__()
        self.layers: nn.ModuleList[nn.ModuleList] = nn.ModuleList()

        # layers
        for dim, act in zip(dims, acts):
            module_list = nn.ModuleList()

            # linear
            linear_layer = nn.Linear(input_dim, dim)
            module_list.append(linear_layer)

            # activation
            if act.upper() != "LINEAR":
                module_list.append(get_act_fn(act))

            self.layers.append(module_list)

            input_dim = dim

    def forward(self, x):
        x = x.float()

        module_list: nn.ModuleList
        for module_list in self.layers:
            for module in module_list:
                x = module(x)

        return x


def get_encoder() -> nn.Module:
    class NNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = FullyConnectedModel(784, [400, 100, 2], ["relu", "relu", "linear"])

        def forward(self, x):
            x = self.fc(x)

            return x

    return NNet()


class VAE(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = FullyConnectedModel(input_dim, [400, 100, 2 * 2], ["relu", "relu", "linear"])
        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = self.fc(x)
        mu = x[:, :2]
        logvar = x[:, 2:]
        sigma = torch.exp(logvar / 2.0)
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        return z


class Decoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = FullyConnectedModel(input_dim, [100, 400, 784], ["relu", "relu", "linear"])

    def forward(self, x):
        x = self.fc(x)

        return x


def get_encoder_variational(cvae: bool) -> nn.Module:
    if cvae:
        return VAE(785)
    else:
        return VAE(784)


def get_decoder(cvae: bool) -> nn.Module:
    if cvae:
        return Decoder(3)
    else:
        return Decoder(2)


class Autoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class AutoencoderCond(Autoencoder):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__(encoder, decoder)

    def forward(self, x):
        labels = x[:, -1:]
        x = self.encoder(x)
        x = self.decoder(torch.cat((x, labels), dim=1))

        return x


def get_ae(encoder: nn.Module, decoder: nn.Module, cvae: bool) -> Autoencoder:
    if cvae:
        return AutoencoderCond(encoder, decoder)
    else:
        return Autoencoder(encoder, decoder)


def train_nnet(nnet: Autoencoder, train_input_np: np.ndarray, train_labels_np: np.ndarray, val_input_np: np.ndarray,
               val_labels_np: np.ndarray, fig, axs, vae: bool, cvae: bool) -> nn.Module:
    # optimization
    train_itr: int = 0
    batch_size: int = 200
    num_itrs: int = 10000
    if vae:
        kl_weight: float = 0.005
    else:
        kl_weight: float = 0.01

    display_itrs = 100
    criterion = nn.MSELoss()
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
        data_labels_b = torch.tensor(train_labels_np[batch_idxs]).float()

        # forward
        if cvae:
            ae_input = torch.cat((data_input_b, torch.unsqueeze(data_labels_b, 1)), dim=1)
        else:
            ae_input = data_input_b

        nnet_output_b: Tensor = nnet(ae_input)
        # cost
        loss_recon = criterion(nnet_output_b, data_input_b)
        loss_kl = loss_recon * 0
        if vae or cvae:
            loss_kl = nnet.encoder.kl
        loss = loss_recon + kl_weight * loss_kl

        # backwards
        loss.backward()

        # step
        optimizer.step()

        # display progress
        if train_itr % display_itrs == 0:
            nnet.eval()

            loss_val = evaluate_nnet(nnet, val_input_np, val_labels_np, fig, axs, cvae)

            nnet.train()

            print("Itr: %i, lr: %.2E, loss_recon: %.2E, loss_kl: %.2E, loss_val: %.2E, Time: %.2f" % (
                train_itr, lr_itr, loss_recon.item(), loss_kl.item(), loss_val, time.time() - start_time))

            start_time = time.time()

        train_itr = train_itr + 1

    return nnet


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=None, help="")
    parser.add_argument('--vae', action='store_true', default=False, help="")
    parser.add_argument('--cvae', action='store_true', default=False, help="")
    args = parser.parse_args()
    plt.ion()

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # parse data
    train_input_np, train_labels_np = pickle.load(open("data/mnist/mnist_train.pkl", "rb"))
    train_input_np = train_input_np.reshape(-1, 28 * 28)

    val_input_np, val_labels_np = pickle.load(open("data/mnist/mnist_val.pkl", "rb"))
    val_input_np = val_input_np.reshape(-1, 28 * 28)

    print(f"Training input shape: {train_input_np.shape}, Validation data shape: {val_input_np.shape}")
    fig, axs = plt.subplots(2, 3)
    fig.show()
    plt.pause(0.01)
    breakpoint()

    # get nnet
    start_time = time.time()
    if args.vae or args.cvae:
        encoder: nn.Module = get_encoder_variational(args.cvae)
        decoder: nn.Module = get_decoder(args.cvae)
    else:
        encoder: nn.Module = get_encoder()
        decoder: nn.Module = get_decoder(args.cvae)

    ae: Autoencoder = get_ae(encoder, decoder, args.cvae)

    train_nnet(ae, train_input_np, train_labels_np, val_input_np, val_labels_np, fig, axs, args.vae, args.cvae)
    loss = evaluate_nnet(ae, val_input_np, val_labels_np, fig, axs, args.cvae)
    print(f"Loss: %.5f, Time: %.2f seconds" % (loss, time.time() - start_time))

    torch.save(encoder.state_dict(), f"{args.save_dir}/encoder.pt")
    torch.save(decoder.state_dict(), f"{args.save_dir}/decoder.pt")


if __name__ == "__main__":
    main()
